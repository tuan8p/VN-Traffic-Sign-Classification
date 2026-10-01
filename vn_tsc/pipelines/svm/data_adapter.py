"""Metadata and feature boundary for the SVM pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from vn_tsc.pipelines.svm.dev_features import extract_dev_features

log = logging.getLogger(__name__)
SPLITS = ("train", "val", "test")
REQUIRED_COLUMNS = {"sample_id", "processed_path", "class_name", "class_id", "split"}
CACHE_VERSION = 1


@dataclass
class SVMDataBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    class_names: list[str]
    label_map: dict[str, int]
    feature_config: dict[str, Any]
    processed_root: Path


@dataclass(frozen=True)
class _Sample:
    sample_id: str
    image_path: Path
    class_name: str
    class_id: int
    split: str


def feature_config_from_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Snapshot feature settings, including the backend, for caching and inference."""
    source = cfg["features"]
    return {"backend": source.get("backend", "dev"), **{
        name: dict(source[name]) for name in ("hog", "lbp", "color_hist")
    }}


def _read_metadata(root: Path) -> tuple[list[_Sample], dict[str, int]]:
    metadata_path = root / "metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Processed metadata not found at {metadata_path}. "
            "Run temporary SVM preprocessing first."
        )
    with metadata_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Metadata missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    samples = []
    label_map: dict[str, int] = {}
    ids: dict[int, str] = {}
    sample_ids = set()
    for line, row in enumerate(rows, start=2):
        sample_id = (row["sample_id"] or "").strip()
        class_name = (row["class_name"] or "").strip()
        path_value = (row["processed_path"] or "").strip()
        split = (row["split"] or "").strip()
        if not sample_id or not class_name or not path_value:
            raise ValueError(f"Metadata row {line} has an empty required field")
        if sample_id in sample_ids:
            raise ValueError(f"Duplicate sample_id in metadata: {sample_id}")
        sample_ids.add(sample_id)
        if split not in SPLITS:
            raise ValueError(f"Invalid split {split!r} for sample {sample_id}")
        try:
            class_id = int(row["class_id"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid class_id for sample {sample_id}") from exc
        if (class_name in label_map and label_map[class_name] != class_id) or (
                class_id in ids and ids[class_id] != class_name):
            raise ValueError(f"Inconsistent class_name/class_id for sample {sample_id}")
        label_map[class_name] = class_id
        ids[class_id] = class_name
        path = Path(path_value)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise FileNotFoundError(f"Processed image missing for sample {sample_id}: {path}")
        samples.append(_Sample(sample_id, path, class_name, class_id, split))
    if sorted(ids) != list(range(len(ids))):
        raise ValueError("class_id values must be contiguous integers starting at 0")
    for split in SPLITS:
        if not any(sample.split == split for sample in samples):
            raise ValueError(f"Processed metadata has an empty {split} split")
    if len({sample.class_id for sample in samples if sample.split == "train"}) < 2:
        raise ValueError("Training split must contain at least two classes")
    return samples, label_map


def _fingerprint(root: Path, samples: list[_Sample]) -> str:
    digest = hashlib.sha256((root / "metadata.csv").read_bytes())
    for sample in samples:
        stat = sample.image_path.stat()
        digest.update(f"{sample.image_path}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def _load_cache(cache_dir: Path, manifest: dict[str, Any], root: Path,
                label_map: dict[str, int], feature_config: dict[str, Any]) -> SVMDataBundle | None:
    try:
        with (cache_dir / "feature_config.json").open(encoding="utf-8") as file:
            if json.load(file) != manifest:
                return None
        arrays = {name: np.load(cache_dir / f"{name}.npy", allow_pickle=False)
                  for split in SPLITS for name in (f"X_{split}", f"y_{split}")}
        dimensions = {arrays[f"X_{split}"].shape[1] for split in SPLITS}
        if len(dimensions) != 1 or any(arrays[f"X_{split}"].ndim != 2 for split in SPLITS):
            return None
        if any(arrays[f"X_{split}"].shape[0] != arrays[f"y_{split}"].shape[0] for split in SPLITS):
            return None
    except (OSError, ValueError, KeyError, IndexError, json.JSONDecodeError):
        return None
    log.info("Loaded SVM feature cache from %s", cache_dir)
    return SVMDataBundle(**arrays, class_names=[name for name, _ in sorted(label_map.items(), key=lambda item: item[1])],
                         label_map=label_map, feature_config=feature_config, processed_root=root)


def _extract(samples: list[_Sample], extractor: Callable[[np.ndarray, dict[str, Any]], np.ndarray],
             feature_config: dict[str, Any], label_map: dict[str, int], root: Path) -> SVMDataBundle:
    arrays: dict[str, np.ndarray] = {}
    expected_dim = None
    for split in SPLITS:
        vectors = []
        labels = []
        selected = [sample for sample in samples if sample.split == split]
        for index, sample in enumerate(selected, start=1):
            try:
                with Image.open(sample.image_path) as image:
                    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
                vector = np.asarray(extractor(rgb, feature_config), dtype=np.float32).ravel()
            except (OSError, ValueError) as exc:
                raise ValueError(f"Cannot extract features for sample {sample.sample_id} at {sample.image_path}: {exc}") from exc
            if vector.size == 0 or not np.all(np.isfinite(vector)):
                raise ValueError(f"Invalid feature vector for sample {sample.sample_id}")
            if expected_dim is None:
                expected_dim = vector.size
            elif vector.size != expected_dim:
                raise ValueError(f"Inconsistent feature dimensions for sample {sample.sample_id}: "
                                 f"expected {expected_dim}, got {vector.size}")
            vectors.append(vector)
            labels.append(sample.class_id)
            if index % 100 == 0:
                log.info("Extracted %s/%s %s samples", index, len(selected), split)
        arrays[f"X_{split}"] = np.stack(vectors).astype(np.float32, copy=False)
        arrays[f"y_{split}"] = np.asarray(labels, dtype=np.int64)
    return SVMDataBundle(**arrays, class_names=[name for name, _ in sorted(label_map.items(), key=lambda item: item[1])],
                         label_map=label_map, feature_config=feature_config, processed_root=root)


def load_svm_data(cfg: dict[str, Any]) -> SVMDataBundle:
    """Read fixed metadata splits and extract or reuse deterministic features."""
    source_type = cfg["data_source"]["type"]
    if source_type not in {"dev", "shared"}:
        raise ValueError(f"Unknown data_source.type: {source_type}")
    root = Path(cfg["data_source"]["processed_root"]).expanduser().resolve()
    feature_config = feature_config_from_cfg(cfg)
    backend = feature_config["backend"]
    if backend == "shared":
        raise NotImplementedError(
            "Shared Role D feature backend is not available yet. "
            "Use features.backend=dev during Role B development."
        )
    if backend != "dev":
        raise ValueError(f"Unknown features.backend: {backend}")
    log.info("Loading SVM metadata from %s", root / "metadata.csv")
    samples, label_map = _read_metadata(root)
    counts = {split: sum(sample.split == split for sample in samples) for split in SPLITS}
    log.info("Metadata samples: train=%s val=%s test=%s", *(counts[split] for split in SPLITS))
    manifest = {"cache_version": CACHE_VERSION, "feature_config": feature_config,
                "data_fingerprint": _fingerprint(root, samples)}
    cache_dir = root / "features"
    if cfg["features"].get("cache", False):
        cached = _load_cache(cache_dir, manifest, root, label_map, feature_config)
        if cached is not None:
            return cached
    bundle = _extract(samples, extract_dev_features, feature_config, label_map, root)
    log.info("SVM feature dimension: %s", bundle.X_train.shape[1])
    if cfg["features"].get("cache", False):
        cache_dir.mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            for prefix in ("X", "y"):
                name = f"{prefix}_{split}"
                np.save(cache_dir / f"{name}.npy", getattr(bundle, name))
        (cache_dir / "feature_config.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return bundle
