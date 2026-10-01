"""Metadata and feature boundary for the SVM pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from vn_tsc.data.dataset import TrafficSignDataset, load_class_table
from vn_tsc.pipelines.svm.dev_features import extract_dev_features
from vn_tsc.utils.io import load_json

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
    common_class_ids: list[int] | None = None
    feature_report: dict[str, Any] | None = None


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
    return {"backend": source.get("backend", "dev"),
            "input_size": source.get("input_size"), **{
        name: dict(source[name]) for name in ("hog", "lbp", "color_hist")
    }}


def _check_shared_metadata(dataset: TrafficSignDataset, labels: dict[str, np.ndarray],
                           class_table) -> None:
    """Check that metadata array_index and labels describe the same crop rows."""
    rows_by_split = {split: [] for split in (*SPLITS, "train_aug")}
    seen_ids: set[str] = set()
    for sample in dataset.index:
        if sample.split not in rows_by_split:
            raise ValueError(f"Unknown shared metadata split: {sample.split}")
        if sample.crop_id in seen_ids:
            raise ValueError(f"Duplicate crop_id in shared metadata: {sample.crop_id}")
        seen_ids.add(sample.crop_id)
        rows_by_split[sample.split].append(sample)
    for split, rows in rows_by_split.items():
        y = labels.get(split)
        if y is None:
            if rows:
                raise ValueError(f"Shared metadata contains {split} rows but its label array is missing")
            continue
        if len(rows) != len(y):
            raise ValueError(f"Shared metadata/array count mismatch for {split}: {len(rows)} vs {len(y)}")
        indexes = [sample.array_index for sample in rows]
        if sorted(indexes) != list(range(len(y))):
            raise ValueError(f"Shared metadata array_index must cover 0..{len(y) - 1} for {split}")
        for sample in rows:
            if sample.class_id != int(y[sample.array_index]):
                raise ValueError(f"Shared metadata label mismatch for {sample.crop_id} in {split}")
            if not 0 <= sample.class_id < len(class_table):
                raise ValueError(f"Unknown class_id {sample.class_id} for {sample.crop_id}")
            if sample.sign_code != class_table[sample.class_id].sign_code:
                raise ValueError(f"Shared metadata sign_code mismatch for {sample.crop_id}")
            if bool(sample.is_aug) != (split == "train_aug"):
                raise ValueError(f"Shared metadata is_aug mismatch for {sample.crop_id}")


def _shared_feature_slices(report: dict[str, Any],
                           feature_config: dict[str, Any]) -> list[slice]:
    """Select precomputed blocks for SVM ablations without re-extracting features."""
    if report.get("input_size") != feature_config["input_size"]:
        raise ValueError("Shared feature store was built with a different input_size; "
                         "rerun python -m tools.run_preprocess --skip-crops")
    stored = report.get("config", {})
    blocks = report.get("layout", {}).get("blocks", {})
    selected = []
    for name in ("hog", "lbp", "color_hist"):
        requested = feature_config[name]
        if not requested.get("enabled", False):
            continue
        source = stored.get(name, {})
        if (not source.get("enabled", False) or
                {key: value for key, value in source.items() if key != "enabled"} !=
                {key: value for key, value in requested.items() if key != "enabled"}):
            raise ValueError(f"Shared feature store has different {name} settings; "
                             "rerun python -m tools.run_preprocess --skip-crops")
        if name not in blocks:
            raise ValueError(f"Shared feature block missing from feature_report.json: {name}")
        block = blocks[name]
        selected.append(slice(int(block["start"]), int(block["end"])))
    if not selected:
        raise ValueError("At least one HOG/LBP/color feature must be enabled")
    return selected


def _select_shared_features(X: np.ndarray, columns: list[slice]) -> np.ndarray:
    if len(columns) == 1:
        return X[:, columns[0]]
    if columns[0].start == 0 and columns[-1].stop == X.shape[1] and all(
            left.stop == right.start for left, right in pairwise(columns)):
        return X
    return np.concatenate([X[:, block] for block in columns], axis=1)


def _load_shared_data(cfg: dict[str, Any], root: Path,
                      feature_config: dict[str, Any]) -> SVMDataBundle:
    metadata_path = root / "metadata.csv"
    class_table_path = root / "class_table.csv"
    report_path = root / "features" / "feature_report.json"
    for path in (metadata_path, class_table_path, report_path):
        if not path.is_file():
            raise FileNotFoundError(f"Shared preprocessing output missing: {path}. "
                                    "Run python -m tools.run_preprocess first.")
    table = load_class_table(root)
    report = load_json(report_path)
    selected_columns = _shared_feature_slices(report, feature_config)
    with_aug = bool(cfg.get("train", {}).get("with_aug", True))
    aug_features = root / "features" / "F_train_aug.npy"
    aug_labels = root / "images" / "y_train_aug.npy"
    if with_aug and (not aug_features.is_file() or not aug_labels.is_file()):
        raise FileNotFoundError("train.with_aug=true but shared train augmentation is missing. "
                                "Rebuild preprocessing or set train.with_aug=false for an ablation.")
    dataset = TrafficSignDataset.from_processed(root)
    labels = {}
    for split in SPLITS:
        path = root / "images" / f"y_{split}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Shared label array missing: {path}")
        labels[split] = np.load(path, mmap_mode="r", allow_pickle=False)
    if aug_labels.is_file():
        labels["train_aug"] = np.load(aug_labels, mmap_mode="r", allow_pickle=False)
        if report.get("rows", {}).get("train_aug") != len(labels["train_aug"]):
            raise ValueError("Shared train_aug row count differs from feature_report.json")
    _check_shared_metadata(dataset, labels, table)
    arrays = {}
    feature_dim = None
    for split in SPLITS:
        feature_path = root / "features" / f"F_{split}.npy"
        if not feature_path.is_file():
            raise FileNotFoundError(f"Shared feature array missing: {feature_path}")
        X = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        y = labels[split]
        if split == "train" and with_aug:
            X_aug = np.load(aug_features, mmap_mode="r", allow_pickle=False)
            y_aug = labels["train_aug"]
            if X_aug.ndim != 2 or len(X_aug) != len(y_aug) or X_aug.shape[1] != X.shape[1]:
                raise ValueError("Shared train augmentation feature/label arrays do not match")
            X = np.concatenate([X, X_aug])
            y = np.concatenate([y, y_aug])
        if X.ndim != 2 or y.ndim != 1 or len(X) == 0 or len(X) != len(y):
            raise ValueError(f"Invalid or empty shared {split} feature/label arrays")
        if X.shape[1] != report.get("layout", {}).get("total"):
            raise ValueError(f"Shared {split} feature dimension differs from feature_report.json")
        if report.get("rows", {}).get(split) != len(labels[split]):
            raise ValueError(f"Shared {split} row count differs from feature_report.json")
        if not np.isfinite(X).all() or not np.isin(y, table.ids).all():
            raise ValueError(f"Shared {split} features contain non-finite values or unknown labels")
        X = _select_shared_features(X, selected_columns)
        if feature_dim is None:
            feature_dim = X.shape[1]
        elif X.shape[1] != feature_dim:
            raise ValueError(f"Selected shared feature dimensions differ in {split}")
        arrays[f"X_{split}"] = X
        arrays[f"y_{split}"] = y
    if len(np.unique(arrays["y_train"])) < 2:
        raise ValueError("Shared training split must contain at least two classes")
    if set(np.unique(arrays["y_train"])) != set(table.ids):
        raise ValueError("Shared training split must contain every class in class_table.csv")
    threshold = int(cfg.get("metrics", {}).get("rare_class_threshold", 30))
    real_counts = {class_id: 0 for class_id in table.ids}
    for sample in dataset.index:
        if not sample.is_aug:
            real_counts[sample.class_id] += 1
    common_ids = [class_id for class_id in table.ids if real_counts[class_id] >= threshold]
    log.info("Loaded shared feature store: train=%s val=%s test=%s dimension=%s augmentation=%s",
             *(arrays[f"X_{split}"].shape[0] for split in SPLITS), feature_dim, with_aug)
    return SVMDataBundle(
        **arrays, class_names=table.labels,
        label_map={name: index for index, name in enumerate(table.labels)},
        feature_config=feature_config, processed_root=root,
        common_class_ids=common_ids, feature_report=report,
    )


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
    if source_type != backend:
        raise ValueError(f"data_source.type={source_type} must match features.backend={backend}")
    if backend == "shared":
        return _load_shared_data(cfg, root, feature_config)
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
