"""Temporary development preprocessing for the SVM pipeline.

This module exists only so Role B can develop independently before the
shared preprocessing pipeline from Role D is available.

Do not use this module as the final fair-comparison preprocessing.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageOps

from vn_tsc.config.resolve import resolve_config

log = logging.getLogger(__name__)
SPLITS = ("train", "val", "test")
METADATA_COLUMNS = (
    "sample_id", "source_path", "processed_path", "class_name", "class_id",
    "split", "original_width", "original_height", "processed_width", "processed_height",
)


@dataclass(frozen=True)
class RawImage:
    path: Path
    class_name: str
    width: int
    height: int


def _settings(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate and snapshot only the shared settings used here."""
    size = cfg["preprocess"]["image_size"]
    if len(size) != 2 or any(type(value) is not int or value <= 0 for value in size):
        raise ValueError("preprocess.image_size must contain two positive integers [width, height]")
    color_mode = str(cfg["preprocess"]["color_mode"]).lower()
    if color_mode != "rgb":
        raise ValueError("Temporary SVM preprocessing requires preprocess.color_mode: rgb")
    keep_aspect = cfg["preprocess"]["keep_aspect"]
    if type(keep_aspect) is not bool:
        raise ValueError("preprocess.keep_aspect must be a boolean")
    strategy = cfg["split"]["strategy"]
    if strategy != "stratified":
        raise ValueError("Temporary SVM preprocessing supports only split.strategy: stratified")
    ratios = {split: float(cfg["split"]["ratios"][split]) for split in SPLITS}
    if any(not math.isfinite(value) or value <= 0 for value in ratios.values()):
        raise ValueError("All train/val/test split ratios must be positive and finite")
    if not math.isclose(sum(ratios.values()), 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0; got {sum(ratios.values()):.6f}")
    extensions = sorted({str(ext).lower().lstrip(".") for ext in cfg["data"]["image_extensions"]})
    if not extensions or any(not ext for ext in extensions):
        raise ValueError("data.image_extensions must contain image extensions")
    seed = cfg["seed"]
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    return {
        "seed": seed,
        "image_size": list(size),
        "color_mode": color_mode,
        "keep_aspect": keep_aspect,
        "split": {"strategy": strategy, "ratios": ratios},
        "image_extensions": extensions,
    }


def discover_images(raw_root: Path, image_extensions: list[str]) -> list[tuple[Path, str]]:
    """Find images below immediate class folders in stable relative-path order."""
    allowed = {f".{ext.lower().lstrip('.')}" for ext in image_extensions}
    paths = []
    for path in raw_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        relative = path.relative_to(raw_root)
        if len(relative.parts) < 2 or relative.parts[0].startswith("."):
            continue
        paths.append((path, relative.parts[0]))
    return sorted(paths, key=lambda item: item[0].relative_to(raw_root).as_posix())


def _valid_images(candidates: list[tuple[Path, str]]) -> tuple[list[RawImage], int]:
    valid = []
    skipped = 0
    for path, class_name in candidates:
        try:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source)
                image.load()
                width, height = image.size
            valid.append(RawImage(path, class_name, width, height))
        except (OSError, ValueError) as exc:
            skipped += 1
            log.warning("Skipping corrupt image %s: %s", path, exc)
    return valid, skipped


def build_label_map(class_names: list[str]) -> dict[str, int]:
    """Assign stable class IDs in alphabetical order."""
    return {name: index for index, name in enumerate(sorted(set(class_names)))}


def resize_with_padding(image: Image.Image, target_size: tuple[int, int], keep_aspect: bool) -> Image.Image:
    """Return an RGB image at target size, with centered black padding if requested."""
    rgb = image.convert("RGB")
    if not keep_aspect:
        return rgb.resize(target_size, Image.Resampling.LANCZOS)
    target_w, target_h = target_size
    scale = min(target_w / rgb.width, target_h / rgb.height)
    new_w = min(target_w, max(1, round(rgb.width * scale)))
    new_h = min(target_h, max(1, round(rgb.height * scale)))
    resized = rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", target_size, (0, 0, 0))
    canvas.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return canvas


def _split_counts(count: int, ratios: dict[str, float]) -> dict[str, int]:
    target = {split: count * ratios[split] for split in SPLITS}
    allocation = {split: max(1, math.floor(target[split])) for split in SPLITS}
    while sum(allocation.values()) < count:
        split = max(SPLITS, key=lambda name: target[name] - allocation[name])
        allocation[split] += 1
    while sum(allocation.values()) > count:
        split = max((name for name in SPLITS if allocation[name] > 1),
                    key=lambda name: allocation[name] - target[name])
        allocation[split] -= 1
    return allocation


def make_stratified_split(images: list[RawImage], ratios: dict[str, float], seed: int) -> list[str]:
    """Assign each image one split, preserving every class in all three splits."""
    by_class: dict[str, list[int]] = {}
    for index, image in enumerate(images):
        by_class.setdefault(image.class_name, []).append(index)
    too_small = {name: len(indexes) for name, indexes in by_class.items() if len(indexes) < len(SPLITS)}
    if too_small:
        details = ", ".join(f"{name} ({count} images)" for name, count in sorted(too_small.items()))
        raise ValueError(f"Each class needs at least 3 valid images for train/val/test: {details}")
    assignments = [""] * len(images)
    rng = random.Random(seed)
    for name in sorted(by_class):
        indexes = by_class[name][:]
        rng.shuffle(indexes)
        allocation = _split_counts(len(indexes), ratios)
        offset = 0
        for split in SPLITS:
            for index in indexes[offset:offset + allocation[split]]:
                assignments[index] = split
            offset += allocation[split]
    return assignments


def _check_output_paths(raw_root: Path, output_root: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    home = Path.home().resolve()
    shared_processed = repo_root / "data" / "processed"
    if output_root == Path(output_root.anchor) or output_root in (home, repo_root):
        raise ValueError(f"Unsafe output root: {output_root}")
    if output_root in repo_root.parents or output_root in home.parents:
        raise ValueError(f"Unsafe output root: {output_root}")
    if (output_root == shared_processed or shared_processed in output_root.parents
            or output_root in shared_processed.parents):
        raise ValueError("Temporary data cannot overwrite data/processed or its parent")
    if output_root == raw_root or output_root in raw_root.parents or raw_root in output_root.parents:
        raise ValueError("Output root must not overlap raw data root")
    if output_root.is_symlink():
        raise ValueError(f"Output root must not be a symlink: {output_root}")


def _write_outputs(
    stage: Path, images: list[RawImage], assignments: list[str],
    label_map: dict[str, int], settings: dict[str, Any], skipped: int,
) -> dict[str, Any]:
    image_dir = stage / "images"
    image_dir.mkdir()
    distribution = {split: {name: 0 for name in label_map} for split in SPLITS}
    with (stage / "metadata.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        for index, (raw, split) in enumerate(zip(images, assignments)):
            sample_id = f"{index:08d}"
            relative_path = Path("images") / f"{sample_id}.png"
            with Image.open(raw.path) as source:
                corrected = ImageOps.exif_transpose(source)
                processed = resize_with_padding(corrected, tuple(settings["image_size"]), settings["keep_aspect"])
                processed.save(stage / relative_path, format="PNG")
            writer.writerow({
                "sample_id": sample_id,
                "source_path": str(raw.path),
                "processed_path": str(relative_path),
                "class_name": raw.class_name,
                "class_id": label_map[raw.class_name],
                "split": split,
                "original_width": raw.width,
                "original_height": raw.height,
                "processed_width": processed.width,
                "processed_height": processed.height,
            })
            distribution[split][raw.class_name] += 1
    split_counts = {split: sum(distribution[split].values()) for split in SPLITS}
    summary = {
        "total_samples": len(images),
        "num_classes": len(label_map),
        "splits": {split: {"count": split_counts[split]} for split in SPLITS},
        "class_distribution": distribution,
        "skipped_images": skipped,
    }
    (stage / "label_map.json").write_text(json.dumps(label_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (stage / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (stage / "preprocessing_config.yaml").write_text(yaml.safe_dump(settings, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return {
        "num_samples": len(images),
        "num_classes": len(label_map),
        "train_count": split_counts["train"],
        "val_count": split_counts["val"],
        "test_count": split_counts["test"],
        "skipped_images": skipped,
    }


def run_dev_preprocess(
    cfg: dict[str, Any], raw_root: str | Path,
    output_root: str | Path = "data/svm_dev_processed", overwrite: bool = False,
) -> dict[str, Any]:
    """Create a disposable processed dataset with stable metadata for SVM development."""
    raw = Path(raw_root).expanduser().resolve()
    out_path = Path(output_root).expanduser()
    if out_path.is_symlink():
        raise ValueError(f"Output root must not be a symlink: {out_path}")
    out = out_path.resolve()
    _check_output_paths(raw, out)
    if not raw.is_dir():
        raise FileNotFoundError(f"Raw data root does not exist: {raw}")
    if out.exists() and not out.is_dir():
        raise ValueError(f"Output root is not a directory: {out}")
    if out.exists() and not overwrite:
        raise FileExistsError(f"Output root already exists: {out}; pass --overwrite to replace it")
    settings = _settings(cfg)
    candidates = discover_images(raw, settings["image_extensions"])
    images, skipped = _valid_images(candidates)
    if not images:
        raise ValueError(f"No valid class-folder images found under {raw}; skipped {skipped} corrupt images")
    label_map = build_label_map([image.class_name for image in images])
    assignments = make_stratified_split(images, settings["split"]["ratios"], settings["seed"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".svm-dev-", dir=out.parent) as temp_dir:
        stage = Path(temp_dir)
        summary = _write_outputs(stage, images, assignments, label_map, settings, skipped)
        if out.exists():
            for child in out.iterdir():
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            for child in stage.iterdir():
                shutil.move(str(child), str(out / child.name))
        else:
            stage.rename(out)
    log.info("SVM dev preprocessing complete: %s; skipped corrupt images: %s; output: %s",
             summary["num_samples"], skipped, out)
    return summary


def main(argv: list[str] | None = None) -> None:
    """Run temporary SVM preprocessing from the command line."""
    parser = argparse.ArgumentParser(description="Temporary SVM development preprocessing")
    parser.add_argument("--data-root", required=True, help="Raw data root with immediate class folders")
    parser.add_argument("--output-root", default="data/svm_dev_processed")
    parser.add_argument("--shared", default="configs/shared.yaml")
    parser.add_argument("--pipeline-config", default="configs/pipelines/svm.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = resolve_config(pipeline_yaml=args.pipeline_config, shared_yaml=args.shared)
    run_dev_preprocess(cfg, args.data_root, args.output_root, args.overwrite)


if __name__ == "__main__":
    main()
