from __future__ import annotations
import csv
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from vn_tsc.data.augment import augment_from_source, plan_assignments, plan_augmentation
from vn_tsc.data.classes import ClassTable
from vn_tsc.data.crops import box_geometry, crop_box, imread_rgb
from vn_tsc.data.discover import discover_yolo_dataset, find_dataset_root
from vn_tsc.data.grouping import group_source_images, leakage_against
from vn_tsc.data.splitting import stratified_group_split
from vn_tsc.utils.io import save_json, save_yaml
from vn_tsc.utils.logging_utils import setup_logger
from vn_tsc.utils.seed import set_seed

log = setup_logger("preprocess_offline")

SPLITS = ("train", "val", "test")

METADATA_COLUMNS = [
    "crop_id", "split", "array_index", "class_id", "sign_code",
    "source_image", "group_id",
    "x1", "y1", "x2", "y2", "box_w_px", "box_h_px", "box_area_px",
    "img_w", "img_h", "clipped", "is_aug", "aug_source_crop_id", "aug_ops",
]


def _cfg(cfg: dict[str, Any], *path: str, default: Any = None) -> Any:
    node: Any = cfg
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def run_offline_preprocess(
    cfg: dict[str, Any], data_root: str | Path, out_root: str | Path
) -> Path:
    """Shared offline preprocess for ALL pipelines. No W&B.

    VNTS ships as YOLO detection data, but this project is classification, so
    every bounding box becomes one labelled sample. The three pipelines then
    read the same arrays, which is what keeps the comparison fair.

    Writes to out_root:
        images/X_{split}.npy, y_{split}.npy   uint8 HWC crops + int16 labels
        images/X_train_aug.npy, y_train_aug.npy   synthetic extras only
        metadata.csv            one row per crop, joins to the arrays by index
        class_table.csv         copy of the class list actually used
        preprocess_report.json  validation, grouping, split and leakage numbers
        resolved_preprocess.yaml
    """
    t0 = time.time()
    data_root, out_root = Path(data_root), Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "images").mkdir(exist_ok=True)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    table = ClassTable.from_config(cfg)
    num_classes = len(table)
    size = tuple(_cfg(cfg, "preprocess", "image_size", default=[224, 224]))
    out_size = (int(size[0]), int(size[1]))
    bbox = _cfg(cfg, "preprocess", "bbox_crop", default={}) or {}
    margin = float(bbox.get("context_margin", 0.15))
    min_box_px = int(bbox.get("min_box_px", 12))
    pad_value = int(bbox.get("pad_value", 114))
    pad_mode = str(bbox.get("pad_mode", "replicate"))

    base = find_dataset_root(
        data_root,
        _cfg(cfg, "data", "images_dirname", default="images"),
        _cfg(cfg, "data", "labels_dirname", default="labels"),
    )
    log.info("dataset root resolved to %s", base)

    # 1) read + validate the YOLO layout
    records, discovery = discover_yolo_dataset(
        data_root, num_classes,
        _cfg(cfg, "data", "images_dirname", default="images"),
        _cfg(cfg, "data", "labels_dirname", default="labels"),
    )
    log.info(
        "discovered %d images / %d boxes (%d empty labels, %d malformed, %d clipped)",
        discovery.n_images, discovery.n_boxes, len(discovery.empty_labels),
        len(discovery.malformed_lines), len(discovery.clipped_boxes),
    )

    # 2) group near-duplicate source frames so they cannot straddle a split
    dedup = _cfg(cfg, "split", "dedup", default={}) or {}
    log.info("hashing %d images for near-duplicate grouping", len(records))
    group_of, grouping = group_source_images(
        [r.image_path for r in records],
        hash_size=int(dedup.get("hash_size", 16)),
        max_hamming=int(dedup.get("max_hamming", 24)),
        enabled=bool(dedup.get("enabled", True)),
    )
    log.info(
        "%d groups for %d images; %d groups hold >1 frame (largest %d)",
        grouping.n_groups, grouping.n_items, grouping.n_multi_groups, grouping.largest_group,
    )

    # 3) build the crop index (one entry per usable box)
    index, dropped_small = _build_crop_index(records, group_of, min_box_px)
    log.info("kept %d crops, dropped %d below %dpx", len(index), dropped_small, min_box_px)
    if not index:
        raise RuntimeError("no usable boxes found — check data root and labels")

    # 4) split whole groups, balancing classes
    ratios = _cfg(cfg, "split", "ratios", default={"train": 0.7, "val": 0.15, "test": 0.15})
    group_split, split_report = stratified_group_split(
        [e["group_id"] for e in index],
        [e["class_id"] for e in index],
        num_classes,
        {k: float(v) for k, v in ratios.items()},
        seed=seed,
        min_train_per_class=int(_cfg(cfg, "split", "min_train_per_class", default=1)),
    )
    for e in index:
        e["split"] = group_split[e["group_id"]]
    split_report.images_per_split = _count_images_per_split(index)
    log.info("split crops %s", split_report.crops_per_split)

    # 5) render crops into per-split memmaps
    arrays = _write_crops(index, out_root, out_size, margin, pad_value, pad_mode)

    # 6) offline augmentation of TRAIN only
    aug_info = _augment_train(cfg, index, arrays, out_root, seed)

    # 7) metadata + reports
    _write_metadata(index, aug_info.get("rows", []), out_root, table)
    shutil.copyfile(
        _cfg(cfg, "data", "class_table", default="configs/classes.csv"),
        out_root / "class_table.csv",
    )

    report = {
        "dataset_root": str(base),
        "seed": seed,
        "image_size": list(out_size),
        "bbox_crop": {
            "context_margin": margin, "min_box_px": min_box_px,
            "pad_value": pad_value, "pad_mode": pad_mode,
            "dropped_below_min_box_px": dropped_small,
        },
        "discovery": discovery.to_dict(),
        "grouping": grouping.to_dict(),
        "split": split_report.to_dict(),
        "leakage": _leakage_section(index, group_of, base, discovery),
        "crop_size_stats": _crop_size_stats(index),
        "augmentation": aug_info.get("report", {"enabled": False}),
        "arrays": {k: list(v.shape) for k, v in arrays.items()},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    save_json(report, out_root / "preprocess_report.json")
    save_yaml(
        {"preprocess": cfg.get("preprocess"), "split": cfg.get("split"),
         "augment": cfg.get("augment"), "seed": seed},
        out_root / "resolved_preprocess.yaml",
    )
    log.info("done in %.1fs -> %s", time.time() - t0, out_root)
    return out_root


def _build_crop_index(
    records: list, group_of: dict[str, int], min_box_px: int
) -> tuple[list[dict[str, Any]], int]:
    """One dict per usable box, with pixel geometry resolved.

    Image dimensions come from the JPEG header only (PIL does not decode the
    pixels for .size), so this pass stays cheap even though it touches every file.
    """
    from PIL import Image

    index: list[dict[str, Any]] = []
    dropped = 0
    for rec in tqdm(records, desc="indexing boxes", unit="img"):
        if not rec.boxes:
            continue
        with Image.open(rec.image_path) as im:
            img_w, img_h = im.size
        for box in rec.boxes:
            geom = box_geometry(box.cx, box.cy, box.w, box.h, img_w, img_h)
            if min(geom.box_w_px, geom.box_h_px) < min_box_px:
                dropped += 1
                continue
            index.append({
                "crop_id": f"{rec.stem}_{box.line_no:02d}",
                "source_image": rec.stem,
                "image_path": rec.image_path,
                "class_id": box.class_id,
                "group_id": group_of[rec.stem],
                "clipped": int(box.clipped),
                "geom": geom,
            })
    return index, dropped


def _count_images_per_split(index: list[dict[str, Any]]) -> dict[str, int]:
    seen: dict[str, set[str]] = {}
    for e in index:
        seen.setdefault(e["split"], set()).add(e["source_image"])
    return {k: len(v) for k, v in sorted(seen.items())}


def _write_crops(
    index: list[dict[str, Any]],
    out_root: Path,
    out_size: tuple[int, int],
    margin: float,
    pad_value: int,
    pad_mode: str,
) -> dict[str, np.memmap]:
    """Render every crop straight into a per-split memmap.

    Crops are grouped by source image so each JPEG is decoded once, and written
    through open_memmap so a multi-GB train array never has to fit in RAM.
    """
    counters = {s: 0 for s in SPLITS}
    for e in index:
        counters[e["split"]] += 1

    arrays: dict[str, np.memmap] = {}
    labels: dict[str, np.ndarray] = {}
    for split, n in counters.items():
        if n == 0:
            continue
        arrays[split] = np.lib.format.open_memmap(
            out_root / "images" / f"X_{split}.npy", mode="w+",
            dtype=np.uint8, shape=(n, out_size[1], out_size[0], 3),
        )
        labels[split] = np.zeros(n, dtype=np.int16)

    by_image: dict[str, list[dict[str, Any]]] = {}
    for e in index:
        by_image.setdefault(e["source_image"], []).append(e)

    written = {s: 0 for s in SPLITS}
    for stem in tqdm(sorted(by_image), desc="cropping", unit="img"):
        entries = by_image[stem]
        img = imread_rgb(entries[0]["image_path"])
        for e in entries:
            split = e["split"]
            i = written[split]
            arrays[split][i] = crop_box(
                img, e["geom"], out_size, margin, pad_value, pad_mode
            )
            labels[split][i] = e["class_id"]
            e["array_index"] = i
            written[split] += 1

    for split, arr in arrays.items():
        arr.flush()
        np.save(out_root / "images" / f"y_{split}.npy", labels[split])
    return arrays


def _augment_train(
    cfg: dict[str, Any],
    index: list[dict[str, Any]],
    arrays: dict[str, np.memmap],
    out_root: Path,
    seed: int,
) -> dict[str, Any]:
    """Generate the shared offline augmentation for the train split only.

    Extras go in their own array: the loader concatenates them onto the clean
    train set on request, so the with/without-augmentation ablation costs no
    extra disk and the untouched arrays stay available for every pipeline.
    """
    acfg = cfg.get("augment", {}) or {}
    if not acfg.get("enabled", False) or "train" not in arrays:
        return {"report": {"enabled": False}}

    train_entries = sorted(
        (e for e in index if e["split"] == "train"), key=lambda e: e["array_index"]
    )
    class_ids = [e["class_id"] for e in train_entries]
    plan = plan_augmentation(
        class_ids,
        target_count=int(acfg.get("target_count", 300)),
        max_multiplier=int(acfg.get("max_multiplier", 8)),
    )
    if acfg.get("hflip") or acfg.get("vflip"):
        raise ValueError(
            "hflip/vflip must stay false: VNTS has mirror-pair classes "
            "(7<->32, 6<->50, 1<->37, 15<->21, 19<->49) that a flip mislabels"
        )
    log.info("augmentation plan: +%d crops across %d classes",
             plan.total_extra, len(plan.per_class_extra))
    if plan.total_extra == 0:
        return {"report": {"enabled": True, **plan.to_dict()}}

    cls, src, ops = _render_augmented_from_source(
        train_entries, class_ids, plan, acfg, cfg, out_root, seed
    )

    rows = []
    for i, (c, s, o) in enumerate(zip(cls, src, ops)):
        source = train_entries[s]
        rows.append({
            "crop_id": f"aug_{i:06d}",
            "split": "train_aug",
            "array_index": i,
            "class_id": c,
            "source_image": source["source_image"],
            "group_id": source["group_id"],
            "geom": source["geom"],
            "clipped": source["clipped"],
            "is_aug": 1,
            "aug_source_crop_id": source["crop_id"],
            "aug_ops": o,
        })
    before = {c: class_ids.count(c) for c in set(class_ids)}
    after = {c: before.get(c, 0) + plan.per_class_extra.get(c, 0) for c in before}
    return {
        "rows": rows,
        "report": {
            "enabled": True,
            **plan.to_dict(),
            "train_before": len(class_ids),
            "train_after": len(class_ids) + len(cls),
            "imbalance_ratio_before": round(max(before.values()) / min(before.values()), 1),
            "imbalance_ratio_after": round(max(after.values()) / min(after.values()), 1),
        },
    }


def _render_augmented_from_source(
    train_entries: list[dict[str, Any]],
    class_ids: list[int],
    plan,
    acfg: dict[str, Any],
    cfg: dict[str, Any],
    out_root: Path,
    seed: int,
) -> tuple[list[int], list[int], list[str]]:
    """Cut every synthetic sample out of its original frame.

    Augmenting the cached 224px crop would rotate its letterbox padding too, and
    since the extra samples are concentrated in the rare classes a tilted grey
    bar would become a giveaway for "rare class". Re-cropping a jittered window
    from the source JPEG keeps the padding axis-aligned and fills the jitter with
    real neighbouring pixels, which is what a detector's loose box looks like.
    """
    size = tuple(_cfg(cfg, "preprocess", "image_size", default=[224, 224]))
    out_size = (int(size[0]), int(size[1]))
    bbox = _cfg(cfg, "preprocess", "bbox_crop", default={}) or {}
    margin = float(bbox.get("context_margin", 0.15))
    pad_value = int(bbox.get("pad_value", 114))
    pad_mode = str(bbox.get("pad_mode", "replicate"))

    def crop_fn(image: np.ndarray, geom) -> np.ndarray:
        return crop_box(image, geom, out_size, margin, pad_value, pad_mode)

    order = plan_assignments(class_ids, plan)
    images = np.lib.format.open_memmap(
        out_root / "images" / "X_train_aug.npy", mode="w+",
        dtype=np.uint8, shape=(len(order), out_size[1], out_size[0], 3),
    )
    ops_used: list[str] = ["" for _ in order]
    rng = np.random.default_rng(seed)

    # One RNG stream shared in source-image order keeps the run reproducible
    # while each JPEG is still decoded only once.
    by_image: dict[str, list[int]] = {}
    for out_i, src in enumerate(order):
        by_image.setdefault(train_entries[src]["source_image"], []).append(out_i)

    for stem in tqdm(sorted(by_image), desc="augmenting", unit="img"):
        slots = by_image[stem]
        img = imread_rgb(train_entries[order[slots[0]]]["image_path"])
        for out_i in slots:
            entry = train_entries[order[out_i]]
            crop, applied = augment_from_source(
                img, entry["geom"], rng, crop_fn, acfg.get("ops")
            )
            images[out_i] = crop
            ops_used[out_i] = "+".join(applied)
    images.flush()

    cls = [int(class_ids[s]) for s in order]
    np.save(out_root / "images" / "y_train_aug.npy", np.asarray(cls, dtype=np.int16))
    return cls, order, ops_used


def _write_metadata(
    index: list[dict[str, Any]],
    aug_rows: list[dict[str, Any]],
    out_root: Path,
    table: ClassTable,
) -> None:
    with open(out_root / "metadata.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        for e in sorted(index, key=lambda e: (e["split"], e["array_index"])):
            writer.writerow(_metadata_row(e, table))
        for e in aug_rows:
            writer.writerow(_metadata_row(e, table))


def _metadata_row(e: dict[str, Any], table: ClassTable) -> dict[str, Any]:
    geom = e["geom"]
    return {
        "crop_id": e["crop_id"],
        "split": e["split"],
        "array_index": e["array_index"],
        "class_id": e["class_id"],
        "sign_code": table[e["class_id"]].sign_code,
        "source_image": e["source_image"],
        "group_id": e["group_id"],
        **geom.to_dict(),
        "clipped": e.get("clipped", 0),
        "is_aug": e.get("is_aug", 0),
        "aug_source_crop_id": e.get("aug_source_crop_id", ""),
        "aug_ops": e.get("aug_ops", ""),
    }


def _leakage_section(
    index: list[dict[str, Any]], group_of: dict[str, int], base: Path, discovery
) -> dict[str, Any]:
    """Our split must have zero straddling groups; the authors' split does not.

    Reporting both is the evidence for changing split.strategy in shared.yaml.
    """
    ours = {e["source_image"]: e["split"] for e in index}
    section: dict[str, Any] = {"ours": leakage_against(group_of, ours)}
    provided: dict[str, str] = {}
    split_dir = base / "split_dataset"
    for name in ("train", "test"):
        f = split_dir / f"{name}_files.txt"
        if f.exists():
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        provided[Path(line.strip()).stem] = name
    if provided:
        section["provided_by_authors"] = leakage_against(group_of, provided)
    return section


def _crop_size_stats(index: list[dict[str, Any]]) -> dict[str, Any]:
    sides = np.array([min(e["geom"].box_w_px, e["geom"].box_h_px) for e in index])
    areas = np.array([e["geom"].area_px for e in index])
    pct = [1, 10, 25, 50, 75, 90, 99]
    return {
        "min_side_px_percentiles": {
            str(p): float(np.percentile(sides, p)) for p in pct
        },
        "area_px_percentiles": {str(p): float(np.percentile(areas, p)) for p in pct},
        "share_below_32x32_px": round(float((areas < 32 * 32).mean()), 4),
        "share_below_16x16_px": round(float((areas < 16 * 16).mean()), 4),
    }
