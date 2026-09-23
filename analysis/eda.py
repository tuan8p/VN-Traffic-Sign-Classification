"""EDA for the processed VNTS crop-classification store. No W&B required.

Run after tools.run_preprocess:
    python -m analysis.eda --processed data/processed --out analysis/figures
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vn_tsc.data.classes import ClassTable
from vn_tsc.utils.io import save_json
from vn_tsc.utils.logging_utils import setup_logger

log = setup_logger("eda")

# Figures label classes by "id_code" rather than the Vietnamese name: the names
# are long, and a Windows cp1252 console cannot print them at all.
SPLIT_COLORS = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path)
    return path


def fig_class_distribution(md: pd.DataFrame, table: ClassTable, out_dir: Path) -> None:
    """Log-scale bar of real crops per class, split-stacked."""
    real = md[md.is_aug == 0]
    order = real.class_id.value_counts().index.tolist()
    labels = [table[c].label for c in order]
    fig, ax = plt.subplots(figsize=(14, 5))
    bottom = np.zeros(len(order))
    for split in ("train", "val", "test"):
        counts = real[real.split == split].class_id.value_counts()
        vals = np.array([counts.get(c, 0) for c in order], dtype=float)
        ax.bar(labels, vals, bottom=bottom, label=split, color=SPLIT_COLORS[split])
        bottom += vals
    ax.set_yscale("log")
    ax.set_ylabel("crops (log scale)")
    ax.set_title(
        f"Class distribution — {len(order)} classes, "
        f"{int(bottom.max())}:{int(bottom.min())} imbalance"
    )
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.legend()
    _save(fig, out_dir, "01_class_distribution.png")


def fig_augmentation_effect(md: pd.DataFrame, table: ClassTable, out_dir: Path) -> None:
    """Train counts before and after the shared offline augmentation."""
    tr = md[(md.split == "train")].class_id.value_counts()
    aug = md[md.split == "train_aug"].class_id.value_counts()
    if aug.empty:
        return
    order = tr.sort_values(ascending=False).index.tolist()
    labels = [table[c].label for c in order]
    before = np.array([tr.get(c, 0) for c in order], dtype=float)
    extra = np.array([aug.get(c, 0) for c in order], dtype=float)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(labels, before, label="real", color="#4C72B0")
    ax.bar(labels, extra, bottom=before, label="offline augmented", color="#C44E52")
    ax.set_yscale("log")
    ax.set_ylabel("train crops (log scale)")
    after = before + extra
    ax.set_title(
        f"Offline augmentation — imbalance {before.max()/before.min():.0f}:1 "
        f"-> {after.max()/after.min():.0f}:1"
    )
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.legend()
    _save(fig, out_dir, "02_augmentation_effect.png")


def fig_box_sizes(md: pd.DataFrame, out_dir: Path) -> None:
    """How small the signs really are — the core difficulty of this dataset."""
    real = md[md.is_aug == 0]
    side = np.minimum(real.box_w_px, real.box_h_px)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(side, bins=60, range=(0, 200), color="#4C72B0")
    axes[0].axvline(32, color="#C44E52", ls="--", label="32 px (COCO 'small')")
    axes[0].set_xlabel("shorter box side (px, original image)")
    axes[0].set_ylabel("crops")
    axes[0].set_title(f"Box size — median {np.median(side):.0f} px")
    axes[0].legend()

    area = real.box_area_px
    axes[1].hist(np.log10(area.clip(lower=1)), bins=60, color="#55A868")
    axes[1].axvline(np.log10(32 * 32), color="#C44E52", ls="--", label="32x32 px")
    axes[1].set_xlabel("log10(box area in px)")
    axes[1].set_title(
        f"{100*(area < 32*32).mean():.0f}% of signs are smaller than 32x32 px"
    )
    axes[1].legend()
    _save(fig, out_dir, "03_box_sizes.png")


def fig_position_heatmap(md: pd.DataFrame, out_dir: Path) -> None:
    """Where signs sit in the frame — a prior a detector could exploit."""
    real = md[md.is_aug == 0]
    cx = ((real.x1 + real.x2) / 2 / real.img_w).clip(0, 1)
    cy = ((real.y1 + real.y2) / 2 / real.img_h).clip(0, 1)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    h = ax.hist2d(cx, cy, bins=(48, 27), cmap="magma")
    ax.invert_yaxis()
    ax.set_xlabel("x (fraction of width)")
    ax.set_ylabel("y (fraction of height)")
    ax.set_title("Sign centre position in the source frame")
    fig.colorbar(h[3], ax=ax, label="crops")
    _save(fig, out_dir, "04_position_heatmap.png")


def fig_leakage(report: dict[str, Any], out_dir: Path) -> None:
    """Near-duplicate clusters, and how the two splits handle them."""
    grouping = report.get("grouping", {})
    hist = {int(k): v for k, v in grouping.get("group_size_hist", {}).items()}
    leak = report.get("leakage", {})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    sizes = sorted(hist)
    axes[0].bar([str(s) for s in sizes], [hist[s] for s in sizes], color="#4C72B0")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("frames per near-duplicate cluster")
    axes[0].set_ylabel("clusters (log)")
    axes[0].set_title(
        f"{grouping.get('n_groups', 0)} clusters from "
        f"{grouping.get('n_items', 0)} frames"
    )

    names, vals = [], []
    for key, label in (("provided_by_authors", "authors' split"), ("ours", "group split (ours)")):
        if key in leak:
            names.append(label)
            vals.append(leak[key]["n_affected_images"])
    axes[1].bar(names, vals, color=["#C44E52", "#55A868"][: len(names)])
    for i, v in enumerate(vals):
        axes[1].text(i, v, f" {v}", va="bottom", ha="center")
    axes[1].set_ylabel("images in a cluster that straddles two splits")
    axes[1].set_title("Train/test leakage")
    _save(fig, out_dir, "05_leakage.png")


def fig_class_grid(
    md: pd.DataFrame, table: ClassTable, processed: Path, out_dir: Path,
    name: str = "06_class_examples.png", class_ids: list[int] | None = None,
    title: str = "One example per class (largest crop)", cols: int = 13,
) -> None:
    """A sheet of real crops — the only way to actually check the labels."""
    X = np.load(processed / "images" / "X_train.npy", mmap_mode="r")
    tr = md[md.split == "train"]
    ids = class_ids if class_ids is not None else table.ids
    picks = []
    for c in ids:
        s = tr[tr.class_id == c]
        if len(s):
            picks.append((c, int(s.nlargest(1, "box_area_px").array_index.iloc[0])))
    rows = (len(picks) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.25, rows * 1.45))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, (c, idx) in zip(np.ravel(axes), picks):
        ax.imshow(np.asarray(X[idx]))
        ax.set_title(table[c].label, fontsize=6)
    fig.suptitle(title, y=1.0)
    _save(fig, out_dir, name)


def fig_mirror_pairs(md: pd.DataFrame, table: ClassTable, processed: Path, out_dir: Path) -> None:
    """Why horizontal flip is banned: each row is one sign and its mirror twin."""
    pairs = table.mirror_pairs()
    if not pairs:
        return
    X = np.load(processed / "images" / "X_train.npy", mmap_mode="r")
    tr = md[md.split == "train"]
    fig, axes = plt.subplots(len(pairs), 2, figsize=(4.4, 2.2 * len(pairs)))
    axes = np.atleast_2d(axes)
    for r, (a, b) in enumerate(pairs):
        for col, c in enumerate((a, b)):
            ax = axes[r, col]
            ax.axis("off")
            s = tr[tr.class_id == c]
            if len(s):
                ax.imshow(np.asarray(X[int(s.nlargest(1, "box_area_px").array_index.iloc[0])]))
            ax.set_title(f"{table[c].label}\n{table[c].name_vi[:30]}", fontsize=6)
    fig.suptitle("Mirror-pair classes: hflip would relabel these", y=1.0, fontsize=9)
    _save(fig, out_dir, "07_mirror_pairs.png")


def fig_confusable_groups(md: pd.DataFrame, table: ClassTable, processed: Path, out_dir: Path) -> None:
    """Classes that differ only in fine detail, at the size they actually occur."""
    groups = table.confusable_groups()
    if not groups:
        return
    X = np.load(processed / "images" / "X_train.npy", mmap_mode="r")
    tr = md[md.split == "train"]
    rows = sorted(groups.items())
    width = max(len(v) for _, v in rows)
    fig, axes = plt.subplots(len(rows), width, figsize=(width * 1.3, len(rows) * 1.55))
    axes = np.atleast_2d(axes)
    for ax in np.ravel(axes):
        ax.axis("off")
    for r, (gname, ids) in enumerate(rows):
        for col, c in enumerate(ids):
            s = tr[tr.class_id == c]
            if not len(s):
                continue
            # Median-sized example, not the biggest: this is how the model sees it.
            med = s.iloc[(s.box_area_px - s.box_area_px.median()).abs().argsort()[:1]]
            axes[r, col].imshow(np.asarray(X[int(med.array_index.iloc[0])]))
            axes[r, col].set_title(f"{table[c].label}", fontsize=6)
        axes[r, 0].set_ylabel(gname, fontsize=7)
    fig.suptitle("Fine-grained groups at their median size", y=1.0, fontsize=9)
    _save(fig, out_dir, "08_confusable_groups.png")


def fig_brightness(md: pd.DataFrame, processed: Path, out_dir: Path) -> None:
    """Lighting spread of the crops — the set advertises day/night and blur."""
    X = np.load(processed / "images" / "X_train.npy", mmap_mode="r")
    step = max(1, len(X) // 2000)
    sample = np.asarray(X[::step]).reshape(-1, X.shape[1] * X.shape[2], 3)
    mean = sample.mean(axis=(1, 2))
    # Variance of the Laplacian is the standard sharpness proxy; low = blurred.
    import cv2
    sharp = np.array([
        cv2.Laplacian(cv2.cvtColor(np.asarray(X[i]), cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
        for i in range(0, len(X), step)
    ])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(mean, bins=50, color="#4C72B0")
    axes[0].set_xlabel("mean crop brightness (0-255)")
    axes[0].set_ylabel("crops")
    axes[0].set_title(f"Brightness — {100*(mean < 60).mean():.1f}% very dark")
    axes[1].hist(np.log10(np.clip(sharp, 1e-3, None)), bins=50, color="#55A868")
    axes[1].set_xlabel("log10 variance of Laplacian (sharpness)")
    axes[1].set_title(f"Sharpness — {100*(sharp < 50).mean():.1f}% likely blurred")
    _save(fig, out_dir, "09_brightness_sharpness.png")


def build_summary(md: pd.DataFrame, table: ClassTable, report: dict[str, Any]) -> dict[str, Any]:
    real = md[md.is_aug == 0]
    counts = Counter(real.class_id)
    rare = sorted(c for c, n in counts.items() if n < 30)
    per_class = []
    for c in table.ids:
        row = {"class_id": c, "code": table[c].label, "name_vi": table[c].name_vi}
        for s in ("train", "val", "test", "train_aug"):
            row[s] = int((md[md.split == s].class_id == c).sum())
        row["total_real"] = counts.get(c, 0)
        per_class.append(row)
    return {
        "n_crops_real": int(len(real)),
        "n_crops_augmented": int((md.is_aug == 1).sum()),
        "n_source_images": int(real.source_image.nunique()),
        "n_classes": len(table),
        "imbalance_ratio": round(max(counts.values()) / min(counts.values()), 1),
        "rare_classes_below_30": rare,
        "n_rare_classes_below_30": len(rare),
        "mirror_pairs": table.mirror_pairs(),
        "leakage": report.get("leakage", {}),
        "crop_size_stats": report.get("crop_size_stats", {}),
        "per_class": per_class,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="EDA over the processed crop store")
    p.add_argument("--processed", default="data/processed")
    p.add_argument("--out", default="analysis/figures")
    args = p.parse_args(argv)

    processed = Path(args.processed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    md = pd.read_csv(processed / "metadata.csv")
    table = ClassTable.load(processed / "class_table.csv")
    report_path = processed / "preprocess_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}

    fig_class_distribution(md, table, out_dir)
    fig_augmentation_effect(md, table, out_dir)
    fig_box_sizes(md, out_dir)
    fig_position_heatmap(md, out_dir)
    fig_leakage(report, out_dir)
    fig_class_grid(md, table, processed, out_dir)
    fig_mirror_pairs(md, table, processed, out_dir)
    fig_confusable_groups(md, table, processed, out_dir)
    fig_brightness(md, processed, out_dir)

    summary = build_summary(md, table, report)
    save_json(summary, out_dir / "eda_summary.json")
    pd.DataFrame(summary["per_class"]).to_csv(
        out_dir / "per_class_counts.csv", index=False, encoding="utf-8"
    )
    log.info(
        "%d real crops / %d augmented, %d classes, imbalance %.0f:1, %d rare classes",
        summary["n_crops_real"], summary["n_crops_augmented"], summary["n_classes"],
        summary["imbalance_ratio"], summary["n_rare_classes_below_30"],
    )


if __name__ == "__main__":
    main()
