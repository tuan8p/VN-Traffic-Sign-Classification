"""Matplotlib/Seaborn plotting utilities for training evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def curves_from_history_json(history_path: str | Path, out_dir: str | Path) -> list[Path]:
    """Plot train/val loss and val metrics from ``history.json``.

    ``history.json`` is a dict mapping stage name → list of epoch dicts, e.g.::

        {
          "stage1": [{"epoch": 1, "train_loss": 0.5, "val_macro_f1": 0.3, ...}, ...],
          "stage2": [...]
        }

    Returns list of saved figure paths.
    """
    import json

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = Path(history_path)

    with history_path.open(encoding="utf-8") as f:
        history: dict[str, list[dict]] = json.load(f)

    saved: list[Path] = []

    # Flatten all stages into one chronological sequence for cross-stage plots.
    all_epochs: list[dict] = []
    for stage_entries in history.values():
        all_epochs.extend(stage_entries)

    if not all_epochs:
        return saved

    def _get(entries, key):
        return [e.get(key) for e in entries if e.get(key) is not None]

    # --- Loss curve ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(_get(all_epochs, "train_loss"), label="train loss")
    ax.plot(_get(all_epochs, "val_loss"), label="val loss")
    # Mark stage boundaries.
    offset = 0
    for stage, entries in history.items():
        if offset > 0:
            ax.axvline(offset - 0.5, linestyle="--", color="gray", alpha=0.5)
            ax.text(offset, ax.get_ylim()[1] * 0.95, stage, fontsize=8, color="gray")
        offset += len(entries)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "loss_curve.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    saved.append(path)

    # --- Metric curves ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric in zip(axes, ("accuracy", "macro_f1", "weighted_f1")):
        vals = _get(all_epochs, f"val_{metric}")
        ax.plot(vals, color="steelblue")
        if vals:
            best_idx = int(np.argmax(vals))
            ax.scatter([best_idx], [vals[best_idx]], color="red", zorder=5,
                       label=f"best={vals[best_idx]:.4f}")
        ax.set_xlabel("Epoch")
        ax.set_title(f"Val {metric}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "metric_curves.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    saved.append(path)

    return saved


def confusion_matrix_png(
    y_true,
    y_pred,
    class_names: list[str],
    out_path: str | Path,
) -> Path:
    """Save a seaborn confusion-matrix heatmap as PNG."""
    try:
        import seaborn as sns
    except ImportError:
        sns = None  # type: ignore[assignment]

    from sklearn.metrics import confusion_matrix

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    size = min(20, max(6, len(class_names) * 0.45))
    fig, ax = plt.subplots(figsize=(size, size))

    if sns is not None:
        sns.heatmap(
            matrix, annot=False, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=ax,
        )
    else:
        im = ax.imshow(matrix, cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
               xticklabels=class_names, yticklabels=class_names)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=6)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
