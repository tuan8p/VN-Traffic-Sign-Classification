from __future__ import annotations
from pathlib import Path
from typing import Any

def curves_from_history_json(history_path: str | Path, out_dir: str | Path) -> list[Path]:
    """TODO(team): matplotlib curves from history.json."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    return []

def confusion_matrix_png(y_true, y_pred, class_names: list[str], out_path: str | Path) -> Path:
    """TODO(team): seaborn heatmap."""
    raise NotImplementedError("TODO(team): confusion matrix plot")
