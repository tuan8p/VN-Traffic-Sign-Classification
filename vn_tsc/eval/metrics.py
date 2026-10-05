from __future__ import annotations
from typing import Any
import numpy as np

def compute_classification_metrics(
    y_true, y_pred, labels=None, rare_labels=None,
) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, f1_score
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    res: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
    }
    if rare_labels is not None and labels is not None:
        rare_set = set(rare_labels)
        common = [c for c in labels if c not in rare_set]
        if common:
            res["macro_f1_common"] = float(
                f1_score(y_true, y_pred, average="macro", labels=common, zero_division=0)
            )
    return res
