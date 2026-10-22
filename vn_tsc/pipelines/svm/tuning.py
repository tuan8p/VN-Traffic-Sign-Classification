"""Cross-validation search over the complete SVM sklearn Pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from vn_tsc.pipelines.svm.model import validate_fit_data


def build_search(estimator: Pipeline, cfg: dict[str, Any]) -> GridSearchCV | RandomizedSearchCV:
    """Create deterministic CV search without fitting scaler or reducer in advance."""
    options = cfg["tuning"]
    folds = int(options.get("cv_folds", 5))
    if folds < 2:
        raise ValueError("tuning.cv_folds must be at least 2")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cfg["seed"])
    method = options.get("method", "grid")
    if method not in ("grid", "random"):
        raise ValueError(f"Unknown tuning.method: {method}")
    candidates = options.get("param_grid" if method == "grid" else "param_distributions")
    if not candidates:
        if method == "random":
            candidates = options.get("param_grid")
        if not candidates:
            raise ValueError("Tuning requires a nonempty parameter grid/distribution")
    keys = candidates.keys() if isinstance(candidates, dict) else (
        key for grid in candidates for key in grid
    )
    invalid = sorted(set(keys) - set(estimator.get_params()))
    if invalid:
        raise ValueError(f"Invalid tuning parameters for current SVM pipeline: {invalid}")
    common = {"estimator": estimator, "scoring": options.get("scoring", "f1_macro"),
              "cv": cv, "n_jobs": int(options.get("n_jobs", -1)), "refit": True,
              "error_score": "raise", "return_train_score": False}
    if method == "grid":
        return GridSearchCV(param_grid=candidates, **common)
    if method == "random":
        n_iter = int(options.get("n_iter", 20))
        if n_iter < 1:
            raise ValueError("tuning.n_iter must be positive")
        return RandomizedSearchCV(param_distributions=candidates, n_iter=n_iter,
                                  random_state=cfg["seed"], **common)


def tune_svm(estimator: Pipeline, X: np.ndarray, y: np.ndarray,
             cfg: dict[str, Any]) -> GridSearchCV | RandomizedSearchCV:
    """Fit CV search using only the training split."""
    search = build_search(estimator, cfg)
    folds = search.cv.n_splits
    _, counts = np.unique(y, return_counts=True)
    if counts.min() < folds:
        raise ValueError(f"Each training class needs at least {folds} samples for CV; "
                         f"smallest class has {counts.min()}")
    max_validation_size = int(np.ceil(X.shape[0] / folds))
    validate_fit_data(cfg, X, y, smallest_fit_size=X.shape[0] - max_validation_size)
    search.fit(X, y)
    return search
