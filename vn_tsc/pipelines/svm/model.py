"""Leakage-safe scaler, optional reduction and SVC estimator."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_svm(cfg: dict[str, Any]) -> Pipeline:
    """Build an unfitted sklearn Pipeline for multiclass SVM classification."""
    reduction_cfg = cfg["dim_reduction"]
    method = reduction_cfg["method"]
    if method == "none":
        reducer = "passthrough"
    elif method == "pca":
        options = reduction_cfg["pca"]
        components = options["n_components"]
        if (type(components) is int and components < 1) or (
                type(components) is float and not 0 < components < 1) or (
                type(components) not in (int, float) and components is not None):
            raise ValueError("PCA n_components must be a positive integer, a float in (0, 1), or null")
        reducer = PCA(n_components=components, whiten=bool(options.get("whiten", False)),
                      svd_solver="auto", random_state=cfg["seed"])
    elif method == "lda":
        options = reduction_cfg["lda"]
        components = options.get("n_components")
        if components is not None and (type(components) is not int or components < 1):
            raise ValueError("LDA n_components must be a positive integer or null")
        solver = options.get("solver", "svd")
        if solver not in ("svd", "eigen"):
            raise ValueError("LDA reduction supports solver 'svd' or 'eigen'")
        reducer = LinearDiscriminantAnalysis(n_components=components, solver=solver)
    else:
        raise ValueError(f"Unknown dim_reduction.method: {method}")
    model_cfg = cfg["model"]
    if model_cfg.get("estimator", "svc") != "svc":
        raise ValueError("SVM pipeline requires model.estimator: svc")
    # SVC uses One-vs-One internally for multiclass fitting.
    svm = SVC(
        kernel=model_cfg["kernel"], C=float(model_cfg["C"]),
        gamma=model_cfg.get("gamma", "scale"), degree=int(model_cfg.get("degree", 3)),
        coef0=float(model_cfg.get("coef0", 0.0)),
        class_weight=model_cfg.get("class_weight"),
        probability=bool(model_cfg.get("probability", False)),
        decision_function_shape=model_cfg.get("decision_function_shape", "ovr"),
        random_state=cfg["seed"],
    )
    return Pipeline([("scaler", StandardScaler()), ("reduce_dim", reducer), ("svm", svm)])


def validate_fit_data(cfg: dict[str, Any], X: np.ndarray, y: np.ndarray,
                      smallest_fit_size: int | None = None) -> None:
    """Fail clearly on data or reduction settings that cannot fit."""
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0] or X.shape[0] == 0:
        raise ValueError("Training X/y must be nonempty aligned 2D/1D arrays")
    if not np.all(np.isfinite(X)):
        raise ValueError("Training feature matrix contains non-finite values")
    n_classes = len(np.unique(y))
    if n_classes < 2:
        raise ValueError("Training split must contain at least two classes")
    method = cfg["dim_reduction"]["method"]
    if method == "lda":
        components = cfg["dim_reduction"]["lda"].get("n_components")
        if components is not None and components > min(n_classes - 1, X.shape[1]):
            raise ValueError(f"LDA n_components={components} exceeds min(n_classes-1, n_features)="
                             f"{min(n_classes - 1, X.shape[1])}")
    elif method == "pca":
        components = cfg["dim_reduction"]["pca"]["n_components"]
        limit = min(smallest_fit_size or X.shape[0], X.shape[1])
        if type(components) is int and components > limit:
            raise ValueError(f"PCA n_components={components} exceeds fit limit {limit}")
