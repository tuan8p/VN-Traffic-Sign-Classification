from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from vn_tsc.config.resolve import resolve_config
from vn_tsc.pipelines.svm.model import build_svm, validate_fit_data


@pytest.fixture
def cfg() -> dict:
    root = Path(__file__).resolve().parents[1]
    return resolve_config(root / "configs/pipelines/svm.yaml", root / "configs/shared.yaml")


def test_build_modes_keep_scaler_and_reducer_inside_pipeline(cfg: dict) -> None:
    for method, expected in (("none", str), ("pca", PCA), ("lda", LinearDiscriminantAnalysis)):
        candidate = deepcopy(cfg)
        candidate["dim_reduction"]["method"] = method
        model = build_svm(candidate)
        assert isinstance(model, Pipeline)
        assert isinstance(model.named_steps["scaler"], StandardScaler)
        assert isinstance(model.named_steps["reduce_dim"], expected)
        assert isinstance(model.named_steps["svm"], SVC)
        assert model.named_steps["svm"].probability is False


def test_invalid_method_and_components_fail_clearly(cfg: dict) -> None:
    cfg["dim_reduction"]["method"] = "foobar"
    with pytest.raises(ValueError, match="foobar"):
        build_svm(cfg)
    cfg["dim_reduction"]["method"] = "lda"
    cfg["dim_reduction"]["lda"]["n_components"] = 3
    X = np.ones((12, 5))
    y = np.repeat([0, 1, 2], 4)
    with pytest.raises(ValueError, match="LDA n_components"):
        validate_fit_data(cfg, X, y)
    cfg["dim_reduction"]["method"] = "pca"
    cfg["dim_reduction"]["pca"]["n_components"] = 20
    with pytest.raises(ValueError, match="PCA n_components"):
        validate_fit_data(cfg, X, y)


@pytest.mark.parametrize("method", ["none", "pca", "lda"])
def test_fit_and_predict_multiclass(cfg: dict, method: str) -> None:
    X, y = make_classification(n_samples=60, n_features=12, n_informative=8,
                               n_redundant=0, n_classes=3, n_clusters_per_class=1,
                               random_state=42)
    cfg["dim_reduction"]["method"] = method
    if method == "lda":
        cfg["dim_reduction"]["lda"]["n_components"] = 2
    model = build_svm(cfg)
    model.fit(X[:45], y[:45])
    predictions = model.predict(X[45:])
    assert predictions.shape == (15,)
    assert set(predictions).issubset({0, 1, 2})
