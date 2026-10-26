from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from sklearn.datasets import make_classification

from vn_tsc.config.resolve import resolve_config
from vn_tsc.pipelines.svm.data_adapter import feature_config_from_cfg
from vn_tsc.pipelines.svm.inference import (
    extract_svm_image_features,
    predict_svm_image,
    svm_class_scores,
)
from vn_tsc.pipelines.svm.model import build_svm


@pytest.fixture
def cfg() -> dict:
    root = Path(__file__).resolve().parents[1]
    config = resolve_config(root / "configs/pipelines/svm.yaml", root / "configs/shared.yaml")
    config["features"]["input_size"] = [32, 32]
    config["features"]["hog"]["pixels_per_cell"] = [8, 8]
    return config


def test_image_inference_uses_saved_classical_feature_contract(cfg: dict) -> None:
    feature_config = feature_config_from_cfg(cfg)
    red = Image.new("RGB", (30, 18), (220, 25, 25))
    blue = Image.new("RGB", (18, 30), (25, 25, 220))
    red_row = extract_svm_image_features(red, feature_config)
    blue_row = extract_svm_image_features(blue, feature_config)
    X = np.concatenate([
        red_row + offset for offset in np.linspace(0.0, 0.002, 6, dtype=np.float32)
    ] + [
        blue_row + offset for offset in np.linspace(0.0, 0.002, 6, dtype=np.float32)
    ])
    y = np.array([0] * 6 + [1] * 6)
    model = build_svm(cfg).fit(X, y)

    result = predict_svm_image(
        model, red, feature_config, top_k=2,
        labels_by_id={0: "red sign", 1: "blue sign"},
    )

    assert result["class_id"] == int(model.predict(red_row)[0])
    assert result["label"] == "red sign"
    assert result["scores_calibrated"] is False
    assert len(result["top_k"]) == 2
    assert sum(item["score"] for item in result["top_k"]) == pytest.approx(1.0)


def test_image_inference_rejects_feature_metadata_from_another_run(cfg: dict) -> None:
    feature_config = feature_config_from_cfg(cfg)
    image = Image.new("RGB", (32, 32), (200, 20, 20))
    row = extract_svm_image_features(image, feature_config)
    model = build_svm(cfg).fit(np.vstack([row, row + 0.1]), np.array([0, 1]))

    wrong = deepcopy(feature_config)
    wrong["color_hist"]["bins"] = 8
    with pytest.raises(ValueError, match="feature dimension mismatch"):
        predict_svm_image(model, image, wrong)


def test_ovo_scores_rank_the_same_class_as_svc_predict(cfg: dict) -> None:
    X, y = make_classification(
        n_samples=120, n_features=12, n_informative=9, n_redundant=0,
        n_classes=4, random_state=7,
    )
    model = build_svm(cfg).fit(X, y)
    predicted = model.predict(X[:30])
    ranked = []
    for row in X[:30]:
        classes, scores, calibrated = svm_class_scores(model, row.reshape(1, -1))
        ranked.append(classes[np.argmax(scores)])
    np.testing.assert_array_equal(ranked, predicted)
    assert calibrated is False


def test_repository_svm_config_records_best_seed_experiment(cfg: dict) -> None:
    assert cfg["seed"] == 42
    assert cfg["experiment_seeds"] == [7, 21, 123, 2026]
    assert cfg["model"]["C"] == 5.0
    assert cfg["model"]["gamma"] == "scale"
