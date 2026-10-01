from __future__ import annotations

import numpy as np
import pytest

from vn_tsc.pipelines.svm.dev_features import (
    extract_color_hist,
    extract_dev_features,
    extract_hog,
    extract_lbp,
)


@pytest.fixture
def image() -> np.ndarray:
    x = np.arange(32, dtype=np.uint8)
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :, 0] = x[None, :] * 7
    image[:, :, 1] = x[:, None] * 7
    image[:, :, 2] = 128
    return image


@pytest.fixture
def feature_cfg() -> dict:
    return {
        "hog": {"enabled": True, "orientations": 9,
                "pixels_per_cell": [8, 8], "cells_per_block": [2, 2]},
        "lbp": {"enabled": True, "p": 8, "r": 1},
        "color_hist": {"enabled": True, "bins": 16},
    }


def test_hog_is_finite_1d_and_deterministic(image: np.ndarray, feature_cfg: dict) -> None:
    first = extract_hog(image, feature_cfg["hog"])
    assert first.ndim == 1 and first.size > 0
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, extract_hog(image, feature_cfg["hog"]))


def test_lbp_histogram_is_normalized(image: np.ndarray, feature_cfg: dict) -> None:
    histogram = extract_lbp(image, feature_cfg["lbp"])
    assert histogram.shape == (10,)
    assert np.isfinite(histogram).all()
    assert histogram.sum() == pytest.approx(1.0)


def test_color_histogram_and_combination(image: np.ndarray, feature_cfg: dict) -> None:
    color = extract_color_hist(image, feature_cfg["color_hist"])
    assert color.shape == (48,)
    assert np.isfinite(color).all()
    assert color.reshape(3, 16).sum(axis=1) == pytest.approx([1, 1, 1])
    hog = extract_hog(image, feature_cfg["hog"])
    lbp = extract_lbp(image, feature_cfg["lbp"])
    combined = extract_dev_features(image, feature_cfg)
    assert combined.ndim == 1
    assert combined.size == hog.size + lbp.size + color.size
    feature_cfg["hog"]["enabled"] = False
    assert extract_dev_features(image, feature_cfg).size == lbp.size + color.size
    feature_cfg["lbp"]["enabled"] = False
    feature_cfg["color_hist"]["enabled"] = False
    with pytest.raises(ValueError, match="At least one"):
        extract_dev_features(image, feature_cfg)
