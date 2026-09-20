from __future__ import annotations
import numpy as np
from typing import Any

def extract_hog(image: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """TODO(team-preprocessing): HOG."""
    raise NotImplementedError("TODO(team-preprocessing): HOG")

def extract_lbp(image: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """TODO(team-preprocessing): LBP."""
    raise NotImplementedError("TODO(team-preprocessing): LBP")

def extract_color_hist(image: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """TODO(team-preprocessing): color hist."""
    raise NotImplementedError("TODO(team-preprocessing): color hist")

def extract_shared_features(image: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    feats = []
    fcfg = cfg.get("features", {})
    if fcfg.get("hog", {}).get("enabled", True):
        feats.append(extract_hog(image, fcfg.get("hog", {})))
    if fcfg.get("lbp", {}).get("enabled", True):
        feats.append(extract_lbp(image, fcfg.get("lbp", {})))
    if fcfg.get("color_hist", {}).get("enabled", True):
        feats.append(extract_color_hist(image, fcfg.get("color_hist", {})))
    if not feats:
        raise ValueError("No classical features enabled")
    return np.concatenate([f.ravel() for f in feats], axis=0)
