"""Temporary classical feature extraction used by Role B.

This exists only to allow independent SVM development before Role D
provides the shared HOG/LBP/color feature implementation.

Do not treat these features as the final fair-comparison implementation.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Features require an RGB uint8 image with shape (height, width, 3)")
    return image


def extract_hog(image: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Compute grayscale HOG using configured cell, block and bin sizes."""
    rgb = _rgb(image)
    height, width = rgb.shape[:2]
    cell_h, cell_w = (int(value) for value in cfg["pixels_per_cell"])
    block_h, block_w = (int(value) for value in cfg["cells_per_block"])
    bins = int(cfg["orientations"])
    if min(cell_h, cell_w, block_h, block_w, bins) <= 0:
        raise ValueError("HOG cell, block and orientation settings must be positive")
    if (height % cell_h or width % cell_w or
            height < cell_h * block_h or width < cell_w * block_w):
        raise ValueError(f"HOG image shape {(height, width)} is incompatible with cell/block settings")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if hasattr(cv2, "HOGDescriptor"):
        descriptor = cv2.HOGDescriptor(
            (width, height), (cell_w * block_w, cell_h * block_h),
            (cell_w, cell_h), (cell_w, cell_h), bins,
        )
        return descriptor.compute(gray).ravel().astype(np.float32, copy=False)
    from skimage.feature import hog
    return hog(
        gray,
        orientations=bins,
        pixels_per_cell=(cell_h, cell_w),
        cells_per_block=(block_h, block_w),
        block_norm="L2-Hys",
        visualize=False,
    ).ravel().astype(np.float32, copy=False)


def extract_lbp(image: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Return a normalized uniform-LBP histogram with P+2 bins."""
    rgb = _rgb(image)
    points, radius = int(cfg["p"]), float(cfg["r"])
    if points <= 0 or radius <= 0 or not np.isfinite(radius):
        raise ValueError("LBP p and r must be positive")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    height, width = gray.shape
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32),
                                 np.arange(height, dtype=np.float32))
    ones = np.zeros((height, width), dtype=np.uint16)
    transitions = np.zeros((height, width), dtype=np.uint16)
    first = previous = None
    for point in range(points):
        angle = 2 * np.pi * point / points
        neighbor = cv2.remap(
            gray, grid_x + np.float32(radius * np.cos(angle)),
            grid_y - np.float32(radius * np.sin(angle)),
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
        )
        bit = neighbor >= gray
        ones += bit
        if previous is not None:
            transitions += bit != previous
        else:
            first = bit
        previous = bit
    transitions += previous != first
    codes = np.where(transitions <= 2, ones, points + 1)
    histogram = np.bincount(codes.ravel(), minlength=points + 2).astype(np.float32)
    histogram /= histogram.sum()
    return histogram


def extract_color_hist(image: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Concatenate individually normalized R, G and B histograms."""
    rgb = _rgb(image)
    bins = int(cfg["bins"])
    if bins <= 0:
        raise ValueError("color_hist.bins must be positive")
    histograms = []
    for channel in range(3):
        histogram = np.histogram(rgb[:, :, channel], bins=bins, range=(0, 256))[0].astype(np.float32)
        histogram /= histogram.sum()
        histograms.append(histogram)
    return np.concatenate(histograms)


def extract_dev_features(image: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Concatenate enabled HOG, uniform-LBP and RGB histogram features."""
    features = []
    for name, extractor in (("hog", extract_hog), ("lbp", extract_lbp),
                            ("color_hist", extract_color_hist)):
        settings = cfg[name]
        if settings.get("enabled", False):
            features.append(extractor(image, settings))
    if not features:
        raise ValueError("At least one HOG/LBP/color feature must be enabled")
    return np.concatenate(features).astype(np.float32, copy=False)
