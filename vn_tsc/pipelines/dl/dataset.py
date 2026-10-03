"""PyTorch Dataset wrapper for Deep Learning pipeline."""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset


class NpyDataset(Dataset):
    """PyTorch Dataset wrapping memory-mapped .npy image/label arrays.

    Applies per-sample online transforms (e.g. normalization, color jitter)
    at batch collation time.

    Args:
        X: uint8 numpy array of shape (N, H, W, 3).
        y: integer numpy array of shape (N,).
        transform: Optional callable applied to each HWC uint8 ndarray,
            returning a float32 CHW Tensor in [0, 1].
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        transform: Callable | None = None,
    ) -> None:
        if X.ndim != 4 or X.shape[-1] != 3:
            raise ValueError(f"X must be (N, H, W, 3), got {X.shape}")
        if y.ndim != 1 or len(X) != len(y):
            raise ValueError(f"y must be 1-D with len={len(X)}, got {y.shape}")
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        image = self.X[idx]  # uint8 (H, W, 3)
        label = int(self.y[idx])
        if self.transform is not None:
            image = self.transform(image)
        return image, label
