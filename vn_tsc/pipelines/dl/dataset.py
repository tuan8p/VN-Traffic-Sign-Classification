"""PyTorch Dataset wrapper for Deep Learning pipeline."""
from __future__ import annotations

from typing import Callable

import numpy as np
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
        X_aug: Optional offline-augmented images (same layout as X).
        y_aug: Labels aligned with ``X_aug``. When set, ``__len__`` is
            ``len(y) + len(y_aug)`` without concatenating arrays in RAM.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        transform: Callable | None = None,
        X_aug: np.ndarray | None = None,
        y_aug: np.ndarray | None = None,
    ) -> None:
        if X.ndim != 4 or X.shape[-1] != 3:
            raise ValueError(f"X must be (N, H, W, 3), got {X.shape}")
        if y.ndim != 1 or len(X) != len(y):
            raise ValueError(f"y must be 1-D with len={len(X)}, got {y.shape}")
        self.X = X
        self.y = y
        self.transform = transform
        self.X_aug = X_aug
        self.y_aug = y_aug
        if X_aug is not None or y_aug is not None:
            if X_aug is None or y_aug is None:
                raise ValueError("X_aug and y_aug must both be set or both omitted")
            if X_aug.ndim != 4 or X_aug.shape[-1] != 3:
                raise ValueError(f"X_aug must be (N, H, W, 3), got {X_aug.shape}")
            if y_aug.ndim != 1 or len(X_aug) != len(y_aug):
                raise ValueError(f"y_aug must be 1-D with len={len(X_aug)}, got {y_aug.shape}")
        self._n_base = len(y)
        self._n_aug = len(y_aug) if y_aug is not None else 0

    def __len__(self) -> int:
        return self._n_base + self._n_aug

    def __getitem__(self, idx: int):
        if idx < self._n_base:
            image = self.X[idx]
            label = int(self.y[idx])
        else:
            j = idx - self._n_base
            image = self.X_aug[j]  # type: ignore[index]
            label = int(self.y_aug[j])  # type: ignore[index]
        # One small contiguous buffer per sample (works with mmap + torchvision).
        image = np.ascontiguousarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label
