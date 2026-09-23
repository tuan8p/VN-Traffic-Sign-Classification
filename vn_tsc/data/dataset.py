from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vn_tsc.data.classes import ClassTable

DEFAULT_PROCESSED = "data/processed"
REAL_SPLITS = ("train", "val", "test")


@dataclass
class Sample:
    """One crop. `array_index` is its row in X_{split}.npy / y_{split}.npy."""

    crop_id: str
    split: str
    array_index: int
    class_id: int
    sign_code: str
    source_image: str
    group_id: int
    box_w_px: int
    box_h_px: int
    box_area_px: int
    is_aug: int

    @property
    def path(self) -> Path:
        """Kept for the original scaffold's interface; crops live in a .npy."""
        return Path(f"{self.split}#{self.array_index}")

    @property
    def label(self) -> str:
        return self.sign_code


class TrafficSignDataset:
    """The index side of the processed store: metadata.csv as Sample objects."""

    def __init__(self, index: list[Sample], root: Path | None = None):
        self.index = index
        self.root = root

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Sample:
        return self.index[i]

    def filter(self, **conditions: Any) -> "TrafficSignDataset":
        def keep(s: Sample) -> bool:
            return all(getattr(s, k) == v for k, v in conditions.items())

        return TrafficSignDataset([s for s in self.index if keep(s)], self.root)

    def class_counts(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for s in self.index:
            out[s.class_id] = out.get(s.class_id, 0) + 1
        return dict(sorted(out.items()))

    @classmethod
    def from_processed(cls, processed_root: str | Path = DEFAULT_PROCESSED) -> "TrafficSignDataset":
        import csv

        root = Path(processed_root)
        meta = root / "metadata.csv"
        if not meta.exists():
            raise FileNotFoundError(
                f"{meta} not found — run `python -m tools.run_preprocess` first"
            )
        index: list[Sample] = []
        with open(meta, "r", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                index.append(
                    Sample(
                        crop_id=r["crop_id"],
                        split=r["split"],
                        array_index=int(r["array_index"]),
                        class_id=int(r["class_id"]),
                        sign_code=r["sign_code"],
                        source_image=r["source_image"],
                        group_id=int(r["group_id"]),
                        box_w_px=int(r["box_w_px"]),
                        box_h_px=int(r["box_h_px"]),
                        box_area_px=int(r["box_area_px"]),
                        is_aug=int(r["is_aug"]),
                    )
                )
        return cls(index, root)


def _load_pair(root: Path, split: str, mmap: bool) -> tuple[np.ndarray, np.ndarray]:
    x = root / "images" / f"X_{split}.npy"
    y = root / "images" / f"y_{split}.npy"
    if not x.exists():
        raise FileNotFoundError(f"{x} not found — run tools.run_preprocess first")
    return np.load(x, mmap_mode="r" if mmap else None), np.load(y)


def load_split(
    split: str,
    processed_root: str | Path = DEFAULT_PROCESSED,
    with_aug: bool = False,
    mmap: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Cached crops for one split: uint8 (N, H, W, 3) RGB and int16 labels.

    `with_aug=True` appends the shared offline augmentation and is only legal
    for train — val and test must stay untouched or the comparison is void.
    Arrays are memory-mapped by default; the train split is ~840 MB.
    """
    root = Path(processed_root)
    X, y = _load_pair(root, split, mmap)
    if not with_aug:
        return X, y
    if split != "train":
        raise ValueError(f"augmented data exists for train only, not {split!r}")
    if not (root / "images" / "X_train_aug.npy").exists():
        return X, y
    Xa, ya = _load_pair(root, "train_aug", mmap)
    return np.concatenate([np.asarray(X), np.asarray(Xa)]), np.concatenate([y, ya])


def load_features(
    split: str,
    processed_root: str | Path = DEFAULT_PROCESSED,
    with_aug: bool = False,
    standardize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Shared HOG + LBP + colour features for SVM and Boosting.

    Applies the StandardScaler that was fitted on train only, so calling this
    for val or test cannot leak their statistics into the model.
    """
    root = Path(processed_root)
    fdir = root / "features"
    path = fdir / f"F_{split}.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python -m tools.run_preprocess` (features step)"
        )
    F = np.load(path)
    _, y = _load_pair(root, split, mmap=True)
    if with_aug:
        if split != "train":
            raise ValueError(f"augmented data exists for train only, not {split!r}")
        aug = fdir / "F_train_aug.npy"
        if aug.exists():
            _, ya = _load_pair(root, "train_aug", mmap=True)
            F = np.concatenate([F, np.load(aug)])
            y = np.concatenate([y, ya])
    if standardize:
        scaler_path = fdir / "scaler.joblib"
        if scaler_path.exists():
            from joblib import load

            F = load(scaler_path).transform(F).astype(np.float32)
    return F, y


def load_class_table(processed_root: str | Path = DEFAULT_PROCESSED) -> ClassTable:
    """The class list that was actually used, copied beside the arrays."""
    root = Path(processed_root)
    local = root / "class_table.csv"
    return ClassTable.load(local if local.exists() else "configs/classes.csv")


def class_weights(
    y: np.ndarray, num_classes: int | None = None
) -> dict[int, float]:
    """Balanced weights for pipelines that cannot resample (n / (k * count))."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    k = num_classes or len(classes)
    return {int(c): float(len(y) / (k * n)) for c, n in zip(classes, counts)}


def rare_classes(
    processed_root: str | Path = DEFAULT_PROCESSED, threshold: int = 30
) -> list[int]:
    """Classes with fewer than `threshold` real (non-augmented) crops overall.

    Reporting macro-F1 with and without these is what metrics.rare_class_threshold
    in shared.yaml is for: with 3 test crops a per-class F1 is close to noise.
    """
    ds = TrafficSignDataset.from_processed(processed_root)
    counts: dict[int, int] = {}
    for s in ds.index:
        if not s.is_aug:
            counts[s.class_id] = counts.get(s.class_id, 0) + 1
    return sorted(c for c, n in counts.items() if n < threshold)
