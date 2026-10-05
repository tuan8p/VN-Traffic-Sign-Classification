from __future__ import annotations
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from tqdm import tqdm

from vn_tsc.utils.io import save_json
from vn_tsc.utils.logging_utils import setup_logger

log = setup_logger("features")

FEATURE_SPLITS = ("train", "val", "test", "train_aug")


def to_feature_image(image: np.ndarray, input_size: tuple[int, int]) -> np.ndarray:
    """Shrink a cached crop to the size the classical features run at.

    HOG on the full 224x224 cache with 8px cells is 26,244-D; an RBF SVM over
    that many dimensions and ~13k samples does not finish in a course project.
    At 64x64 the same cell geometry gives 1,764-D and keeps the sign's outline.
    """
    w, h = input_size
    if image.shape[:2] == (h, w):
        return image
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)


def extract_hog(image: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Shape of the sign: edge-orientation histograms over a fixed cell grid."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    return hog(
        gray,
        orientations=int(cfg.get("orientations", 9)),
        pixels_per_cell=tuple(cfg.get("pixels_per_cell", (8, 8))),
        cells_per_block=tuple(cfg.get("cells_per_block", (2, 2))),
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)


def extract_lbp(image: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Local texture, histogrammed over a spatial grid.

    A single global histogram throws away where the texture sits, which is what
    separates a sign's patterned interior from its plain border, so the crop is
    tiled first and one uniform-LBP histogram is taken per tile.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    p = int(cfg.get("p", 8))
    r = float(cfg.get("r", 1))
    gy, gx = (int(v) for v in cfg.get("grid", (4, 4)))
    codes = local_binary_pattern(gray, p, r, method="uniform")
    n_bins = p + 2
    h, w = codes.shape
    out = []
    for i in range(gy):
        for j in range(gx):
            tile = codes[
                i * h // gy : (i + 1) * h // gy, j * w // gx : (j + 1) * w // gx
            ]
            hist, _ = np.histogram(tile, bins=n_bins, range=(0, n_bins), density=False)
            total = hist.sum()
            out.append(hist / total if total else hist.astype(np.float64))
    return np.concatenate(out).astype(np.float32)


def extract_color_hist(image: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Colour signature. HSV by default: a sign's hue is part of its meaning
    (red = cấm, blue = hiệu lệnh, yellow = cảnh báo), and separating hue from
    brightness keeps that cue stable across the set's lighting extremes."""
    bins = int(cfg.get("bins", 32))
    space = str(cfg.get("color_space", "hsv")).lower()
    img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) if space == "hsv" else image
    out = []
    for c in range(img.shape[2]):
        hist = cv2.calcHist([img], [c], None, [bins], [0, 256]).ravel()
        total = hist.sum()
        out.append(hist / total if total else hist)
    return np.concatenate(out).astype(np.float32)


def extract_shared_features(image: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """The one feature vector SVM and Boosting both consume."""
    feats = []
    fcfg = cfg.get("features", cfg)
    if fcfg.get("hog", {}).get("enabled", True):
        feats.append(extract_hog(image, fcfg.get("hog", {})))
    if fcfg.get("lbp", {}).get("enabled", True):
        feats.append(extract_lbp(image, fcfg.get("lbp", {})))
    if fcfg.get("color_hist", {}).get("enabled", True):
        feats.append(extract_color_hist(image, fcfg.get("color_hist", {})))
    if not feats:
        raise ValueError("No classical features enabled")
    return np.concatenate([f.ravel() for f in feats], axis=0)


def feature_layout(cfg: dict[str, Any], input_size: tuple[int, int]) -> dict[str, Any]:
    """Which slice of the vector each block owns — needed to read importances."""
    fcfg = cfg.get("features", cfg)
    probe = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
    layout: dict[str, Any] = {"blocks": {}, "total": 0}
    start = 0
    for name, fn in (
        ("hog", extract_hog), ("lbp", extract_lbp), ("color_hist", extract_color_hist)
    ):
        if not fcfg.get(name, {}).get("enabled", True):
            continue
        n = int(fn(probe, fcfg.get(name, {})).size)
        layout["blocks"][name] = {"start": start, "end": start + n, "dim": n}
        start += n
    layout["total"] = start
    return layout


def _iter_batches(path: Path, batch: int) -> Iterator[np.ndarray]:
    arr = np.load(path, mmap_mode="r")
    for i in range(0, len(arr), batch):
        yield np.asarray(arr[i : i + batch])


def build_feature_store(
    cfg: dict[str, Any],
    processed_root: str | Path,
    batch: int = 512,
) -> Path:
    """Turn the cached crops into the shared classical feature matrices.

    The scaler is fitted on TRAIN ONLY and reused for val/test, so no statistic
    from the evaluation splits can leak into training. PCA is deliberately left
    out: it is a per-pipeline key in svm.yaml / boosting.yaml, so each pipeline
    fits its own on top of this store.
    """
    from joblib import dump
    from sklearn.preprocessing import StandardScaler

    processed_root = Path(processed_root)
    out_dir = processed_root / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    fcfg = cfg.get("features", {})
    size = tuple(fcfg.get("input_size", [64, 64]))
    input_size = (int(size[0]), int(size[1]))
    layout = feature_layout(cfg, input_size)
    log.info("feature dim = %d %s", layout["total"],
             {k: v["dim"] for k, v in layout["blocks"].items()})

    written: dict[str, int] = {}
    for split in FEATURE_SPLITS:
        src = processed_root / "images" / f"X_{split}.npy"
        if not src.exists():
            continue
        n = len(np.load(src, mmap_mode="r"))
        out = np.lib.format.open_memmap(
            out_dir / f"F_{split}.npy", mode="w+",
            dtype=np.float32, shape=(n, layout["total"]),
        )
        w = 0
        for chunk in tqdm(
            _iter_batches(src, batch), total=(n + batch - 1) // batch,
            desc=f"features {split}", unit="batch",
        ):
            for img in chunk:
                out[w] = extract_shared_features(
                    to_feature_image(img, input_size), cfg
                )
                w += 1
        out.flush()
        written[split] = n

    if fcfg.get("standardize", True) and "train" in written:
        scaler = StandardScaler()
        train = np.load(out_dir / "F_train.npy", mmap_mode="r")
        for i in range(0, len(train), batch):
            scaler.partial_fit(np.asarray(train[i : i + batch]))
        # Augmented train rows are training data, so they may inform the scaler
        # too; val and test must not.
        aug_path = out_dir / "F_train_aug.npy"
        if aug_path.exists():
            aug = np.load(aug_path, mmap_mode="r")
            for i in range(0, len(aug), batch):
                scaler.partial_fit(np.asarray(aug[i : i + batch]))
        dump(scaler, out_dir / "scaler.joblib")
        log.info("fitted StandardScaler on train (+aug) and saved scaler.joblib")

    save_json(
        {
            "input_size": list(input_size),
            "layout": layout,
            "rows": written,
            "standardized": bool(fcfg.get("standardize", True)),
            "config": {k: fcfg.get(k) for k in ("hog", "lbp", "color_hist")},
        },
        out_dir / "feature_report.json",
    )
    return out_dir
