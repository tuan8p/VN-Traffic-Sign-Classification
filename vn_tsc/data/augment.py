from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np

# Augmentations that are FORBIDDEN on this dataset, and why:
#
#   horizontal flip - VNTS contains mirror-pair classes where a flip turns one
#       valid sign into a different valid sign: 7<->32 (cấm rẽ trái / phải),
#       6<->50, 1<->37, 15<->21, 19<->49. Flipping silently mislabels them.
#       Other directional signs (3, 42, 45, 47) have no mirror class at all, so
#       a flip invents a sign that does not exist.
#   vertical flip - no traffic sign is ever seen upside down.
#   hue shift    - colour is the primary semantic cue (red = cấm, blue = hiệu
#       lệnh, yellow = cảnh báo). Rotating hue rewrites the sign's category.
#   large rotation - beyond ~15 degrees an arrow's direction becomes ambiguous.


@dataclass
class AugPlan:
    """How many extra copies each class receives, and why."""

    per_class_extra: dict[int, int]
    target_count: int
    max_multiplier: int

    @property
    def total_extra(self) -> int:
        return sum(self.per_class_extra.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_count": self.target_count,
            "max_multiplier": self.max_multiplier,
            "total_extra": self.total_extra,
            "per_class_extra": {str(k): v for k, v in sorted(self.per_class_extra.items())},
        }


def plan_augmentation(
    class_ids: Sequence[int],
    target_count: int = 300,
    max_multiplier: int = 8,
) -> AugPlan:
    """Decide how many synthetic copies each class needs.

    Two rules, both needed:
      - lift rare classes toward `target_count`, so macro-F1 is not decided by
        whichever class happens to be biggest;
      - never exceed `max_multiplier` copies per real sample. Class 44 has three
        crops: blowing it up to 300 would be 100 near-identical clones of the
        same three photos, which teaches the model those photos, not the sign.
    """
    counts: dict[int, int] = {}
    for c in class_ids:
        counts[int(c)] = counts.get(int(c), 0) + 1
    extra = {
        c: min(max(0, target_count - n), n * (max_multiplier - 1))
        for c, n in counts.items()
    }
    return AugPlan(
        per_class_extra={c: e for c, e in extra.items() if e > 0},
        target_count=target_count,
        max_multiplier=max_multiplier,
    )


def _affine(img: np.ndarray, rng: np.random.Generator, ops: dict[str, Any]) -> np.ndarray:
    """Fallback geometric jitter, applied to an already-letterboxed crop.

    Prefer jitter_geometry + a fresh crop: rotating a letterboxed image also
    rotates its grey padding bars, and because augmented samples cluster in the
    rare classes the model could read "tilted padding" as a class cue.
    """
    h, w = img.shape[:2]
    angle = rng.uniform(-ops["rotate_deg"], ops["rotate_deg"])
    scale = 1.0 + rng.uniform(-ops["scale_jitter"], ops["scale_jitter"])
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    m[0, 2] += rng.uniform(-ops["translate_frac"], ops["translate_frac"]) * w
    m[1, 2] += rng.uniform(-ops["translate_frac"], ops["translate_frac"]) * h
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def rotate_about(img: np.ndarray, center: tuple[float, float], angle: float) -> np.ndarray:
    """Rotate the full frame about a point, keeping real pixels around it."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def jitter_geometry(
    geom: Any, rng: np.random.Generator, ops: dict[str, Any]
) -> tuple[Any, float]:
    """Randomly shift and rescale a crop window, and pick a rotation angle.

    Working in source-image coordinates means the jittered window is filled with
    genuine surrounding pixels, exactly as a real detector's imperfect box would
    be, and the letterbox padding added afterwards stays axis-aligned.
    """
    from dataclasses import replace

    scale = 1.0 + rng.uniform(-ops["scale_jitter"], ops["scale_jitter"])
    dx = rng.uniform(-ops["translate_frac"], ops["translate_frac"]) * geom.box_w_px
    dy = rng.uniform(-ops["translate_frac"], ops["translate_frac"]) * geom.box_h_px
    cx = (geom.x1 + geom.x2) / 2 + dx
    cy = (geom.y1 + geom.y2) / 2 + dy
    bw, bh = geom.box_w_px * scale, geom.box_h_px * scale
    jittered = replace(
        geom,
        x1=int(round(cx - bw / 2)), y1=int(round(cy - bh / 2)),
        x2=int(round(cx + bw / 2)), y2=int(round(cy + bh / 2)),
        box_w_px=max(1, int(round(bw))), box_h_px=max(1, int(round(bh))),
    )
    return jittered, rng.uniform(-ops["rotate_deg"], ops["rotate_deg"])


def _brightness_contrast(img: np.ndarray, rng: np.random.Generator, ops: dict[str, Any]) -> np.ndarray:
    alpha = 1.0 + rng.uniform(-ops["contrast"], ops["contrast"])
    beta = rng.uniform(-ops["brightness"], ops["brightness"]) * 255.0
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def _saturation(img: np.ndarray, rng: np.random.Generator, ops: dict[str, Any]) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] *= 1.0 + rng.uniform(-ops["saturation"], ops["saturation"])
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _gaussian_blur(img: np.ndarray, rng: np.random.Generator, ops: dict[str, Any]) -> np.ndarray:
    sigma = rng.uniform(0.3, ops["gaussian_blur_sigma"])
    return cv2.GaussianBlur(img, (0, 0), sigma)


def _motion_blur(img: np.ndarray, rng: np.random.Generator, ops: dict[str, Any]) -> np.ndarray:
    k = int(ops["motion_blur_ksize"])
    k = max(3, k if k % 2 == 1 else k + 1)
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    m = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1.0)
    kernel = cv2.warpAffine(kernel, m, (k, k))
    s = kernel.sum()
    if s > 0:
        kernel /= s
    return cv2.filter2D(img, -1, kernel)


def _noise(img: np.ndarray, rng: np.random.Generator, ops: dict[str, Any]) -> np.ndarray:
    std = rng.uniform(4.0, ops["gaussian_noise_std"])
    out = img.astype(np.float32) + rng.normal(0.0, std, img.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def _jpeg(img: np.ndarray, rng: np.random.Generator, ops: dict[str, Any]) -> np.ndarray:
    lo, hi = ops["jpeg_quality"]
    q = int(rng.integers(int(lo), int(hi) + 1))
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        return img
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


PHOTOMETRIC = {
    "brightness_contrast": _brightness_contrast,
    "saturation": _saturation,
    "gaussian_blur": _gaussian_blur,
    "motion_blur": _motion_blur,
    "noise": _noise,
    "jpeg": _jpeg,
}

DEFAULT_OPS: dict[str, Any] = {
    "rotate_deg": 10,
    "scale_jitter": 0.10,
    "translate_frac": 0.05,
    "brightness": 0.25,
    "contrast": 0.25,
    "saturation": 0.15,
    "gaussian_blur_sigma": 1.2,
    "motion_blur_ksize": 7,
    "gaussian_noise_std": 12,
    "jpeg_quality": [40, 90],
}


def apply_photometric(
    img: np.ndarray, rng: np.random.Generator, ops: dict[str, Any], k_max: int = 3
) -> tuple[np.ndarray, list[str]]:
    """Apply one to k_max photometric effects, in random order."""
    out = img
    applied: list[str] = []
    names = list(PHOTOMETRIC)
    k = int(rng.integers(1, k_max + 1))
    for name in rng.permutation(names)[:k]:
        out = PHOTOMETRIC[str(name)](out, rng, ops)
        applied.append(str(name))
    return out, applied


def augment_once(
    img: np.ndarray, rng: np.random.Generator, ops: dict[str, Any] | None = None
) -> tuple[np.ndarray, list[str]]:
    """Geometric jitter plus photometric effects on an already-cropped image.

    Used only when the source frame is unavailable; see augment_from_source for
    the path the offline preprocess actually takes.
    """
    ops = {**DEFAULT_OPS, **(ops or {})}
    out, applied = apply_photometric(_affine(img, rng, ops), rng, ops)
    return out, ["affine", *applied]


def augment_from_source(
    img: np.ndarray,
    geom: Any,
    rng: np.random.Generator,
    crop_fn: Any,
    ops: dict[str, Any] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Re-cut a jittered, rotated crop from the full frame, then jitter its look.

    `crop_fn(image, geometry) -> letterboxed crop` is injected so this module
    does not have to know the project's crop settings.
    """
    ops = {**DEFAULT_OPS, **(ops or {})}
    jittered, angle = jitter_geometry(geom, rng, ops)
    center = ((jittered.x1 + jittered.x2) / 2.0, (jittered.y1 + jittered.y2) / 2.0)
    rotated = rotate_about(img, center, angle) if abs(angle) > 1e-3 else img
    out, applied = apply_photometric(crop_fn(rotated, jittered), rng, ops)
    return out, ["window_jitter", "rotate", *applied]


def plan_assignments(
    class_ids: Sequence[int], plan: AugPlan
) -> list[int]:
    """Expand a plan into the list of source indices to augment.

    Sources are cycled round-robin inside each class so every real crop is used
    about equally often, rather than the RNG over-sampling a few of them.
    """
    by_class: dict[int, list[int]] = {}
    for i, c in enumerate(class_ids):
        by_class.setdefault(int(c), []).append(i)
    out: list[int] = []
    for cls in sorted(plan.per_class_extra):
        sources = by_class.get(cls, [])
        for j in range(plan.per_class_extra[cls] if sources else 0):
            out.append(sources[j % len(sources)])
    return out


def generate_augmented(
    images: np.ndarray,
    class_ids: Sequence[int],
    plan: AugPlan,
    ops: dict[str, Any] | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, list[int], list[int], list[str]]:
    """Build the extra samples from cached crops (no source frames available).

    Returns (images, class_ids, source_indices, op_descriptions).
    """
    order = plan_assignments(class_ids, plan)
    if not order:
        return np.empty((0, *images.shape[1:]), dtype=images.dtype), [], [], []
    out = np.empty((len(order), *images.shape[1:]), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    out_ops: list[str] = []
    for w, src in enumerate(order):
        img, applied = augment_once(images[src], rng, ops)
        out[w] = img
        out_ops.append("+".join(applied))
    return out, [int(class_ids[s]) for s in order], order, out_ops
