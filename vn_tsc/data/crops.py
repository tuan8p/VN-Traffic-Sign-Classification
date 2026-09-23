from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PAD_MODES = {"replicate": cv2.BORDER_REPLICATE, "constant": cv2.BORDER_CONSTANT}


def imread_rgb(path: str | Path) -> np.ndarray:
    """Read an image as RGB uint8.

    Goes through imdecode rather than cv2.imread so non-ASCII paths work on
    Windows, where imread silently returns None for them.
    """
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot decode image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _resize(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """INTER_AREA shrinks without aliasing, INTER_CUBIC keeps upscaled edges
    sharp. The median VNTS sign is ~28 px, so most crops take the cubic path."""
    w, h = size
    shrinking = w * h < img.shape[0] * img.shape[1]
    interp = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
    return cv2.resize(img, (w, h), interpolation=interp)


def letterbox(img: np.ndarray, out_size: tuple[int, int], pad_value: int = 114) -> np.ndarray:
    """Resize preserving aspect ratio, then pad to out_size (w, h).

    Traffic signs are circles and triangles; a plain stretch to a square would
    turn a P.102 disc into an ellipse and destroy the shape cue HOG relies on.
    """
    out_w, out_h = out_size
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("empty crop")
    scale = min(out_w / w, out_h / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = _resize(img, (new_w, new_h))
    top = (out_h - new_h) // 2
    bottom = out_h - new_h - top
    left = (out_w - new_w) // 2
    right = out_w - new_w - left
    if top or bottom or left or right:
        resized = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=(pad_value, pad_value, pad_value),
        )
    return resized


@dataclass
class CropGeometry:
    """Pixel geometry of one crop, kept for metadata and size-stratified eval."""

    x1: int
    y1: int
    x2: int
    y2: int
    box_w_px: int
    box_h_px: int
    img_w: int
    img_h: int

    @property
    def area_px(self) -> int:
        return self.box_w_px * self.box_h_px

    def to_dict(self) -> dict[str, Any]:
        return {
            "x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2,
            "box_w_px": self.box_w_px, "box_h_px": self.box_h_px,
            "box_area_px": self.area_px,
            "img_w": self.img_w, "img_h": self.img_h,
        }


def box_geometry(
    cx: float, cy: float, w: float, h: float, img_w: int, img_h: int
) -> CropGeometry:
    """Normalised YOLO box -> pixel box, without any margin applied."""
    bw, bh = w * img_w, h * img_h
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    return CropGeometry(
        x1=int(round(x1)), y1=int(round(y1)),
        x2=int(round(x1 + bw)), y2=int(round(y1 + bh)),
        box_w_px=int(round(bw)), box_h_px=int(round(bh)),
        img_w=img_w, img_h=img_h,
    )


def crop_box(
    img: np.ndarray,
    geom: CropGeometry,
    out_size: tuple[int, int],
    context_margin: float = 0.15,
    pad_value: int = 114,
    pad_mode: str = "replicate",
) -> np.ndarray:
    """Cut one sign out of a full frame and letterbox it to out_size.

    When the margin pushes the window past the frame edge we pad instead of
    clamping, so the sign stays centred. Clamping would shove signs photographed
    at the image border off-centre and make them look different from the rest.
    """
    img_h, img_w = img.shape[:2]
    mx = geom.box_w_px * context_margin
    my = geom.box_h_px * context_margin
    x1, y1 = int(np.floor(geom.x1 - mx)), int(np.floor(geom.y1 - my))
    x2, y2 = int(np.ceil(geom.x2 + mx)), int(np.ceil(geom.y2 + my))

    pad_l, pad_t = max(0, -x1), max(0, -y1)
    pad_r, pad_b = max(0, x2 - img_w), max(0, y2 - img_h)
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(img_w, x2), min(img_h, y2)
    if cx2 <= cx1 or cy2 <= cy1:
        raise ValueError("crop window falls outside the image")

    patch = img[cy1:cy2, cx1:cx2]
    if pad_l or pad_t or pad_r or pad_b:
        border = PAD_MODES.get(pad_mode, cv2.BORDER_REPLICATE)
        patch = cv2.copyMakeBorder(
            patch, pad_t, pad_b, pad_l, pad_r, border,
            value=(pad_value, pad_value, pad_value),
        )
    return letterbox(patch, out_size, pad_value)
