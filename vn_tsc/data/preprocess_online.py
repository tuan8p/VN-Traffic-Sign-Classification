"""Online image transforms for the DL pipeline.

These are applied *per-sample at load time* (not offline). They are intentionally
kept lightweight because the shared preprocessing already handles crop + resize to
224×224 and the offline augmentation step already wrote ``X_train_aug.npy``.

EfficientNetV2-B0 was pretrained on ImageNet — we normalise with ImageNet stats.
"""
from __future__ import annotations

from typing import Any

import torch
from torchvision import transforms


# ImageNet normalisation constants (used because the backbone is pretrained on it).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transforms(cfg: dict[str, Any]) -> transforms.Compose:
    """Online augmentation applied to each training sample.

    Input: uint8 numpy array (H, W, 3) coming from the .npy store.
    Output: float32 tensor (3, H, W) normalised with ImageNet stats.

    Augmentation ops are read from ``cfg['aug']``:
      - ``hflip`` (bool, default True)   — random horizontal flip
      - ``color_jitter`` (float 0-1)     — brightness/contrast/saturation jitter
      - ``random_resized_crop`` (bool)   — RRC from 0.8–1.0 scale range

    Note: traffic signs in this dataset have mirror-pair classes (see shared.yaml),
    so hflip is intentionally controlled per config.
    """
    aug_cfg = cfg.get("aug", {})
    img_size = cfg.get("preprocess", {}).get("image_size", [224, 224])
    h, w = img_size[1], img_size[0]  # [width, height] → (H, W)

    ops: list[Any] = []

    # Convert numpy uint8 HWC → float tensor CHW in [0, 1].
    ops.append(transforms.ToTensor())

    # Optional: random resized crop (disabled by default for small crops).
    if aug_cfg.get("random_resized_crop", False):
        ops.append(
            transforms.RandomResizedCrop(
                (h, w),
                scale=(0.8, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            )
        )

    # Random horizontal flip (disabled by default — mirror-pair classes exist).
    if aug_cfg.get("hflip", True):
        ops.append(transforms.RandomHorizontalFlip())

    # Colour jitter.
    jitter = float(aug_cfg.get("color_jitter", 0.0))
    if jitter > 0:
        ops.append(transforms.ColorJitter(
            brightness=jitter, contrast=jitter, saturation=jitter,
        ))

    # ImageNet normalisation.
    ops.append(transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD))

    return transforms.Compose(ops)


def build_eval_transforms(cfg: dict[str, Any]) -> transforms.Compose:
    """Minimal transform for validation and test: only ToTensor + normalise.

    Input: uint8 numpy array (H, W, 3).
    Output: float32 tensor (3, H, W).
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
