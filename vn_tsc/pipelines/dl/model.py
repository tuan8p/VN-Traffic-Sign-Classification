"""EfficientNetV2-B0 model builder for DL pipeline."""
from __future__ import annotations

from typing import Any

import timm
import torch
import torch.nn as nn


def build_model(cfg: dict[str, Any], num_classes: int) -> nn.Module:
    """Build a pretrained EfficientNetV2-B0 with a custom classification head.

    Uses timm's ``tf_efficientnetv2_b0`` (EfficientNet V2 B0) with pretrained
    ImageNet weights. The final classifier is replaced with a dropout + linear
    head matching ``num_classes``.

    Args:
        cfg: Resolved pipeline config dict (reads ``model.*``).
        num_classes: Number of output classes (inferred from data, not hardcoded).

    Returns:
        ``nn.Module`` with backbone frozen (ready for Stage-1 head training).
        Call :func:`unfreeze_backbone` to unlock for Stage-2.
    """
    model_cfg = cfg.get("model", {})
    backbone = model_cfg.get("backbone", "tf_efficientnetv2_b0")
    pretrained = bool(model_cfg.get("pretrained", True))
    dropout = float(model_cfg.get("dropout", 0.2))

    # timm creates the model with pretrained weights and replaces the head.
    model = timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout,
    )
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters except the final classifier head.

    Works for any timm model that exposes a ``classifier`` or ``head``
    attribute (EfficientNetV2 exposes ``classifier``).
    """
    # Freeze everything first.
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the classification head.
    head = _get_head(model)
    for param in head.parameters():
        param.requires_grad = True


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze all parameters for Stage-2 full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


def _get_head(model: nn.Module) -> nn.Module:
    """Return the classification head regardless of timm naming convention."""
    if hasattr(model, "get_classifier"):
        candidate = model.get_classifier()
        if isinstance(candidate, nn.Module):
            return candidate
    for attr in ("classifier", "head", "fc"):
        if hasattr(model, attr):
            candidate = getattr(model, attr)
            if isinstance(candidate, nn.Module):
                return candidate
    raise AttributeError(
        f"Cannot find classification head on {type(model).__name__}. "
        "Expected one of: classifier, head, fc."
    )


def count_trainable(model: nn.Module) -> int:
    """Return number of trainable parameters (useful for logging)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
