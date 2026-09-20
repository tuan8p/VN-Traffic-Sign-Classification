from __future__ import annotations
from typing import Any

# Recommended primary: efficientnet_b0 (CNN)
# Backup Transformer (timm names):
BACKUP_TRANSFORMERS = (
    "vit_tiny_patch16_224",
    "vit_small_patch16_224",
    "deit_tiny_patch16_224",
)

CNN_DEFAULTS = (
    "efficientnet_b0",
    "resnet18",
    "mobilenetv3_small_100",
    "convnext_tiny",
)


def build_model(cfg: dict[str, Any]):
    """TODO(team-dl): timm.create_model(backbone, pretrained=..., num_classes=...).

    Primary path: CNN (efficientnet_b0).
    Backup path: set model.backbone to a name in BACKUP_TRANSFORMERS
    (and model.family: transformer). Flash Attention only if use_flash_attn
    and backbone is Transformer — do not enable for CNN.
    """
    model_cfg = cfg.get("model", {})
    backbone = model_cfg.get("backbone", "efficientnet_b0")
    family = model_cfg.get("family", "cnn")
    _ = (backbone, family, model_cfg.get("transformer", {}))
    raise NotImplementedError(
        "TODO(team-dl): build_model — primary CNN; backup ViT via timm "
        f"(e.g. {BACKUP_TRANSFORMERS[0]})"
    )
