# DL backbones

## Primary (recommended)
- efficientnet_b0 (CNN) — best fit for traffic-sign classification on 2×T4.

## Backup (Transformer)
timm names ready in config comments / n_tsc/pipelines/dl/model.py:
- it_tiny_patch16_224
- it_small_patch16_224
- deit_tiny_patch16_224

Set in configs/pipelines/dl.yaml or notebook OVERRIDES:
`yaml
model:
  backbone: vit_tiny_patch16_224
  family: transformer
  transformer:
    use_flash_attn: false
`

Flash Attention: only consider for Transformer path, and only if the Kaggle torch build supports it. Default **off**. CNN path never needs it.

Fair-compare tip: report CNN as main result; ViT as ablation/backup in the report.
