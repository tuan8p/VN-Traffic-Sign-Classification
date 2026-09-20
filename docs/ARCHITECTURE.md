# Architecture

```
raw images → offline preprocess (shared) → feature store / tensors
        ├─ SVM
        ├─ Boosting
        └─ DL (timm + optional 2×GPU DDP)
                 ↓
            outputs/<pipeline>/<run_id>/ + W&B
```

Fair-compare: same seed, splits, image size, label map, metrics from `configs/shared.yaml`.
