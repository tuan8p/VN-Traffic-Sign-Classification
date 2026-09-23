# Data contract — what preprocessing hands to SVM / Boosting / DL

Everything below is produced by one command and consumed identically by all
three pipelines. If a pipeline loads data any other way, the comparison in
`docs/FAIR_COMPARE.md` no longer holds.

```bash
python -m tools.run_preprocess --data-root <path to VNTS>
```

## What VNTS actually is, and what we turned it into

VNTS ships as a **YOLO detection** set: 3,216 street photos plus one `.txt` of
bounding boxes each. This project is **classification**, so preprocessing cuts
every box out of its frame and treats it as one labelled sample.

| | count |
|---|---|
| source images | 3,216 (25 have no sign and are dropped) |
| boxes in the labels | 8,334 |
| boxes kept (`min_box_px = 12`) | 7,989 |
| classes | 52 |
| crops after offline augmentation | 15,440 |

## Files

```
data/processed/
  images/
    X_train.npy        (5564, 224, 224, 3) uint8, RGB
    y_train.npy        (5564,) int16, class id 0..51
    X_val.npy          (1212, ...)      y_val.npy
    X_test.npy         (1213, ...)      y_test.npy
    X_train_aug.npy    (7451, ...)      y_train_aug.npy   <- extras only
  features/
    F_train.npy        (5564, 2020) float32   HOG | LBP | colour
    F_val.npy, F_test.npy, F_train_aug.npy
    scaler.joblib      StandardScaler fitted on TRAIN (+aug) only
    feature_report.json
  metadata.csv         one row per crop, joins to the arrays by array_index
  class_table.csv      id, sign code, Vietnamese name, mirror pair
  preprocess_report.json
  resolved_preprocess.yaml
```

`X_train_aug.npy` holds **only** the synthetic extras, not a second copy of
train. Ask for them with `with_aug=True` and the loader concatenates.

## How to load

```python
from vn_tsc.data.dataset import load_split, load_features, load_class_table

# DL — uint8 HWC RGB, memory-mapped (train is ~840 MB)
X, y = load_split("train", with_aug=True)
Xv, yv = load_split("val")

# SVM / Boosting — shared classical features, scaler already applied
F, y = load_features("train", with_aug=True)
Fv, yv = load_features("val")

table = load_class_table()
table[7].name_vi          # 'Cấm rẽ trái'
table.mirror_pairs()      # [(1, 37), (6, 50), (7, 32), (15, 21), (19, 49)]
```

Normalisation for DL is **not** baked in: divide by 255 and apply whatever
mean/std your backbone expects, in `vn_tsc/data/preprocess_online.py`.

## Rules that keep the comparison fair

1. **`with_aug=True` is for train only.** The loader raises on val/test. The
   augmented crops are shared by all three pipelines precisely so that no one
   gets a data advantage.
2. **Never refit the scaler on val or test.** `load_features` applies the saved
   train-fitted scaler. PCA is deliberately *not* in the store: it is a
   per-pipeline key (`features.pca` in `svm.yaml` / `boosting.yaml`), so fit it
   on train inside your pipeline if you enable it.
3. **Report the split you used.** The default store is the leakage-free group
   split. `--no-augment` rebuilds the same store without synthetic data, for the
   with/without ablation.

## Two dataset properties that will bite you

**Horizontal flip is forbidden.** VNTS contains five mirror-pair classes where a
flip turns one valid sign into a *different* valid sign:

| | |
|---|---|
| 7 `P.123a` Cấm rẽ trái | 32 `P.123b` Cấm rẽ phải |
| 6 `W.201a` Ngoặt vòng bên trái | 50 `W.201b` Ngoặt vòng bên phải |
| 1 `W.205c` Ngã ba bên phải | 37 `W.205b` Ngã ba bên trái |
| 15 `W.203c` Thu hẹp phía phải | 21 `W.203b` Thu hẹp phía trái |
| 19 `P.124d` Cấm rẽ phải và quay đầu | 49 `P.124c` Cấm rẽ trái và quay đầu |

Other directional signs (3, 42, 45, 47) have no mirror class at all, so flipping
them invents a sign that does not exist. See
`analysis/figures/07_mirror_pairs.png`.

> **`configs/pipelines/dl.yaml` currently sets `aug.hflip: true`.** That needs to
> become `false` — team-dl owns that file.

**Macro-F1 over 52 classes is noisy.** 13 classes have fewer than 30 crops in
total; class 44 (`W.233`) has 3, which cannot reach all three splits — it gets
2 train / 1 val / 0 test. Report macro-F1 twice: over all 52, and over the
classes above `metrics.rare_class_threshold` (30). `vn_tsc.data.dataset.rare_classes()`
returns the list.

## metadata.csv columns

| column | meaning |
|---|---|
| `crop_id` | `<source image>_<label line>`, or `aug_NNNNNN` |
| `split` | `train` / `val` / `test` / `train_aug` |
| `array_index` | row in `X_{split}.npy` — the join key |
| `class_id`, `sign_code` | label |
| `source_image` | stem of the original JPEG |
| `group_id` | near-duplicate cluster; never spans two splits |
| `x1,y1,x2,y2`, `box_w_px`, `box_h_px`, `box_area_px` | geometry in the ORIGINAL image |
| `img_w`, `img_h` | source frame size |
| `clipped` | box edge ran past the frame and was clipped |
| `is_aug`, `aug_source_crop_id`, `aug_ops` | provenance of synthetic rows |

`box_area_px` is worth keeping: 44% of signs are smaller than 32×32 px, so
breaking accuracy down by original size explains far more than a single number.
