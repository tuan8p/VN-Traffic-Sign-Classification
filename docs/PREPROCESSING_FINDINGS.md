# Preprocessing findings — measured, not assumed

Every number here comes from `data/processed/preprocess_report.json` and
`analysis/figures/eda_summary.json`, reproducible with
`python -m tools.run_preprocess && python -m analysis.eda`.

## 1. VNTS is a detection set; we made it a classification set

3,216 street photos with 8,334 YOLO boxes over 52 classes. Each box becomes one
224×224 crop with a 15% context margin, letterboxed so circular signs stay circular.

| step | crops |
|---|---|
| boxes in the labels | 8,334 |
| dropped, shorter side < 12 px | 345 (4.1%) |
| **kept** | **7,989** |
| + offline augmentation (train only) | 15,440 total |

Dropping sub-12px boxes costs no class more than 30% of its samples and wipes out
none — the 345 losses all fall in common classes. 25 images carry no sign at all
and produce no crops.

## 2. Label validation

| check | result |
|---|---|
| images without a label / labels without an image | 0 / 0 |
| malformed label lines | 0 |
| boxes with zero or negative size | 0 |
| boxes genuinely outside the frame | **1** (`0248`, class 49, overshoots by 1.9% of width) |

A naive bounds check flags 29 boxes, but 28 of those overshoot by exactly 5e-7 —
an artefact of YOLO's 6-decimal format, not a defect. `discover.CLIP_TOL` separates
the two so the report is not inflated.

## 3. Class imbalance is the real problem

In the raw labels, `P.130` has 1,071 boxes against **3** for `W.233` — 357:1. After
the sub-12px filter the kept crops run 1,014 : 3 (**338:1**), and within the train
split alone 710 : 2 (**355:1**). 13 classes have fewer than 30 crops in total.

Class 44 (`W.233`, 3 crops) mathematically cannot reach all three splits; the
splitter prioritises train and it ends up 2 / 1 / 0. This is reported, not hidden.

Offline class-aware augmentation takes train from 5,564 to 13,015 crops and the
imbalance from **355:1 to 44:1**, capped at 8× per real sample so a 3-crop class
becomes 16 varied samples rather than 300 clones of the same three photos.

## 4. Horizontal flip would corrupt the labels

Five mirror-pair classes exist, where a flip produces a *different valid sign*:

| | |
|---|---|
| 7 `P.123a` Cấm rẽ trái | 32 `P.123b` Cấm rẽ phải |
| 6 `W.201a` Ngoặt vòng bên trái | 50 `W.201b` Ngoặt vòng bên phải |
| 1 `W.205c` Ngã ba bên phải | 37 `W.205b` Ngã ba bên trái |
| 15 `W.203c` Thu hẹp phía phải | 21 `W.203b` Thu hẹp phía trái |
| 19 `P.124d` Cấm rẽ phải và quay đầu | 49 `P.124c` Cấm rẽ trái và quay đầu |

Classes 3, 42, 45 and 47 are directional with *no* mirror class, so flipping them
invents a sign that does not exist. `analysis/figures/07_mirror_pairs.png` shows
the pairs side by side. The augmentation module refuses to run with `hflip: true`.

**`configs/pipelines/dl.yaml` still has `aug.hflip: true` — team-dl must change it.**

## 5. Signs are small

Median shorter side is 34 px; **44% of crops are smaller than 32×32 px**, the COCO
definition of a small object. Everything is upscaled to 224×224 with cubic
interpolation, so the cache looks sharp but carries no detail the original lacked.
`metadata.csv` keeps `box_area_px` so accuracy can be broken down by real size —
a far more informative table than one headline number.

## 6. Leakage: real, fixed, and smaller than expected

VNTS frames come from video. Perceptual-hash clustering (dHash, 256-bit,
Hamming ≤ 24) finds **2,912 clusters among 3,216 frames**; 199 clusters hold more
than one frame, the largest 11.

| split | images inside a cluster that straddles two splits |
|---|---|
| the dataset's own `split_dataset/` | **211** |
| our group split | **0** |

**But the measured effect on results is negligible.** With the training-set size
held equal, a naive random split scores *no better* than our leakage-free split:

| model | group split (ours) | naive random split | difference |
|---|---|---|---|
| 1-NN (pure memorisation) | 97.44% / 96.66 macro-F1 | 96.87% / 96.39 | −0.6 / −0.3 pts |
| LinearSVC + PCA-256 | 93.98% / 91.90 macro-F1 | 93.90% / 91.13 | −0.1 / −0.8 pts |

The naive split is, if anything, marginally *worse* — the differences sit inside
run-to-run noise. This holds even though in that naive split **79.5% of test crops
have a crop from the same photo sitting in train**: sibling crops from one street
photo are usually *different* signs, so they do not help classify one another.

And there is no residual leakage hiding at the sign level: cross-checking every
test crop against every train crop, only **1 of 1,213 (0.1%)** has a near-identical
partner in train, and removing it changes 1-NN accuracy by 0.00 points
(97.44% either way).

> An earlier comparison appeared to show leakage inflating scores by ~0.5–1.0
> points, but the naive split there had 6,776 training crops against our 5,564.
> The gain came from the extra data, not from leakage. Matched sizes show no gain.

**Why so small?** Traffic signs are standardised objects — a `P.130` looks the same
everywhere — so having a duplicate frame in train adds little over having any other
example of that class. That is also why 1-NN on hand-crafted features already
reaches 97%.

**Conclusion:** keep the group split. It is the methodologically correct choice, it
removes a confound for free, and it lets the report state a measured fact rather
than an assumption. It is not, however, the headline result — the class imbalance
and the flip hazard matter far more for this dataset.

## 7. Feature store sizing

HOG on the full 224×224 cache at 8px cells is **26,244-D**, which makes an RBF SVM
over ~13k samples impractical. Classical features therefore run on a 64×64
downscale, keeping the same cell geometry:

| block | dim |
|---|---|
| HOG (9 orientations, 8px cells, 2×2 blocks) | 1,764 |
| LBP (uniform, P=8, R=1, 4×4 spatial grid) | 160 |
| Colour histogram (HSV, 32 bins × 3) | 96 |
| **total** | **2,020** |

The `StandardScaler` is fitted on train (plus its augmented rows) only. PCA is
deliberately left out of the store: it is a per-pipeline key in `svm.yaml` and
`boosting.yaml`, so each pipeline fits its own.

## 8. Baseline, for calibration only

LinearSVC + PCA-256 on the shared features, our split, no tuning:
**93.98% accuracy, 91.90 macro-F1**. Offline augmentation moved it +0.2 / +0.1 —
within noise for a linear model, though CNNs typically gain more from augmentation.
This is a floor for the three real pipelines, not a result to report.

## 9. Config changes requiring a team vote

`configs/shared.yaml` is locked. One existing key changed:

```diff
 split:
-  strategy: stratified
+  strategy: stratified_group
```

Everything else is additive: `data.class_table`, `preprocess.bbox_crop`,
`split.dedup`, `split.group_by`, `split.min_train_per_class`,
`features.input_size`, `features.lbp.grid`, `features.color_hist.color_space`,
`features.standardize`, the whole `augment` block, and
`metrics.rare_class_threshold`.
