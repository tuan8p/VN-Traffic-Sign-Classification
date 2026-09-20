# Fair comparison contract

Locked in `configs/shared.yaml`:

- seed, stratified split ratios
- image size / color mode
- metrics (accuracy, macro-F1, weighted-F1)

May differ: model hyperparameters, DL online aug (must be logged).

Default: SVM & Boosting share classical HOG/LBP/color features.
