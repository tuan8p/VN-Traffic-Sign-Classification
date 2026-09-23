# Meeting checklist (19:00)

- [ ] Confirm dataset path / Kaggle slug → `DATA_ROOT`
- [ ] Create role branches
- [ ] Preprocessing: `vn_tsc/data/preprocess_offline.py` + classical features
- [ ] SVM / Boosting / DL: fill pipeline TODOs
- [ ] Demo: Streamlit predict
- [ ] Do not change `configs/shared.yaml` without vote
- [ ] Join W&B entity `P4AIDS_ML` / project `BTL` (train only)

## Preprocessing branch — needs a decision (added by team-preprocessing)

- [ ] **Vote on `configs/shared.yaml`**: one existing key changes,
      `split.strategy: stratified` -> `stratified_group`. Everything else is
      additive. Rationale + measurements in `docs/PREPROCESSING_FINDINGS.md`.
- [ ] **team-dl: set `aug.hflip: false` in `configs/pipelines/dl.yaml`.**
      VNTS has five mirror-pair classes (7<->32, 6<->50, 1<->37, 15<->21,
      19<->49); a horizontal flip relabels them. This is a correctness bug.
- [ ] Everyone reads `docs/DATA_CONTRACT.md` before writing a pipeline.
- [ ] Report macro-F1 twice: over all 52 classes and over classes with
      >= `metrics.rare_class_threshold` (30) samples. Class 44 has 3 crops and
      gets 0 test samples — its per-class F1 is meaningless.
- [ ] Publish `data/processed` once as a Kaggle Dataset; do not re-run
      preprocessing inside the training notebooks.
