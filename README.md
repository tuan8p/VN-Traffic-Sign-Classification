# VN Traffic Sign Classification

Fair comparison of **classical ML** (SVM, LightGBM) and **deep learning** (EfficientNetV2) on **52 Vietnamese traffic sign classes**, built from the [VNTS](https://www.kaggle.com/datasets/maitam/vietnamese-traffic-signs) detection dataset converted to **crop-level classification**.

Course / team project (P4AIDS + Machine Learning). Shared preprocessing, locked metrics, and a single processed store keep experiments comparable across pipelines.

---

## Highlights

- **One preprocess → three trainers:** shared crops (`X_*.npy`), classical features (`F_*.npy`), stratified splits, offline augmentation (no horizontal flip — mirror-pair classes).
- **Pipelines:** `vn_tsc/pipelines/{svm,boosting,dl}/` with YAML config merge (`configs/shared.yaml` + pipeline + runtime).
- **Primary metric:** **macro-F1** on validation (early stopping / checkpoint selection); test for final reporting.
- **Streamlit demo:** side-by-side inference and latency for DL, SVM, and Boosting ([demo/README.md](demo/README.md)).
- **EDA:** report figures from `metadata.csv` ([analysis/README.md](analysis/README.md)).

---

## Repository layout

```text
VN-Traffic-Sign-Classification/
├── analysis/           # EDA scripts & figures (no W&B)
├── configs/
│   ├── shared.yaml     # Fair-compare keys (seed, splits, metrics, augment rules)
│   ├── classes.csv     # 52 classes, mirror pairs, confusable groups
│   ├── pipelines/      # svm.yaml, boosting.yaml, dl.yaml
│   └── runtime/        # local.yaml, kaggle_2xt4.yaml
├── demo/               # Streamlit app + inference engine
├── saved_models/       # Demo deployment artifacts (checkpoints, scaler, class_table)
├── tools/              # CLI entrypoints (preprocess, train, eval, pack)
├── vn_tsc/             # Library: data, features, pipelines, eval, runtime
├── data/               # Local data root (raw + processed; often gitignored)
└── outputs/            # Training runs (metrics, checkpoints, figures)
```

---

## Requirements

- **Python ≥ 3.10**
- **GPU:** optional for SVM/Boosting; recommended for DL (Kaggle 2×T4 supported via DDP).
- **Weights & Biases:** required for `tools.run_train` unless disabled (`--no-wandb` / config).

---

## Installation

```bash
python -m venv .venv
# Windows PowerShell:
#   .\.venv\Scripts\Activate.ps1
# Linux/macOS:
#   source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
# or modular install:
# pip install -e ".[boosting,dl,demo,dev]"
```

Copy environment template and set paths / W&B:

```bash
cp .env.example .env
# WANDB_API_KEY=...
# DATA_ROOT=...   # folder containing VNTS YOLO layout
```

---

## Workflow

### 1. Offline preprocess (team-shared)

Turns YOLO boxes into fixed **224×224 RGB crops**, writes `data/processed/`, extracts **HOG + LBP + color** features at 64×64, optional offline aug:

```bash
python -m tools.run_preprocess --data-root <path-to-VNTS>
```

Typical outputs: `images/X_{train,val,test}.npy`, `features/F_*.npy`, `metadata.csv`, `class_table.csv`, `scaler.joblib`.

**Kaggle:** attach a preprocessed dataset and point trainers at it, e.g.:

```bash
set SVM_PROCESSED_ROOT=/kaggle/input/.../vnts-processed
python -m tools.run_train --pipeline svm --data-root %SVM_PROCESSED_ROOT%
```

### 2. EDA (optional, no W&B)

```bash
python -m analysis.eda --processed data/processed --out analysis/figures
```

### 3. Train

```bash
python -m tools.run_train --pipeline svm
python -m tools.run_train --pipeline boosting
python -m tools.run_train --pipeline dl --runtime configs/runtime/local.yaml
```

DL on **2× GPU** (example):

```bash
torchrun --nproc_per_node=2 -m tools.run_train --pipeline dl \
  --config configs/pipelines/dl.yaml \
  --runtime configs/runtime/kaggle_2xt4.yaml \
  --data-root <processed_root>
```

### 4. Evaluate & pack

```bash
python -m tools.run_eval --pipeline dl --run-dir outputs/dl/<run_id>
python -m tools.pack_outputs --run-dir outputs/svm/<run_id>
python -m tools.plot_from_json --run-dir outputs/dl/<run_id>
```

### 5. Demo

Deploy checkpoints under `saved_models/` (see [demo/README.md](demo/README.md)), then:

```bash
streamlit run demo/app.py
```

---

## Configuration

| Layer | Path | Role |
| ------- | ------ | ------ |
| Shared contract | `configs/shared.yaml` | Seed, split ratios, image size, **no hflip**, metric names, rare-class threshold |
| Class ontology | `configs/classes.csv` | IDs, sign codes, Vietnamese names, mirror/confusable metadata |
| Pipeline | `configs/pipelines/*.yaml` | Model & train hyperparameters |
| Runtime | `configs/runtime/*.yaml` | Device, `num_workers`, DDP |
| Resolved run config | `outputs/<pipeline>/<run_id>/resolved_config.yaml` | Effective config after merge + notebook overrides |

Override at CLI: `--config`, `--runtime`, `--data-root`, `--run-name`, `--run-dir`.

---

## Run artifacts

Each training run creates:

```text
outputs/<pipeline>/<run_id>/
  resolved_config.yaml
  metrics.json          # validation (+ test if auto_eval_test)
  history.json
  figures/              # confusion matrix, curves
  checkpoints/          # best model
  run.zip               # optional bundle (config flag)
```

---

## Fair comparison rules (summary)

1. **Same splits** — do not augment val/test; `train.with_aug` only affects training features/crops.
2. **No horizontal flip** — mirror-pair classes (see `configs/shared.yaml`).
3. **Select configs by validation macro-F1**, not test, when tuning ablations.
4. **Report** accuracy, macro-F1, weighted-F1; macro-F1 on **common** (non-rare) classes where applicable.

---

## Development

```bash
pytest
ruff check vn_tsc tools demo analysis
```

Package entry points (after `pip install -e .`): `vn-tsc-preprocess`, `vn-tsc-train`, `vn-tsc-eval`, `vn-tsc-pack`.

---

## License

MIT — see [LICENSE](LICENSE).
