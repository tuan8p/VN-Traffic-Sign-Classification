# VN-Traffic-Sign-Classification

Course project (P4AIDS + Machine Learning): fair comparison of **SVM**, **Boosting**, and **Deep Learning** on Vietnamese traffic-sign images.

Learning scaffold — teammates fill `TODO` blocks. Shared offline preprocessing + locked keys in `configs/shared.yaml` keep comparisons fair.

## Team roles / branches

| Role | Branch | Owns |
|------|--------|------|
| Preprocessing | `feat/preprocessing` | `vn_tsc/data/`, `vn_tsc/features/`, `tools/run_preprocess.py` |
| SVM | `feat/svm` | `vn_tsc/pipelines/svm/`, `configs/pipelines/svm.yaml`, `notebooks/02_train_svm.ipynb` |
| Boosting | `feat/boosting` | `vn_tsc/pipelines/boosting/`, `configs/pipelines/boosting.yaml`, `notebooks/03_train_boosting.ipynb` |
| Deep Learning | `feat/dl` | `vn_tsc/pipelines/dl/`, `configs/pipelines/dl.yaml`, `notebooks/04_train_dl.ipynb` |
| Demo | `feat/demo` | `demo/`, `notebooks/05_demo.ipynb` |

## Quick start (local)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -U pip && pip install -e ".[all]"
cp .env.example .env   # set WANDB_API_KEY + DATA_ROOT
```

## Kaggle 2×T4

1. Enable GPU T4 ×2.
2. Open `notebooks/00_kaggle_bootstrap.ipynb`.
3. Training **requires** W&B (`entity=P4AIDS_ML`, `project=BTL`). Preprocess / EDA / demo do **not**.

## Data

VNTS is a **YOLO detection** set; this project is classification, so preprocessing
crops every bounding box into one labelled sample (8,334 boxes → 7,989 crops over
52 classes). Read **[docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md)** before writing
any pipeline — it specifies the arrays, the loaders and two rules that are easy to
break (no horizontal flip; macro-F1 must be reported twice).

```bash
python -m tools.run_preprocess --data-root <path to VNTS>   # ~3 min, ~2.2 GB
python -m analysis.eda                                      # figures for the report
```

## Config

- `configs/shared.yaml` — locked fair-compare keys
- `configs/classes.csv` — the 52 class ids, sign codes, Vietnamese names, mirror pairs
- `configs/pipelines/{svm,boosting,dl}.yaml` — per-pipeline
- `configs/runtime/{local,kaggle_2xt4}.yaml` — devices / DDP
- Notebooks may override; overrides land in `outputs/.../resolved_config.yaml`

## Outputs

```
outputs/<pipeline>/<run_id>/
  resolved_config.yaml
  metrics.json
  history.json
  logs/train.log
  figures/
  checkpoints/
  run.zip
```

## CLI

```bash
python -m tools.run_preprocess --data-root <path>
python -m tools.run_train --pipeline svm
python -m tools.run_eval --pipeline svm --run-dir outputs/svm/<run_id>
python -m tools.pack_outputs --run-dir outputs/svm/<run_id>
streamlit run demo/app.py
```

## License

MIT
