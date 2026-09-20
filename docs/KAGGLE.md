# Kaggle 2×T4

- Accelerator: GPU T4 ×2
- `pip install -r requirements-kaggle.txt && pip install -e .`
- `DATA_ROOT=/kaggle/input/<slug>`
- Runtime: `configs/runtime/kaggle_2xt4.yaml`
- DL: DDP stubs in `vn_tsc/pipelines/dl/train.py`
- SVM/Boosting: high `n_jobs`; LightGBM GPU optional (verify driver)
- Secret: `WANDB_API_KEY`
