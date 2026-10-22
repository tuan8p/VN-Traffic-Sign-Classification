import pytest
from pathlib import Path
from vn_tsc.runtime.wandb_gate import build_base_run_name, generate_run_name, require_wandb


def test_require_wandb_raises_without_key(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        require_wandb()


def test_require_wandb_disabled():
    assert require_wandb(enabled=False) is None


def test_build_base_run_name_svm():
    cfg = {
        "model": {"kernel": "rbf", "C": 1.0},
        "dim_reduction": {"method": "none"},
        "train": {"with_aug": True},
    }
    assert build_base_run_name("svm", cfg) == "svm_rbf_C1.0_none_aug"

    cfg_pca = {
        "model": {"kernel": "linear", "C": 0.5},
        "dim_reduction": {"method": "pca", "pca": {"n_components": 0.95}},
        "train": {"with_aug": False},
        "tuning": {"enabled": True},
    }
    assert build_base_run_name("svm", cfg_pca) == "svm_linear_C0.5_pca0.95_noaug_cv"


def test_build_base_run_name_boosting():
    cfg = {
        "model": {"backend": "lgbm", "num_leaves": 63, "learning_rate": 0.05, "n_estimators": 150},
        "features": {"pca": {"enabled": False}},
        "train": {"with_aug": True},
    }
    assert build_base_run_name("boosting", cfg) == "boosting_lgbm_leaves63_lr0.05_est150_aug"


def test_build_base_run_name_dl():
    cfg = {
        "model": {"backbone": "tf_efficientnetv2_b0"},
        "train": {"epochs": 30, "warmup_epochs": 5, "lr": 3e-4, "batch_size": 64, "with_aug": True},
    }
    assert build_base_run_name("dl", cfg) == "dl_effnetv2b0_ep30_warm5_lr0.0003_bs64_aug"


def test_build_base_run_name_custom_user_name():
    assert build_base_run_name("dl", user_name="my_custom_exp") == "dl_my_custom_exp"
    assert build_base_run_name("dl", user_name="dl_already_prefixed") == "dl_already_prefixed"


def test_generate_run_name_extracts_timestamp_from_run_dir():
    run_dir = Path("outputs/dl/20261004_174500_dl_effnetv2b0")
    cfg = {
        "model": {"backbone": "tf_efficientnetv2_b0"},
        "train": {"epochs": 30, "warmup_epochs": 5, "lr": 3e-4, "batch_size": 64, "with_aug": True},
    }
    name = generate_run_name("dl", cfg, run_dir=run_dir)
    assert name == "dl_effnetv2b0_ep30_warm5_lr0.0003_bs64_aug_20261004_174500"

