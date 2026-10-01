from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from PIL import Image
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from vn_tsc.config.resolve import resolve_config
from vn_tsc.pipelines.svm.dev_preprocess import run_dev_preprocess
from vn_tsc.pipelines.svm.model import build_svm
from vn_tsc.pipelines.svm.train import SVMPipeline
from vn_tsc.pipelines.svm.tuning import build_search, tune_svm


@pytest.fixture
def cfg(tmp_path: Path) -> dict:
    repo = Path(__file__).resolve().parents[1]
    config = resolve_config(repo / "configs/pipelines/svm.yaml", repo / "configs/shared.yaml")
    config["preprocess"]["image_size"] = [32, 32]
    config["data_source"]["processed_root"] = str(tmp_path / "processed")
    config["features"]["hog"]["enabled"] = False
    config["features"]["lbp"]["enabled"] = False
    config["features"]["color_hist"]["bins"] = 8
    config["train"]["require_wandb"] = False
    config["outputs"]["zip_after_train"] = False
    return config


def _prepare_data(cfg: dict, tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    for class_index, name in enumerate(("apple", "banana", "zebra")):
        folder = raw / name
        folder.mkdir(parents=True)
        for index in range(8):
            color = [15, 15, 15]
            color[class_index] = 180 + index
            Image.new("RGB", (24, 16), tuple(color)).save(folder / f"{index}.png")
    run_dev_preprocess(cfg, raw, cfg["data_source"]["processed_root"])


def test_end_to_end_artifacts_evaluate_and_no_early_test(cfg: dict, tmp_path: Path) -> None:
    _prepare_data(cfg, tmp_path)
    run_dir = tmp_path / "outputs" / "svm" / "run"
    pipeline = SVMPipeline(cfg, run_dir)
    result = pipeline.fit()
    assert set(result["validation"]) == {"accuracy", "macro_f1", "weighted_f1"}
    assert "test" not in result
    checkpoint = run_dir / "checkpoints" / "model.joblib"
    assert checkpoint.is_file()
    model = joblib.load(checkpoint)
    assert isinstance(model, Pipeline)
    assert model.predict(np.zeros((1, result["feature_dimension"]))).shape == (1,)
    assert (run_dir / "figures" / "confusion_matrix_val.png").is_file()
    assert (run_dir / "metadata" / "feature_config.json").is_file()
    assert json.loads((run_dir / "metadata" / "model_metadata.json").read_text())["feature_backend"] == "dev"
    test_result = pipeline.evaluate()
    assert set(test_result) == {"accuracy", "macro_f1", "weighted_f1"}
    assert "test" in json.loads((run_dir / "metrics.json").read_text())


def test_wandb_payload_and_final_test_option(cfg: dict, tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    _prepare_data(cfg, tmp_path)

    class FakeConfig(dict):
        def update(self, values, allow_val_change=False):
            super().update(values)

    class FakeRun:
        def __init__(self):
            self.config = FakeConfig()
            self.tags = ()
            self.logged = []
            self.finished = False
            self.name = ""

        def log(self, values):
            self.logged.append(values)

        def finish(self):
            self.finished = True

    fake = FakeRun()
    monkeypatch.setattr("vn_tsc.pipelines.svm.train.require_wandb", lambda **kwargs: fake)
    cfg["train"]["require_wandb"] = True
    cfg["evaluation"]["evaluate_test_after_train"] = True
    result = SVMPipeline(cfg, tmp_path / "wandb_run").fit()
    assert "test" in result
    assert fake.finished
    assert fake.config["feature_backend"] == "dev"
    assert fake.config["temporary_experiment"] is True
    assert "development" in fake.tags
    assert fake.logged[0]["validation/macro_f1"] == result["validation"]["macro_f1"]
    assert "test_macro_f1" in fake.logged[0]


def test_grid_and_random_search_use_training_folds(cfg: dict) -> None:
    X, y = make_classification(n_samples=45, n_features=8, n_informative=6,
                               n_redundant=0, n_classes=3, n_clusters_per_class=1,
                               random_state=42)
    cfg["tuning"].update({"enabled": True, "cv_folds": 3, "n_jobs": 1,
                          "param_grid": {"svm__C": [0.1, 1.0]}, "n_iter": 2})
    for method, search_type in (("grid", GridSearchCV), ("random", RandomizedSearchCV)):
        cfg["tuning"]["method"] = method
        search = build_search(build_svm(cfg), cfg)
        assert isinstance(search, search_type)
        assert isinstance(search.estimator, Pipeline)
        assert isinstance(search.cv, StratifiedKFold)
        assert search.cv.shuffle and search.cv.random_state == cfg["seed"]
        fitted = tune_svm(build_svm(cfg), X, y, cfg)
        assert isinstance(fitted.best_estimator_, Pipeline)
        assert fitted.best_estimator_.predict(X[:3]).shape == (3,)


def test_pipeline_tuning_writes_cv_results(cfg: dict, tmp_path: Path) -> None:
    _prepare_data(cfg, tmp_path)
    cfg["tuning"].update({"enabled": True, "method": "grid", "cv_folds": 2,
                          "n_jobs": 1, "param_grid": {"svm__C": [0.1, 1.0]}})
    run_dir = tmp_path / "tuned"
    metrics = SVMPipeline(cfg, run_dir).fit()
    assert "best_cv_macro_f1" in metrics
    assert metrics["best_params"]["svm__C"] in (0.1, 1.0)
    assert (run_dir / "best_params.json").is_file()
    assert (run_dir / "cv_results.csv").is_file()


def test_tuning_rejects_too_few_training_samples(cfg: dict) -> None:
    cfg["tuning"].update({"cv_folds": 5, "n_jobs": 1,
                          "param_grid": {"svm__C": [1.0]}})
    X = np.arange(32, dtype=np.float32).reshape(8, 4)
    y = np.repeat([0, 1], 4)
    with pytest.raises(ValueError, match="at least 5 samples"):
        tune_svm(build_svm(cfg), X, y, cfg)
