"""SVM training, evaluation and run artifacts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

from vn_tsc.eval.metrics import compute_classification_metrics
from vn_tsc.pipelines.base import BasePipeline
from vn_tsc.pipelines.svm.data_adapter import SVMDataBundle, load_svm_data
from vn_tsc.pipelines.svm.model import build_svm, validate_fit_data
from vn_tsc.pipelines.svm.tuning import tune_svm
from vn_tsc.runtime.pack import zip_run_dir
from vn_tsc.runtime.wandb_gate import require_wandb
from vn_tsc.utils.io import load_json, save_json, save_yaml

log = logging.getLogger(__name__)


def _confusion_matrix_png(y_true, y_pred, class_names: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    size = min(18, max(6, len(class_names) * 0.45))
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           xlabel="Predicted", ylabel="True")
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _score_split(model: Pipeline, X, y, class_names: list[str],
                 figure: Path | None) -> dict[str, float]:
    predictions = model.predict(X)
    metrics = compute_classification_metrics(y, predictions, labels=list(range(len(class_names))))
    if figure is not None:
        _confusion_matrix_png(y, predictions, class_names, figure)
    return metrics


def _model_details(model: Pipeline, data: SVMDataBundle, cfg: dict[str, Any]) -> dict[str, Any]:
    reducer = model.named_steps["reduce_dim"]
    details = {
        "pipeline": "svm",
        "feature_backend": data.feature_config["backend"],
        "data_source": cfg["data_source"]["type"],
        "dim_reduction": cfg["dim_reduction"]["method"],
        "dim_reduction_config": cfg["dim_reduction"],
        "feature_combination": [name for name in ("hog", "lbp", "color_hist")
                                if data.feature_config[name].get("enabled")],
        "num_features_before_reduction": int(data.X_train.shape[1]),
        "num_classes": len(data.class_names),
        "class_names": data.class_names,
        "svm_parameters": model.named_steps["svm"].get_params(),
    }
    if hasattr(reducer, "explained_variance_ratio_"):
        details["pca_components_fitted"] = int(reducer.n_components_)
        details["pca_variance_retained"] = float(reducer.explained_variance_ratio_.sum())
    elif cfg["dim_reduction"]["method"] == "lda":
        scaled = model.named_steps["scaler"].transform(data.X_train[:1])
        details["lda_components_fitted"] = int(reducer.transform(scaled).shape[1])
    return details


class SVMPipeline(BasePipeline):
    """Train and evaluate a classical SVM using fixed metadata splits."""

    name = "svm"

    def fit(self) -> dict[str, Any]:
        """Train on metadata train split and evaluate on validation by default."""
        project = self.cfg["project"]
        run = require_wandb(
            entity=project["wandb_entity"], project=project["wandb_project"],
            enabled=self.cfg.get("train", {}).get("require_wandb", True),
        )
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            save_yaml(self.cfg, self.run_dir / "resolved_config.yaml")
            data = load_svm_data(self.cfg)
            log.info("SVM data shapes: train=%s val=%s test=%s", data.X_train.shape,
                     data.X_val.shape, data.X_test.shape)
            model = build_svm(self.cfg)
            validate_fit_data(self.cfg, data.X_train, data.y_train)
            log.info("Training SVM: kernel=%s C=%s gamma=%s reduction=%s",
                     self.cfg["model"]["kernel"], self.cfg["model"]["C"],
                     self.cfg["model"]["gamma"], self.cfg["dim_reduction"]["method"])
            search = None
            if self.cfg.get("tuning", {}).get("enabled", False):
                search = tune_svm(model, data.X_train, data.y_train, self.cfg)
                model = search.best_estimator_
                save_json(search.best_params_, self.run_dir / "best_params.json")
                pd.DataFrame(search.cv_results_).to_csv(self.run_dir / "cv_results.csv", index=False)
            else:
                model.fit(data.X_train, data.y_train)
            figure = (self.run_dir / "figures" / "confusion_matrix_val.png"
                      if self.cfg["outputs"].get("save_confusion_matrix", False) else None)
            val_metrics = _score_split(model, data.X_val, data.y_val, data.class_names, figure)
            metrics: dict[str, Any] = {
                "validation": val_metrics,
                "train_samples": int(data.X_train.shape[0]),
                "val_samples": int(data.X_val.shape[0]),
                "feature_dimension": int(data.X_train.shape[1]),
                "dim_reduction": self.cfg["dim_reduction"]["method"],
                "primary_metric": self.cfg["metrics"]["primary"],
            }
            if search is not None:
                metrics["best_cv_macro_f1"] = float(search.best_score_)
                metrics["best_params"] = search.best_params_
            if self.cfg.get("evaluation", {}).get("evaluate_test_after_train", False):
                figure = (self.run_dir / "figures" / "confusion_matrix_test.png"
                          if self.cfg["outputs"].get("save_confusion_matrix", False) else None)
                metrics["test"] = _score_split(model, data.X_test, data.y_test, data.class_names, figure)
                metrics["test_samples"] = int(data.X_test.shape[0])
            if self.cfg.get("train", {}).get("save_model", True):
                checkpoint = self.run_dir / "checkpoints" / "model.joblib"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, checkpoint)
                log.info("Saved complete SVM Pipeline at %s", checkpoint)
            metadata_dir = self.run_dir / "metadata"
            save_json(data.label_map, metadata_dir / "label_map.json")
            save_json(data.feature_config, metadata_dir / "feature_config.json")
            save_json(_model_details(model, data, self.cfg), metadata_dir / "model_metadata.json")
            save_json(metrics, self.run_dir / "metrics.json")
            log.info("Validation: accuracy=%.4f macro_f1=%.4f weighted_f1=%.4f",
                     val_metrics["accuracy"], val_metrics["macro_f1"], val_metrics["weighted_f1"])
            if run is not None:
                temporary = (data.feature_config["backend"] == "dev" or
                             self.cfg["data_source"]["type"] == "dev")
                enabled = [name for name in ("hog", "lbp", "color_hist")
                           if data.feature_config[name].get("enabled")]
                run.name = f"svm_{'-'.join(enabled)}_{self.cfg['dim_reduction']['method']}_{self.cfg['model']['kernel']}"
                if temporary:
                    run.tags = tuple(set(run.tags or ()) | {"development", "temporary-preprocessing"})
                run.config.update(self.cfg, allow_val_change=True)
                run.config.update({"feature_backend": data.feature_config["backend"],
                                   "temporary_experiment": temporary}, allow_val_change=True)
                logged = {f"validation/{key}": value for key, value in val_metrics.items()}
                logged.update({"feature_dimension": metrics["feature_dimension"],
                               "train_samples": metrics["train_samples"],
                               "val_samples": metrics["val_samples"]})
                if search is not None:
                    logged["best_cv_macro_f1"] = metrics["best_cv_macro_f1"]
                    logged["best_params"] = metrics["best_params"]
                if "test" in metrics:
                    logged.update({f"test_{key}": value for key, value in metrics["test"].items()})
                run.log(logged)
            if self.cfg["outputs"].get("zip_after_train", False):
                zip_run_dir(self.run_dir)
            return metrics
        finally:
            if run is not None:
                run.finish()

    def evaluate(self) -> dict[str, Any]:
        """Evaluate a saved checkpoint on the held-out test split once selected."""
        checkpoint = self.run_dir / "checkpoints" / "model.joblib"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SVM checkpoint not found: {checkpoint}")
        data = load_svm_data(self.cfg)
        saved_features = load_json(self.run_dir / "metadata" / "feature_config.json")
        saved_labels = load_json(self.run_dir / "metadata" / "label_map.json")
        if saved_features != data.feature_config or saved_labels != data.label_map:
            raise ValueError("Current feature configuration or label map differs from saved model")
        model = joblib.load(checkpoint)
        figure = (self.run_dir / "figures" / "confusion_matrix_test.png"
                  if self.cfg["outputs"].get("save_confusion_matrix", False) else None)
        result = _score_split(model, data.X_test, data.y_test, data.class_names, figure)
        metrics_path = self.run_dir / "metrics.json"
        metrics = load_json(metrics_path) if metrics_path.exists() else {}
        metrics["test"] = result
        metrics["test_samples"] = int(data.X_test.shape[0])
        save_json(metrics, metrics_path)
        return result
