from __future__ import annotations
from pathlib import Path
from typing import Any
from vn_tsc.pipelines.base import BasePipeline
from vn_tsc.runtime.wandb_gate import require_wandb
from vn_tsc.utils.io import load_yaml, save_json, save_yaml
from vn_tsc.runtime.pack import zip_run_dir

import shutil
import joblib
import lightgbm as lgb
import numpy as np
from wandb.integration.lightgbm import wandb_callback
from vn_tsc.data.dataset import load_features, load_class_table
from vn_tsc.eval.metrics import compute_classification_metrics
from vn_tsc.pipelines.boosting.model import build_booster
from vn_tsc.data.classes import ClassTable
from vn_tsc.utils.io import load_json
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score
from vn_tsc.data.dataset import rare_classes

def compute_boosting_metrics(y_true, y_pred, table, common_ids):
    # Keep the same class list across runs.
    metrics = compute_classification_metrics(
        y_true,
        y_pred,
        labels=table.ids,
    )

    # Also report F1 without rare classes.
    metrics["macro_f1_common"] = float(
        f1_score(
            y_true,
            y_pred,
            labels=common_ids,
            average="macro",
            zero_division=0,
        )
    )
    return metrics

class BoostingPipeline(BasePipeline):
    name = "boosting"

    def fit(self) -> dict[str, Any]:

        entity = self.cfg.get("project", {}).get("wandb_entity", "P4AIDS_ML")
        project = self.cfg.get("project", {}).get("wandb_project", "BTL")
        run = require_wandb(
            entity=entity, project=project,
            enabled=self.cfg.get("train", {}).get("require_wandb", True),
        )
        try:
            save_yaml(self.cfg, self.run_dir / "resolved_config.yaml")

            # Check config and input files.
            root = Path(self.cfg.get("data", {}).get("processed_root", "data/processed"))
            train_cfg = self.cfg.get("train", {})
            with_aug = bool(train_cfg.get("with_aug", True))
            if self.cfg.get("features", {}).get("pca", {}).get("enabled", False):
                raise NotImplementedError("Baseline requires features.pca.enabled=false")

            required = ["features/scaler.joblib", "class_table.csv"]
            if with_aug:
                required += [
                    "features/F_train_aug.npy", "images/X_train_aug.npy",
                    "images/y_train_aug.npy",
                ]
            for relative_path in required:
                if not (root / relative_path).exists():
                    raise FileNotFoundError(root / relative_path)

            # Load scaled features; augmentation is train-only.
            F_train, y_train = load_features("train", root, with_aug=with_aug)
            F_val, y_val = load_features("val", root)
            table = load_class_table(root)

            # Select classes for the common-class F1.
            threshold = int(
                self.cfg.get("metrics", {}).get("rare_class_threshold", 30)
            )

            rare_ids = set(rare_classes(root, threshold=threshold))
            common_ids = [
                class_id for class_id in table.ids
                if class_id not in rare_ids
            ]

            if not common_ids:
                raise ValueError("No class reached the common threshold.")
            
            # Check shapes, values and labels.
            for name, features, labels in (
                ("train", F_train, y_train), ("val", F_val, y_val),
            ):
                if features.ndim != 2 or labels.ndim != 1:
                    raise ValueError(f"{name}: invalid feature/label dimensions")
                if not len(labels) or len(features) != len(labels):
                    raise ValueError(f"{name}: empty or mismatched features/labels")
                if not np.isfinite(features).all():
                    raise ValueError(f"{name}: features contain NaN or infinity")
                if not set(np.unique(labels)).issubset(table.ids):
                    raise ValueError(f"{name}: unknown class IDs")
            if F_train.shape[1] != F_val.shape[1]:
                raise ValueError("Train and validation feature dimensions differ")
            if set(np.unique(y_train)) != set(table.ids):
                raise ValueError("Training data must contain every class")

            # Set up the model and training logs.
            model = build_booster(self.cfg)
            history = {}
            callbacks = [
                lgb.record_evaluation(history),
                lgb.log_evaluation(period=10),
                lgb.early_stopping(
                    stopping_rounds=int(train_cfg.get("early_stopping_rounds", 30)),
                    first_metric_only=True,
                ),
            ]
            if run is not None:
                run.name = self.run_dir.name
                run.config.update(self.cfg)
                callbacks.append(wandb_callback())

            # Train with early stopping on validation loss.
            model.fit(
                F_train, y_train,
                eval_set=[(F_train, y_train), (F_val, y_val)],
                eval_names=["train", "val"],
                eval_metric="multi_logloss",
                callbacks=callbacks,
            )
            # Score validation predictions.
            y_pred = model.predict(F_val)
            metrics = compute_boosting_metrics(y_val, y_pred, table, common_ids,)

            # Save the model and its preprocessing files.
            checkpoint_dir = self.run_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            save_json(
                {
                    "rare_class_threshold": threshold,
                    "common_class_ids": common_ids,
                },
                checkpoint_dir / "metric_policy.json",
            )

            joblib.dump(model, checkpoint_dir / "model.joblib")

            shutil.copy2(
                root / "features" / "scaler.joblib",
                checkpoint_dir / "scaler.joblib",
            )
            shutil.copy2(
                root / "class_table.csv",
                checkpoint_dir / "class_table.csv",
            )

            feature_report_path = root / "features" / "feature_report.json"
            if not feature_report_path.exists():
                raise FileNotFoundError(feature_report_path)

            shutil.copy2(
                feature_report_path,
                checkpoint_dir / "feature_report.json",
            )

            # Save history and log final scores.
            save_json(history, self.run_dir / "history.json")
            np.savez_compressed(
                self.run_dir / "val_predictions.npz", y_true=y_val, y_pred=y_pred,
            )
            if run is not None:
                run.summary.update({f"val/{key}": value for key, value in metrics.items()})
                run.summary["best_iteration"] = int(model.best_iteration_)
                run.summary["n_train"] = len(y_train)
                run.summary["n_val"] = len(y_val)
                run.summary["n_features"] = F_train.shape[1]

            # Write metrics and zip the run.
            save_json(metrics, self.run_dir / "metrics.json")
            if self.cfg.get("outputs", {}).get("zip_after_train", True):
                zip_run_dir(self.run_dir)
        # Mark failed runs before raising the error.
        except BaseException:
            if run is not None:
                run.finish(exit_code=1)
            raise
        else:
            if run is not None:
                run.finish()
        return metrics


    def evaluate(self) -> dict[str, Any]:
        # Choose val or test from config.
        split = self.cfg.get("eval", {}).get("split", "val")
        if split not in ("val", "test"):
            raise ValueError("Evaluation split must be 'val' or 'test'")

        # Read the saved training config.
        saved_cfg = load_yaml(self.run_dir / "resolved_config.yaml")
        if saved_cfg.get("features", {}).get("pca", {}).get("enabled", False):
            raise NotImplementedError("Baseline requires features.pca.enabled=false")
        
        # The dataset path may change between sessions.
        root = Path(self.cfg.get("data", {}).get(
            "processed_root",
            saved_cfg.get("data", {}).get("processed_root", "data/processed"),
        ))
        
        # Check checkpoint files.
        checkpoint_dir = self.run_dir / "checkpoints"

        required = [
            checkpoint_dir / "model.joblib",
            checkpoint_dir / "scaler.joblib",
            checkpoint_dir / "class_table.csv",
            checkpoint_dir / "feature_report.json",
            root / "features" / "feature_report.json",
        ]

        for path in required:
            if not path.exists():
                raise FileNotFoundError(path)

        # Load the model, scaler and raw features.
        model = joblib.load(checkpoint_dir / "model.joblib")
        scaler = joblib.load(checkpoint_dir / "scaler.joblib")

        features, labels = load_features(
            split,
            root,
            with_aug=False,
            standardize=False,
        )

        # Check labels and features against the training version.
        table = ClassTable.load(checkpoint_dir / "class_table.csv")
        current_table = load_class_table(root)

        saved_classes = [
            (c.class_id, c.sign_code, c.name_vi)
            for c in table.classes
        ]
        current_classes = [
            (c.class_id, c.sign_code, c.name_vi)
            for c in current_table.classes
        ]
        if saved_classes != current_classes:
            raise ValueError("The current dataset has a different label table compared to the training phase.")

        saved_report = load_json(checkpoint_dir / "feature_report.json")
        current_report = load_json(root / "features" / "feature_report.json")

        for key in ("input_size", "layout", "config"):
            if saved_report.get(key) != current_report.get(key):
                raise ValueError(f"The characteristic configuration has changed: {key}")

        # Use the saved scaler, then check inputs.
        features = scaler.transform(features).astype(np.float32)
        
        if features.ndim != 2 or labels.ndim != 1:
            raise ValueError(f"{split}: invalid feature/label dimensions")
        if not len(labels) or len(features) != len(labels):
            raise ValueError(f"{split}: empty or mismatched features/labels")
        if not np.isfinite(features).all():
            raise ValueError(f"{split}: features contain NaN or infinity")
        if features.shape[1] != model.n_features_in_:
            raise ValueError("Feature dimension does not match the saved model")
        if set(model.classes_) != set(table.ids):
            raise ValueError("Class table does not match the saved model")
        if not set(np.unique(labels)).issubset(table.ids):
            raise ValueError(f"{split}: unknown class IDs")

        # Predict and score with the saved class groups.
        predictions = model.predict(features)
        policy = load_json(checkpoint_dir / "metric_policy.json")

        metrics = compute_boosting_metrics(
            labels,
            predictions,
            table,
            policy["common_class_ids"],
        )
        # Save reports for this split.
        out_dir = self.run_dir / "evaluation" / split
        save_json(metrics, out_dir / "metrics.json")
        save_json(
            classification_report(
                labels, predictions, labels=table.ids,
                target_names=table.labels, output_dict=True, zero_division=0,
            ),
            out_dir / "classification_report.json",
        )
        save_json(
            {"split": split, "processed_root": str(root), "n_samples": len(labels),
             "class_ids": table.ids, "class_names": table.names_vi},
            out_dir / "evaluation_info.json",
        )
        np.savez_compressed(out_dir / "predictions.npz", y_true=labels, y_pred=predictions)
        np.save(
            out_dir / "confusion_matrix.npy",
            confusion_matrix(labels, predictions, labels=table.ids),
        )
        # Include evaluation results in the zip.
        if saved_cfg.get("outputs", {}).get("zip_after_train", True):
            zip_run_dir(self.run_dir)
        return metrics
