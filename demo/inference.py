from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
from torchvision import transforms

from vn_tsc.pipelines.svm.inference import extract_svm_image_features, svm_class_scores


class TrafficSignPredictor:
    """Unified inference engine for Deep Learning, SVM, and Boosting."""

    def __init__(
        self,
        models_dir: str | None = None,
        img_size: Tuple[int, int] = (224, 224),
    ) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        self.models_dir = Path(models_dir) if models_dir else repo_root / "saved_models"
        self.img_size = img_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = self._load_labels()
        self.loaded_models: Dict[str, Any] = {}
        self.scaler = None
        self.svm_feature_config: Dict[str, Any] | None = None
        self.svm_uses_external_scaler = False
        self.model_status: Dict[str, bool] = {
            "dl": False,
            "svm": False,
            "boosting": False,
        }

        self.dl_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        self._discover_and_load_models()

    def _load_labels(self) -> List[str]:
        """Load the 52 class labels in class-id order."""
        repo_root = Path(__file__).resolve().parents[1]
        candidates = [
            self.models_dir / "class_table.csv",
            repo_root / "configs" / "classes.csv",
            repo_root / "class_table.csv",
        ]
        for path in candidates:
            if not path.is_file():
                continue
            try:
                with path.open("r", encoding="utf-8") as file:
                    rows = sorted(csv.DictReader(file), key=lambda row: int(row["class_id"]))
                labels = []
                for row in rows:
                    code = (row.get("sign_code") or row.get("id") or "").strip()
                    name = (row.get("name_vi") or row.get("sign_name") or row.get("name") or "").strip()
                    labels.append(f"{code} - {name}" if name else code)
                if len(labels) == 52:
                    print(f"[INFO] Loaded {len(labels)} semantic labels from {path}")
                    return labels
            except Exception as exc:
                print(f"[WARN] Failed to read {path}: {exc}")
        return [f"Class {index}" for index in range(52)]

    @staticmethod
    def _read_json(path: Path) -> Any:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _load_feature_config(self, model_path: Path) -> None:
        roots = [self.models_dir, model_path.parent, model_path.parent.parent]
        candidates = [self.models_dir / "svm_feature_config.json"]
        for root in roots:
            candidates.extend((
                root / "feature_config.json",
                root / "feature_report.json",
                root / "metadata" / "feature_config.json",
                root / "metadata" / "feature_report.json",
            ))
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is not None:
            data = self._read_json(path)
            if "config" in data:  # feature_report.json
                data = {
                    "backend": "shared",
                    "input_size": data.get("input_size", [64, 64]),
                    **data["config"],
                }
            self.svm_feature_config = data
            return

        from vn_tsc.config.resolve import resolve_config
        from vn_tsc.pipelines.svm.data_adapter import feature_config_from_cfg

        repo_root = Path(__file__).resolve().parents[1]
        cfg = resolve_config(
            repo_root / "configs/pipelines/svm.yaml",
            repo_root / "configs/shared.yaml",
        )
        self.svm_feature_config = feature_config_from_cfg(cfg)
        print("[WARN] SVM feature metadata missing; using repository configuration.")

    def _discover_and_load_models(self) -> None:
        import joblib

        scaler_path = self.models_dir / "scaler.joblib"
        if scaler_path.is_file():
            try:
                self.scaler = joblib.load(scaler_path)
                print(f"[INFO] Loaded scaler from {scaler_path}")
            except Exception as exc:
                print(f"[WARN] Failed to load scaler: {exc}")

        dl_candidates = [
            self.models_dir / "checkpoint_best.pt",
            self.models_dir / "dl_model.pt",
        ]
        dl_path = next((path for path in dl_candidates if path.is_file()), None)
        if dl_path is not None:
            try:
                import timm

                model = timm.create_model(
                    "tf_efficientnetv2_b0", pretrained=False, num_classes=len(self.labels)
                )
                checkpoint = torch.load(dl_path, map_location=self.device)
                state_dict = checkpoint.get("state_dict", checkpoint)
                cleaned = {
                    key.replace("module.", "").replace("model.", ""): value
                    for key, value in state_dict.items()
                }
                model.load_state_dict(cleaned, strict=True)
                model.to(self.device).eval()
                self.loaded_models["dl"] = model
                self.model_status["dl"] = True
                print(f"[INFO] Loaded DL checkpoint from {dl_path}")
            except Exception as exc:
                print(f"[ERROR] Failed to load DL model: {exc}")

        svm_candidates = [
            self.models_dir / name for name in (
                "svm.joblib", "svm.pkl", "vm.joblib", "svm_pipeline.joblib",
                "svm_pipeline.pkl", "svm_model.joblib",
            )
        ]
        svm_candidates.append(self.models_dir / "svm" / "checkpoints" / "model.joblib")
        for path in svm_candidates:
            if not path.is_file():
                continue
            try:
                model = joblib.load(path)
                self.loaded_models["svm"] = model
                self._load_feature_config(path)
                steps = getattr(model, "named_steps", {})
                self.svm_uses_external_scaler = self.scaler is not None and "scaler" not in steps
                self.model_status["svm"] = True
                print(f"[INFO] Loaded SVM model from {path}")
                break
            except Exception as exc:
                print(f"[WARN] Failed to load SVM from {path}: {exc}")

        for name in (
            "boosting.joblib", "boost.joblib", "model.joblib",
            "boosting_pipeline.joblib", "boosting_pipeline.pkl", "model.pkl",
        ):
            path = self.models_dir / name
            if not path.is_file():
                continue
            try:
                self.loaded_models["boosting"] = joblib.load(path)
                self.model_status["boosting"] = True
                print(f"[INFO] Loaded Boosting model from {path}")
                break
            except Exception as exc:
                print(f"[WARN] Failed to load Boosting model from {path}: {exc}")

    def _mock_predict(self, pipeline_key: str, top_k: int) -> Dict[str, Any]:
        start_time = time.perf_counter()
        time.sleep(0.007)
        k = min(top_k, len(self.labels))
        scores = [random.uniform(0.1, 1.0) for _ in range(k)]
        total = sum(scores) + random.uniform(0.05, 0.25)
        normalized = sorted((score / total for score in scores), reverse=True)
        chosen = random.sample(self.labels, k=k)
        return {
            "top_class": chosen[0],
            "confidence": normalized[0],
            "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
            "predictions": dict(zip(chosen, normalized)),
            "is_mock": True,
        }

    def _predict_svm(self, image: Image.Image, top_k: int) -> Dict[str, Any]:
        model = self.loaded_models["svm"]
        features = extract_svm_image_features(image, self.svm_feature_config or {})
        if self.svm_uses_external_scaler:
            features = self.scaler.transform(features)

        expected = getattr(model, "n_features_in_", None)
        if expected is not None and features.shape[1] != int(expected):
            raise ValueError(
                f"SVM expects {int(expected)} features but inference extracted "
                f"{features.shape[1]}; use metadata from the same training run."
            )
        predicted_id = int(np.asarray(model.predict(features))[0])
        classes, scores, calibrated = svm_class_scores(model, features)
        ranked = np.argsort(scores)[::-1].tolist()
        predicted_position = np.flatnonzero(classes == predicted_id)
        if predicted_position.size and ranked[0] != int(predicted_position[0]):
            ranked.remove(int(predicted_position[0]))
            ranked.insert(0, int(predicted_position[0]))
        ranked = ranked[:min(max(1, top_k), len(classes))]
        predictions = {
            self.labels[int(classes[position])]: round(float(scores[position]), 4)
            for position in ranked
        }
        score_position = int(np.flatnonzero(classes == predicted_id)[0])
        return {
            "top_class": self.labels[predicted_id],
            "confidence": float(scores[score_position]),
            "predictions": predictions,
            "is_mock": False,
            "confidence_calibrated": calibrated,
        }

    def predict(self, img_pil: Image.Image, pipeline_key: str, top_k: int = 5) -> Dict[str, Any]:
        if not self.model_status.get(pipeline_key, False):
            return self._mock_predict(pipeline_key, top_k)

        start_time = time.perf_counter()
        try:
            if pipeline_key == "svm":
                result = self._predict_svm(img_pil, top_k)
                result["latency_ms"] = round((time.perf_counter() - start_time) * 1000, 2)
                return result

            if pipeline_key == "dl":
                tensor = self.dl_transform(img_pil.convert("RGB")).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.loaded_models["dl"](tensor)
                    probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            else:
                if self.svm_feature_config is None:
                    self._load_feature_config(self.models_dir / "boosting.joblib")
                features = extract_svm_image_features(
                    img_pil, self.svm_feature_config or {}
                )
                model = self.loaded_models["boosting"]
                steps = getattr(model, "named_steps", {})
                if self.scaler is not None and "scaler" not in steps:
                    features = self.scaler.transform(features)
                probabilities = np.asarray(model.predict_proba(features)[0], dtype=np.float64)

            k = min(max(1, top_k), len(probabilities))
            top_indices = np.argsort(probabilities)[::-1][:k]
            return {
                "top_class": self.labels[int(top_indices[0])],
                "confidence": float(probabilities[top_indices[0]]),
                "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
                "predictions": {
                    self.labels[int(index)]: round(float(probabilities[index]), 4)
                    for index in top_indices
                },
                "is_mock": False,
                "confidence_calibrated": True,
            }
        except Exception as exc:
            raise RuntimeError(f"Inference failed for loaded model '{pipeline_key}': {exc}") from exc
