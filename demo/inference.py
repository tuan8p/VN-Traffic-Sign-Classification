from __future__ import annotations

import csv
import json
import os
import random
import time
from typing import Any, Dict, List, Tuple
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

DEFAULT_CLASSES: List[str] = [
    "P.102 - No Entry",
    "P.130 - No Stopping or Parking",
    "P.127 - Speed Limit (60 km/h)",
    "W.224 - Pedestrian Crossing",
    "W.245a - Slow Down",
    "R.301 - Mandatory Direction",
    "P.103a - No Cars",
    "I.407a - One-Way Traffic",
]


class TrafficSignPredictor:
    """Unified inference engine for Deep Learning, SVM, and Boosting pipelines."""

    def __init__(
        self,
        models_dir: str = "saved_models",
        labels_json_path: str = "saved_models/class_labels.json",
        labels_csv_path: str = "saved_models/class_table.csv",
        img_size: Tuple[int, int] = (224, 224),
    ) -> None:
        self.models_dir = models_dir
        self.img_size = img_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = self._load_labels(labels_json_path, labels_csv_path)
        self.loaded_models: Dict[str, Any] = {}
        self.model_status: Dict[str, bool] = {
            "dl": False,
            "svm": False,
            "boosting": False,
        }

        # Chuẩn hóa tiền xử lý đầu vào theo chuẩn EfficientNetV2
        self.dl_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self._discover_and_load_models()

    def _load_labels(self, json_path: str, csv_path: str) -> List[str]:
        """Loads semantic class labels from JSON or CSV."""
        # 1. Thử load từ class_labels.json
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return [data[str(i)] for i in range(len(data))]
                    elif isinstance(data, list):
                        return data
            except Exception as e:
                print(f"[WARN] Failed to load {json_path}: {e}")

        # 2. Thử load từ class_table.csv
        if os.path.exists(csv_path):
            try:
                classes = []
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        sign_name = row.get("sign_name") or row.get("name") or row.get("class_name")
                        sign_code = row.get("sign_code") or row.get("id") or ""
                        if sign_name:
                            classes.append(f"{sign_code} - {sign_name}" if sign_code else sign_name)
                if classes:
                    return classes
            except Exception as e:
                print(f"[WARN] Failed to load {csv_path}: {e}")

        return DEFAULT_CLASSES

    def _discover_and_load_models(self) -> None:
        """Discovers and loads trained checkpoints."""
        # 1. Pipeline A: Deep Learning (EfficientNetV2-B0)
        # Hỗ trợ cả dl_model.pt, checkpoint_best.pt, dl_model.pth
        dl_candidates = [
            os.path.join(self.models_dir, "checkpoint_best.pt"),
            os.path.join(self.models_dir, "dl_model.pt"),
            os.path.join(self.models_dir, "dl_model.pth"),
        ]
        dl_path = next((p for p in dl_candidates if os.path.exists(p)), None)

        if dl_path:
            try:
                num_classes = len(self.labels)
                # Khởi tạo backbone EfficientNetV2-B0
                model = models.efficientnet_v2_s(weights=None) if "v2_s" in dl_path else models.efficientnet_v2_b0(weights=None)
                in_features = model.classifier[1].in_features
                model.classifier[1] = nn.Linear(in_features, num_classes)

                # Nạp checkpoint
                checkpoint = torch.load(dl_path, map_location=self.device)
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["state_dict"])
                elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                elif isinstance(checkpoint, dict):
                    model.load_state_dict(checkpoint)
                else:
                    model = checkpoint

                model.to(self.device)
                model.eval()
                self.loaded_models["dl"] = model
                self.model_status["dl"] = True
                print(f"[INFO] Successfully loaded PyTorch DL checkpoint from {dl_path}")
            except Exception as e:
                print(f"[ERROR] Failed to load DL model from {dl_path}: {e}")

        # 2. Pipeline B: Machine Learning - SVM
        for ext in ["joblib", "pkl"]:
            svm_path = os.path.join(self.models_dir, f"svm_pipeline.{ext}")
            if os.path.exists(svm_path):
                try:
                    import joblib
                    self.loaded_models["svm"] = joblib.load(svm_path)
                    self.model_status["svm"] = True
                    print(f"[INFO] Loaded SVM pipeline from {svm_path}")
                    break
                except Exception as e:
                    print(f"[WARN] Failed to load SVM model: {e}")

        # 3. Pipeline C: Machine Learning - Boosting
        for ext in ["joblib", "pkl"]:
            boost_path = os.path.join(self.models_dir, f"boosting_pipeline.{ext}")
            if os.path.exists(boost_path):
                try:
                    import joblib
                    self.loaded_models["boosting"] = joblib.load(boost_path)
                    self.model_status["boosting"] = True
                    print(f"[INFO] Loaded Boosting pipeline from {boost_path}")
                    break
                except Exception as e:
                    print(f"[WARN] Failed to load Boosting model: {e}")

    def _mock_predict(self, pipeline_key: str, top_k: int) -> Dict[str, Any]:
        """Simulation fallback returning realistic synthetic metrics."""
        start_time = time.perf_counter()
        simulated_delay = 0.022 if pipeline_key == "dl" else 0.007
        time.sleep(simulated_delay)

        k = min(top_k, len(self.labels))
        scores = [random.uniform(0.1, 1.0) for _ in range(k)]
        total = sum(scores) + random.uniform(0.05, 0.25)
        normalized = sorted([s / total for s in scores], reverse=True)
        chosen_labels = random.sample(self.labels, k=k)

        latency = (time.perf_counter() - start_time) * 1000

        predictions = {label: round(prob, 4) for label, prob in zip(chosen_labels, normalized)}
        return {
            "top_class": chosen_labels[0],
            "confidence": normalized[0],
            "latency_ms": round(latency, 2),
            "predictions": predictions,
            "is_mock": True,
        }

    def predict(
        self, img_pil: Image.Image, pipeline_key: str, top_k: int = 5
    ) -> Dict[str, Any]:
        """Runs single-image inference."""
        # Fallback chế độ mock nếu chưa có model thật
        if not self.model_status.get(pipeline_key, False):
            return self._mock_predict(pipeline_key, top_k)

        start_time = time.perf_counter()
        img_rgb = img_pil.convert("RGB")

        try:
            if pipeline_key == "dl":
                model = self.loaded_models["dl"]
                tensor = self.dl_transform(img_rgb).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    logits = model(tensor)
                    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            else:
                model = self.loaded_models[pipeline_key]
                # Chuẩn hóa ảnh cho ML truyền thống
                img_resized = img_rgb.resize((64, 64))
                flat_input = (np.array(img_resized, dtype=np.float32) / 255.0).reshape(1, -1)
                probs = model.predict_proba(flat_input)[0]

            k = min(top_k, len(probs))
            top_indices = np.argsort(probs)[::-1][:k]
            predictions = {
                self.labels[idx] if idx < len(self.labels) else f"Class {idx}": round(float(probs[idx]), 4)
                for idx in top_indices
            }
            top_class = self.labels[top_indices[0]] if top_indices[0] < len(self.labels) else f"Class {top_indices[0]}"
            top_conf = float(probs[top_indices[0]])
            latency = (time.perf_counter() - start_time) * 1000

            return {
                "top_class": top_class,
                "confidence": top_conf,
                "latency_ms": round(latency, 2),
                "predictions": predictions,
                "is_mock": False,
            }
        except Exception as e:
            print(f"[ERROR] Inference error in '{pipeline_key}': {e}. Falling back to mock.")
            return self._mock_predict(pipeline_key, top_k)