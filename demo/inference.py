from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Dict, List, Tuple
import numpy as np
from PIL import Image

# Default class names (used if saved_models/class_labels.json is missing)
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
    """Unified inference engine for Deep Learning, SVM, and Boosting pipelines.
    
    Serves as an upgrade from DummyPredictor: automatically detects actual model
    artifacts in `saved_models/` and seamlessly falls back to simulation mode
    when checkpoints are not yet ready.
    """

    def __init__(
        self,
        models_dir: str = "saved_models",
        labels_path: str = "saved_models/class_labels.json",
        img_size: Tuple[int, int] = (64, 64),
    ) -> None:
        self.models_dir = models_dir
        self.img_size = img_size
        self.labels = self._load_labels(labels_path)
        self.loaded_models: Dict[str, Any] = {}
        self.model_status: Dict[str, bool] = {
            "dl": False,
            "svm": False,
            "boosting": False,
        }

        self._discover_and_load_models()

    def _load_labels(self, labels_path: str) -> List[str]:
        """Loads semantic class labels from JSON or falls back to defaults."""
        if os.path.exists(labels_path):
            try:
                with open(labels_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return [data[str(i)] for i in range(len(data))]
                    elif isinstance(data, list):
                        return data
            except Exception as e:
                print(f"[WARN] Failed to load {labels_path}: {e}. Using defaults.")
        return DEFAULT_CLASSES

    def _discover_and_load_models(self) -> None:
        """Discovers and loads trained checkpoints exported by teammates A, B, and C."""
        # 1. Pipeline A: Deep Learning (.onnx preferred, fallback to .pth)
        dl_onnx = os.path.join(self.models_dir, "dl_model.onnx")
        dl_pth = os.path.join(self.models_dir, "dl_model.pth")
        
        if os.path.exists(dl_onnx):
            try:
                import onnxruntime as ort
                self.loaded_models["dl"] = ort.InferenceSession(dl_onnx)
                self.model_status["dl"] = True
                print("[INFO] DL Model loaded successfully via ONNX Runtime.")
            except Exception as e:
                print(f"[WARN] ONNX Runtime failed to load: {e}")
        elif os.path.exists(dl_pth):
            try:
                import torch
                self.loaded_models["dl"] = torch.load(dl_pth, map_location=torch.device("cpu"))
                if hasattr(self.loaded_models["dl"], "eval"):
                    self.loaded_models["dl"].eval()
                self.model_status["dl"] = True
                print("[INFO] DL Model loaded successfully via PyTorch.")
            except Exception as e:
                print(f"[WARN] PyTorch checkpoint load failed: {e}")

        # 2. Pipeline B: Machine Learning - SVM (.joblib or .pkl)
        for ext in ["joblib", "pkl"]:
            svm_path = os.path.join(self.models_dir, f"svm_pipeline.{ext}")
            if os.path.exists(svm_path):
                try:
                    import joblib
                    self.loaded_models["svm"] = joblib.load(svm_path)
                    self.model_status["svm"] = True
                    print(f"[INFO] SVM Pipeline loaded successfully from {svm_path}.")
                    break
                except Exception as e:
                    print(f"[WARN] SVM load failed: {e}")

        # 3. Pipeline C: Machine Learning - Boosting (.joblib or .pkl)
        for ext in ["joblib", "pkl"]:
            boost_path = os.path.join(self.models_dir, f"boosting_pipeline.{ext}")
            if os.path.exists(boost_path):
                try:
                    import joblib
                    self.loaded_models["boosting"] = joblib.load(boost_path)
                    self.model_status["boosting"] = True
                    print(f"[INFO] Boosting Pipeline loaded successfully from {boost_path}.")
                    break
                except Exception as e:
                    print(f"[WARN] Boosting load failed: {e}")

    def preprocess_image(self, img_pil: Image.Image) -> np.ndarray:
        """Preprocesses input image matching offline dataset specifications."""
        img = img_pil.convert("RGB").resize(self.img_size)
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr

    def _mock_predict(self, pipeline_key: str, top_k: int) -> Dict[str, Any]:
        """Simulation fallback returning realistic synthetic metrics when weights are absent."""
        start_time = time.perf_counter()
        
        # Simulate realistic latency differences between ML and DL architectures
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
        """Runs single-image inference on specified architecture with latency profiling."""
        # Fall back to simulation if the requested checkpoint is not available
        if not self.model_status.get(pipeline_key, False):
            return self._mock_predict(pipeline_key, top_k)

        start_time = time.perf_counter()
        arr = self.preprocess_image(img_pil)

        try:
            if pipeline_key == "dl":
                model_obj = self.loaded_models["dl"]
                # ONNX Runtime inference
                if hasattr(model_obj, "get_inputs"):
                    tensor = np.transpose(arr, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
                    input_name = model_obj.get_inputs()[0].name
                    raw_out = model_obj.run(None, {input_name: tensor})[0][0]
                else:
                    # PyTorch inference
                    import torch
                    tensor = torch.from_numpy(np.transpose(arr, (2, 0, 1))).unsqueeze(0).float()
                    with torch.no_grad():
                        raw_out = model_obj(tensor).squeeze(0).cpu().numpy()

                exp_vals = np.exp(raw_out - np.max(raw_out))
                probs = exp_vals / np.sum(exp_vals)
            else:
                # Scikit-learn / XGBoost pipeline inference
                model = self.loaded_models[pipeline_key]
                flat_input = arr.reshape(1, -1)
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