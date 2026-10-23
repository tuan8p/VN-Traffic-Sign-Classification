from __future__ import annotations

import csv
import json
import os
import random
import time
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms


class TrafficSignPredictor:
    """Unified inference engine for Deep Learning, SVM, and Boosting pipelines."""

    def __init__(
        self,
        models_dir: str | None = None,
        img_size: Tuple[int, int] = (224, 224),
    ) -> None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = models_dir or os.path.join(base_dir, "saved_models")
        
        self.img_size = img_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = self._load_labels()
        self.loaded_models: Dict[str, Any] = {}
        self.scaler = None
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
                std=[0.229, 0.224, 0.225]
            )
        ])

        self._discover_and_load_models()

    def _load_labels(self) -> List[str]:
        """Tự động đọc danh sách 52 nhãn trực tiếp từ class_table.csv."""
        csv_candidates = [
            os.path.join(self.models_dir, "class_table.csv"),
            os.path.join(os.path.dirname(self.models_dir), "configs", "classes.csv"),
            os.path.join(os.path.dirname(self.models_dir), "class_table.csv"),
        ]

        for p in csv_candidates:
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        labels = []
                        for row in reader:
                            code = row.get("sign_code") or row.get("id") or ""
                            name = row.get("name_vi") or row.get("sign_name") or row.get("name") or ""
                            if code and name:
                                labels.append(f"{code.strip()} - {name.strip()}")
                            elif code:
                                labels.append(code.strip())
                        if len(labels) == 52:
                            print(f"[INFO] Loaded {len(labels)} semantic labels from {p}")
                            return labels
                except Exception as e:
                    print(f"[WARN] Failed to read {p}: {e}")

        json_path = os.path.join(self.models_dir, "class_labels.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        return [f"Class {i}" for i in range(52)]

    def _extract_ml_features(self, img_pil: Image.Image) -> np.ndarray:
        """Trích xuất 2020 chiều đặc trưng: HOG (1764) + LBP (160) + HSV Color Hist (96)."""
        from skimage.feature import hog as ski_hog
        from skimage.feature import local_binary_pattern

        img_rgb = np.array(img_pil.convert("RGB").resize((64, 64)))
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # 1. HOG Feature (1764 dims)
        hog_feat = ski_hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
        ).astype(np.float32)

        # 2. LBP Feature (160 dims)
        lbp_img = local_binary_pattern(gray, P=8, R=1, method="default")
        lbp_hist = []
        for r in range(4):
            for c in range(4):
                patch = lbp_img[r * 16 : (r + 1) * 16, c * 16 : (c + 1) * 16]
                h, _ = np.histogram(patch, bins=10, range=(0, 256))
                lbp_hist.extend(h)
        lbp_feat = np.array(lbp_hist, dtype=np.float32)

        # 3. HSV Color Hist (96 dims)
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        h_hist, _ = np.histogram(hsv[:, :, 0], bins=32, range=(0, 180))
        s_hist, _ = np.histogram(hsv[:, :, 1], bins=32, range=(0, 256))
        v_hist, _ = np.histogram(hsv[:, :, 2], bins=32, range=(0, 256))
        color_feat = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)

        feat = np.concatenate([hog_feat, lbp_feat, color_feat]).reshape(1, -1)
        if self.scaler is not None:
            feat = self.scaler.transform(feat)
        return feat

    def _discover_and_load_models(self) -> None:
        import joblib

        # Nạp scaler chuẩn hóa
        scaler_path = os.path.join(self.models_dir, "scaler.joblib")
        if os.path.exists(scaler_path):
            try:
                self.scaler = joblib.load(scaler_path)
                print(f"[INFO] Loaded Scaler from {scaler_path}")
            except Exception as e:
                print(f"[WARN] Failed to load scaler: {e}")

        # 1. Pipeline A: Deep Learning (EfficientNetV2-B0)
        dl_candidates = [
            os.path.join(self.models_dir, "checkpoint_best.pt"),
            os.path.join(self.models_dir, "dl_model.pt"),
        ]
        dl_path = next((p for p in dl_candidates if os.path.exists(p)), None)
        if dl_path:
            try:
                import timm
                num_classes = len(self.labels)
                model = timm.create_model("tf_efficientnetv2_b0", pretrained=False, num_classes=num_classes)
                checkpoint = torch.load(dl_path, map_location=self.device)
                state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
                cleaned_state = {k.replace("module.", "").replace("model.", ""): v for k, v in state_dict.items()}
                model.load_state_dict(cleaned_state, strict=True)
                model.to(self.device).eval()
                self.loaded_models["dl"] = model
                self.model_status["dl"] = True
                print(f"[INFO] Successfully loaded PyTorch DL checkpoint from {dl_path}")
            except Exception as e:
                print(f"[ERROR] Failed to load DL model: {e}")

        # 2. Pipeline B: Machine Learning - SVM
        svm_filenames = [
            "svm.joblib", "svm.pkl", "vm.joblib",
            "svm_pipeline.joblib", "svm_pipeline.pkl", "svm_model.joblib"
        ]
        for filename in svm_filenames:
            svm_path = os.path.join(self.models_dir, filename)
            if os.path.exists(svm_path):
                try:
                    self.loaded_models["svm"] = joblib.load(svm_path)
                    self.model_status["svm"] = True
                    print(f"[INFO] Loaded SVM model from {svm_path}")
                    break
                except Exception as e:
                    print(f"[WARN] Failed to load SVM from {svm_path}: {e}")

        # 3. Pipeline C: Machine Learning - Boosting
        boost_filenames = [
            "boosting.joblib", "boost.joblib", "model.joblib",
            "boosting_pipeline.joblib", "boosting_pipeline.pkl", "model.pkl"
        ]
        for filename in boost_filenames:
            boost_path = os.path.join(self.models_dir, filename)
            if os.path.exists(boost_path):
                try:
                    self.loaded_models["boosting"] = joblib.load(boost_path)
                    self.model_status["boosting"] = True
                    print(f"[INFO] Loaded Boosting model from {boost_path}")
                    break
                except Exception as e:
                    print(f"[WARN] Failed to load Boosting model from {boost_path}: {e}")

    def _mock_predict(self, pipeline_key: str, top_k: int) -> Dict[str, Any]:
        start_time = time.perf_counter()
        time.sleep(0.007)
        k = min(top_k, len(self.labels))
        scores = [random.uniform(0.1, 1.0) for _ in range(k)]
        total = sum(scores) + random.uniform(0.05, 0.25)
        normalized = sorted([s / total for s in scores], reverse=True)
        chosen_labels = random.sample(self.labels, k=k)
        latency = (time.perf_counter() - start_time) * 1000
        return {
            "top_class": chosen_labels[0],
            "confidence": normalized[0],
            "latency_ms": round(latency, 2),
            "predictions": {label: round(prob, 4) for label, prob in zip(chosen_labels, normalized)},
            "is_mock": True,
        }

    def predict(self, img_pil: Image.Image, pipeline_key: str, top_k: int = 5) -> Dict[str, Any]:
        if not self.model_status.get(pipeline_key, False):
            return self._mock_predict(pipeline_key, top_k)

        start_time = time.perf_counter()
        try:
            num_classes = len(self.labels)
            if pipeline_key == "dl":
                img_rgb = img_pil.convert("RGB")
                tensor = self.dl_transform(img_rgb).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.loaded_models["dl"](tensor)
                    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            else:
                model = self.loaded_models[pipeline_key]
                features = self._extract_ml_features(img_pil)

                # Trường hợp 1: Mô hình có xác suất dự đoán (LightGBM/XGBoost)
                probs = None
                if hasattr(model, "predict_proba") and getattr(model, "probability", True):
                    try:
                        probs = model.predict_proba(features)[0]
                    except Exception:
                        probs = None

                # Trường hợp 2: Mô hình SVM (không có predict_proba hoặc SVC probability=False)
                if probs is None:
                    pred_class = int(model.predict(features)[0])
                    probs = np.zeros(num_classes, dtype=np.float32)

                    if hasattr(model, "decision_function"):
                        df = model.decision_function(features)[0]
                        # One-vs-Rest (đúng 52 giá trị)
                        if len(df) == num_classes:
                            exp_d = np.exp(df - np.max(df))
                            probs = exp_d / exp_d.sum()
                        else:
                            # One-vs-One (1326 giá trị cặp), tính điểm số biểu quyết (voting)
                            votes = np.zeros(num_classes, dtype=np.float32)
                            idx = 0
                            for i in range(num_classes):
                                for j in range(i + 1, num_classes):
                                    if df[idx] > 0:
                                        votes[i] += 1
                                    else:
                                        votes[j] += 1
                                    idx += 1
                            probs = votes / votes.sum()
                            probs[pred_class] = max(probs[pred_class], 0.5)
                            probs = probs / probs.sum()
                    else:
                        probs[pred_class] = 1.0

            k = min(top_k, len(probs))
            top_indices = np.argsort(probs)[::-1][:k]
            predictions = {
                self.labels[idx]: round(float(probs[idx]), 4)
                for idx in top_indices
            }
            top_class = self.labels[top_indices[0]]
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