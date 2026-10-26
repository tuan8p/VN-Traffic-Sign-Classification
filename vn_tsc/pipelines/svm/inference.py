"""Single-image inference that mirrors the SVM training feature pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from vn_tsc.data.crops import letterbox
from vn_tsc.features.classical import extract_shared_features


def extract_svm_image_features(
    image: Image.Image | np.ndarray,
    feature_config: dict[str, Any],
    *,
    pad_value: int = 114,
) -> np.ndarray:
    """Convert one already-cropped sign image to its training-time feature row.

    Training sees RGB, aspect-ratio-preserving letterboxed sign crops and then
    extracts HOG/LBP/colour features at ``features.input_size``.  Inference must
    reproduce that contract; flattening resized RGB pixels is not equivalent.
    """
    if isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        rgb = np.asarray(image)
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, axis=2)
        if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
            raise ValueError("SVM inference image must be HxW grayscale, RGB, or RGBA")
        rgb = rgb[..., :3]
        if np.issubdtype(rgb.dtype, np.floating):
            if not np.isfinite(rgb).all():
                raise ValueError("SVM inference image contains non-finite values")
            if rgb.size and float(rgb.max()) <= 1.0:
                rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    size = feature_config.get("input_size", [64, 64])
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError("SVM feature metadata must contain input_size: [width, height]")
    input_size = (int(size[0]), int(size[1]))
    if min(input_size) < 1:
        raise ValueError("SVM feature input_size values must be positive")

    feature_image = letterbox(rgb, input_size, pad_value=pad_value)
    row = extract_shared_features(feature_image, feature_config).astype(np.float32, copy=False)
    if row.ndim != 1 or row.size == 0 or not np.isfinite(row).all():
        raise ValueError("SVM feature extraction produced an invalid vector")
    return row.reshape(1, -1)


def _ovo_to_ovr(pairwise: np.ndarray, n_classes: int) -> np.ndarray:
    """Aggregate libsvm pairwise margins into class-level ranking scores."""
    pairwise = np.atleast_2d(pairwise)
    expected = n_classes * (n_classes - 1) // 2
    if pairwise.shape[1] != expected:
        raise ValueError(
            f"Expected {expected} one-vs-one margins for {n_classes} classes, "
            f"got {pairwise.shape[1]}"
        )
    votes = np.zeros((pairwise.shape[0], n_classes), dtype=np.float64)
    confidence_sums = np.zeros_like(votes)
    k = 0
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            # Matches sklearn's OVO -> OVR transformation for SVC. A positive
            # libsvm margin votes for class i; a negative one votes for class j.
            confidence = -pairwise[:, k]
            confidence_sums[:, i] -= confidence
            confidence_sums[:, j] += confidence
            votes[pairwise[:, k] >= 0, i] += 1
            votes[pairwise[:, k] < 0, j] += 1
            k += 1
    transformed = confidence_sums / (3.0 * (np.abs(confidence_sums) + 1.0))
    return votes + transformed


def svm_class_scores(model: Any, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return model class IDs, normalized ranking scores, and calibration flag."""
    estimator = model.named_steps.get("svm") if hasattr(model, "named_steps") else model
    classes = np.asarray(getattr(estimator, "classes_", []))
    if classes.size < 2:
        raise ValueError("Loaded SVM is not fitted or has fewer than two classes")

    if bool(getattr(estimator, "probability", False)):
        probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
        return classes, probabilities[0], True

    raw = np.asarray(model.decision_function(features), dtype=np.float64)
    raw = np.atleast_2d(raw)
    if classes.size == 2 and raw.shape[1] == 1:
        scores = np.column_stack((-raw[:, 0], raw[:, 0]))
    elif raw.shape[1] == classes.size:
        scores = raw
    else:
        scores = _ovo_to_ovr(raw, int(classes.size))

    # SVC without probability calibration has margins, not probabilities.  A
    # softmax provides a normalized display/ranking score while the boolean
    # return value ensures callers do not present it as calibrated confidence.
    scores = scores - scores.max(axis=1, keepdims=True)
    normalized = np.exp(scores)
    normalized /= normalized.sum(axis=1, keepdims=True)
    return classes, normalized[0], False


def predict_svm_image(
    model: Any,
    image: Image.Image | np.ndarray,
    feature_config: dict[str, Any],
    *,
    top_k: int = 5,
    labels_by_id: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Predict a cropped sign image with feature and dimension validation."""
    features = extract_svm_image_features(image, feature_config)
    expected = getattr(model, "n_features_in_", None)
    if expected is None and hasattr(model, "named_steps"):
        expected = getattr(model.named_steps.get("scaler"), "n_features_in_", None)
    if expected is not None and features.shape[1] != int(expected):
        raise ValueError(
            "SVM feature dimension mismatch: model expects "
            f"{int(expected)}, inference extracted {features.shape[1]}. "
            "Use the feature_config.json saved beside this checkpoint."
        )

    predicted_id = int(np.asarray(model.predict(features))[0])
    classes, scores, calibrated = svm_class_scores(model, features)
    k = min(max(1, int(top_k)), len(classes))
    ranked = np.argsort(scores)[::-1].tolist()
    predicted_pos = np.flatnonzero(classes == predicted_id)
    if predicted_pos.size and ranked[0] != int(predicted_pos[0]):
        ranked.remove(int(predicted_pos[0]))
        ranked.insert(0, int(predicted_pos[0]))
    ranked = ranked[:k]

    labels = labels_by_id or {}
    entries = [
        {
            "class_id": int(classes[pos]),
            "label": labels.get(int(classes[pos]), f"Class {int(classes[pos])}"),
            "score": float(scores[pos]),
        }
        for pos in ranked
    ]
    return {
        "class_id": predicted_id,
        "label": labels.get(predicted_id, f"Class {predicted_id}"),
        "score": entries[0]["score"],
        "scores_calibrated": calibrated,
        "top_k": entries,
    }
