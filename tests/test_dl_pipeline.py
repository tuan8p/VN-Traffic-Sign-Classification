"""Comprehensive unit tests for DL pipeline modules."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vn_tsc.config.resolve import resolve_config
from vn_tsc.pipelines.dl.dataset import NpyDataset
from vn_tsc.data.preprocess_online import build_eval_transforms, build_train_transforms
from vn_tsc.eval.metrics import compute_classification_metrics
from vn_tsc.eval.plots import confusion_matrix_png, curves_from_history_json
from vn_tsc.pipelines.dl.model import (
    build_model,
    count_trainable,
    freeze_backbone,
    unfreeze_backbone,
)
from vn_tsc.pipelines.dl.train import (
    _eval_epoch,
    _grad_scaler,
    _make_optimizer,
    _make_scheduler,
    _num_classes_from_table,
    _train_epoch,
)


@pytest.fixture
def dl_config():
    """Return resolved DL config with pretrained=False for fast testing."""
    root = Path(__file__).resolve().parents[1]
    cfg = resolve_config(
        pipeline_yaml=root / "configs/pipelines/dl.yaml",
        shared_yaml=root / "configs/shared.yaml",
        runtime_yaml=root / "configs/runtime/local.yaml",
        overrides={
            "model": {"pretrained": False, "num_classes": 5},
            "train": {"amp": False, "batch_size": 4},
            "runtime_cfg": {"num_workers": 0, "device": "cpu"},
        },
    )
    return cfg


# ===========================================================================
# 1. Model & Backbone Freeze/Unfreeze Tests
# ===========================================================================

def test_model_build_and_forward(dl_config):
    """Test model instantiation and forward pass with correct logits shape."""
    num_classes = 5
    model = build_model(dl_config, num_classes=num_classes)
    assert isinstance(model, nn.Module)

    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy_input)

    assert out.shape == (2, num_classes)
    assert not torch.isnan(out).any()


def test_freeze_and_unfreeze_backbone(dl_config):
    """Verify Stage 1 (freeze backbone) and Stage 2 (unfreeze) parameter states."""
    model = build_model(dl_config, num_classes=5)
    total_params = sum(p.numel() for p in model.parameters())

    # Stage 1: Freeze
    freeze_backbone(model)
    trainable_stage1 = count_trainable(model)
    assert 0 < trainable_stage1 < total_params

    # Head parameters must require grad, backbone parameters must NOT
    head_param_ids = {id(p) for p in model.get_classifier().parameters()}
    for param in model.parameters():
        if id(param) in head_param_ids:
            assert param.requires_grad, "Classifier head param should require grad"
        else:
            assert not param.requires_grad, "Backbone param should be frozen"

    # Stage 2: Unfreeze
    unfreeze_backbone(model)
    trainable_stage2 = count_trainable(model)
    assert trainable_stage2 == total_params
    for name, param in model.named_parameters():
        assert param.requires_grad, f"Param {name} should require grad after unfreeze"


# ===========================================================================
# 2. Preprocessing & Online Transforms Tests
# ===========================================================================

def test_online_transforms(dl_config):
    """Verify online train & eval transforms output valid normalized tensors."""
    sample_uint8 = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

    eval_tf = build_eval_transforms(dl_config)
    eval_tensor = eval_tf(sample_uint8)
    assert isinstance(eval_tensor, torch.Tensor)
    assert eval_tensor.shape == (3, 224, 224)
    assert eval_tensor.dtype == torch.float32

    # Train transforms with jitter
    train_tf = build_train_transforms(dl_config)
    train_tensor = train_tf(sample_uint8)
    assert isinstance(train_tensor, torch.Tensor)
    assert train_tensor.shape == (3, 224, 224)
    assert train_tensor.dtype == torch.float32

    # Test with random_resized_crop enabled
    crop_cfg = dict(dl_config)
    crop_cfg["aug"] = {"random_resized_crop": True, "hflip": False, "color_jitter": 0.2}
    rrc_tf = build_train_transforms(crop_cfg)
    rrc_tensor = rrc_tf(sample_uint8)
    assert rrc_tensor.shape == (3, 224, 224)


# ===========================================================================
# 3. Dataset Wrapper Tests
# ===========================================================================

def test_npy_dataset_and_loader(dl_config):
    """Test NpyDataset and DataLoader batch collation."""
    num_samples = 8
    X = np.random.randint(0, 256, (num_samples, 224, 224, 3), dtype=np.uint8)
    y = np.random.randint(0, 5, size=(num_samples,), dtype=np.int64)

    eval_tf = build_eval_transforms(dl_config)
    dataset = NpyDataset(X, y, transform=eval_tf)
    assert len(dataset) == num_samples

    img, label = dataset[0]
    assert img.shape == (3, 224, 224)
    assert isinstance(label, int)

    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    for bx, by in loader:
        assert bx.shape == (4, 3, 224, 224)
        assert by.shape == (4,)
        assert by.dtype == torch.int64
        break


# ===========================================================================
# 4. Metric Computation Tests
# ===========================================================================

def test_metrics_calculation():
    """Verify accuracy, macro_f1, and weighted_f1 calculation."""
    # 4 classes: 0, 1, 2, 3
    y_true = [0, 0, 1, 1, 2, 2, 3, 3]
    y_pred = [0, 0, 1, 0, 2, 2, 3, 3]  # 1 mistake on class 1 (predicted 0)

    # 7 out of 8 correct -> accuracy = 0.875
    # Class 0: precision = 2/3, recall = 2/2 = 1.0, f1 = 0.8
    # Class 1: precision = 1/1 = 1.0, recall = 1/2 = 0.5, f1 = 2/3
    # Class 2: precision = 1.0, recall = 1.0, f1 = 1.0
    # Class 3: precision = 1.0, recall = 1.0, f1 = 1.0
    metrics = compute_classification_metrics(y_true, y_pred, labels=[0, 1, 2, 3])

    assert metrics["accuracy"] == pytest.approx(7 / 8)
    expected_macro = (0.8 + (2 / 3) + 1.0 + 1.0) / 4.0
    assert metrics["macro_f1"] == pytest.approx(expected_macro)
    assert "weighted_f1" in metrics
    assert 0.0 <= metrics["weighted_f1"] <= 1.0


def test_metrics_zero_division():
    """Verify robust handling of unobserved classes."""
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    # Class 2 is not present in true or pred
    metrics = compute_classification_metrics(y_true, y_pred, labels=[0, 1, 2])
    assert metrics["accuracy"] == 1.0
    # Class 0: f1=1, Class 1: f1=1, Class 2: f1=0 (due to zero_division=0)
    assert metrics["macro_f1"] == pytest.approx((1.0 + 1.0 + 0.0) / 3.0)


# ===========================================================================
# 5. Training & Evaluation Epoch Tests
# ===========================================================================

def test_train_and_eval_epochs(dl_config):
    """Test single training and validation epoch execution."""
    num_classes = 4
    model = build_model(dl_config, num_classes=num_classes)
    device = torch.device("cpu")
    model = model.to(device)

    # Synthetic batch of 6 samples
    X = np.random.randint(0, 256, (6, 224, 224, 3), dtype=np.uint8)
    y = np.array([0, 1, 2, 3, 0, 1], dtype=np.int64)

    tf = build_eval_transforms(dl_config)
    dataset = NpyDataset(X, y, transform=tf)
    loader = DataLoader(dataset, batch_size=3, shuffle=False)

    optimizer = _make_optimizer(model, lr=1e-3, weight_decay=1e-4)
    scheduler = _make_scheduler(optimizer, epochs=5)
    criterion = nn.CrossEntropyLoss()
    scaler = _grad_scaler(enabled=False)

    # Train epoch
    loss = _train_epoch(
        model, loader, optimizer, criterion, scaler,
        device=device, use_amp=False, sampler=None, epoch=1,
    )
    assert isinstance(loss, float)
    assert loss > 0.0
    scheduler.step()

    # Eval epoch
    val_loss, metrics, preds, targets = _eval_epoch(
        model, loader, criterion, device=device, use_amp=False, num_classes=num_classes,
    )
    assert isinstance(val_loss, float)
    assert len(preds) == len(X)
    assert len(targets) == len(y)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0


# ===========================================================================
# 6. Evaluation Plots & Figures Tests
# ===========================================================================

def test_curves_and_confusion_matrix_plots(tmp_path):
    """Verify plot generation functions."""
    # Mock history.json
    history_data = {
        "stage1": [
            {"epoch": 1, "train_loss": 1.5, "val_loss": 1.2, "val_accuracy": 0.6, "val_macro_f1": 0.55, "val_weighted_f1": 0.58},
            {"epoch": 2, "train_loss": 1.1, "val_loss": 0.9, "val_accuracy": 0.7, "val_macro_f1": 0.65, "val_weighted_f1": 0.68},
        ],
        "stage2": [
            {"epoch": 3, "train_loss": 0.7, "val_loss": 0.6, "val_accuracy": 0.8, "val_macro_f1": 0.78, "val_weighted_f1": 0.79},
        ],
    }
    history_file = tmp_path / "history.json"
    history_file.write_text(json.dumps(history_data), encoding="utf-8")

    out_dir = tmp_path / "plots"
    saved = curves_from_history_json(history_file, out_dir)
    assert len(saved) == 2
    assert (out_dir / "loss_curve.png").exists()
    assert (out_dir / "metric_curves.png").exists()

    # Confusion matrix
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 1, 0, 1, 2]
    class_names = ["c0", "c1", "c2"]
    cm_path = out_dir / "confusion_matrix.png"
    result_path = confusion_matrix_png(y_true, y_pred, class_names, cm_path)
    assert Path(result_path).exists()
    assert Path(result_path).stat().st_size > 0
