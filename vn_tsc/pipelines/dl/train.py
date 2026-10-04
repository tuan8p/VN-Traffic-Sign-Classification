"""EfficientNetV2-B0 fine-tuning pipeline with two-stage training.

Architecture
============
Stage 1 — Head warm-up
  Backbone frozen; only the final classifier trains for ``train.stage1_epochs``
  epochs at ``train.lr``. This prevents early gradient noise from destroying
  pretrained features.

Stage 2 — Full fine-tune
  All parameters unfrozen; trains for ``train.stage2_epochs`` epochs at
  ``train.lr_stage2`` (typically 10× smaller).

Both stages share the same optimiser type (AdamW) and cosine LR scheduler.

Data
====
Reads ``data/processed/images/X_*.npy`` (uint8 RGB crops, shape N×H×W×3) and
the corresponding ``y_*.npy`` label arrays written by shared preprocessing.
With ``train.with_aug: true``, the offline-augmented ``X_train_aug.npy`` is
appended to the training set before Stage 1.

Distributed / AMP
=================
DistributedDataParallel (DDP) is activated when ``torch.distributed`` is
already initialised (Kaggle 2×T4 via ``torchrun``). Single-GPU / CPU runs
work without any changes.

Mixed precision (AMP) is controlled by ``train.amp`` in config.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any
import warnings

warnings.filterwarnings("ignore", message=".*The given NumPy array is not writable.*")
warnings.filterwarnings("ignore", message=".*find_unused_parameters=True was specified in DDP constructor.*")
warnings.filterwarnings("ignore", message=".*barrier\\(\\): using the device under current context.*")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
    def _autocast(enabled: bool):
        return torch.amp.autocast(device_type="cuda", enabled=enabled)

    def _grad_scaler(enabled: bool):
        return torch.amp.GradScaler("cuda", enabled=enabled)
else:
    from torch.cuda.amp import GradScaler as _GradScaler, autocast as _autocast_cuda

    def _autocast(enabled: bool):
        return _autocast_cuda(enabled=enabled)

    def _grad_scaler(enabled: bool):
        return _GradScaler(enabled=enabled)

from vn_tsc.data.dataset import load_split
from vn_tsc.data.preprocess_online import build_eval_transforms, build_train_transforms
from vn_tsc.eval.metrics import compute_classification_metrics
from vn_tsc.pipelines.base import BasePipeline
from vn_tsc.pipelines.dl.dataset import NpyDataset
from vn_tsc.pipelines.dl.model import (
    build_model,
    count_trainable,
    freeze_backbone,
    unfreeze_backbone,
)
from vn_tsc.runtime.pack import zip_run_dir
from vn_tsc.runtime.wandb_gate import require_wandb
from vn_tsc.utils.io import save_json, save_yaml
from vn_tsc.utils.seed import set_seed

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_main_process() -> bool:
    """True for the rank-0 process (or when not using DDP)."""
    return int(os.environ.get("RANK", 0)) == 0


def _setup_distributed(cfg: dict[str, Any]) -> tuple[bool, int, int]:
    """Initialize torch.distributed if launched via torchrun or multi-GPU environment.

    Returns:
        (use_ddp, rank, local_rank)
    """
    world_size_env = os.environ.get("WORLD_SIZE")
    rank_env = os.environ.get("RANK")
    local_rank_env = os.environ.get("LOCAL_RANK")

    if world_size_env is not None and int(world_size_env) > 1:
        rank = int(rank_env or 0)
        local_rank = int(local_rank_env or 0)
        backend = str(cfg.get("distributed", {}).get("backend", "nccl" if torch.cuda.is_available() else "gloo"))
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
            torch.distributed.init_process_group(
                backend=backend,
                init_method="env://",
            )
        return True, rank, local_rank
    elif torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        local_rank = int(local_rank_env or rank)
        return True, rank, local_rank

    return False, 0, 0


def _device(cfg: dict[str, Any], local_rank: int = 0) -> torch.device:
    runtime = cfg.get("runtime_cfg", {})
    device_str = runtime.get("device", None)
    if device_str == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        if device_str and device_str.startswith("cuda:"):
            return torch.device(device_str)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")



def _num_workers(cfg: dict[str, Any]) -> int:
    return int(cfg.get("runtime_cfg", {}).get("num_workers", 4))


def _pin_memory(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("runtime_cfg", {}).get("pin_memory", True))


def _build_loader(
    X: np.ndarray,
    y: np.ndarray,
    transform,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    use_ddp: bool,
    seed: int = 42,
) -> DataLoader:
    dataset = NpyDataset(X, y, transform=transform)
    sampler = DistributedSampler(dataset, shuffle=shuffle, seed=seed) if use_ddp else None
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=shuffle,   # drop incomplete batch for train only
        # persistent_workers requires num_workers > 0 and is not supported on
        # Windows with the default 'spawn' start method when using mmap arrays.
        persistent_workers=False,
        generator=generator,
    )


def _make_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int = 0,
) -> Any:
    if warmup_epochs > 0 and epochs > warmup_epochs:
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)


def _num_classes_from_table(processed_root: str, cfg: dict[str, Any] | None = None) -> int:
    """Read the authoritative class count from config or class_table.csv.

    This is the only correct source: inferring from ``np.max(y)`` is unsafe
    because a split may not contain every class (especially val/test with rare
    classes), which would cause the model head to have the wrong output dim and
    make checkpoint loading fail.
    """
    if cfg is not None and cfg.get("model", {}).get("num_classes") is not None:
        return int(cfg["model"]["num_classes"])

    try:
        from vn_tsc.data.dataset import load_class_table
        return len(load_class_table(processed_root))
    except Exception:
        pass

    if cfg is not None and cfg.get("data", {}).get("num_classes") is not None:
        return int(cfg["data"]["num_classes"])

    from pathlib import Path as _Path
    import csv as _csv

    for candidate in (
        _Path(processed_root) / "class_table.csv",
        _Path("configs/classes.csv"),
    ):
        if candidate.exists():
            with candidate.open(encoding="utf-8", newline="") as f:
                rows = list(_csv.DictReader(f))
            return len(rows)
    raise FileNotFoundError(
        "class_table.csv not found in processed_root or configs/. "
        "Run `python -m tools.run_preprocess` first, or specify model.num_classes in config."
    )


# ---------------------------------------------------------------------------
# Per-epoch train / eval
# ---------------------------------------------------------------------------

def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: Any,
    device: torch.device,
    use_amp: bool,
    sampler: DistributedSampler | None,
    epoch: int,
    total_epochs: int = 1,
    stage_name: str = "train",
    is_main: bool = True,
) -> float:
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)
    total_loss = 0.0
    total_samples = 0
    pbar = tqdm(
        loader,
        desc=f"[{stage_name}] Epoch {epoch:2d}/{total_epochs:2d}",
        disable=not is_main,
        leave=False,
        dynamic_ncols=True,
    )
    for X_batch, y_batch in pbar:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(enabled=use_amp):
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        bs = len(y_batch)
        total_loss += loss.item() * bs
        total_samples += bs
        if is_main:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def _eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    num_classes: int,
    rare_classes_list: list[int] | None = None,
) -> tuple[float, dict[str, float], list[int], list[int]]:
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        with _autocast(enabled=use_amp):
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(y_batch.cpu().numpy().tolist())
    avg_loss = total_loss / max(len(loader.dataset), 1)
    metrics = compute_classification_metrics(
        np.array(all_labels), np.array(all_preds),
        labels=list(range(num_classes)),
        rare_labels=rare_classes_list,
    )
    return avg_loss, metrics, all_preds, all_labels


# ---------------------------------------------------------------------------
# Training stage (shared by Stage 1 and Stage 2)
# ---------------------------------------------------------------------------


def _run_stage(
    stage_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    use_amp: bool,
    num_classes: int,
    early_stopping_patience: int,
    best_checkpoint_path: Path,
    run,                        # wandb run or None
    history: dict,
    sampler: DistributedSampler | None,
    is_main: bool,
    rare_classes_list: list[int] | None = None,
    epoch_offset: int = 0,
    global_best_score: float = -1.0,
    warmup_epochs: int = 0,
    primary_metric: str = "macro_f1",
) -> tuple[dict[str, float], float]:
    """Run one training stage; returns (best_val_metrics_of_stage, updated_global_best_score)."""
    optimizer = _make_optimizer(model, lr, weight_decay)
    scheduler = _make_scheduler(optimizer, epochs, warmup_epochs=warmup_epochs)
    scaler = _grad_scaler(enabled=use_amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    history.setdefault(stage_name, [])
    patience_counter = 0
    best_metrics: dict[str, float] = {}
    stage_best_score = -1.0

    if is_main:
        log.info("[%s] trainable params: %s", stage_name, f"{count_trainable(model):,}")

    for epoch in range(1, epochs + 1):
        global_epoch = epoch_offset + epoch

        train_loss = _train_epoch(
            model, train_loader, optimizer, criterion, scaler,
            device, use_amp, sampler, epoch,
            total_epochs=epochs, stage_name=stage_name, is_main=is_main,
        )
        scheduler.step()

        val_loss, val_metrics, _, _ = _eval_epoch(
            model, val_loader, criterion, device, use_amp, num_classes, rare_classes_list,
        )
        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["macro_f1"]

        entry = {
            "epoch": global_epoch,
            "stage_epoch": epoch,
            "stage": stage_name,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            **{f"val_{k}": round(v, 5) for k, v in val_metrics.items()},
            "lr": scheduler.get_last_lr()[0],
        }
        history[stage_name].append(entry)

        # Track score by primary_metric (default: macro_f1)
        score = val_f1 if primary_metric == "macro_f1" else val_acc
        metric_label = "MACRO F1" if primary_metric == "macro_f1" else "ACC"

        # Track best metrics for this stage & early stopping
        if score > stage_best_score:
            stage_best_score = score
            best_metrics = dict(val_metrics)
            patience_counter = 0
        else:
            patience_counter += 1

        # Checkpoint saving across all stages: tracked by BEST GLOBAL SCORE
        is_global_best = score > global_best_score
        if is_global_best:
            global_best_score = score
            if is_main:
                _save_checkpoint(model, best_checkpoint_path)

        if is_main:
            best_tag = f" -> [BEST VAL {metric_label}: {score:.4f}]" if is_global_best else ""
            print(
                f"[{stage_name}] Epoch {epoch:2d}/{epochs:2d} (Global {global_epoch:2d}) | "
                f"train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | "
                f"val_acc: {val_acc:.4f} | val_macro_f1: {val_f1:.4f} | "
                f"lr: {scheduler.get_last_lr()[0]:.2e}{best_tag}",
                flush=True,
            )
            log.info(
                "[%s] epoch %d/%d (global %d)  train_loss=%.4f  val_loss=%.4f  "
                "val_acc=%.4f  val_macro_f1=%.4f%s",
                stage_name, epoch, epochs, global_epoch, train_loss, val_loss,
                val_acc, val_f1, best_tag,
            )
            if run is not None:
                current_lr = scheduler.get_last_lr()[0]
                # Log unified continuous metrics (single loss and accuracy curves for all stages)
                run.log({
                    "epoch": global_epoch,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                    "val/macro_f1": val_f1,
                    "val/macro_f1_common": val_metrics.get("macro_f1_common", val_f1),
                    "val/weighted_f1": val_metrics.get("weighted_f1", val_acc),
                    "lr": current_lr,
                })

        if patience_counter >= early_stopping_patience:
            if is_main:
                print(
                    f"[{stage_name}] early stopping at epoch {epoch} (patience={early_stopping_patience})",
                    flush=True,
                )
                log.info(
                    "[%s] early stopping at epoch %d (patience=%d)",
                    stage_name, epoch, early_stopping_patience,
                )
            break

    return best_metrics, global_best_score


def _save_checkpoint(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unwrap DDP before saving.
    raw = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    torch.save(raw.state_dict(), path)


# ---------------------------------------------------------------------------
# DLPipeline
# ---------------------------------------------------------------------------

class DLPipeline(BasePipeline):
    """Two-stage or single-stage fine-tuning of EfficientNetV2-B0 on Vietnamese traffic signs."""

    name = "dl"

    def fit(self) -> dict[str, Any]:  # noqa: PLR0912, PLR0915
        cfg = self.cfg
        train_cfg = cfg.get("train", {})

        use_ddp, rank, local_rank = _setup_distributed(cfg)
        is_main = (rank == 0)

        # Synchronize run_dir across all DDP ranks so everyone logs/saves to identical path
        if use_ddp:
            obj = [str(self.run_dir)] if is_main else [None]
            torch.distributed.broadcast_object_list(obj, src=0)
            self.run_dir = Path(obj[0])

        entity = cfg.get("project", {}).get("wandb_entity", "P4AIDS_ML")
        project = cfg.get("project", {}).get("wandb_project", "BTL")
        run = require_wandb(
            entity=entity, project=project,
            enabled=bool(train_cfg.get("require_wandb", True)) and is_main,
        )

        device = _device(cfg, local_rank=local_rank)
        use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
        num_workers = _num_workers(cfg)
        pin_memory = _pin_memory(cfg) and device.type == "cuda"

        batch_size = int(train_cfg.get("batch_size", 64))
        weight_decay = float(train_cfg.get("weight_decay", 1e-4))
        lr = float(train_cfg.get("lr", 3e-4))
        lr_stage2 = float(train_cfg.get("lr_stage2", lr / 10))
        warmup_epochs = int(train_cfg.get("warmup_epochs", 5))
        two_stage = bool(train_cfg.get("two_stage", False))
        primary_metric = str(train_cfg.get("primary_metric", "macro_f1")).lower()
        if primary_metric not in ("macro_f1", "accuracy"):
            primary_metric = "macro_f1"
        epochs = int(train_cfg.get("epochs", int(train_cfg.get("stage1_epochs", 0)) + int(train_cfg.get("stage2_epochs", 30))))
        if epochs <= 0:
            epochs = 30
        stage1_epochs = int(train_cfg.get("stage1_epochs", 5))
        stage2_epochs = int(train_cfg.get("stage2_epochs", 25))
        early_stop_patience = int(
            cfg.get("train_common", {}).get("early_stopping_patience", 10)
        )
        with_aug = bool(train_cfg.get("with_aug", True))

        if is_main:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            save_yaml(cfg, self.run_dir / "resolved_config.yaml")
            log_dir = self.run_dir / "logs"
            log_dir.mkdir(exist_ok=True)

        # ---- Data ----
        if is_main:
            log.info("Loading training data (with_aug=%s)…", with_aug)
        X_train, y_train = load_split("train", cfg["data"]["processed_root"], with_aug=with_aug)
        X_val, y_val = load_split("val", cfg["data"]["processed_root"])

        num_classes = _num_classes_from_table(cfg["data"]["processed_root"], cfg)

        if is_main:
            log.info(
                "Data: train=%d val=%d num_classes=%d image_shape=%s",
                len(X_train), len(X_val), num_classes, X_train.shape[1:],
            )

        train_tf = build_train_transforms(cfg)
        eval_tf = build_eval_transforms(cfg)

        train_sampler: DistributedSampler | None = None
        seed = int(cfg.get("seed", 42))
        set_seed(seed)

        train_loader = _build_loader(X_train, y_train, train_tf, batch_size,
                                     shuffle=True, num_workers=num_workers,
                                     pin_memory=pin_memory, use_ddp=use_ddp,
                                     seed=seed)
        val_loader = _build_loader(X_val, y_val, eval_tf, batch_size * 2,
                                   shuffle=False, num_workers=num_workers,
                                   pin_memory=pin_memory, use_ddp=False,
                                   seed=seed)
        if use_ddp:
            train_sampler = train_loader.sampler  # type: ignore[assignment]

        # ---- Model ----
        raw_model = build_model(cfg, num_classes=num_classes)
        raw_model = raw_model.to(device)
        find_unused = bool(cfg.get("distributed", {}).get("find_unused_parameters", True))

        checkpoint_path = self.run_dir / "checkpoints" / "checkpoint_best.pt"
        history: dict[str, list] = {}

        if is_main and run is not None:
            run.name = f"dl_effnetv2b0_{self.run_dir.name}"
            run.config.update(cfg, allow_val_change=True)
            run.config.update(
                {"num_classes": num_classes, "with_aug": with_aug,
                 "epochs": epochs, "two_stage": two_stage,
                 "primary_metric": primary_metric},
                allow_val_change=True,
            )
            # Define metrics so W&B automatically builds unified single loss and accuracy charts vs epoch
            try:
                run.define_metric("epoch")
                run.define_metric("train/loss", step_metric="epoch")
                run.define_metric("val/loss", step_metric="epoch")
                run.define_metric("val/accuracy", step_metric="epoch")
                run.define_metric("val/macro_f1", step_metric="epoch")
                run.define_metric("val/macro_f1_common", step_metric="epoch")
                run.define_metric("val/weighted_f1", step_metric="epoch")
                run.define_metric("lr", step_metric="epoch")
            except Exception:
                pass

        rare_cls: list[int] | None = None
        try:
            from vn_tsc.data.dataset import rare_classes
            threshold = int(cfg.get("metrics", {}).get("rare_class_threshold", 30))
            rare_cls = rare_classes(cfg["data"]["processed_root"], threshold=threshold)
            if is_main and rare_cls:
                log.info("Identified %d rare classes (<%d samples)", len(rare_cls), threshold)
        except Exception:
            pass

        if two_stage:
            # ---- Stage 1: Head warm-up ----
            if is_main:
                print(f"\n=======================================================", flush=True)
                print(f"=== Stage 1: Head warm-up ({stage1_epochs} epochs, lr={lr:.2e}) ===", flush=True)
                print(f"=======================================================\n", flush=True)
                log.info("=== Stage 1: Head warm-up (%d epochs, lr=%.2e) ===",
                         stage1_epochs, lr)
            freeze_backbone(raw_model)
            model = (
                nn.parallel.DistributedDataParallel(
                    raw_model,
                    device_ids=[local_rank] if device.type == "cuda" else None,
                    output_device=local_rank if device.type == "cuda" else None,
                    find_unused_parameters=find_unused,
                )
                if use_ddp
                else raw_model
            )

            best_stage1, global_best_score = _run_stage(
                "stage1", model, train_loader, val_loader,
                epochs=stage1_epochs, lr=lr, weight_decay=weight_decay,
                device=device, use_amp=use_amp, num_classes=num_classes,
                early_stopping_patience=early_stop_patience,
                best_checkpoint_path=checkpoint_path,
                run=run if is_main else None,
                history=history, sampler=train_sampler, is_main=is_main,
                rare_classes_list=rare_cls,
                epoch_offset=0,
                global_best_score=-1.0,
                warmup_epochs=0,
                primary_metric=primary_metric,
            )

            # ---- Stage 2: Full fine-tune ----
            if is_main:
                print(f"\n=======================================================", flush=True)
                print(f"=== Stage 2: Full fine-tune ({stage2_epochs} epochs, lr={lr_stage2:.2e}) ===", flush=True)
                print(f"=======================================================\n", flush=True)
                log.info("=== Stage 2: Full fine-tune (%d epochs, lr=%.2e) ===",
                         stage2_epochs, lr_stage2)
            unfreeze_backbone(raw_model)
            model = (
                nn.parallel.DistributedDataParallel(
                    raw_model,
                    device_ids=[local_rank] if device.type == "cuda" else None,
                    output_device=local_rank if device.type == "cuda" else None,
                    find_unused_parameters=find_unused,
                )
                if use_ddp
                else raw_model
            )

            best_stage2, global_best_score = _run_stage(
                "stage2", model, train_loader, val_loader,
                epochs=stage2_epochs, lr=lr_stage2, weight_decay=weight_decay,
                device=device, use_amp=use_amp, num_classes=num_classes,
                early_stopping_patience=early_stop_patience,
                best_checkpoint_path=checkpoint_path,
                run=run if is_main else None,
                history=history, sampler=train_sampler, is_main=is_main,
                rare_classes_list=rare_cls,
                epoch_offset=stage1_epochs,
                global_best_score=global_best_score,
                warmup_epochs=0,
                primary_metric=primary_metric,
            )

            # Best overall = max primary_metric across both stages
            best_val = (
                best_stage2
                if best_stage2.get(primary_metric, -1) >= best_stage1.get(primary_metric, -1)
                else best_stage1
            )
            stage1_metrics = best_stage1
            stage2_metrics = best_stage2
        else:
            # ---- Full End-to-End Training (Single Stage) ----
            if is_main:
                print(f"\n=======================================================", flush=True)
                print(f"=== Full Training End-to-End ({epochs} epochs, lr={lr:.2e}, warmup={warmup_epochs}, metric={primary_metric}) ===", flush=True)
                print(f"=======================================================\n", flush=True)
                log.info("=== Full Training End-to-End (%d epochs, lr=%.2e, warmup=%d, metric=%s) ===",
                         epochs, lr, warmup_epochs, primary_metric)
            unfreeze_backbone(raw_model)
            model = (
                nn.parallel.DistributedDataParallel(
                    raw_model,
                    device_ids=[local_rank] if device.type == "cuda" else None,
                    output_device=local_rank if device.type == "cuda" else None,
                    find_unused_parameters=find_unused,
                )
                if use_ddp
                else raw_model
            )

            best_val, global_best_score = _run_stage(
                "train", model, train_loader, val_loader,
                epochs=epochs, lr=lr, weight_decay=weight_decay,
                device=device, use_amp=use_amp, num_classes=num_classes,
                early_stopping_patience=early_stop_patience,
                best_checkpoint_path=checkpoint_path,
                run=run if is_main else None,
                history=history, sampler=train_sampler, is_main=is_main,
                rare_classes_list=rare_cls,
                epoch_offset=0,
                global_best_score=-1.0,
                warmup_epochs=warmup_epochs,
                primary_metric=primary_metric,
            )
            stage1_metrics = best_val
            stage2_metrics = best_val

        metrics: dict[str, Any] = {
            "validation": best_val,
            "train_samples": int(len(X_train)),
            "val_samples": int(len(X_val)),
            "num_classes": num_classes,
            "stage1_best": stage1_metrics,
            "stage2_best": stage2_metrics,
        }


        # Auto-evaluate on test split right after training completes
        auto_eval = bool(cfg.get("eval", {}).get("auto_eval_test", True))
        if auto_eval and is_main:
            print(f"\n=======================================================", flush=True)
            print(f"=== Running automatic evaluation on test split ===", flush=True)
            print(f"=======================================================\n", flush=True)
            log.info("=== Running automatic evaluation on test split ===")
            try:
                test_metrics = self.evaluate(split="test")
                metrics["test"] = test_metrics
                if "samples" in test_metrics:
                    metrics["test_samples"] = test_metrics["samples"]
                if run is not None:
                    run.summary.update({
                        f"test/{k}": v for k, v in test_metrics.items() if isinstance(v, (int, float))
                    })
                    cm_test_path = self.run_dir / "confusion_matrix_test.png"
                    if cm_test_path.is_file():
                        run.summary["confusion_matrix_test"] = str(cm_test_path)
                        try:
                            import wandb
                            run.log({"media/confusion_matrix_test": wandb.Image(str(cm_test_path))})
                        except Exception:
                            pass
                print(
                    f"\n[Test Result] Accuracy: {test_metrics['accuracy']:.4f} "
                    f"({test_metrics['accuracy']*100:.2f}%) | "
                    f"Macro F1: {test_metrics['macro_f1']:.4f} | "
                    f"Weighted F1: {test_metrics['weighted_f1']:.4f}\n",
                    flush=True,
                )
            except Exception as e:
                log.error("Failed to run auto-evaluation on test set: %s", e)

        if is_main:
            save_json(history, self.run_dir / "history.json")
            save_json(metrics, self.run_dir / "metrics.json")

            if run is not None:
                run.summary.update({
                    f"best_val/{k}": v for k, v in best_val.items() if isinstance(v, (int, float))
                })
                run.summary["best_val_accuracy"] = best_val.get("accuracy", 0.0)
                run.summary["best_val_macro_f1"] = best_val.get("macro_f1", 0.0)
                run.summary["checkpoint"] = str(checkpoint_path)

            log.info(
                "Training complete. Best val → accuracy=%.4f macro_f1=%.4f",
                best_val.get("accuracy", 0), best_val.get("macro_f1", 0),
            )
            if "test" in metrics:
                log.info(
                    "Test evaluation → accuracy=%.4f macro_f1=%.4f (common=%.4f)",
                    metrics["test"].get("accuracy", 0),
                    metrics["test"].get("macro_f1", 0),
                    metrics["test"].get("macro_f1_common", metrics["test"].get("macro_f1", 0)),
                )

            if cfg.get("outputs", {}).get("save_curves", True):
                try:
                    from vn_tsc.eval.plots import curves_from_history_json
                    curve_paths = curves_from_history_json(self.run_dir / "history.json", self.run_dir / "figures")
                    if run is not None:
                        try:
                            import wandb
                            for cp in curve_paths:
                                run.log({f"media/{cp.stem}": wandb.Image(str(cp))})
                        except Exception:
                            pass
                except Exception as e:
                    log.warning("Could not generate loss/metric curves: %s", e)

            if cfg.get("outputs", {}).get("zip_after_train", True):
                zip_run_dir(self.run_dir)

        if run is not None:
            run.finish()

        if use_ddp and torch.distributed.is_initialized():
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()

        return metrics



    def evaluate(self, split: str | None = None) -> dict[str, Any]:
        """Evaluate saved checkpoint on val or test split.

        Reads ``eval.split`` from config (default ``"test"``) or uses explicit ``split``.
        """
        cfg = self.cfg
        split = split or cfg.get("eval", {}).get("split", "test")
        if split not in ("val", "test"):
            raise ValueError(f"eval.split must be 'val' or 'test', got '{split}'")

        checkpoint_path = self.run_dir / "checkpoints" / "checkpoint_best.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        device = _device(cfg)
        use_amp = bool(cfg.get("train", {}).get("amp", True)) and device.type == "cuda"
        num_workers = _num_workers(cfg)
        pin_memory = _pin_memory(cfg) and device.type == "cuda"
        batch_size = int(cfg.get("train", {}).get("batch_size", 64))

        X, y = load_split(split, cfg["data"]["processed_root"])
        num_classes = _num_classes_from_table(cfg["data"]["processed_root"], cfg)

        rare_cls: list[int] | None = None
        try:
            from vn_tsc.data.dataset import rare_classes
            threshold = int(cfg.get("metrics", {}).get("rare_class_threshold", 30))
            rare_cls = rare_classes(cfg["data"]["processed_root"], threshold=threshold)
        except Exception:
            pass

        eval_tf = build_eval_transforms(cfg)
        loader = _build_loader(X, y, eval_tf, batch_size * 2,
                               shuffle=False, num_workers=num_workers,
                               pin_memory=pin_memory, use_ddp=False)

        model = build_model(cfg, num_classes=num_classes)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        _, metrics, preds, targets = _eval_epoch(
            model, loader, criterion, device, use_amp, num_classes, rare_cls
        )
        metrics["samples"] = int(len(X))

        log.info(
            "Evaluate [%s] → accuracy=%.4f macro_f1=%.4f (common=%.4f)",
            split, metrics["accuracy"], metrics["macro_f1"],
            metrics.get("macro_f1_common", metrics["macro_f1"]),
        )

        if cfg.get("outputs", {}).get("save_confusion_matrix", True):
            try:
                from vn_tsc.data.dataset import load_class_table
                from vn_tsc.eval.plots import confusion_matrix_png
                table = load_class_table(cfg["data"]["processed_root"])
                class_labels = table.labels if len(table) == num_classes else [str(i) for i in range(num_classes)]
                cm_out = self.run_dir / f"confusion_matrix_{split}.png"
                confusion_matrix_png(targets, preds, class_labels, cm_out)
                log.info("Saved confusion matrix: %s", cm_out)
            except Exception as e:
                log.warning("Could not save confusion matrix: %s", e)

        metrics_path = self.run_dir / "metrics.json"
        existing: dict[str, Any] = {}
        if metrics_path.exists():
            from vn_tsc.utils.io import load_json
            existing = load_json(metrics_path)
        existing[split] = metrics
        existing[f"{split}_samples"] = int(len(X))
        save_json(existing, metrics_path)

        return metrics

