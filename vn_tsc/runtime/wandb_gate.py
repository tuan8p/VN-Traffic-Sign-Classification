from __future__ import annotations
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def build_base_run_name(
    pipeline: str,
    cfg: dict[str, Any] | None = None,
    user_name: str | None = None,
) -> str:
    """Build a standardized base run name (without timestamp) identifying pipeline & key hyperparams."""
    if user_name and str(user_name).strip():
        name = str(user_name).strip()
        name = re.sub(r"[^\w\.-]", "_", name)
        pipe_prefix = f"{pipeline}_"
        pipe_dash = f"{pipeline}-"
        if not name.startswith(pipe_prefix) and not name.startswith(pipe_dash):
            name = f"{pipeline}_{name}"
        return name

    cfg = cfg or {}
    pipeline = pipeline.lower()

    if pipeline == "svm":
        model_cfg = cfg.get("model", {})
        kernel = model_cfg.get("kernel", "rbf")
        c_val = model_cfg.get("C", 1.0)
        dim_cfg = cfg.get("dim_reduction", {})
        dim_method = dim_cfg.get("method", "none")
        if dim_method == "pca":
            n_comp = dim_cfg.get("pca", {}).get("n_components", 0.95)
            dim_tag = f"pca{n_comp}"
        else:
            dim_tag = "none"
        with_aug = cfg.get("train", {}).get("with_aug", True)
        aug_tag = "aug" if with_aug else "noaug"
        tuning_tag = "_cv" if cfg.get("tuning", {}).get("enabled", False) else ""
        return f"svm_{kernel}_C{c_val}_{dim_tag}_{aug_tag}{tuning_tag}"

    elif pipeline == "boosting":
        model_cfg = cfg.get("model", {})
        backend = model_cfg.get("backend", "lgbm")
        num_leaves = model_cfg.get("num_leaves", 63)
        lr = model_cfg.get("learning_rate", 0.05)
        n_est = model_cfg.get("n_estimators", 150)
        pca_enabled = cfg.get("features", {}).get("pca", {}).get("enabled", False)
        pca_tag = f"_pca{cfg.get('features', {}).get('pca', {}).get('n_components', 256)}" if pca_enabled else ""
        with_aug = cfg.get("train", {}).get("with_aug", True)
        aug_tag = "aug" if with_aug else "noaug"
        return f"boosting_{backend}_leaves{num_leaves}_lr{lr}_est{n_est}{pca_tag}_{aug_tag}"

    elif pipeline == "dl":
        model_cfg = cfg.get("model", {})
        backbone = str(model_cfg.get("backbone", "tf_efficientnetv2_b0")).lower()
        clean_bb = backbone
        if clean_bb.startswith("tf_"):
            clean_bb = clean_bb[3:]
        if "efficientnet" in clean_bb:
            clean_bb = clean_bb.replace("efficientnet", "effnet")
        clean_bb = clean_bb.replace("_", "")

        train_cfg = cfg.get("train", {})
        epochs = train_cfg.get("epochs", 30)
        warmup = train_cfg.get("warmup_epochs", 5)
        lr = train_cfg.get("lr", 3e-4)
        bs = train_cfg.get("batch_size", 64)
        with_aug = train_cfg.get("with_aug", True)
        aug_tag = "aug" if with_aug else "noaug"
        lr_str = f"{lr:g}"
        return f"dl_{clean_bb}_ep{epochs}_warm{warmup}_lr{lr_str}_bs{bs}_{aug_tag}"

    return f"{pipeline}_run"


def generate_run_name(
    pipeline: str,
    cfg: dict[str, Any] | None = None,
    user_name: str | None = None,
    run_dir: str | Path | None = None,
) -> str:
    """Generate a standardized, unique run name for W&B: {base_name}_{timestamp}."""
    base = build_base_run_name(pipeline, cfg=cfg, user_name=user_name)

    stamp = None
    if run_dir is not None:
        dir_name = Path(run_dir).name
        match = re.match(r"^(\d{8}_\d{6})", dir_name)
        if match:
            stamp = match.group(1)

    if stamp is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if re.search(r"\d{8}_\d{6}", base):
        return base
    return f"{base}_{stamp}"


def require_wandb(
    entity: str = "P4AIDS_ML",
    project: str = "BTL",
    enabled: bool = True,
    name: str | None = None,
    config: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    **kwargs: Any,
) -> Any:
    """Training MUST call this. Preprocess/EDA/demo should NOT."""
    if not enabled:
        return None

    import wandb
    if wandb.run is not None:
        if name and not wandb.run.name:
            try:
                wandb.run.name = name
            except Exception:
                pass
        return wandb.run

    mode = os.environ.get("WANDB_MODE", "").lower()
    if mode in ("offline", "disabled"):
        return wandb.init(
            entity=entity,
            project=project,
            name=name,
            config=config,
            tags=tags,
            resume="allow",
            mode=mode,
            **kwargs,
        )

    key = os.environ.get("WANDB_API_KEY", "").strip()
    if not key:
        try:
            s = wandb.setup().settings
            if s and s.api_key:
                key = str(s.api_key).strip()
                os.environ["WANDB_API_KEY"] = key
        except Exception:
            key = ""

    if not key:
        raise RuntimeError(
            "Training requires W&B. Set WANDB_API_KEY (e.g. export WANDB_API_KEY=... "
            "or run wandb.login(key=...)) and use entity="
            f"{entity} project={project}. To train offline or skip W&B, pass --no-wandb "
            "or set WANDB_MODE=offline or train.require_wandb: false."
        )

    return wandb.init(
        entity=entity,
        project=project,
        name=name,
        config=config,
        tags=tags,
        resume="allow",
        **kwargs,
    )

