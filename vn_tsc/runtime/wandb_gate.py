from __future__ import annotations
import os
from typing import Any

def require_wandb(entity: str = "P4AIDS_ML", project: str = "BTL", enabled: bool = True) -> Any:
    """Training MUST call this. Preprocess/EDA/demo should NOT."""
    if not enabled:
        return None
    mode = os.environ.get("WANDB_MODE", "").lower()
    if mode in ("offline", "disabled"):
        import wandb
        return wandb.init(entity=entity, project=project, resume="allow", mode=mode)

    key = os.environ.get("WANDB_API_KEY", "").strip()
    if not key:
        try:
            import wandb
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
    import wandb
    return wandb.init(entity=entity, project=project, resume="allow")

