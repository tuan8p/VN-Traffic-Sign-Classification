from __future__ import annotations
import os
from typing import Any

def require_wandb(entity: str = "P4AIDS_ML", project: str = "BTL", enabled: bool = True) -> Any:
    """Training MUST call this. Preprocess/EDA/demo should NOT."""
    if not enabled:
        return None
    key = os.environ.get("WANDB_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "Training requires W&B. Set WANDB_API_KEY "
            f"and use entity={entity} project={project}."
        )
    import wandb
    return wandb.init(entity=entity, project=project, resume="allow")
