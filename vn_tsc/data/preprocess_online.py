from __future__ import annotations
from typing import Any

def build_train_transforms(cfg: dict[str, Any]):
    """TODO(team-dl): online aug for DL only."""
    raise NotImplementedError("TODO(team-dl): online transforms")

def build_eval_transforms(cfg: dict[str, Any]):
    """TODO(team-dl): eval transforms."""
    raise NotImplementedError("TODO(team-dl): eval transforms")
