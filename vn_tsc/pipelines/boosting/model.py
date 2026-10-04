from __future__ import annotations
from typing import Any
from lightgbm import LGBMClassifier

def build_booster(cfg: dict[str, Any]):
    params = dict(cfg["model"])

    backend = params.pop("backend", "lightgbm")
    if backend != "lightgbm":
        raise ValueError(f"Unknown backend: {backend}")

    params.update(
        objective="multiclass",
        random_state=cfg.get("seed", 42),
        n_jobs=cfg.get("train", {}).get("n_jobs", -1),
    )

    return LGBMClassifier(**params)
