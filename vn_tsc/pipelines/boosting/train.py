from __future__ import annotations
from pathlib import Path
from typing import Any
from vn_tsc.pipelines.base import BasePipeline
from vn_tsc.runtime.wandb_gate import require_wandb
from vn_tsc.utils.io import save_json, save_yaml
from vn_tsc.runtime.pack import zip_run_dir

class BoostingPipeline(BasePipeline):
    name = "boosting"

    def fit(self) -> dict[str, Any]:
        entity = self.cfg.get("project", {}).get("wandb_entity", "P4AIDS_ML")
        project = self.cfg.get("project", {}).get("wandb_project", "BTL")
        require_wandb(entity=entity, project=project, enabled=self.cfg.get("train", {}).get("require_wandb", True))
        save_yaml(self.cfg, self.run_dir / "resolved_config.yaml")
        # TODO(team-boosting): shared features → LightGBM/XGBoost
        metrics = {"accuracy": None, "macro_f1": None, "status": "TODO"}
        save_json(metrics, self.run_dir / "metrics.json")
        if self.cfg.get("outputs", {}).get("zip_after_train", True):
            zip_run_dir(self.run_dir)
        return metrics

    def evaluate(self) -> dict[str, Any]:
        raise NotImplementedError("TODO(team-boosting): evaluate")
