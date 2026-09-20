from __future__ import annotations
from pathlib import Path
from typing import Any
from vn_tsc.utils.logging_utils import setup_logger

log = setup_logger("preprocess_offline")

def run_offline_preprocess(cfg: dict[str, Any], data_root: str | Path, out_root: str | Path) -> Path:
    """Shared offline preprocess for ALL pipelines. No W&B."""
    data_root = Path(data_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    log.info("TODO(team-preprocessing): implement offline preprocess")
    log.info("data_root=%s out_root=%s size=%s", data_root, out_root, cfg.get("preprocess", {}).get("image_size"))
    raise NotImplementedError("TODO(team-preprocessing): offline preprocess")
