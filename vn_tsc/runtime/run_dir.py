from __future__ import annotations
from datetime import datetime
from pathlib import Path
import uuid

def make_run_dir(outputs_root: str | Path, pipeline: str, run_name: str | None = None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = run_name or uuid.uuid4().hex[:8]
    path = Path(outputs_root) / pipeline / f"{stamp}_{suffix}"
    for sub in ("logs", "figures", "checkpoints"):
        (path / sub).mkdir(parents=True, exist_ok=True)
    return path
