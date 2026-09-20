from __future__ import annotations
from pathlib import Path
import zipfile

def zip_run_dir(run_dir: str | Path, zip_name: str = "run.zip") -> Path:
    run_dir = Path(run_dir)
    out = run_dir / zip_name
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in run_dir.rglob("*"):
            if p.is_file() and p.name != zip_name:
                zf.write(p, arcname=str(p.relative_to(run_dir)))
    return out
