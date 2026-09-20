from pathlib import Path
from vn_tsc.config.resolve import resolve_config

def test_shared_loads():
    root = Path(__file__).resolve().parents[1]
    cfg = resolve_config(shared_yaml=root / "configs/shared.yaml")
    assert cfg["seed"] == 42
    assert cfg["project"]["wandb_entity"] == "P4AIDS_ML"
