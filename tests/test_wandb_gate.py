import pytest
from vn_tsc.runtime.wandb_gate import require_wandb

def test_require_wandb_raises_without_key(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        require_wandb()
