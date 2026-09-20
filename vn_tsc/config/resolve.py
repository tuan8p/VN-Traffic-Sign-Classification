from __future__ import annotations
import copy
import os
from pathlib import Path
from typing import Any
from vn_tsc.utils.io import load_yaml

def _deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out

def _expand_env(obj: Any) -> Any:
    if isinstance(obj, str):
        if obj.startswith("${") and obj.endswith("}"):
            inner = obj[2:-1]
            if ":" in inner:
                var, default = inner.split(":", 1)
                return os.environ.get(var, default)
            return os.environ.get(inner, "")
        return obj
    if isinstance(obj, list):
        return [_expand_env(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    return obj

def resolve_config(
    pipeline_yaml: str | Path | None = None,
    shared_yaml: str | Path = "configs/shared.yaml",
    runtime_yaml: str | Path | None = None,
    overrides: dict | None = None,
) -> dict[str, Any]:
    cfg = load_yaml(shared_yaml)
    if pipeline_yaml:
        p = load_yaml(pipeline_yaml)
        p = {k: v for k, v in p.items() if k != "inherits"}
        cfg = _deep_merge(cfg, p)
    if runtime_yaml:
        cfg = _deep_merge(cfg, {"runtime_cfg": load_yaml(runtime_yaml)})
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return _expand_env(cfg)
