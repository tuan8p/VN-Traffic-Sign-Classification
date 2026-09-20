from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

class BasePipeline(ABC):
    name: str

    def __init__(self, cfg: dict[str, Any], run_dir: Path):
        self.cfg = cfg
        self.run_dir = run_dir

    @abstractmethod
    def fit(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def evaluate(self) -> dict[str, Any]:
        ...
