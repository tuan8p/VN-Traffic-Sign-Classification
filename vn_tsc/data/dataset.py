from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Sample:
    path: Path
    label: str
    split: str

class TrafficSignDataset:
    """TODO(team-preprocessing): load index from processed metadata."""

    def __init__(self, index: list[Sample]):
        self.index = index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Sample:
        return self.index[i]

    @classmethod
    def from_processed(cls, processed_root: str | Path) -> "TrafficSignDataset":
        raise NotImplementedError("TODO(team-preprocessing): read processed index")
