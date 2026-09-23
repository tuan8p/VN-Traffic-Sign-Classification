from __future__ import annotations
import csv
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CLASS_TABLE = "configs/classes.csv"


def ascii_fold(text: str) -> str:
    """Strip Vietnamese diacritics so a string is safe for cp1252 consoles.

    Four sign codes carry a Vietnamese qualifier (P.106a*Xe tải, and the three
    S.505a lane plates), and printing them raw crashes on a Windows terminal.
    NFD splits off the combining accents; đ/Đ has no decomposition, so it is
    mapped by hand.
    """
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if not unicodedata.combining(c)
    )


@dataclass(frozen=True)
class SignClass:
    class_id: int
    sign_code: str
    name_vi: str
    mirror_of: int | None
    confusable_group: str | None

    @property
    def label(self) -> str:
        """ASCII-safe label for logs and axis ticks (Windows consoles are cp1252).

        The id prefix keeps labels unique: three classes share the S.505a code
        and differ only in their Vietnamese qualifier.
        """
        return f"{self.class_id:02d}_{ascii_fold(self.sign_code)}"


class ClassTable:
    """id -> sign code / Vietnamese name / mirror pair, loaded from configs/classes.csv."""

    def __init__(self, classes: list[SignClass]):
        self.classes = sorted(classes, key=lambda c: c.class_id)
        self._by_id = {c.class_id: c for c in self.classes}
        expected = list(range(len(self.classes)))
        if [c.class_id for c in self.classes] != expected:
            raise ValueError(f"class ids must be contiguous 0..{len(self.classes) - 1}")

    def __len__(self) -> int:
        return len(self.classes)

    def __getitem__(self, class_id: int) -> SignClass:
        return self._by_id[class_id]

    @property
    def ids(self) -> list[int]:
        return [c.class_id for c in self.classes]

    @property
    def labels(self) -> list[str]:
        return [c.label for c in self.classes]

    @property
    def names_vi(self) -> list[str]:
        return [c.name_vi for c in self.classes]

    def mirror_pairs(self) -> list[tuple[int, int]]:
        """Unordered (a, b) pairs where a horizontal flip maps class a onto class b."""
        seen: set[tuple[int, int]] = set()
        for c in self.classes:
            if c.mirror_of is not None:
                seen.add((min(c.class_id, c.mirror_of), max(c.class_id, c.mirror_of)))
        return sorted(seen)

    def confusable_groups(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for c in self.classes:
            if c.confusable_group:
                out.setdefault(c.confusable_group, []).append(c.class_id)
        return out

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CLASS_TABLE) -> "ClassTable":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"class table not found: {path}")
        rows: list[SignClass] = []
        with open(path, "r", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                mirror = r.get("mirror_of", "").strip()
                group = (r.get("confusable_group") or "").strip()
                rows.append(
                    SignClass(
                        class_id=int(r["class_id"]),
                        sign_code=r["sign_code"].strip(),
                        name_vi=r["name_vi"].strip(),
                        mirror_of=int(mirror) if mirror else None,
                        confusable_group=group or None,
                    )
                )
        table = cls(rows)
        for a, b in table.mirror_pairs():
            if table[b].mirror_of != a:
                raise ValueError(f"mirror_of is not symmetric for classes {a} and {b}")
        return table

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "ClassTable":
        return cls.load(cfg.get("data", {}).get("class_table", DEFAULT_CLASS_TABLE))

    def to_records(self) -> list[dict[str, Any]]:
        return [
            {
                "class_id": c.class_id,
                "sign_code": c.sign_code,
                "name_vi": c.name_vi,
                "label": c.label,
                "mirror_of": c.mirror_of,
                "confusable_group": c.confusable_group,
            }
            for c in self.classes
        ]
