from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


def dhash(path: str | Path, hash_size: int = 16) -> np.ndarray:
    """Gradient ("difference") hash as a flat bool array of hash_size**2 bits.

    Compares each pixel with its right neighbour, so the signature survives the
    brightness and contrast changes present in this dataset. An average hash
    would key on absolute intensity and merge unrelated dark frames.
    """
    with Image.open(path) as im:
        small = im.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    a = np.asarray(small, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).ravel()


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


@dataclass
class GroupingReport:
    n_items: int = 0
    n_groups: int = 0
    n_multi_groups: int = 0
    n_items_in_multi_groups: int = 0
    largest_group: int = 0
    max_hamming: int = 0
    hash_bits: int = 0
    group_size_hist: dict[int, int] = field(default_factory=dict)
    examples: list[list[str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_items": self.n_items,
            "n_groups": self.n_groups,
            "n_multi_groups": self.n_multi_groups,
            "n_items_in_multi_groups": self.n_items_in_multi_groups,
            "largest_group": self.largest_group,
            "max_hamming": self.max_hamming,
            "hash_bits": self.hash_bits,
            "group_size_hist": {str(k): v for k, v in sorted(self.group_size_hist.items())},
            "examples": self.examples,
        }


def pairwise_hamming(hashes: np.ndarray) -> np.ndarray:
    """Full Hamming distance matrix for a (n, bits) bool array.

    Maps bits to +/-1 so one matmul yields every distance at once: for 3.2k
    images that is a 3216x3216 float matrix (~40 MB), far cheaper than looping.
    """
    signed = (hashes.astype(np.float32) * 2.0) - 1.0
    bits = hashes.shape[1]
    return (bits - signed @ signed.T) / 2.0


def cluster_near_duplicates(
    hashes: np.ndarray,
    names: Sequence[str],
    max_hamming: int = 24,
) -> tuple[dict[str, int], GroupingReport]:
    """Union-find over every pair closer than max_hamming.

    Returns name -> group_id. Merging is transitive on purpose: a slow pan
    produces a chain of frames where only neighbours are close, yet the whole
    chain shows the same signs and must not be split across train and test.
    """
    if hashes.shape[0] != len(names):
        raise ValueError("hashes and names length mismatch")
    n = len(names)
    dist = pairwise_hamming(hashes)
    np.fill_diagonal(dist, np.inf)

    uf = _UnionFind(n)
    for i, j in np.argwhere(dist <= max_hamming):
        if i < j:
            uf.union(int(i), int(j))

    roots = [uf.find(i) for i in range(n)]
    # Number groups by first appearance so ids are stable across runs.
    remap: dict[int, int] = {}
    group_of: dict[str, int] = {}
    for name, root in zip(names, roots):
        if root not in remap:
            remap[root] = len(remap)
        group_of[name] = remap[root]

    members: dict[int, list[str]] = {}
    for name, gid in group_of.items():
        members.setdefault(gid, []).append(name)
    sizes = [len(v) for v in members.values()]
    multi = [sorted(v) for v in members.values() if len(v) > 1]

    hist: dict[int, int] = {}
    for s in sizes:
        hist[s] = hist.get(s, 0) + 1

    report = GroupingReport(
        n_items=n,
        n_groups=len(members),
        n_multi_groups=len(multi),
        n_items_in_multi_groups=sum(len(m) for m in multi),
        largest_group=max(sizes) if sizes else 0,
        max_hamming=max_hamming,
        hash_bits=int(hashes.shape[1]),
        group_size_hist=hist,
        examples=sorted(multi, key=len, reverse=True)[:5],
    )
    return group_of, report


def group_source_images(
    image_paths: Sequence[Path],
    hash_size: int = 16,
    max_hamming: int = 24,
    enabled: bool = True,
) -> tuple[dict[str, int], GroupingReport]:
    """Assign every source image a group id; near-duplicate frames share one."""
    names = [p.stem for p in image_paths]
    if not enabled:
        report = GroupingReport(
            n_items=len(names), n_groups=len(names), largest_group=1 if names else 0
        )
        return {name: i for i, name in enumerate(names)}, report
    hashes = np.stack([dhash(p, hash_size) for p in image_paths])
    return cluster_near_duplicates(hashes, names, max_hamming)


def leakage_against(
    group_of: dict[str, int], assignment: dict[str, str]
) -> dict[str, Any]:
    """Count groups that straddle two splits under `assignment` (stem -> split).

    Used to quantify the leakage in the authors' provided split and to assert
    our own split has none.
    """
    by_group: dict[int, set[str]] = {}
    for stem, split in assignment.items():
        gid = group_of.get(stem)
        if gid is not None:
            by_group.setdefault(gid, set()).add(split)
    straddling = {g: s for g, s in by_group.items() if len(s) > 1}
    affected = [
        stem
        for stem, split in assignment.items()
        if group_of.get(stem) in straddling
    ]
    per_split: dict[str, int] = {}
    for stem in affected:
        per_split[assignment[stem]] = per_split.get(assignment[stem], 0) + 1
    return {
        "n_straddling_groups": len(straddling),
        "n_affected_images": len(affected),
        "affected_per_split": per_split,
    }
