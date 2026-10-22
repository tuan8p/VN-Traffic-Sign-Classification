from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

SPLIT_ORDER = ("train", "val", "test")


@dataclass
class SplitReport:
    ratios_requested: dict[str, float] = field(default_factory=dict)
    ratios_achieved: dict[str, float] = field(default_factory=dict)
    n_groups: int = 0
    groups_per_split: dict[str, int] = field(default_factory=dict)
    images_per_split: dict[str, int] = field(default_factory=dict)
    crops_per_split: dict[str, int] = field(default_factory=dict)
    per_class_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    classes_missing_per_split: dict[str, list[int]] = field(default_factory=dict)
    repaired_classes: list[int] = field(default_factory=list)
    straddling_groups: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ratios_requested": self.ratios_requested,
            "ratios_achieved": self.ratios_achieved,
            "n_groups": self.n_groups,
            "groups_per_split": self.groups_per_split,
            "images_per_split": self.images_per_split,
            "crops_per_split": self.crops_per_split,
            "per_class_counts": self.per_class_counts,
            "classes_missing_per_split": {
                k: v for k, v in self.classes_missing_per_split.items() if v
            },
            "repaired_classes": self.repaired_classes,
            "straddling_groups": self.straddling_groups,
        }


def stratified_group_split(
    group_ids: Sequence[int],
    class_ids: Sequence[int],
    num_classes: int,
    ratios: dict[str, float],
    seed: int = 42,
    min_train_per_class: int = 1,
) -> tuple[dict[int, str], SplitReport]:
    """Split whole groups across train/val/test while balancing class counts.

    One entry per crop; `group_ids[i]` is the leakage group of crop i (the
    perceptual-hash cluster of its source image). Every crop of a group lands in
    the same split, so neither duplicate frames nor two crops of one photo can
    span the train/test boundary.

    sklearn's StratifiedGroupKFold only yields equal folds, which cannot express
    70/15/15 exactly and handles 3-sample classes poorly. Instead we walk the
    groups rarest-class-first and give each one to whichever split is furthest
    below its per-class quota — the standard greedy fill, which keeps tiny
    classes from all landing in the same place.

    Returns group_id -> split name.
    """
    if len(group_ids) != len(class_ids):
        raise ValueError("group_ids and class_ids length mismatch")
    splits = [s for s in SPLIT_ORDER if ratios.get(s, 0) > 0]
    if not splits:
        raise ValueError(f"no positive ratios in {ratios}")

    group_ids = np.asarray(group_ids)
    class_ids = np.asarray(class_ids)
    uniq_groups = np.unique(group_ids)

    # counts[g_index, class] -> number of crops
    g_index = {int(g): i for i, g in enumerate(uniq_groups)}
    counts = np.zeros((len(uniq_groups), num_classes), dtype=np.int64)
    for g, c in zip(group_ids, class_ids):
        counts[g_index[int(g)], int(c)] += 1

    totals = counts.sum(axis=0).astype(np.float64)
    ratio_vec = np.array([ratios[s] for s in splits], dtype=np.float64)
    ratio_vec = ratio_vec / ratio_vec.sum()
    targets = np.outer(ratio_vec, totals)  # (n_splits, num_classes)
    current = np.zeros_like(targets)

    # Rarest class in a group decides its priority: place the scarce classes
    # while every split still has room, otherwise they all fall into whichever
    # split happens to be filled last.
    safe_totals = np.where(totals > 0, totals, np.inf)
    rarity = np.where(counts > 0, 1.0 / safe_totals, 0.0).max(axis=1)
    rng = np.random.default_rng(seed)
    jitter = rng.random(len(uniq_groups)) * 1e-9
    order = np.lexsort((jitter, -counts.sum(axis=1), -rarity))

    assignment: dict[int, str] = {}
    quota = targets.sum(axis=1)
    for gi in order:
        need = np.maximum(targets - current, 0.0)
        # Reward the split that still needs this group's classes the most,
        # normalised so a 1000-crop class cannot drown out a 3-crop one.
        score = (need / np.maximum(targets, 1.0)) @ counts[gi]
        size_deficit = (quota - current.sum(axis=1)) / np.maximum(quota, 1.0)
        best = int(np.argmax(score + 1e-6 * size_deficit))
        current[best] += counts[gi]
        assignment[int(uniq_groups[gi])] = splits[best]

    repaired = _repair_min_train(
        assignment, uniq_groups, g_index, counts, splits, current, min_train_per_class
    )
    return assignment, _build_report(
        assignment, group_ids, class_ids, counts, g_index, uniq_groups,
        splits, ratios, num_classes, repaired,
    )


def _repair_min_train(
    assignment: dict[int, str],
    uniq_groups: np.ndarray,
    g_index: dict[int, int],
    counts: np.ndarray,
    splits: list[str],
    current: np.ndarray,
    min_train_per_class: int,
) -> list[int]:
    """Pull a group into train for any class train did not receive.

    A class with only two or three crops cannot reach all three splits; we
    prefer train, so the model at least sees the class during fitting. Every
    move is reported rather than silently applied.
    """
    if "train" not in splits or min_train_per_class <= 0:
        return []
    train_i = splits.index("train")
    repaired: list[int] = []
    for cls in range(counts.shape[1]):
        if counts[:, cls].sum() == 0 or current[train_i, cls] >= min_train_per_class:
            continue
        # Smallest donor group containing this class, so we disturb the least.
        donors = [
            gi for gi in range(len(uniq_groups))
            if counts[gi, cls] > 0 and assignment[int(uniq_groups[gi])] != "train"
        ]
        if not donors:
            continue
        gi = min(donors, key=lambda g: (counts[g].sum(), int(uniq_groups[g])))
        src = splits.index(assignment[int(uniq_groups[gi])])
        current[src] -= counts[gi]
        current[train_i] += counts[gi]
        assignment[int(uniq_groups[gi])] = "train"
        repaired.append(cls)
    return repaired


def _build_report(
    assignment: dict[int, str],
    group_ids: np.ndarray,
    class_ids: np.ndarray,
    counts: np.ndarray,
    g_index: dict[int, int],
    uniq_groups: np.ndarray,
    splits: list[str],
    ratios: dict[str, float],
    num_classes: int,
    repaired: list[int],
) -> SplitReport:
    per_class: dict[str, dict[str, int]] = {s: {} for s in splits}
    crops_per_split = {s: 0 for s in splits}
    groups_per_split = {s: 0 for s in splits}
    for g in uniq_groups:
        s = assignment[int(g)]
        groups_per_split[s] += 1
        row = counts[g_index[int(g)]]
        crops_per_split[s] += int(row.sum())
        for cls in np.nonzero(row)[0]:
            per_class[s][str(int(cls))] = per_class[s].get(str(int(cls)), 0) + int(row[cls])

    total_crops = sum(crops_per_split.values())
    present = {cls for cls in range(num_classes) if counts[:, cls].sum() > 0}
    missing = {
        s: sorted(cls for cls in present if str(cls) not in per_class[s]) for s in splits
    }
    return SplitReport(
        ratios_requested={s: float(ratios[s]) for s in splits},
        ratios_achieved={
            s: round(crops_per_split[s] / total_crops, 4) if total_crops else 0.0
            for s in splits
        },
        n_groups=len(uniq_groups),
        groups_per_split=groups_per_split,
        crops_per_split=crops_per_split,
        per_class_counts=per_class,
        classes_missing_per_split=missing,
        repaired_classes=sorted(set(repaired)),
        straddling_groups=0,
    )


def assign_splits(
    group_of_stem: dict[str, int], group_split: dict[int, str]
) -> dict[str, str]:
    """Lift a group -> split map onto source-image stems."""
    return {stem: group_split[gid] for stem, gid in group_of_stem.items() if gid in group_split}
