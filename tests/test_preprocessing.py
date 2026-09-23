"""Unit tests for the preprocessing pipeline. No dataset required."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vn_tsc.data.augment import DEFAULT_OPS, augment_once, plan_augmentation
from vn_tsc.data.classes import ClassTable
from vn_tsc.data.crops import CropGeometry, box_geometry, crop_box, letterbox
from vn_tsc.data.discover import CLIP_TOL, DiscoveryReport, _parse_label_file
from vn_tsc.data.grouping import cluster_near_duplicates, leakage_against, pairwise_hamming
from vn_tsc.data.splitting import stratified_group_split

ROOT = Path(__file__).resolve().parents[1]


# --- class table -------------------------------------------------------------

def test_class_table_is_complete_and_mirror_symmetric():
    table = ClassTable.load(ROOT / "configs/classes.csv")
    assert len(table) == 52
    assert table.ids == list(range(52))
    for a, b in table.mirror_pairs():
        assert table[a].mirror_of == b and table[b].mirror_of == a
    # Labels must be ASCII: they go to Windows consoles and matplotlib ticks.
    for label in table.labels:
        label.encode("ascii")


# --- label parsing -----------------------------------------------------------

def _parse(tmp_path: Path, text: str) -> tuple[list, DiscoveryReport]:
    p = tmp_path / "x.txt"
    p.write_text(text, encoding="utf-8")
    report = DiscoveryReport()
    return _parse_label_file(p, "x", 52, report), report


def test_parse_rejects_malformed_and_out_of_range(tmp_path: Path):
    boxes, report = _parse(
        tmp_path,
        "0 0.5 0.5 0.1 0.1\n"       # good
        "1 0.5 0.5\n"                # too few fields
        "2 a b c d\n"                # non-numeric
        "99 0.5 0.5 0.1 0.1\n"       # class id out of range
        "3 0.5 0.5 0 0.1\n",         # zero width
    )
    assert len(boxes) == 1 and boxes[0].class_id == 0
    assert len(report.malformed_lines) == 3
    assert len(report.dropped_boxes) == 1


def test_parse_clips_overflowing_box_but_ignores_rounding_noise(tmp_path: Path):
    # 6-decimal YOLO rounding puts edges ~5e-7 outside; that is not a real defect.
    boxes, report = _parse(
        tmp_path,
        "0 0.9999995 0.5 0.000001 0.1\n"   # rounding noise only
        "1 0.99 0.5 0.10 0.1\n",           # genuinely past the right edge
    )
    assert len(boxes) == 2
    assert len(report.clipped_boxes) == 1, "only the real overflow should be reported"
    for b in boxes:
        assert 0 <= b.cx - b.w / 2 and b.cx + b.w / 2 <= 1 + CLIP_TOL
        assert 0 <= b.cy - b.h / 2 and b.cy + b.h / 2 <= 1 + CLIP_TOL


def test_parse_records_empty_label(tmp_path: Path):
    boxes, report = _parse(tmp_path, "\n\n")
    assert boxes == [] and report.empty_labels == ["x"]


# --- cropping ----------------------------------------------------------------

def test_letterbox_preserves_aspect_ratio_and_pads():
    wide = np.full((10, 100, 3), 200, np.uint8)
    out = letterbox(wide, (224, 224), pad_value=114)
    assert out.shape == (224, 224, 3)
    assert (out[0, 0] == 114).all(), "corner should be padding"
    assert (out[112, 112] == 200).all(), "centre should be content"
    # content band is 224 * 10/100 ~= 22 px tall, not stretched to 224
    content_rows = np.where((out == 200).all(axis=2).any(axis=1))[0]
    assert 15 <= len(content_rows) <= 30


def test_crop_box_keeps_sign_centred_when_margin_leaves_the_frame():
    img = np.zeros((100, 100, 3), np.uint8)
    img[0:20, 0:20] = 255  # sign flush against the top-left corner
    geom = CropGeometry(0, 0, 20, 20, 20, 20, 100, 100)
    out = crop_box(img, geom, (64, 64), context_margin=0.5, pad_mode="replicate")
    assert out.shape == (64, 64, 3)
    # With the window centred on the sign, the middle must be sign, not background.
    assert out[32, 32].mean() > 200


def test_box_geometry_matches_pixel_arithmetic():
    g = box_geometry(0.5, 0.5, 0.1, 0.2, 960, 540)
    assert (g.box_w_px, g.box_h_px) == (96, 108)
    assert (g.x1, g.y1, g.x2, g.y2) == (432, 216, 528, 324)
    assert g.area_px == 96 * 108


# --- near-duplicate grouping -------------------------------------------------

def test_pairwise_hamming_matches_bit_counting():
    rng = np.random.default_rng(0)
    h = rng.random((20, 64)) > 0.5
    d = pairwise_hamming(h)
    for i in (0, 5, 19):
        for j in (1, 7, 18):
            assert d[i, j] == pytest.approx(np.count_nonzero(h[i] != h[j]))


def test_clustering_is_transitive_along_a_chain():
    # A-B and B-C are close, A-C is not: a pan across a scene. All three must
    # still land in one group or the split can tear the sequence apart.
    bits = 64
    a = np.zeros(bits, bool)
    b = a.copy(); b[:5] = True
    c = a.copy(); c[:10] = True
    far = np.ones(bits, bool)
    groups, report = cluster_near_duplicates(
        np.stack([a, b, c, far]), ["a", "b", "c", "far"], max_hamming=5
    )
    assert groups["a"] == groups["b"] == groups["c"]
    assert groups["far"] != groups["a"]
    assert report.n_groups == 2 and report.largest_group == 3


def test_leakage_detection_flags_a_straddling_group():
    groups = {"a": 0, "b": 0, "c": 1}
    leaky = leakage_against(groups, {"a": "train", "b": "test", "c": "train"})
    assert leaky["n_straddling_groups"] == 1 and leaky["n_affected_images"] == 2
    clean = leakage_against(groups, {"a": "train", "b": "train", "c": "test"})
    assert clean["n_straddling_groups"] == 0


# --- splitting ---------------------------------------------------------------

def test_split_never_puts_one_group_in_two_splits():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(300), 3)          # 3 crops per group
    classes = rng.integers(0, 10, size=groups.size)
    assignment, _ = stratified_group_split(
        groups, classes, 10, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=42
    )
    assert set(assignment) == set(range(300))
    # assignment is per group, so a group cannot span splits by construction
    assert set(assignment.values()) == {"train", "val", "test"}


def test_split_hits_requested_ratios_and_is_deterministic():
    rng = np.random.default_rng(1)
    groups = rng.integers(0, 800, size=4000)
    classes = rng.integers(0, 20, size=4000)
    ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
    a1, r1 = stratified_group_split(groups, classes, 20, ratios, seed=42)
    a2, _ = stratified_group_split(groups, classes, 20, ratios, seed=42)
    assert a1 == a2, "same seed must give the same split"
    for split, want in ratios.items():
        assert r1.ratios_achieved[split] == pytest.approx(want, abs=0.03)


def test_split_keeps_an_ultra_rare_class_in_train():
    # One class with a single crop must still reach train, as min_train_per_class asks.
    groups = list(range(50))
    classes = [0] * 49 + [7]
    _, report = stratified_group_split(
        groups, classes, 8, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=0
    )
    assert report.per_class_counts["train"].get("7", 0) >= 1


# --- augmentation ------------------------------------------------------------

def test_plan_lifts_rare_classes_without_cloning_them_to_death():
    classes = [0] * 500 + [1] * 50 + [2] * 3
    plan = plan_augmentation(classes, target_count=300, max_multiplier=8)
    assert 0 not in plan.per_class_extra, "a class above target needs no extras"
    assert plan.per_class_extra[1] == 250, "50 -> 300 is within 8x"
    assert plan.per_class_extra[2] == 3 * 7, "3 samples cap out at 8x, not 300"


def test_augment_is_deterministic_and_leaves_no_black_border():
    flat = np.full((64, 64, 3), 200, np.uint8)
    ops = {**DEFAULT_OPS, "brightness": 0, "contrast": 0, "saturation": 0,
           "gaussian_noise_std": 4.01, "jpeg_quality": [95, 95]}
    a, ops_a = augment_once(flat, np.random.default_rng(3), ops)
    b, ops_b = augment_once(flat, np.random.default_rng(3), ops)
    assert np.array_equal(a, b) and ops_a == ops_b
    # Rotation replicates the border instead of filling with black, which would
    # otherwise be a visual marker for "this sample is synthetic".
    assert a.min() > 100


def test_flip_is_not_among_the_available_operations():
    from vn_tsc.data import augment

    assert "hflip" not in augment.PHOTOMETRIC
    assert "vflip" not in augment.PHOTOMETRIC
