"""Contract tests for the disposable SVM development preprocessing."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from vn_tsc.config.resolve import resolve_config
from vn_tsc.pipelines.svm.dev_preprocess import (
    METADATA_COLUMNS,
    build_label_map,
    resize_with_padding,
    run_dev_preprocess,
)


@pytest.fixture
def cfg() -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    return resolve_config(
        pipeline_yaml=repo_root / "configs/pipelines/svm.yaml",
        shared_yaml=repo_root / "configs/shared.yaml",
    )


@pytest.fixture
def raw_root(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    for class_name in ("zebra", "apple", "banana"):
        nested = root / class_name / "nested"
        nested.mkdir(parents=True)
        for index in range(10):
            image = (Image.new("L", (100, 50), 128) if index == 0 else
                     Image.new("RGB", (100, 50), (index + 20, 100, 200)))
            image.save(nested / f"{index:02d}.jpg")
        (nested / "notes.txt").write_text("not an image", encoding="utf-8")
    return root


def _rows(output_root: Path) -> list[dict[str, str]]:
    with (output_root / "metadata.csv").open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def test_image_size_rgb_and_metadata_schema(cfg: dict, raw_root: Path, tmp_path: Path) -> None:
    output = tmp_path / "processed"
    summary = run_dev_preprocess(cfg, raw_root, output)
    assert summary["num_samples"] == 30
    assert summary["skipped_images"] == 0
    rows = _rows(output)
    assert len(rows) == 30
    assert set(METADATA_COLUMNS).issubset(rows[0])
    assert [row["sample_id"] for row in rows] == [f"{i:08d}" for i in range(30)]
    for row in rows:
        with Image.open(output / row["processed_path"]) as image:
            assert image.size == tuple(cfg["preprocess"]["image_size"])
            assert image.mode == "RGB"
        assert row["original_width"] == "100"
        assert row["original_height"] == "50"
    assert (output / "preprocessing_config.yaml").is_file()
    assert (output / "split_summary.json").is_file()


def test_keep_aspect_and_centered_padding() -> None:
    image = Image.new("RGB", (100, 50), (255, 0, 0))
    result = resize_with_padding(image, (224, 224), keep_aspect=True)
    assert result.size == (224, 224)
    assert result.getpixel((112, 0)) == (0, 0, 0)
    assert result.getpixel((112, 55)) == (0, 0, 0)
    assert result.getpixel((112, 56)) == (255, 0, 0)
    assert result.getpixel((112, 167)) == (255, 0, 0)
    assert result.getpixel((112, 168)) == (0, 0, 0)


def test_label_map_is_alphabetical() -> None:
    assert build_label_map(["zebra", "apple", "banana", "apple"]) == {
        "apple": 0, "banana": 1, "zebra": 2,
    }


def test_split_disjoint_stratified_and_deterministic(cfg: dict, raw_root: Path, tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    run_dev_preprocess(cfg, raw_root, first)
    run_dev_preprocess(cfg, raw_root, second)
    rows = _rows(first)
    ids = {split: {row["sample_id"] for row in rows if row["split"] == split}
           for split in ("train", "val", "test")}
    assert ids["train"].isdisjoint(ids["val"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["val"].isdisjoint(ids["test"])
    assert set.union(*ids.values()) == {row["sample_id"] for row in rows}
    assert all({row["split"] for row in rows if row["class_name"] == name}
               == {"train", "val", "test"} for name in ("apple", "banana", "zebra"))
    for name in ("apple", "banana", "zebra"):
        assert {split: sum(row["class_name"] == name and row["split"] == split for row in rows)
                for split in ids} == {"train": 7, "val": 2, "test": 1}
    assert [(row["sample_id"], row["split"], row["class_id"]) for row in rows] == [
        (row["sample_id"], row["split"], row["class_id"]) for row in _rows(second)
    ]
    assert json.loads((first / "label_map.json").read_text(encoding="utf-8")) == {
        "apple": 0, "banana": 1, "zebra": 2,
    }
    summary = json.loads((first / "split_summary.json").read_text(encoding="utf-8"))
    assert sum(summary["splits"][split]["count"] for split in ids) == 30


def test_existing_output_requires_overwrite(cfg: dict, raw_root: Path, tmp_path: Path) -> None:
    output = tmp_path / "processed"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError, match="--overwrite"):
        run_dev_preprocess(cfg, raw_root, output)
    assert marker.read_text(encoding="utf-8") == "old"
    run_dev_preprocess(cfg, raw_root, output, overwrite=True)
    assert not marker.exists()
    assert (output / "metadata.csv").is_file()


def test_corrupt_image_is_skipped_and_reported(cfg: dict, raw_root: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    (raw_root / "apple" / "broken.png").write_bytes(b"not a PNG")
    output = tmp_path / "processed"
    summary = run_dev_preprocess(cfg, raw_root, output)
    assert summary["skipped_images"] == 1
    assert summary["num_samples"] == 30
    assert "broken.png" in caplog.text
    assert json.loads((output / "split_summary.json").read_text(encoding="utf-8"))["skipped_images"] == 1


def test_too_few_images_fail_before_writing(cfg: dict, tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "tiny").mkdir(parents=True)
    for index in range(2):
        Image.new("RGB", (10, 10)).save(raw / "tiny" / f"{index}.png")
    output = tmp_path / "processed"
    with pytest.raises(ValueError, match=r"tiny \(2 images\)"):
        run_dev_preprocess(cfg, raw, output)
    assert not output.exists()


def test_invalid_ratios_and_shared_output_guard(cfg: dict, raw_root: Path, tmp_path: Path) -> None:
    cfg["split"]["ratios"]["train"] = 0.6
    with pytest.raises(ValueError, match="sum to 1.0"):
        run_dev_preprocess(cfg, raw_root, tmp_path / "processed")
    repo_root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="data/processed"):
        run_dev_preprocess(cfg, raw_root, repo_root / "data/processed", overwrite=True)
    with pytest.raises(ValueError, match="data/processed"):
        run_dev_preprocess(cfg, raw_root, repo_root / "data", overwrite=True)
