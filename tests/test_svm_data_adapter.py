from __future__ import annotations

import csv
from pathlib import Path

import pytest
from PIL import Image

from vn_tsc.config.resolve import resolve_config
from vn_tsc.pipelines.svm.data_adapter import load_svm_data


@pytest.fixture
def cfg(tmp_path: Path) -> dict:
    repo = Path(__file__).resolve().parents[1]
    config = resolve_config(repo / "configs/pipelines/svm.yaml", repo / "configs/shared.yaml")
    config["data_source"]["processed_root"] = str(tmp_path / "processed")
    config["features"]["hog"]["enabled"] = False
    config["features"]["lbp"]["enabled"] = False
    config["features"]["color_hist"]["bins"] = 8
    return config


@pytest.fixture
def processed(cfg: dict) -> Path:
    root = Path(cfg["data_source"]["processed_root"])
    (root / "images").mkdir(parents=True)
    rows = []
    for label, class_id in (("apple", 0), ("zebra", 1)):
        for split, count in (("train", 3), ("val", 1), ("test", 1)):
            for index in range(count):
                sample_id = f"{label}_{split}_{index}"
                relative = Path("images") / f"{sample_id}.png"
                Image.new("RGB", (32, 32), (class_id * 100 + index, 50, 150)).save(root / relative)
                rows.append({"sample_id": sample_id, "processed_path": str(relative),
                             "class_name": label, "class_id": class_id, "split": split})
    with (root / "metadata.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return root


def test_adapter_reads_fixed_splits_and_cache(cfg: dict, processed: Path) -> None:
    data = load_svm_data(cfg)
    assert (data.X_train.shape[0], data.X_val.shape[0], data.X_test.shape[0]) == (6, 2, 2)
    assert data.class_names == ["apple", "zebra"]
    assert data.label_map == {"apple": 0, "zebra": 1}
    assert data.X_train.shape[1] == 24
    assert (processed / "features" / "feature_config.json").exists()
    assert load_svm_data(cfg).X_train.shape == data.X_train.shape
    cfg["features"]["color_hist"]["bins"] = 4
    assert load_svm_data(cfg).X_train.shape[1] == 12


def test_missing_metadata_and_columns(cfg: dict, processed: Path) -> None:
    metadata = processed / "metadata.csv"
    metadata.unlink()
    with pytest.raises(FileNotFoundError, match="Processed metadata not found"):
        load_svm_data(cfg)
    metadata.write_text("sample_id,split\n1,train\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Metadata missing columns"):
        load_svm_data(cfg)


def test_missing_image_and_empty_split(cfg: dict, processed: Path) -> None:
    image = next((processed / "images").iterdir())
    image.unlink()
    with pytest.raises(FileNotFoundError, match="Processed image missing"):
        load_svm_data(cfg)
    image = processed / "images" / image.name
    Image.new("RGB", (32, 32)).save(image)
    metadata = processed / "metadata.csv"
    lines = metadata.read_text(encoding="utf-8").replace(",test\n", ",val\n")
    metadata.write_text(lines, encoding="utf-8")
    with pytest.raises(ValueError, match="empty test split"):
        load_svm_data(cfg)


def test_shared_feature_backend_is_explicitly_unavailable(cfg: dict) -> None:
    cfg["features"]["backend"] = "shared"
    with pytest.raises(NotImplementedError, match="Role D"):
        load_svm_data(cfg)
