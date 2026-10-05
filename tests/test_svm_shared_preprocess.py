"""SVM integration against the merged, official YOLO-to-feature preprocessing."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools.run_eval import main as run_eval_main
from vn_tsc.config.resolve import resolve_config
from vn_tsc.data.preprocess_offline import run_offline_preprocess
from vn_tsc.features.classical import build_feature_store
from vn_tsc.pipelines.svm.data_adapter import load_svm_data
from vn_tsc.pipelines.svm.train import SVMPipeline


@pytest.fixture
def shared_store(tmp_path: Path) -> tuple[dict, Path]:
    repo = Path(__file__).resolve().parents[1]
    cfg = resolve_config(repo / "configs/pipelines/svm.yaml", repo / "configs/shared.yaml")
    raw = tmp_path / "raw"
    (raw / "images").mkdir(parents=True)
    (raw / "labels").mkdir()
    for class_id in (0, 1):
        for index in range(12):
            name = f"sign_{class_id}_{index:02d}"
            color = (180 + index, 30, 30) if class_id == 0 else (30, 30, 180 + index)
            Image.new("RGB", (48, 48), color).save(raw / "images" / f"{name}.png")
            (raw / "labels" / f"{name}.txt").write_text(
                f"{class_id} 0.5 0.5 0.5 0.5\n", encoding="utf-8"
            )
    class_table = tmp_path / "classes.csv"
    class_table.write_text(
        "class_id,sign_code,name_vi,mirror_of,confusable_group\n"
        "0,A,Class A,,\n1,B,Class B,,\n", encoding="utf-8",
    )
    cfg["data"]["class_table"] = str(class_table)
    cfg["data"]["num_classes"] = 2
    cfg["split"]["dedup"]["enabled"] = False
    cfg["preprocess"]["image_size"] = [32, 32]
    cfg["features"]["input_size"] = [32, 32]
    cfg["augment"]["target_count"] = 12
    cfg["augment"]["max_multiplier"] = 2
    cfg["metrics"]["rare_class_threshold"] = 4
    cfg["train"]["require_wandb"] = False
    cfg["outputs"]["zip_after_train"] = False
    cfg["outputs"]["save_confusion_matrix"] = False
    processed = tmp_path / "processed"
    cfg["data_source"]["processed_root"] = str(processed)
    run_offline_preprocess(cfg, raw, processed)
    build_feature_store(cfg, processed)
    return cfg, processed


def test_shared_adapter_uses_raw_features_and_train_augmentation(
    shared_store: tuple[dict, Path],
) -> None:
    cfg, processed = shared_store
    data = load_svm_data(cfg)
    base = np.load(processed / "features" / "F_train.npy")
    augmented = np.load(processed / "features" / "F_train_aug.npy")
    np.testing.assert_array_equal(data.X_train, np.concatenate([base, augmented]))
    assert data.X_val.shape[0] == len(np.load(processed / "images" / "y_val.npy"))
    assert data.X_test.shape[0] == len(np.load(processed / "images" / "y_test.npy"))
    assert data.X_train.shape[1] == data.feature_report["layout"]["total"]
    assert data.class_names == ["00_A", "01_B"]
    assert data.common_class_ids == [0, 1]
    assert data.feature_config["backend"] == "shared"
    for crop_array in (processed / "images").glob("X_*.npy"):
        crop_array.unlink()
    np.testing.assert_array_equal(load_svm_data(cfg).X_train, data.X_train)


def test_shared_training_and_saved_provenance(shared_store: tuple[dict, Path], tmp_path: Path) -> None:
    cfg, processed = shared_store
    run_dir = tmp_path / "svm_run"
    pipeline = SVMPipeline(cfg, run_dir)
    result = pipeline.fit()
    assert "macro_f1_common" in result["validation"]
    assert result["train_samples"] == len(np.load(processed / "images" / "y_train.npy")) + len(
        np.load(processed / "images" / "y_train_aug.npy")
    )
    assert (run_dir / "checkpoints" / "model.joblib").is_file()
    assert (run_dir / "metadata" / "class_table.csv").is_file()
    assert (run_dir / "metadata" / "feature_report.json").is_file()
    provenance = json.loads((run_dir / "metadata" / "model_metadata.json").read_text())
    assert provenance["feature_backend"] == "shared"
    assert provenance["train_with_aug"] is True
    assert "macro_f1_common" in pipeline.evaluate()
    run_eval_main(["--pipeline", "svm", "--run-dir", str(run_dir),
                   "--split", "val", "--processed-root", str(processed)])
    assert "macro_f1_common" in json.loads((run_dir / "metrics.json").read_text())["val"]


def test_shared_ablation_and_config_drift(shared_store: tuple[dict, Path]) -> None:
    cfg, processed = shared_store
    cfg["train"]["with_aug"] = False
    data = load_svm_data(cfg)
    assert data.X_train.shape[0] == len(np.load(processed / "features" / "F_train.npy"))
    cfg["features"]["hog"]["enabled"] = False
    cfg["features"]["lbp"]["enabled"] = False
    color_only = load_svm_data(cfg)
    block = color_only.feature_report["layout"]["blocks"]["color_hist"]
    np.testing.assert_array_equal(
        color_only.X_train,
        np.load(processed / "features" / "F_train.npy")[:, block["start"]:block["end"]],
    )
    cfg["features"]["color_hist"]["bins"] = 8
    with pytest.raises(ValueError, match="different color_hist settings"):
        load_svm_data(cfg)
    cfg["features"]["color_hist"]["enabled"] = False
    with pytest.raises(ValueError, match="At least one"):
        load_svm_data(cfg)


def test_shared_metadata_alignment_is_checked(shared_store: tuple[dict, Path]) -> None:
    cfg, processed = shared_store
    path = processed / "metadata.csv"
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        rows = list(reader)
    rows[0]["class_id"] = "1" if rows[0]["class_id"] == "0" else "0"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="metadata label mismatch"):
        load_svm_data(cfg)
