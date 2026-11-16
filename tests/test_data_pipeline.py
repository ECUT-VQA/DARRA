"""Data and metric integration tests without pretrained weights or transformers."""
import importlib.util
import json
from pathlib import Path

import pytest

from src.data_config import DEFAULT_LABEL_CONFIG, generated_metadata_paths, resolve_metadata_paths
from src.label_schema import pair_quality_targets
from src.evaluate.metrics import evaluate_file

ROOT = Path(__file__).resolve().parents[1]


def _generator():
    source = ROOT / "build_soft_labels" / "gen_soft_label.py"
    spec = importlib.util.spec_from_file_location("soft_labels_integration", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_and_evaluation_consume_generator_filenames(tmp_path):
    config = json.loads(DEFAULT_LABEL_CONFIG.read_text(encoding="utf-8"))
    expected_train = [v["save_train"] for v in config["dataset_params"].values()]
    expected_test = [v["save_test"] for v in config["dataset_params"].values()]
    assert generated_metadata_paths("train") == expected_train
    assert generated_metadata_paths("test") == expected_test
    assert all(p.endswith("_new.json") for p in expected_train + expected_test)
    assert all(str(tmp_path) in x for x in resolve_metadata_paths(expected_train, tmp_path))


def test_scoring_prefers_normalized_targets_over_raw_mos():
    sample = {"gt_score": 75, "std": 11, "gt_score_norm": 4, "std_norm": 0.44}
    assert pair_quality_targets(sample) == pytest.approx((4, 0.44))
    assert pair_quality_targets({"gt_score": 2.5, "std": 0.35}) == pytest.approx((2.5, 0.35))


@pytest.mark.parametrize("sample", [
    {"gt_score": 75, "std": 11},
    {"gt_score": 75, "std": 11, "gt_score_norm": 4},
    {"gt_score_norm": 4, "std_norm": -0.1},
    {"gt_score_norm": float("nan"), "std_norm": 0.5},
])
def test_invalid_or_incomplete_labels_fail_before_training(sample):
    with pytest.raises(ValueError):
        pair_quality_targets(sample)


def test_label_generation_and_direct_metrics_integration(tmp_path):
    metas = tmp_path / "KONIQ" / "metas"
    metas.mkdir(parents=True)
    (metas / "split.json").write_text(json.dumps({"train": ["a.png", "b.png", "c.png"],
                                                  "test": ["d.png"]}), encoding="utf-8")
    (metas / "mos.json").write_text(json.dumps({
        "a.png": {"mos": 20, "std": 0}, "b.png": {"mos": 40, "std": 7},
        "c.png": {"mos": 60, "std": 4}, "d.png": {"mos": 80, "std": 8}
    }), encoding="utf-8")
    cfg = {
        "split_json": "KONIQ/metas/split.json", "mos_json": "KONIQ/metas/mos.json",
        "save_train": "KONIQ/metas/train_koniq_7k_new.json",
        "save_test": "KONIQ/metas/test_koniq_2k_new.json",
        "img_dir": "KONIQ/images", "density_type": "pdf", "thre_std": 0.2,
        "thre_diff": 0.1,
    }
    generator = _generator()
    generator.main(generator.resolve_dataset_paths(cfg, tmp_path), "The quality of the image is {}.")
    train = json.loads((metas / "train_koniq_7k_new.json").read_text(encoding="utf-8"))
    test = json.loads((metas / "test_koniq_2k_new.json").read_text(encoding="utf-8"))
    assert len(train) == 3 and len(test) == 1
    assert train[0]["gt_score"] == 20 and train[0]["gt_score_norm"] == pytest.approx(1)
    assert pair_quality_targets(train[0]) == pytest.approx((1, 0))
    for record in train:
        assert sum(record["level_probs"]) == pytest.approx(1, abs=1e-6)
        assert min(record["level_probs"]) >= 0
    assert test[0]["gt_score_norm"] == pytest.approx(5)
    pred_file = tmp_path / "predictions.jsonl"
    pred_file.write_text("\n".join(json.dumps({
        "quality_score": i + 1, "gt_score": (i + 1) * 20,
        "gt_score_norm": i + 1,
    }) for i in range(4)), encoding="utf-8")
    result = evaluate_file(pred_file)
    assert result["gt_field"] == "gt_score_norm"
    assert result["SRCC"] == pytest.approx(1)
    assert result["MAE"] == pytest.approx(0)
    with pytest.raises(ValueError, match="already normalized"):
        evaluate_file(pred_file, gt_range=(0, 100))
    assert evaluate_file(pred_file, gt_field="gt_score", gt_range=(0, 80))["MAE"] > 0


def test_metrics_reject_raw_mos_without_scale(tmp_path):
    pred_file = tmp_path / "predictions.jsonl"
    pred_file.write_text("\n".join(json.dumps({"quality_score": v, "gt_score": v * 20})
                                    for v in (1, 2, 3)), encoding="utf-8")
    with pytest.raises(ValueError, match="not on the 1–5 scale"):
        evaluate_file(pred_file)
