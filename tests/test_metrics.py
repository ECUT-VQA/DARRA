"""Direct-score metrics and controlled score normalization tests."""
import importlib.util
import json
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "src/evaluate/metrics.py"
spec = importlib.util.spec_from_file_location("darra_metrics", SOURCE)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


def write_predictions(tmp_path, pairs):
    path = tmp_path / "predictions.jsonl"
    path.write_text("\n".join(json.dumps({"quality_score": pred, "gt_score": gt})
                              for pred, gt in pairs), encoding="utf-8")
    return path


def test_direct_metrics_perfect_case(tmp_path):
    file = write_predictions(tmp_path, [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)])
    result = metrics.evaluate_file(file)
    assert result["SRCC"] == pytest.approx(1.0)
    assert result["PLCC"] == pytest.approx(1.0)
    assert result["KRCC"] == pytest.approx(1.0)
    assert result["MAE"] == pytest.approx(0.0)
    assert result["RMSE"] == pytest.approx(0.0)


def test_fixed_ground_truth_scale_conversion(tmp_path):
    file = write_predictions(tmp_path, [(1.0, 0.0), (3.0, 50.0), (5.0, 100.0)])
    assert metrics.evaluate_file(file, gt_range=(0.0, 100.0))["RMSE"] == pytest.approx(0.0)
    invert_file = write_predictions(tmp_path, [(5.0, 0.0), (3.0, 50.0), (1.0, 100.0)])
    assert metrics.evaluate_file(invert_file, gt_range=(0.0, 100.0), invert_gt=True)["MAE"] == pytest.approx(0.0)


def test_invalid_gt_range_rejected(tmp_path):
    file = write_predictions(tmp_path, [(1.0, 0.0), (3.0, 50.0), (5.0, 100.0)])
    with pytest.raises(ValueError, match="GT max"):
        metrics.evaluate_file(file, gt_range=(100.0, 0.0))
