"""Direct SRCC, PLCC, KRCC, MAE and RMSE for DARRA-IER scores."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr


def evaluate_file(path, *, gt_range=None, invert_gt=False, gt_field="auto"):
    entries = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    entries = [x for x in entries if x.get("quality_score") is not None]
    if gt_field not in {"auto", "gt_score", "gt_score_norm"}:
        raise ValueError("gt_field must be auto, gt_score or gt_score_norm")
    if gt_field == "auto":
        # Select one field for the entire file, never a mixture of score scales.
        if entries and all(x.get("gt_score_norm") is not None for x in entries):
            selected_field = "gt_score_norm"
        elif entries and all(x.get("gt_score") is not None for x in entries):
            selected_field = "gt_score"
        else:
            raise ValueError(f"Inconsistent or absent ground-truth fields in {path}; specify --gt-field")
    else:
        selected_field = gt_field
    if gt_range is not None and selected_field == "gt_score_norm":
        raise ValueError("gt_score_norm is already normalized; use --gt-field gt_score with --gt-range")
    if invert_gt and gt_range is None:
        raise ValueError("--invert-gt requires --gt-range to define the raw MOS scale")
    if any(x.get(selected_field) is None for x in entries):
        raise ValueError(f"Missing {selected_field} in some entries of {path}")
    if len(entries) < 3:
        raise ValueError(f"Need at least 3 scored examples with GT values: {path}")
    pred = np.asarray([float(x["quality_score"]) for x in entries], dtype=np.float64)
    gt = np.asarray([float(x[selected_field]) for x in entries], dtype=np.float64)
    if not np.isfinite(pred).all() or not np.isfinite(gt).all():
        raise ValueError("Non-finite predicted or ground-truth scores")
    if gt_range is not None:
        low, high = gt_range
        if not np.isfinite([low, high]).all() or high <= low:
            raise ValueError("GT max must be greater than min")
        if np.any((gt < low) | (gt > high)):
            raise ValueError("Some GT values are outside the specified range")
        gt = 1.0 + 4.0 * (gt - low) / (high - low)
    elif np.any((gt < 1.0 - 1e-6) | (gt > 5.0 + 1e-6)):
        raise ValueError(f"{selected_field} is not on the 1–5 scale; supply --gt-range MIN MAX for raw MOS")
    if invert_gt:
        gt = 6.0 - gt
    if np.std(pred) < 1e-10 or np.std(gt) < 1e-10:
        raise ValueError("Constant scores: correlations are undefined")
    return {
        "count": len(entries),
        "gt_field": selected_field,
        "SRCC": float(spearmanr(pred, gt).statistic),
        "PLCC": float(pearsonr(pred, gt).statistic),
        "KRCC": float(kendalltau(pred, gt, variant="b").statistic),
        "MAE": float(np.mean(np.abs(pred - gt))),
        "RMSE": float(np.sqrt(np.mean((pred - gt)**2))),
    }


def main():
    parser = argparse.ArgumentParser(description="Direct DARRA-IER image-quality metrics")
    parser.add_argument("--pred-paths", nargs="+", required=True)
    parser.add_argument("--gt-field", choices=["auto", "gt_score", "gt_score_norm"], default="auto")
    parser.add_argument("--gt-range", nargs=2, type=float, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--invert-gt", action="store_true")
    args = parser.parse_args()
    for path in args.pred_paths:
        print(path, json.dumps(evaluate_file(path, gt_range=args.gt_range,
                                             invert_gt=args.invert_gt,
                                             gt_field=args.gt_field), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
