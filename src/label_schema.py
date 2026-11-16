"""Consistent perceptual-quality supervision on the 1–5 scoring scale."""
from __future__ import annotations

import math


def pair_quality_targets(record: dict, *, location: str = "sample") -> tuple[float, float]:
    """Obtain normalized MOS and standard deviation for pairwise fidelity.

    Generated metadata has both raw MOS and normalized MOS. Legacy metadata
    may supply only gt_score/std when those fields are *already* in the 1–5
    scale. Never silently train with raw 0–100 MOS against 1–5 predictions.
    """
    normalized = "gt_score_norm" in record or "std_norm" in record
    if normalized:
        if "gt_score_norm" not in record or "std_norm" not in record:
            raise ValueError(f"{location}: gt_score_norm and std_norm must be provided together")
        score = record["gt_score_norm"]
        deviation = record["std_norm"]
    else:
        if "gt_score" not in record or "std" not in record:
            raise ValueError(f"{location}: require normalized score/std or pre-normalized gt_score/std")
        score = record["gt_score"]
        deviation = record["std"]
    try:
        score = float(score)
        deviation = float(deviation)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location}: quality targets must be numeric") from exc
    if not math.isfinite(score) or not (1.0 - 1e-6 <= score <= 5.0 + 1e-6):
        raise ValueError(f"{location}: score={score} is outside the 1–5 quality scale; run the soft-label generator")
    if not math.isfinite(deviation) or deviation < 0:
        raise ValueError(f"{location}: quality deviation must be finite and nonnegative")
    return score, deviation
