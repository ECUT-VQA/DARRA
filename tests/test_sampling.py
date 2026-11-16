"""Dataset-balanced co-training sampler tests."""
import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "src/train/sampling.py"
spec = importlib.util.spec_from_file_location("sampling_module", SOURCE)
sampling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampling)


def test_balanced_global_epoch_counts():
    sampler = sampling.UniformDatasetSampler([3, 7, 5], seed=41)
    items = list(sampler)
    assert len(items) == 15
    assert len(set(items)) <= 15  # Small groups are intentionally resampled.
    assert sum(0 <= i < 3 for i in items) == 5
    assert sum(3 <= i < 10 for i in items) == 5
    assert sum(10 <= i < 15 for i in items) == 5
    assert all(0 <= i < 15 for i in items)


def test_reproducible_sampler_epoch_seed():
    first = sampling.UniformDatasetSampler([7, 9, 8], seed=17)
    second = sampling.UniformDatasetSampler([7, 9, 8], seed=17)
    assert list(first) == list(second)
    first.set_epoch(6)
    second.set_epoch(6)
    assert list(first) == list(second)


@pytest.mark.parametrize("sizes", [[0, 5], [1, 2], []])
def test_unusable_datasets_rejected(sizes):
    with pytest.raises(ValueError):
        sampling.UniformDatasetSampler(sizes)
