"""Balanced sampling of image-quality datasets for co-training."""
import torch
from torch.utils.data import Sampler


class UniformDatasetSampler(Sampler[int]):
    """Give each dataset equal representation in each global training epoch.

    The sampler yields indices into the concatenated PairDataset. It does not
    shard by DDP rank: Trainer/Accelerate shards the global batch sampler.
    """

    def __init__(self, sizes, seed: int = 42):
        self.sizes = tuple(int(n) for n in sizes)
        if not self.sizes or any(n <= 1 for n in self.sizes):
            raise ValueError("Pair training requires at least two samples per dataset")
        self.seed = int(seed)
        self._epoch = 0
        self.total = sum(self.sizes)

    def __len__(self):
        return self.total

    def set_epoch(self, epoch):
        self._epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self._epoch)
        self._epoch += 1
        num_groups = len(self.sizes)
        target = [self.total // num_groups + int(i < self.total % num_groups)
                  for i in range(num_groups)]
        offsets = [0]
        for n in self.sizes[:-1]:
            offsets.append(offsets[-1] + n)
        groups = []
        for size, offset, count in zip(self.sizes, offsets, target):
            parts = []
            remaining = count
            while remaining > 0:
                permutation = torch.randperm(size, generator=generator)
                parts.append(permutation[:remaining])
                remaining -= min(remaining, size)
            groups.append(torch.cat(parts) + offset)
        indices = torch.cat(groups)
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        return iter(indices.tolist())

