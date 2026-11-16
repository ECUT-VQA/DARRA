"""Utilities shared by the distributional IQA model."""
import torch


def extend_list(data_list, count, smallest):
    if smallest == 0:
        return []
    if not data_list:
        raise ValueError("Cannot pad an empty task subset")
    while len(data_list) < count:
        data_list.extend(data_list[:count - len(data_list)])
    return data_list


def find_prefix(input_ids, prefix):
    """Find the quality-scoring text prefix in each tokenized sequence."""
    if prefix.ndim != 1 or prefix.numel() == 0 or input_ids.shape[1] < prefix.numel():
        raise ValueError("Invalid or missing quality-scoring prefix")
    matches = (input_ids.unfold(1, prefix.numel(), 1) == prefix).all(dim=2)
    if not bool(matches.any(dim=1).all()):
        raise ValueError("The quality-scoring prefix was not found in every sample")
    return matches.to(torch.int64).argmax(dim=1)
