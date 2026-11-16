"""Pairwise image-quality training data."""
from .pair_dataset import make_pair_data_module


def make_data_module(tokenizer, data_args):
    if data_args.dataset_type != "pair":
        raise ValueError("DARRA training requires dataset_type='pair'")
    return make_pair_data_module(tokenizer, data_args)
