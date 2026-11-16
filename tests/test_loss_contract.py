"""Regression tests for independent IQA losses and consistent prompt templates."""
import ast
from pathlib import Path

import pytest
import torch

from src.losses import pair_score_objective
from src import conversation


ROOT = Path(__file__).resolve().parents[1]


def test_score_loss_weights_do_not_scale_kl_by_next_token_coefficient():
    rank, next_token, kl, level = [torch.tensor(x) for x in (0.2, 2.0, 1.0, 0.5)]
    actual = pair_score_objective(rank, next_token, kl, level,
                                  weight_rank=1.0, weight_next_token=0.05,
                                  weight_softkl=1.0, weight_in_level=None)
    assert torch.allclose(actual, torch.tensor(1.3))
    assert not torch.allclose(actual, torch.tensor(0.35))
    assert torch.allclose(pair_score_objective(rank, next_token, kl, level,
                                              weight_next_token=0.0,
                                              weight_softkl=0.0,
                                              weight_in_level=2.0), torch.tensor(1.2))
    with pytest.raises(ValueError, match="nonnegative"):
        pair_score_objective(rank, next_token, kl, level, weight_softkl=-1)


def test_training_uses_same_chat_template_as_inference():
    train = (ROOT / "src/train/train_darra.py").read_text(encoding="utf-8")
    infer = (ROOT / "src/evaluate/iqa_eval.py").read_text(encoding="utf-8")
    assert 'conversation_lib.conv_templates["mplug_owl2"]' in train
    assert 'conv_templates["mplug_owl2"]' in infer
    assert conversation.conv_templates["mplug_owl2"].sep_style.name == "TWO_NO_SYS"


def test_score_forward_explicitly_tracks_ce_and_kl_components():
    tree = ast.parse((ROOT / "src/model/modeling_mplug_owl2.py").read_text(encoding="utf-8"))
    classes = [x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "MPLUGOwl2LlamaForCausalLM"]
    assert len(classes) == 1
    source = ast.unparse(classes[0])
    assert "result.loss_next_token = loss_next_token" in source
    assert "result.loss_softkl = loss_kl" in source
    assert "pair_score_objective(" in source
