"""Pure-Torch invariants; run without transformers/peft or a 7B checkpoint."""
import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
try:
    import transformers  # noqa: F401
    from src.model import darra
except ModuleNotFoundError:
    # The minimal CI image has torch only. Direct load avoids importing full 7B deps.
    for name in ("src", "src.model"):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(PROJECT / name.replace(".", "/"))]
            sys.modules[name] = module
    spec = importlib.util.spec_from_file_location("src.model.darra", PROJECT / "src/model/darra.py")
    darra = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = darra
    spec.loader.exec_module(darra)


def test_ier_formula_and_bounds():
    ier = darra.InvertedEnergyReweighting(ier_lambda=0.01, ier_eps=1e-4)
    x = torch.tensor([[[[0., 0.5], [1., 10.]]]], dtype=torch.float32)
    mean = x.mean((2, 3), keepdim=True)
    variance = ((x - mean) ** 2).mean((2, 3), keepdim=True)
    d = (x - mean).square() / (4 * (variance + 0.01)) + 0.5
    expected = torch.sigmoid(1 / (d + 1e-4))
    assert torch.allclose(ier.retention_weights(x), expected, rtol=1e-6)
    assert bool(torch.all((expected > 0.5) & (expected <= torch.sigmoid(torch.tensor(2.0)))))
    # A larger d produces a smaller retention weight.
    ds = torch.tensor([0.5, 1.0, 5.0])
    weights = torch.sigmoid(1.0 / (ds + ier.ier_eps))
    assert torch.all(weights[:-1] > weights[1:])
    assert torch.allclose(ier(x), x * expected)


def test_ier_half_dtype():
    x = torch.randn(2, 4, 3, 3).half()
    y = darra.InvertedEnergyReweighting()(x)
    assert y.dtype == x.dtype and torch.isfinite(y).all()


def test_adapter_identity_gradient_and_shape():
    torch.manual_seed(5)
    cfg = darra.DARRAConfig(start_layer=1, end_layer=1)
    adapter = darra.DARRAAdapter(16, cfg)
    x = torch.randn(2, 17, 16, requires_grad=True)
    y = adapter(x)
    assert y.shape == x.shape
    assert torch.count_nonzero(y) == 0
    (y[:, 1:, :] * torch.randn_like(y[:, 1:, :])).sum().backward()
    assert adapter.up_proj.weight.grad is not None
    assert adapter.up_proj.weight.grad.abs().sum() > 0
    with torch.no_grad():
        adapter.up_proj.weight.add_(0.0001)
    adapter.zero_grad(set_to_none=True)
    out = adapter(x.detach())
    out[:, 1:, :].square().sum().backward()
    assert adapter.residual_down[0].weight.grad is not None
    assert adapter.residual_down[0].weight.grad.abs().sum() > 0
    assert out[:, 1:, :].abs().sum() > 0
    assert (out[:, 0] == 0).all()
    with pytest.raises(ValueError, match="square"):
        adapter(torch.randn(2, 18, 16))


class FakeAttention(torch.nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.query_key_value = torch.nn.Linear(hidden_size, 3 * hidden_size)
        self.dense = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        return self.dense(hidden_states), None


class DummyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = FakeAttention()


class DummyRoot(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.vision_model = torch.nn.Module()
        self.model.vision_model.encoder = torch.nn.Module()
        self.model.vision_model.encoder.layers = torch.nn.ModuleList([DummyLayer() for _ in range(24)])

    def get_model(self):
        return self.model


def test_injection_range_identity_save_and_strict_load(tmp_path, monkeypatch):
    fake = types.ModuleType("src.model.visual_encoder")
    fake.MplugOwlVisionAttention = FakeAttention
    monkeypatch.setitem(sys.modules, "src.model.visual_encoder", fake)
    model = DummyRoot()
    initial = model.model.vision_model.encoder.layers[12].self_attn
    cfg = darra.DARRAConfig()
    injected = darra.inject_darra(model, cfg)
    assert injected == list(range(13, 25))
    assert isinstance(model.model.vision_model.encoder.layers[11].self_attn, FakeAttention)
    assert model.model.vision_model.encoder.layers[12].self_attn.base_attention is initial
    x = torch.randn(1, 5, 8)
    baseline = initial(x)[0]
    adapted = model.model.vision_model.encoder.layers[12].self_attn(x)[0]
    assert torch.equal(baseline, adapted), "Zero-up-proj must exactly preserve base MHSA"

    model.requires_grad_(False)
    unfrozen = darra.unfreeze_darra_only(model)
    assert unfrozen and all(".ier_adapter." in n for n in unfrozen)
    assert not model.model.vision_model.encoder.layers[12].self_attn.base_attention.dense.weight.requires_grad
    with torch.no_grad():
        model.model.vision_model.encoder.layers[12].self_attn.ier_adapter.up_proj.bias.fill_(0.015)
    darra.save_darra(model, tmp_path, cfg)
    with torch.no_grad():
        model.model.vision_model.encoder.layers[12].self_attn.ier_adapter.up_proj.bias.zero_()
    assert model.model.vision_model.encoder.layers[12].self_attn.ier_adapter.up_proj.bias.abs().sum() == 0
    assert darra.load_darra(model, tmp_path, cfg) == cfg
    assert torch.allclose(model.model.vision_model.encoder.layers[12].self_attn.ier_adapter.up_proj.bias,
                          torch.full((8,), .015))
    state = torch.load(tmp_path / "non_lora_trainables.bin", map_location="cpu", weights_only=True)
    state.pop(next(iter(state)))
    torch.save(state, tmp_path / "non_lora_trainables.bin")
    with pytest.raises(RuntimeError, match="key mismatch"):
        darra.load_darra(model, tmp_path, cfg)


def test_foreign_checkpoint_cannot_be_loaded(tmp_path):
    model = DummyRoot()
    darra.DARRAConfig().save(tmp_path)
    torch.save({"model.vision_model.encoder.layers.0.self_attn.unrelated_adapter.attention.weight":
                torch.ones(1)}, tmp_path / "non_lora_trainables.bin")
    with pytest.raises(ValueError, match="Invalid DARRA state"):
        darra.load_darra(model, tmp_path)


def test_peft_prefix_normalization(monkeypatch):
    fake = types.ModuleType("src.model.visual_encoder")
    fake.MplugOwlVisionAttention = FakeAttention
    monkeypatch.setitem(sys.modules, "src.model.visual_encoder", fake)
    model = DummyRoot()
    darra.inject_darra(model, darra.DARRAConfig())
    wrapped = torch.nn.Module()
    wrapped.base_model = torch.nn.Module()
    wrapped.base_model.model = model
    state = darra.adapter_state_dict(wrapped)
    assert state
    assert all(k.startswith("model.vision_model.encoder.layers.") for k in state)
    assert all("base_model.model." not in k for k in state)
