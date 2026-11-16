"""Distortion-Aware Residual Recalibration Adapter with Inverted Energy Reweighting."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DARRAConfig:
    # Paper: ViT blocks 13-24, 1-based inclusive, lambda=1e-2, epsilon=1e-4.
    start_layer: int = 13
    end_layer: int = 24
    ier_lambda: float = 1e-2
    ier_eps: float = 1e-4
    bottleneck_ratio: float = 0.5
    architecture: str = "DARRA-IER-v1"

    def validate(self, num_layers: Optional[int] = None) -> None:
        if self.architecture != "DARRA-IER-v1":
            raise ValueError(f"Unrecognized adapter architecture: {self.architecture}")
        if self.start_layer < 1 or self.end_layer < self.start_layer:
            raise ValueError("Invalid 1-based adapter layer range")
        if num_layers is not None and self.end_layer > num_layers:
            raise ValueError(f"DARRA end_layer={self.end_layer} exceeds ViT layer count={num_layers}")
        if self.ier_lambda <= 0 or self.ier_eps <= 0:
            raise ValueError("IER lambda and epsilon must be positive")
        if not (0 < self.bottleneck_ratio <= 1):
            raise ValueError("bottleneck_ratio must be in (0, 1]")

    def save(self, directory) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "darra_config.json").write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, directory) -> "DARRAConfig":
        path = Path(directory) / "darra_config.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing DARRA configuration: {path}")
        config = cls(**json.loads(path.read_text(encoding="utf-8")))
        config.validate()
        return config


class InvertedEnergyReweighting(nn.Module):
    """Paper equations (5-6): w=sigmoid(1/(d+eps)), output=u*w.

    The weights attenuate spatially distinctive residual responses and
    preserve weaker, spatially distributed local residual evidence.
    Statistics are per sample/channel over HxW, biased variance (M divisor).
    """

    def __init__(self, ier_lambda: float = 1e-2, ier_eps: float = 1e-4):
        super().__init__()
        if ier_lambda <= 0 or ier_eps <= 0:
            raise ValueError("IER lambda and epsilon must be positive")
        self.ier_lambda = float(ier_lambda)
        self.ier_eps = float(ier_eps)

    def retention_weights(self, residual: torch.Tensor) -> torch.Tensor:
        if residual.ndim != 4:
            raise ValueError(f"IER expects [B, C, H, W], got {tuple(residual.shape)}")
        # Compute local statistics in FP32 to avoid overflow/underflow with FP16 ViT.
        u = residual.float()
        mean = u.mean(dim=(2, 3), keepdim=True)
        variance = (u - mean).square().mean(dim=(2, 3), keepdim=True)
        distinctiveness = (u - mean).square() / (4.0 * (variance + self.ier_lambda)) + 0.5
        weights = torch.sigmoid(1.0 / (distinctiveness + self.ier_eps))
        return weights.to(residual.dtype)

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        return residual * self.retention_weights(residual)


class ChannelLayerNorm2d(nn.Module):
    """Per-pixel channel layer normalization."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Avoid FP16 variance path and preserve output dtype.
        dtype = x.dtype
        x = x.float().permute(0, 2, 3, 1)
        x = F.layer_norm(x, (x.shape[-1],), self.weight.float(), self.bias.float(), self.eps)
        return x.permute(0, 3, 1, 2).to(dtype)


class DARRAAdapter(nn.Module):
    """Local residual, structure/residual streams, IER, cross-gating, zero-up-proj."""

    def __init__(self, hidden_size: int, config: DARRAConfig):
        super().__init__()
        config.validate()
        middle = max(1, int(hidden_size * config.bottleneck_ratio))
        self.structure_down = nn.Sequential(
            nn.Conv2d(hidden_size, middle, 1), ChannelLayerNorm2d(middle), nn.SiLU()
        )
        self.residual_down = nn.Sequential(
            nn.Conv2d(hidden_size, middle, 1), ChannelLayerNorm2d(middle), nn.SiLU()
        )
        self.structure_extract = nn.Sequential(
            nn.Conv2d(middle, middle, 3, padding=2, dilation=2, groups=middle),
            nn.SiLU(), nn.Conv2d(middle, middle, 1),
        )
        self.residual_extract = nn.Sequential(
            nn.Conv2d(middle, middle, 3, padding=1, groups=middle), nn.SiLU()
        )
        self.ier = InvertedEnergyReweighting(config.ier_lambda, config.ier_eps)
        self.gate_residual_from_structure = nn.Conv2d(middle, middle, 1)
        self.gate_structure_from_residual = nn.Conv2d(middle, middle, 1)
        self.up_proj = nn.Conv2d(middle, hidden_size, 1)

        # The residual path is exactly zero at initialization.
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
        for gate in (self.gate_residual_from_structure, self.gate_structure_from_residual):
            nn.init.normal_(gate.weight, std=1e-3)
            nn.init.zeros_(gate.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("DARRA expects tokens of shape [B, 1+H*W, C]")
        batch, count, channels = tokens.shape
        patches = count - 1
        side = math.isqrt(patches)
        if side * side != patches or patches <= 0:
            raise ValueError(f"DARRA requires a square spatial patch grid, got {patches} patches")
        x = tokens[:, 1:, :].transpose(1, 2).reshape(batch, channels, side, side)
        x_residual = x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        structure = self.structure_extract(self.structure_down(x))
        residual = self.residual_extract(self.residual_down(x_residual))
        residual = self.ier(residual)

        g_residual = torch.sigmoid(self.gate_residual_from_structure(structure))
        g_structure = torch.sigmoid(self.gate_structure_from_residual(residual))
        fused = residual * (1.0 + g_residual) + structure * (1.0 + g_structure)
        delta = self.up_proj(fused).flatten(2).transpose(1, 2)
        return torch.cat([torch.zeros_like(tokens[:, :1, :]), delta], dim=1)


def inject_darra(model: nn.Module, cfg: DARRAConfig) -> list[int]:
    """Inject ONLY the requested 1-based late ViT blocks, preserving pretrained QKV/dense."""
    from .visual_encoder import MplugOwlVisionAttention

    vision = model.get_model().vision_model
    layers = vision.encoder.layers
    cfg.validate(num_layers=len(layers))
    selected = []
    for idx in range(cfg.start_layer - 1, cfg.end_layer):
        original = layers[idx].self_attn
        if isinstance(original, DARRAAttention):
            selected.append(idx + 1)
            continue
        if not isinstance(original, MplugOwlVisionAttention):
            raise TypeError(f"Layer {idx+1} has unsupported attention: {type(original)}")
        replacement = DARRAAttention(original, cfg)
        layers[idx].self_attn = replacement
        selected.append(idx + 1)
    return selected


class DARRAAttention(nn.Module):
    """Pretrained visual attention with a parallel DARRA residual branch."""

    def __init__(self, original: nn.Module, config: DARRAConfig):
        super().__init__()
        self.base_attention = original
        # Access the base attention under a single module (no duplicated weights).
        hidden_size = original.hidden_size
        self.ier_adapter = DARRAAdapter(hidden_size, config).to(
            device=original.query_key_value.weight.device, dtype=original.query_key_value.weight.dtype
        )

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        attention_output, attention_probs = self.base_attention(
            hidden_states, head_mask=head_mask, output_attentions=output_attentions
        )
        return attention_output + self.ier_adapter(hidden_states), attention_probs


def unfreeze_darra_only(model: nn.Module) -> list[str]:
    """PEFT freezes the base model; restore adapter training after wrapping with PEFT."""
    trained = []
    for name, param in model.named_parameters():
        if ".ier_adapter." in name:
            param.requires_grad_(True)
            trained.append(name)
    if not trained:
        raise RuntimeError("DARRA parameters were not found after injection/PEFT wrapping")
    return trained


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """CPU state; gather ZeRO-3 partitions if present. Keys omit PEFT wrapper prefix."""
    state = {}
    for key, param in model.named_parameters():
        if ".ier_adapter." not in key:
            continue
        if hasattr(param, "ds_id"):
            raise NotImplementedError("ZeRO-partitioned DARRA weights require distributed checkpoint gathering")
        value = param.detach().cpu().clone()
        if key.startswith("base_model.model."):
            key = key[len("base_model.model."):]
        state[key] = value
    if not state:
        raise RuntimeError("No DARRA parameters to save")
    return state


def save_darra(model: nn.Module, directory, cfg: DARRAConfig) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = adapter_state_dict(model)
    if not all(".ier_adapter." in k for k in state):
        raise AssertionError("Non-DARRA parameter present in DARRA checkpoint")
    torch.save(state, directory / "non_lora_trainables.bin")
    cfg.save(directory)


def load_darra(model: nn.Module, directory, cfg: Optional[DARRAConfig] = None) -> DARRAConfig:
    """Fail closed on missing/mismatched keys (strict=False alone is not acceptable)."""
    directory = Path(directory)
    saved_config = DARRAConfig.load(directory)
    if cfg is not None and cfg != saved_config:
        raise ValueError(f"DARRA architecture mismatch: requested {cfg}, checkpoint {saved_config}")
    path = directory / "non_lora_trainables.bin"
    if not path.is_file():
        raise FileNotFoundError(f"Required trained DARRA checkpoint is missing: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or any(".ier_adapter." not in k for k in state):
        raise ValueError(f"Invalid DARRA state in {path}: unexpected parameter keys")
    expected = {k for k in model.state_dict() if ".ier_adapter." in k}
    actual = set(state)
    if actual != expected:
        raise RuntimeError(
            f"DARRA checkpoint key mismatch: missing={sorted(expected-actual)[:8]} "
            f"unexpected={sorted(actual-expected)[:8]} "
            f"(totals {len(expected-actual)}/{len(actual-expected)})"
        )
    for k in expected:
        if state[k].shape != model.state_dict()[k].shape:
            raise RuntimeError(f"DARRA tensor shape mismatch: {k}")
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys or any(".ier_adapter." in k for k in result.missing_keys):
        raise RuntimeError("DARRA weights did not load correctly")
    return saved_config
