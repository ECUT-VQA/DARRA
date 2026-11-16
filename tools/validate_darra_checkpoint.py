"""Check PEFT+DARRA checkpoint completeness without loading the 7B backbone."""
import argparse
import json
import re
from pathlib import Path

import torch


def validate(path):
    path = Path(path)
    config_path = path / "darra_config.json"
    weight_path = path / "non_lora_trainables.bin"
    lora_config_path = path / "adapter_config.json"
    missing = [str(p) for p in (config_path, weight_path, lora_config_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint is incomplete; missing: {missing}")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if cfg.get("architecture") != "DARRA-IER-v1":
        raise ValueError("Checkpoint architecture is not DARRA-IER-v1")
    if not (cfg.get("ier_lambda", 0) > 0 and cfg.get("ier_eps", 0) > 0):
        raise ValueError("Invalid IER parameters")
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError("DARRA state must be a non-empty tensor dict")
    expected = set(range(cfg["start_layer"] - 1, cfg["end_layer"]))
    found = set()
    prefix = re.compile(r"^model\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.ier_adapter\.")
    for key, value in state.items():
        match = prefix.match(key)
        if match is None:
            raise ValueError(f"Unexpected / old architecture state key: {key}")
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError(f"Invalid tensor in checkpoint: {key}")
        found.add(int(match.group(1)))
    if found != expected:
        raise ValueError(f"Wrong injected layer set. missing={expected-found}, extra={found-expected}")
    for layer in expected:
        head = f"model.vision_model.encoder.layers.{layer}.self_attn.ier_adapter."
        for tail in (
            "up_proj.weight", "up_proj.bias", "residual_down.0.weight",
            "structure_down.0.weight", "gate_residual_from_structure.weight",
            "gate_structure_from_residual.weight",
        ):
            if head + tail not in state:
                raise ValueError(f"Incomplete DARRA layer {layer+1}: missing {head+tail}")
    lora = json.loads(lora_config_path.read_text(encoding="utf-8"))
    if not (path / "adapter_model.bin").is_file() and not (path / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"LoRA adapter weight is missing in {path}")
    return {"layers_1based": [i + 1 for i in sorted(found)], "darra_tensors": len(state),
            "ier_lambda": cfg["ier_lambda"], "ier_eps": cfg["ier_eps"],
            "lora_r": lora.get("r"), "status": "check_passed"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.checkpoint), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
