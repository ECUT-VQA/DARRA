"""mPLUG-Owl2 checkpoint loading with DARRA-IER and PEFT LoRA."""
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.models.clip.image_processing_clip import CLIPImageProcessor

from src.model import MPLUGOwl2Config, MPLUGOwl2LlamaForCausalLM
from src.model.darra import DARRAConfig, inject_darra, load_darra


def load_pretrained_model(
    model_path, model_base=None, model_name=None, load_8bit=False, load_4bit=False,
    device_map="auto", device="cuda", preprocessor_path=None, use_darra=False,
):
    if load_8bit and load_4bit:
        raise ValueError("Cannot use both 8-bit and 4-bit loading")
    if not torch.cuda.is_available() and device.startswith("cuda"):
        raise RuntimeError(f"CUDA requested ({device}) but CUDA is unavailable")

    kwargs = {"low_cpu_mem_usage": True}
    if device_map == "auto":
        # Keep inference on the selected logical device.
        kwargs["device_map"] = {"": device}
    else:
        kwargs["device_map"] = device_map
    if load_8bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float32 if device == "cpu" else torch.float16

    model_path = str(model_path)
    checkpoint_dir = Path(model_path)
    peft_checkpoint = checkpoint_dir.is_dir() and (checkpoint_dir / "adapter_config.json").is_file()

    if use_darra:
        if model_base is None:
            raise ValueError("DARRA checkpoints require --model-base for original mPLUG-Owl2 weights")
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"DARRA checkpoint folder not found: {model_path}")
        if not peft_checkpoint:
            raise FileNotFoundError(f"Missing LoRA adapter_config.json in {model_path}")
        cfg = DARRAConfig.load(checkpoint_dir)
    else:
        cfg = None
        if checkpoint_dir.is_dir() and (checkpoint_dir / "darra_config.json").exists():
            raise ValueError("This checkpoint contains DARRA. Pass --use-darra; do not evaluate it as baseline")

    tokenizer_path = str(preprocessor_path or model_base or model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)

    if model_base is not None:
        full_weights = checkpoint_dir.is_dir() and (checkpoint_dir / "config.json").is_file() and any(
            (checkpoint_dir / filename).exists() for filename in (
                "pytorch_model.bin", "model.safetensors",
                "pytorch_model.bin.index.json", "model.safetensors.index.json",
            )
        )
        if not use_darra and not peft_checkpoint and full_weights:
            # A full checkpoint supplies the complete language and vision weights.
            model = MPLUGOwl2LlamaForCausalLM.from_pretrained(model_path, **kwargs)
        else:
            # Load the pretrained architecture before adding its residual adapters.
            model = MPLUGOwl2LlamaForCausalLM.from_pretrained(str(model_base), **kwargs)
            if use_darra:
                layers = inject_darra(model, cfg)
                if len(layers) != cfg.end_layer - cfg.start_layer + 1:
                    raise RuntimeError("Incorrect DARRA injection count")
                load_darra(model, checkpoint_dir, cfg)
            if peft_checkpoint:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, model_path)
            elif checkpoint_dir.is_dir() and checkpoint_dir.resolve() != Path(str(model_base)).resolve():
                raise FileNotFoundError(
                    f"Expected PEFT adapter or full model weights in {model_path}"
                )
    else:
        # Detect the architecture using config.json rather than a model/directory name.
        # The mPLUG-Owl2 config class is registered when src.model is imported.
        architecture = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
        if isinstance(architecture, MPLUGOwl2Config):
            model = MPLUGOwl2LlamaForCausalLM.from_pretrained(model_path, **kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    image_processor = CLIPImageProcessor.from_pretrained(tokenizer_path)
    model.eval()
    context_len = getattr(model.config, "max_sequence_length", 2048)
    return tokenizer, model, image_processor, context_len
