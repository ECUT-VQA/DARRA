"""DARRA-IER training for pairwise distributional image quality regression."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import transformers
from peft import LoraConfig, get_peft_model
from transformers import CLIPImageProcessor

from src import conversation as conversation_lib
from src.datasets import make_data_module
from src.data_config import DEFAULT_LABEL_CONFIG, generated_metadata_paths, resolve_metadata_paths
from src.model import MPLUGOwl2LlamaForCausalLM
from src.model.darra import (DARRAConfig, inject_darra, load_darra,
                             unfreeze_darra_only)
from src.train.darra_trainer import DARRATrainer


@dataclass
class DataArguments:
    dataset_type: str = "pair"
    data_paths: List[str] = field(default_factory=list)
    data_weights: List[int] = field(default_factory=list)
    lazy_preprocess: bool = False
    is_multimodal: bool = True
    image_folder: Optional[str] = None
    image_aspect_ratio: str = "pad"
    image_grid_pinpoints: Optional[str] = None


@dataclass
class DARRATrainingArguments(transformers.TrainingArguments):
    remove_unused_columns: bool = False
    optim: str = "adamw_torch"
    model_max_length: int = 2048
    group_by_modality_length: bool = False
    visual_abstractor_lr: Optional[float] = None
    lora_enable: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    # Distributional score and pairwise fidelity loss coefficients.
    weight_desp: float = 1.0
    weight_rank: float = 1.0
    weight_softkl: float = 1.0
    weight_next_token: float = 0.05
    weight_in_level: Optional[float] = None
    continuous_rating_loss: bool = True
    closeset_rating_loss: bool = True
    use_fix_std: bool = True
    detach_pred_std: bool = True
    binary_rating_loss: str = "fidelity"
    softkl_loss: bool = True
    dataset_sampling: str = "balanced"


LEVELS = ("excellent", "good", "fair", "poor", "bad")


def parse_args():
    p = argparse.ArgumentParser(description="DARRA-IER training")
    p.add_argument("--model-base", default="ModelZoo/mplug-owl2-llama2-7b")
    p.add_argument("--preprocessor-path", default="preprocessor",
                   help="Tokenizer and CLIP image processor directory")
    p.add_argument("--data-root", default="Data-DARRA")
    p.add_argument("--output-dir", default="checkpoints/DARRA-IER")
    p.add_argument("--label-config", default=str(DEFAULT_LABEL_CONFIG),
                   help="Soft-label configuration used for default training metadata paths")
    p.add_argument("--train-meta", nargs="+", default=None,
                   help="Override default training metadata paths defined by --label-config")
    p.add_argument("--data-weights", nargs="+", type=int, default=[1, 1, 1])
    p.add_argument("--dataset-sampling", choices=["balanced", "proportional"], default="balanced")
    p.add_argument("--epochs", type=float, default=4.0)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=32)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--dataloader-workers", type=int, default=4)
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--ier-lambda", type=float, default=1e-2)
    p.add_argument("--ier-eps", type=float, default=1e-4)
    p.add_argument("--bottleneck-ratio", type=float, default=0.5)
    p.add_argument("--start-layer", type=int, default=13)
    p.add_argument("--end-layer", type=int, default=24)
    p.add_argument("--resume-from-checkpoint", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def lora_targets(model):
    result = []
    for name, module in model.named_modules():
        if "vision_model" in name or "visual_abstractor" in name:
            continue
        # mPLUG-Owl2 has a MultiwayNetwork under v_proj (not a Linear).
        # mPLUG image modality uses v_proj.multiway.1 for visual tokens.
        if isinstance(module, torch.nn.Linear) and (
            name.endswith("q_proj") or name.endswith("v_proj.multiway.1")
        ):
            result.append(name)
    if not result:
        raise RuntimeError("No LLaMA q_proj / v_proj LoRA targets found; check model version")
    return sorted(result)


def main():
    args = parse_args()
    transformers.set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if not torch.cuda.is_available():
        raise RuntimeError("Training a 7B model requires CUDA; CPU runs are supported only for unit tests")
    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)
    train_meta = args.train_meta if args.train_meta is not None else generated_metadata_paths(
        "train", args.label_config)
    if len(train_meta) != len(args.data_weights):
        raise ValueError("Training dataset count and --data-weights must have equal lengths")
    if args.dataset_sampling == "balanced" and any(w != 1 for w in args.data_weights):
        raise ValueError("Balanced sampling requires --data-weights 1 for every dataset")
    if any(n <= 0 for n in args.data_weights):
        raise ValueError("Data weights must be positive")
    if args.per_device_batch_size < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid batch size or gradient accumulation")

    root = Path(args.data_root).expanduser().resolve()
    base = Path(args.model_base).expanduser().resolve()
    processor_path = Path(args.preprocessor_path).expanduser().resolve() if args.preprocessor_path else base
    output = Path(args.output_dir).expanduser().resolve()
    data_paths = resolve_metadata_paths(train_meta, root)
    for path in (str(base), str(processor_path), str(root), *data_paths):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Required model or training data is missing: {path}. "
                "For *_new.json run build_soft_labels/gen_soft_label.py first, "
                "or supply --train-meta for already prepared metadata."
            )

    config = DARRAConfig(
        start_layer=args.start_layer, end_layer=args.end_layer,
        ier_lambda=args.ier_lambda, ier_eps=args.ier_eps,
        bottleneck_ratio=args.bottleneck_ratio,
    )
    config.validate()
    if args.resume_from_checkpoint is not None:
        if DARRAConfig.load(args.resume_from_checkpoint) != config:
            raise ValueError("Resume checkpoint's DARRA configuration does not match requested values")

    supports_bf16 = torch.cuda.is_bf16_supported()
    train_args = DARRATrainingArguments(
        output_dir=str(output), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, weight_decay=0.01, warmup_ratio=0.03,
        lr_scheduler_type="cosine", save_strategy="steps", save_steps=args.save_steps,
        save_total_limit=2, logging_steps=10, gradient_checkpointing=False,
        bf16=supports_bf16, fp16=not supports_bf16, tf32=False,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=args.dataloader_workers,
        report_to=[], save_safetensors=False,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        dataset_sampling=args.dataset_sampling, seed=args.seed,
    )
    dtype = torch.bfloat16 if supports_bf16 else torch.float16
    # Explicit local mapping keeps the pretrained model on the correct DDP GPU.
    model = MPLUGOwl2LlamaForCausalLM.from_pretrained(
        str(base), low_cpu_mem_usage=True, torch_dtype=dtype, device_map={"": rank}
    )
    # Train DARRA and the scoring-branch LoRA without updating the pretrained ViT.
    model.requires_grad_(False)
    injected = inject_darra(model, config)
    print(f"[rank {rank}] DARRA injected into 1-based ViT layers: {injected}")

    # Keep frozen visual modules outside reentrant checkpointing.
    model.get_model().vision_model.encoder.gradient_checkpointing = False
    model.get_model().visual_abstractor.encoder.gradient_checkpointing = False
    model.get_model().visual_abstractor.encoder.config.gradient_checkpointing = False
    model.config.use_cache = False
    # Checkpoint the language decoder while preserving the visual adapter graph.
    model.model.gradient_checkpointing = True
    model.config.gradient_checkpointing = True

    targets = lora_targets(model)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=targets, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
    ))
    unfreeze = unfreeze_darra_only(model)
    assert len(unfreeze) > 0
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        model.get_input_embeddings().register_forward_hook(
            lambda _, __, output: output.requires_grad_(True)
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(processor_path), model_max_length=train_args.model_max_length,
        padding_side="right", use_fast=False
    )
    tokenizer.pad_token = tokenizer.unk_token
    # Five ordered quality-token anchors.
    level_ids = [tokenizer(x, add_special_tokens=False).input_ids for x in LEVELS]
    if not all(len(ids) == 1 for ids in level_ids):
        raise RuntimeError(f"All quality levels must map to one token: {dict(zip(LEVELS, level_ids))}")
    inner = model.base_model.model
    prefix = tokenizer("The quality of the image is", add_special_tokens=False).input_ids
    inner.config.level_prefix = prefix
    inner.config.level_ids = [ids[0] for ids in level_ids]
    for field_name in (
        "weight_desp", "weight_rank", "weight_softkl", "weight_next_token",
        "weight_in_level", "continuous_rating_loss", "closeset_rating_loss",
        "use_fix_std", "detach_pred_std", "binary_rating_loss", "softkl_loss",
    ):
        setattr(inner.config, field_name, getattr(train_args, field_name))
    inner.config.image_aspect_ratio = "pad"
    inner.config.tune_visual_abstractor = False
    inner.config.freeze_vision_model = True
    inner.config.darra_enabled = True
    inner.config.darra_architecture = config.architecture
    conversation_lib.default_conversation = conversation_lib.conv_templates["mplug_owl2"]

    data_args = DataArguments(
        data_paths=data_paths, data_weights=args.data_weights, image_folder=str(root),
    )
    data_args.image_processor = CLIPImageProcessor.from_pretrained(str(processor_path))
    data_module = make_data_module(tokenizer=tokenizer, data_args=data_args)

    trainer = DARRATrainer(
        model=model, tokenizer=tokenizer, args=train_args,
        darra_config=config, **data_module,
    )
    if args.resume_from_checkpoint:
        # Restore the trained residual path alongside the LoRA state.
        load_darra(model.base_model.model, args.resume_from_checkpoint, config)
    if trainer.is_world_process_zero():
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        world = int(os.environ.get("WORLD_SIZE", "1"))
        print(f"Trainable parameters: {trainable:,}/{total:,}")
        print(f"Effective batch size = {args.per_device_batch_size * args.gradient_accumulation_steps * world}")
        print("Pretrained ViT: frozen; DARRA and language LoRA: trainable")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output))
    trainer.save_state()
    trainer.save_darra_final()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(output))
        print(f"Final LoRA + DARRA checkpoint saved at {output}")


if __name__ == "__main__":
    main()
