"""Training utilities for DARRA and dataset-balanced IQA pair sampling."""
from pathlib import Path

from src.model.darra import DARRAConfig, save_darra
from src.train.sampling import UniformDatasetSampler
from transformers import Trainer


class DARRATrainer(Trainer):
    def __init__(self, *args, darra_config: DARRAConfig, **kwargs):
        self.darra_config = darra_config
        if getattr(kwargs.get("args"), "deepspeed", None):
            raise ValueError("Use DDP (torchrun) for DARRA checkpointing; ZeRO checkpoints are not supported")
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self):
        if self.train_dataset is None:
            return None
        if getattr(self.args, "dataset_sampling", "balanced") == "balanced":
            return UniformDatasetSampler(self.train_dataset.nums_eachdata,
                                         seed=self.args.seed)
        return super()._get_train_sampler()

    def _save_checkpoint(self, model, trial, metrics=None):
        super()._save_checkpoint(model, trial, metrics)
        if self.is_world_process_zero():
            checkpoint = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
            save_darra(self.model, checkpoint, self.darra_config)

    def save_darra_final(self):
        if self.is_world_process_zero():
            save_darra(self.model, self.args.output_dir, self.darra_config)
