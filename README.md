# DARRA-IER

Official implementation of **DARRA: A Distortion-Aware Residual Recalibration Adapter for MLLM-based No-Reference Image Quality Assessment**, with Inverted Energy Reweighting (IER).

**Paper:** *Signal Processing*, 251 (2027), 110859 · [DOI: 10.1016/j.sigpro.2026.110859](https://doi.org/10.1016/j.sigpro.2026.110859)

DARRA adapts the mPLUG-Owl2 vision encoder (blocks 13–24) for no-reference image quality assessment. We used **8 × NVIDIA RTX 4090 GPUs for full fine-tuning**. A separate **2 × RTX 4090 LoRA fine-tuning** configuration is also available as a more resource-efficient implementation.

## Setup

Python 3.10, CUDA and a compatible PyTorch environment are required.

```bash
pip install -e '.[test]'
python -m pytest -q tests/
```

The pretrained **mPLUG-Owl2-LLaMA2-7B** model, IQA datasets and trained checkpoints are **not included**. Place the model in `ModelZoo/mplug-owl2-llama2-7b/` and datasets in `Data-DARRA/`, or change the paths below.

## Prepare data

For KonIQ, SPAQ and KADID10K, prepare `split.json` and `mos.json`, adjust `build_soft_labels/config.json`, and generate soft labels:

```bash
python build_soft_labels/gen_soft_label.py --data-root Data-DARRA
```

## Training

We used two training configurations:

| Configuration | Hardware | Training mode |
| --- | --- | --- |
| **Full fine-tuning (used in our experiments)** | **8 × NVIDIA RTX 4090 (24 GB)** | Full-parameter fine-tuning |
| **LoRA fine-tuning** | **2 × NVIDIA RTX 4090 (24 GB)** | DARRA + language LoRA, with the pretrained backbone frozen |

### LoRA: 2 × RTX 4090

The following command uses the **LoRA training entrypoint currently included in this repository**:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m src.train.train_darra \
  --model-base ModelZoo/mplug-owl2-llama2-7b \
  --preprocessor-path preprocessor \
  --data-root Data-DARRA \
  --output-dir checkpoints/DARRA-IER \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 32 \
  --epochs 4
```

Default LoRA rank is **64** (alpha **128**); the effective global batch size is **64**. Actual memory use depends on the environment.

### Full fine-tuning: 8 × RTX 4090

**Full fine-tuning was carried out on 8 × RTX 4090 GPUs**

## Inference & evaluation

```bash
python run_infer_darra.py \
  --checkpoint checkpoints/DARRA-IER \
  --model-base ModelZoo/mplug-owl2-llama2-7b \
  --preprocessor-path preprocessor \
  --data-root Data-DARRA \
  --save-dir results/DARRA-IER --gpu-id 0

python -m src.evaluate.metrics \
  --pred-paths results/DARRA-IER/test_koniq_2k_new.jsonl
```

Evaluation metrics include **SRCC, PLCC, KRCC, MAE, and RMSE**. Ensure that ground-truth MOS and predictions use consistent score scales and directions.

## Citation

If you use this work, please cite:

> T. Wang, H. Luo, C. Yang, and H. Jia, “DARRA: A distortion-aware residual recalibration adapter for MLLM-based no-reference image quality assessment,” *Signal Processing*, vol. 251, art. 110859, 2027. https://doi.org/10.1016/j.sigpro.2026.110859


