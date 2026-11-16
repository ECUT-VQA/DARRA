#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_GPUS="${NUM_GPUS:-2}"
python -m torch.distributed.run --standalone --nproc_per_node="$NUM_GPUS" \
  -m src.train.train_darra "$@"
