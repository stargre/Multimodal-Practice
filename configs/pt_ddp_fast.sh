#!/usr/bin/env bash
set -euo pipefail
PROJECT=/mnt/data2/zhiyu/first-task/qwen2.5/multimodal_training
DATA_ROOT=/mnt/data2/zhiyu/first-task/datasets/llava_stage2
cd "$PROJECT"
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

# Choose visible GPU indices from your server's nvidia-smi output:
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_GPUS=$(python - <<'PY'
import os
ids = [x.strip() for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if x.strip()]
if not ids:
    raise SystemExit('CUDA_VISIBLE_DEVICES is empty')
print(len(ids))
PY
)

# GRAD_ACCUM is PER GPU; global batch = NUM_GPUS * GRAD_ACCUM.
GRAD_ACCUM="${GRAD_ACCUM:-4}"
RUN_OUT="${RUN_OUT:-runs/pt_558k_ddp}"

exec torchrun --standalone --nproc_per_node="$NUM_GPUS" \
  scripts/pt/train_pt_ddp_fast.py \
  --json "$DATA_ROOT/pretrain/blip_laion_cc_sbu_558k.json" \
  --root "$DATA_ROOT/pretrain" \
  --out "$RUN_OUT" \
  --epochs 1 \
  --grad-accum "$GRAD_ACCUM" \
  --lr 1e-4 \
  --warmup-ratio 0.03 \
  --val-size 128 \
  --max-tokens 3072 \
  --log-every 50 \
  --save-every 2500 \
  --eval-every 5000 \
  "$@"
