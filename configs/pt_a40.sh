#!/usr/bin/env bash
set -euo pipefail
PROJECT=/mnt/data2/zhiyu/first-task/qwen2.5/multimodal_training
DATA_ROOT=/mnt/data2/zhiyu/first-task/datasets/llava_stage2
cd "$PROJECT"
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
python scripts/pt/train_pt.py \
  --json "$DATA_ROOT/pretrain/blip_laion_cc_sbu_558k.json" \
  --root "$DATA_ROOT/pretrain" \
  --out logs/pt_558k_a40 \
  --epochs 1 \
  --grad-accum 8 \
  --lr 1e-4 \
  --warmup-ratio 0.03 \
  --val-size 128 \
  --max-tokens 3072 \
  --log-every 50 \
  --save-every 2500 \
  --eval-every 5000 \
  "$@"
