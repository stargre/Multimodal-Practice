#!/usr/bin/env bash
set -euo pipefail
PROJECT=/mnt/data2/zhiyu/first-task/qwen2.5/multimodal_training
DATA_ROOT=/mnt/data2/zhiyu/first-task/datasets/llava_stage2
cd "$PROJECT"
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
VISIBLE="${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES, e.g. 2,3,4,5}"
IFS=',' read -r -a GPUS <<< "$VISIBLE"
WORLD=${#GPUS[@]}
if (( WORLD < 1 )); then echo 'No GPUs'; exit 1; fi
OUT="${RUN_OUT:-runs/sft_150k_ddp_${WORLD}gpu}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
SPLIT="${SFT_SPLIT:-data_splits/sft_150k_filtered_4096}"
PT="${PT_CKPT:-runs/pt_558k_ddp_fast_4gpu/best_merger.pt}"
exec torchrun --standalone --nproc_per_node "$WORLD" \
  scripts/sft/train_sft_ddp.py \
  --json "$DATA_ROOT/sft/llava_instruct_150k.json" \
  --image-root /mnt/data3/public_datasets/coco/train2017 \
  --train-indices "$SPLIT/train_indices.json" \
  --val-indices "$SPLIT/val_indices.json" \
  --pt-checkpoint "$PT" \
  --out "$OUT" \
  --epochs 1 \
  --grad-accum "$GRAD_ACCUM" \
  --max-tokens 4096 \
  --lr-llm 2e-5 \
  --lr-merger 5e-5 \
  --warmup-ratio 0.03 \
  --save-every 2000 \
  --eval-every 2000 \
  --log-every 100 \
  "$@"
