#!/usr/bin/env bash

set -euo pipefail


PROJECT=/mnt/data2/zhiyu/first-task/qwen2.5/multimodal_training


cd "$PROJECT"


export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"



CUDA_VISIBLE_DEVICES=2,3 \
torchrun \
    --standalone \
    --nproc_per_node=2 \
    scripts/dpo/train_dpo_ddp.py \
    --train-json data_splits/mmrlhf_dpo/train_1000.json \
    --val-json data_splits/mmrlhf_dpo/val.json \
    --sft-checkpoint runs/sft_150k_ddp_6gpu/best_model.pt \
    --out runs/dpo_smoke-1000 \
    --epochs 1 \
    --lr 5e-7 \
    --beta 0.1 \
    --grad-accum 4 \
    --log-every 5 \
    --eval-every 50 \
    --save-every 100