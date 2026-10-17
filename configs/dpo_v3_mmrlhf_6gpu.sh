#!/usr/bin/env bash

set -euo pipefail

PROJECT=/mnt/data2/zhiyu/first-task/qwen2.5/multimodal_training

cd "$PROJECT"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"

export TOKENIZERS_PARALLELISM=false

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

# Keep expandable segments enabled to reduce allocator fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT_DIR="$PROJECT/runs/dpo_mmrlhf_v3"
RESUME_CKPT="$OUT_DIR/resume_latest.pt"

mkdir -p "$OUT_DIR"

# Automatic resume:
# - first run: no resume checkpoint -> start from SFT
# - later run after interruption/OOM: resume_latest.pt is loaded
RESUME_ARGS=()
if [[ -f "$RESUME_CKPT" ]]; then
    echo "============================================================"
    echo "AUTO-RESUME: $RESUME_CKPT"
    echo "============================================================"
    RESUME_ARGS+=(--resume "$RESUME_CKPT")
else
    echo "============================================================"
    echo "FRESH DPO-v3 RUN: starting from SFT"
    echo "============================================================"
fi

torchrun \
    --standalone \
    --nproc_per_node=6 \
    stage3lab/train_dpo_ddp_v3.py \
    --data \
        "$PROJECT/data_splits/mmrlhf_dpo/train.json" \
    --val_data \
        "$PROJECT/data_splits/mmrlhf_dpo/val.json" \
    --image_root \
        "/mnt/data2/zhiyu/first-task/datasets/MMRLHF_ROOT" \
    --epochs 1 \
    --batch_size 1 \
    --grad_accum 4 \
    --lr 5e-7 \
    --merger_lr 1e-6 \
    --init-policy "" \
    --beta 0.1 \
    --anchor_weight 0.25 \
    --anchor_delta 0.0 \
    --weight_decay 0.0 \
    --warmup_ratio 0.1 \
    --max_grad_norm 1.0 \
    --max_steps 2232 \
    --log_every 20 \
    --eval_every 500 \
    --eval_limit 256 \
    --save_every 50 \
    --num_workers 4 \
    --max_tokens 0 \
    --seed 42 \
    --out \
        "$OUT_DIR" \
    "${RESUME_ARGS[@]}"
