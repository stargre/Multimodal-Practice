#!/usr/bin/env bash

set -euo pipefail


PROJECT=/mnt/data2/zhiyu/first-task/qwen2.5/multimodal_training


cd "$PROJECT"


export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5



torchrun \
--standalone \
--nproc_per_node=6 \
stage3lab/train_dpo_ddp.py \
\
--epochs 2 \
--batch_size 1 \
--grad_accum 4 \
--lr 5e-7 \
--beta 0.1 \
--log_every 20 \
--save_every 500 \
--max_steps 5200 \
--out runs/dpo_mmrlhf_v2_freeze_vit