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
--max_steps 20 \
--grad_accum 2 \
--log_every 5 \
--save_every 10 \
--out runs/dpo_smoke