#!/usr/bin/env bash
set -euo pipefail
PROJECT=/mnt/data2/zhiyu/first-task/qwen2.5/multimodal_training
cd "$PROJECT"
export PYTHONPATH="$PROJECT:$(printenv PYTHONPATH 2>/dev/null || true)"
export TOKENIZERS_PARALLELISM=false

VISIBLE="$(printenv CUDA_VISIBLE_DEVICES || true)"
if [[ -z "$VISIBLE" ]]; then
  echo "Set CUDA_VISIBLE_DEVICES, e.g. 0,1,2,3,4,5" >&2
  exit 1
fi
WORLD="$(echo "$VISIBLE" | awk -F, '{print NF}')"
if (( WORLD < 1 )); then
  echo "No GPUs selected" >&2
  exit 1
fi

SFT_JSON="$(printenv SFT_JSON || echo data_splits/sft_mixed/sft_mixed.json)"
SPLIT="$(printenv SFT_SPLIT || echo data_splits/sft_mixed_4096)"
PT="$(printenv PT_CKPT || echo runs/pt_558k_ddp_fast_4gpu/best_merger.pt)"
OUT="$(printenv RUN_OUT || printf 'runs/sft_mixed_v1_%sgpu' "$WORLD")"
IMAGE_ROOT="$(printenv IMAGE_ROOT || echo /mnt/data3/public_datasets/coco/train2017)"
GRAD_ACCUM="$(printenv GRAD_ACCUM || echo 2)"

if [[ ! -f "$SFT_JSON" ]]; then
  echo "Missing mixed SFT JSON: $SFT_JSON" >&2
  exit 1
fi
if [[ ! -f "$SPLIT/train_indices.json" || ! -f "$SPLIT/val_indices.json" ]]; then
  echo "Missing prepared indices under $SPLIT; run build_mixed_sft.py and prepare_sft.py first." >&2
  exit 1
fi
if [[ ! -f "$SPLIT/prepare_report.json" ]]; then
  echo "Missing $SPLIT/prepare_report.json; run prepare_sft.py first." >&2
  exit 1
fi
if [[ ! -f "$PT" ]]; then
  echo "Missing PT merger checkpoint: $PT" >&2
  exit 1
fi

exec torchrun --standalone --nproc_per_node "$WORLD" \
  scripts/sft/train_sft_ddp.py \
  --json "$SFT_JSON" \
  --image-root "$IMAGE_ROOT" \
  --train-indices "$SPLIT/train_indices.json" \
  --val-indices "$SPLIT/val_indices.json" \
  --pt-checkpoint "$PT" \
  --out "$OUT" \
  --epochs 1 \
  --grad-accum "$GRAD_ACCUM" \
  --max-tokens 4096 \
  --lr-llm 1e-5 \
  --lr-merger 3e-5 \
  --warmup-ratio 0.03 \
  --min-lr-ratio 0.1 \
  --weight-decay 0.01 \
  --save-every 2000 \
  --eval-every 2000 \
  --log-every 100 \
  "$@"
