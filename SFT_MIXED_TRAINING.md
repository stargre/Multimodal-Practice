# Mixed SFT v1: vision + general text + math

This workflow keeps old SFT/DPO runs untouched. It initializes from the PT merger checkpoint,
original Qwen3-0.6B-Base, and frozen InternViT; it does not load old SFT/DPO weights.

## 1. Export text data to JSONL

The builder accepts JSON arrays or JSONL. It supports messages(role/content), conversations
(from/value), instruction/input/output, question/answer, query/response, problem/solution,
and prompt/completion records.

~~~bash
mkdir -p data_splits/source_json
python - <<'PY'
from datasets import load_dataset

ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
ds.to_json("data_splits/source_json/ultrachat_train.jsonl", orient="records", lines=True)

math_ds = load_dataset("meta-math/MetaMathQA", split="train")
math_ds.to_json("data_splits/source_json/metamathqa_train.jsonl", orient="records", lines=True)
PY
~~~

If a dataset ID or split differs in your installed datasets version, inspect available splits
and export the intended training split. Do not include evaluation records used for final reporting.

## 2. Build the mixed dataset

Defaults keep the existing image-disjoint LLaVA split and sample 44,900 text + 22,500 math
training records, nominally 70/20/10: 157,200 vision / 44,900 text / 22,500 math.

~~~bash
python scripts/sft/build_mixed_sft.py \
  --llava-json /mnt/data2/zhiyu/first-task/datasets/llava_stage2/sft/llava_instruct_150k.json \
  --llava-split-dir data_splits/sft_150k \
  --text-json data_splits/source_json/ultrachat_train.jsonl \
  --math-json data_splits/source_json/metamathqa_train.jsonl \
  --out-json data_splits/sft_mixed/sft_mixed.json \
  --out-split-dir data_splits/sft_mixed_raw \
  --text-train-count 44900 --text-val-count 1000 \
  --math-train-count 22500 --math-val-count 1000 --seed 42
~~~

The builder fails if either source lacks enough valid unique records; it does not silently shrink
the target counts. Review data_splits/sft_mixed/build_report.json.

## 3. Prepare all samples

~~~bash
python scripts/sft/prepare_sft.py \
  --json data_splits/sft_mixed/sft_mixed.json \
  --image-root /mnt/data3/public_datasets/coco/train2017 \
  --split-dir data_splits/sft_mixed_raw \
  --out data_splits/sft_mixed_4096 --max-tokens 4096
~~~

Check prepare_report.json. It must show the expected sample types, no unexpected drops, and
scan_limit equal to zero.

## 4. Smoke test

~~~bash
export PT_CKPT=/absolute/path/to/best_merger.pt
python scripts/sft/test_mixed_sft.py \
  --json data_splits/sft_mixed/sft_mixed.json \
  --image-root /mnt/data3/public_datasets/coco/train2017 \
  --indices data_splits/sft_mixed_4096/train_indices.json \
  --pt-checkpoint "$PT_CKPT" --max-tokens 4096
~~~

It runs one forward/backward pass per type, without an optimizer or checkpoint writes. All types
must pass before full training.

## 5. Launch six-GPU SFT

~~~bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export PT_CKPT=/absolute/path/to/best_merger.pt
export SFT_JSON=data_splits/sft_mixed/sft_mixed.json
export SFT_SPLIT=data_splits/sft_mixed_4096
export RUN_OUT=runs/sft_mixed_v1_6gpu
export GRAD_ACCUM=2
bash configs/sft_mixed_ddp.sh
~~~

The first run uses one epoch, LLM LR 1e-5, merger LR 3e-5, max length 4096, accumulation 2,
and a new output directory. Compare PT-only, old SFT, and new SFT on identical fixed prompts
for text reasoning, visual QA/OCR/counting, blank-image hallucination, and format validity.
Token loss alone is not a quality metric. Do not start new DPO training until the new SFT passes.
