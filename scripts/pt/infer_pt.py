"""Qualitative real-image generation with a baseline or a trained PT merger.

Single image: --image PATH [--question ...]
Held-out examples: --json PATH --root DIR --val-size 128 --examples 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps

from pt_common import (LLaVAPT, fix_seed, load_base, load_merger_weights,
                       split_ids, to_gpu)


def generate_one(model, processor, image_file, question, max_new_tokens):
    with Image.open(image_file) as f:
        image = ImageOps.exif_transpose(f).convert('RGB')
    prefix = processor.encode_images(
        [processor.user_prompt(question, num_images=1)], [[image]]
    )
    length = int(prefix['input_ids'].shape[1])
    tokenizer = processor.tokenizer
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    eos_ids = [tokenizer.eos_token_id,
               tokenizer.convert_tokens_to_ids('<|im_end|>')]
    eos_ids = list(dict.fromkeys(int(x) for x in eos_ids if x is not None and x >= 0))
    model.model.rope_deltas = None
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        ids = model.generate(**to_gpu(prefix), max_new_tokens=max_new_tokens,
                             num_beams=1, do_sample=False, use_cache=True,
                             pad_token_id=pad_id, eos_token_id=eos_ids)
    suffix = ids[0, length:]
    return tokenizer.decode(suffix, skip_special_tokens=True).strip()


def main():
    p = argparse.ArgumentParser()
    sources = p.add_mutually_exclusive_group(required=True)
    sources.add_argument('--image', default=None)
    sources.add_argument('--json', default=None)
    p.add_argument('--root', default=None)
    p.add_argument('--question', default='Describe this image in detail.')
    p.add_argument('--checkpoint', default=None, help='best_merger.pt/latest_merger.pt')
    p.add_argument('--max-new-tokens', type=int, default=96)
    p.add_argument('--examples', type=int, default=8)
    p.add_argument('--val-size', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out-json', default=None)
    a = p.parse_args()
    if a.json and not a.root:
        p.error('--root is required with --json')
    if a.examples < 1 or a.max_new_tokens < 1:
        p.error('examples and max-new-tokens must be positive')
    fix_seed(a.seed)
    model, proc = load_base(train=False)
    if a.checkpoint:
        print('Loaded merger tensors:', load_merger_weights(model, a.checkpoint),
              'from', a.checkpoint)
    else:
        print('Baseline: stage-1 initial merger (not PT-trained)')
    if a.image:
        items = [{'image_path': a.image, 'question': a.question,
                  'answer': None, 'id': Path(a.image).stem}]
    else:
        data = LLaVAPT(a.json, a.root)
        _, heldout = split_ids(len(data), a.seed, a.val_size)
        items = [data[i] for i in heldout[:a.examples]]
    records = []
    for n, sample in enumerate(items, 1):
        pred = generate_one(model, proc, sample['image_path'],
                            sample['question'], a.max_new_tokens)
        record = {'id': sample['id'], 'image_path': sample['image_path'],
                  'question': sample['question'],
                  'reference': sample.get('answer'), 'prediction': pred}
        records.append(record)
        print(f'\n[{n}/{len(items)}] {record["id"]}\n'
              f'IMAGE: {record["image_path"]}\n'
              f'Q: {record["question"]}\n'
              f'GT: {record["reference"]}\n'
              f'PRED: {record["prediction"]}', flush=True)
    if a.out_json:
        dst = Path(a.out_json)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')
        print('Saved:', dst)
    print('PASS: inference completed')


if __name__ == '__main__':
    main()
