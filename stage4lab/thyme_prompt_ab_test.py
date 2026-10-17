#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A/B prompt test for DPO-v3 on a Thyme-RL sample. Inference only; no training."""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='runs/dpo_mmrlhf_v3/last_model.pt')
    p.add_argument('--out', default='runs/thyme_prompt_ab_test')
    p.add_argument('--max_new_tokens', type=int, default=256)
    p.add_argument('--sample_repeats', type=int, default=3,
                   help='Sampled generations per prompt in addition to one greedy generation; 0 disables sampling.')
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--top_p', type=float, default=0.9)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--dataset_index', type=int, default=0)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items() if v is not None}


def build_prompt(processor, system_prompt, user_text):
    system_block = '<|im_start|>system\n' + system_prompt + '<|im_end|>\n'
    return system_block + processor.user_prompt(user_text, num_images=1)


def make_original_user_text(sample, image_path):
    text = re.sub(r'^\s*<image>\s*', '', sample['user_prompt'], count=1)
    w, h = sample['image'].size
    text = re.sub(r'(?m)(###\s*User Image Path:\*\*\s*")[^"]*(")',
                  lambda m: m.group(1) + str(image_path) + m.group(2), text)
    text = re.sub(r'(?m)(###\s*User Image Size:\*\*\s*")[^"]*(")',
                  lambda m: m.group(1) + f'{w}x{h}' + m.group(2), text)
    return text


def get_eos_ids(tokenizer):
    out = []
    eos = getattr(tokenizer, 'eos_token_id', None)
    if isinstance(eos, (tuple, list)):
        out.extend(int(x) for x in eos if x is not None)
    elif eos is not None:
        out.append(int(eos))
    try:
        im_end = tokenizer.convert_tokens_to_ids('<|im_end|>')
        unk = getattr(tokenizer, 'unk_token_id', None)
        if im_end is not None and im_end >= 0 and im_end != unk:
            out.append(int(im_end))
    except Exception:
        pass
    return sorted(set(out))


def trim_at_stop(ids, stop_ids, pad_id):
    ids = ids.tolist() if torch.is_tensor(ids) else list(ids)
    stop_set = set(stop_ids)
    for i, token in enumerate(ids):
        if token in stop_set or (pad_id is not None and int(token) == int(pad_id)):
            return ids[:i]
    return ids


def diagnostic_answer(text):
    m = re.search(r'<answer>\s*(.*?)\s*</answer>', text, flags=re.I | re.S)
    if m:
        return m.group(1).strip()
    nums = re.findall(r'[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*/\s*\d+)?', text)
    return nums[-1].strip() if nums else ''


@torch.inference_mode()
def generate_once(model, processor, image, prompt, device, max_new_tokens,
                  do_sample=False, temperature=0.7, top_p=0.9, seed=2026):
    set_seed(seed)
    batch = processor.encode_images([prompt], images_per_sample=[[image]], padding=True)
    batch = move_batch_to_device(batch, device)
    if 'input_ids' not in batch:
        raise RuntimeError('encode_images() did not return input_ids.')
    if 'pixel_values' not in batch:
        raise RuntimeError('encode_images() did not return pixel_values; image input is missing.')
    core = getattr(model, 'model', None)
    if core is not None and hasattr(core, 'rope_deltas'):
        core.rope_deltas = None

    tokenizer = processor.tokenizer
    eos_ids = get_eos_ids(tokenizer)
    pad_id = getattr(tokenizer, 'pad_token_id', None)
    gen = {'max_new_tokens': max_new_tokens, 'do_sample': do_sample, 'num_beams': 1}
    if pad_id is not None:
        gen['pad_token_id'] = int(pad_id)
    if eos_ids:
        gen['eos_token_id'] = eos_ids
    if do_sample:
        gen.update(temperature=temperature, top_p=top_p)

    t0 = time.time()
    output_ids = model.generate(**batch, **gen)
    elapsed = time.time() - t0
    input_len = batch['input_ids'].shape[1]
    response_ids = output_ids[0, input_len:] if output_ids.shape[1] >= input_len else output_ids[0]
    trimmed = trim_at_stop(response_ids, eos_ids, pad_id)
    text = tokenizer.decode(trimmed, skip_special_tokens=False,
                             clean_up_tokenization_spaces=False)
    return {'text': text, 'diagnostic_extracted_answer_only': diagnostic_answer(text),
            'generated_tokens_after_trim': len(trimmed), 'seconds': round(elapsed, 3),
            'do_sample': do_sample, 'seed': seed,
            'batch_shapes': {k: list(v.shape) for k, v in batch.items() if torch.is_tensor(v)}}


def main():
    args = parse_args()
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')

    print('=' * 88, flush=True)
    print('THYME PROMPT A/B TEST (INFERENCE ONLY)', flush=True)
    print('Project root:', PROJECT_ROOT, flush=True)
    print('Checkpoint:', checkpoint, flush=True)
    print('CUDA available:', torch.cuda.is_available(), flush=True)

    from stage4lab.thyme_dataset import ThymeRLStream
    from stage4lab.grpo_runtime import load_dpo_v2_policy

    sample = None
    for idx, item in enumerate(ThymeRLStream(split='train')):
        if idx == args.dataset_index:
            sample = item
            break
    if sample is None:
        raise IndexError(f'Cannot read Thyme dataset index {args.dataset_index}')

    image = sample['image'].convert('RGB')
    image_path = out_dir / f'sample_{args.dataset_index:06d}.png'
    image.save(image_path)
    print('Dataset index:', sample.get('index', args.dataset_index), flush=True)
    print('Question:', sample['question'], flush=True)
    print('Gold solution:', repr(sample['solution']), flush=True)
    print('Image size:', image.size, 'saved to:', image_path, flush=True)

    print('\nLoading DPO-v3 checkpoint...', flush=True)
    model, processor = load_dpo_v2_policy(checkpoint_path=str(checkpoint), freeze_visual=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    device = next(model.parameters()).device
    print('Model:', type(model).__name__, 'Processor:', type(processor).__name__, flush=True)
    print('Model device:', device, flush=True)

    # A: original Thyme system/user prompt (only stale image path/size are updated).
    user_a = make_original_user_text(sample, image_path)
    prompt_a = build_prompt(processor, sample['system_prompt'], user_a)

    # B: simplified diagnostic prompt. Image and question are unchanged.
    system_b = (
        'You are a careful visual math assistant. Read the image accurately. '
        'Identify all relevant values before calculating. Check the count of values and arithmetic. '
        'Give a concise explanation and clearly state the final numerical answer.'
    )
    user_b = (
        'Read every score in the table carefully. List all the scores you can see. '
        'Count the number of scores. Calculate their arithmetic mean. '
        f'Question: {sample["question"]}'
    )
    prompt_b = build_prompt(processor, system_b, user_b)

    print('\n--- PROMPT A: ORIGINAL THYME (tail) ---\n', prompt_a[-1600:], flush=True)
    print('\n--- PROMPT B: SIMPLIFIED (tail) ---\n', prompt_b[-1000:], flush=True)

    records = []
    for label, prompt in [('A_original', prompt_a), ('B_simplified', prompt_b)]:
        print('\n' + '#' * 88 + f'\n{label}: GREEDY', flush=True)
        result = generate_once(model, processor, image, prompt, device,
                               args.max_new_tokens, do_sample=False, seed=args.seed)
        record = {'prompt': label, 'mode': 'greedy', 'question': sample['question'],
                  'solution': sample['solution'], **result}
        records.append(record)
        print('Tensor shapes:', result['batch_shapes'], flush=True)
        print('Seconds:', result['seconds'], 'tokens:', result['generated_tokens_after_trim'], flush=True)
        print('Diagnostic extracted answer (not a reward):', repr(result['diagnostic_extracted_answer_only']), flush=True)
        print('Completion:\n' + result['text'], flush=True)

        for j in range(args.sample_repeats):
            seed = args.seed + 100 + j
            print(f'\n{label}: SAMPLED {j + 1}/{args.sample_repeats}, seed={seed}', flush=True)
            result = generate_once(model, processor, image, prompt, device,
                                   args.max_new_tokens, do_sample=True,
                                   temperature=args.temperature, top_p=args.top_p, seed=seed)
            records.append({'prompt': label, 'mode': 'sampled', 'question': sample['question'],
                            'solution': sample['solution'], **result})
            print('Seconds:', result['seconds'], 'tokens:', result['generated_tokens_after_trim'], flush=True)
            print('Diagnostic extracted answer (not a reward):', repr(result['diagnostic_extracted_answer_only']), flush=True)
            print('Completion:\n' + result['text'], flush=True)

    result_path = out_dir / 'ab_results.json'
    with result_path.open('w', encoding='utf-8') as f:
        json.dump({'checkpoint': str(checkpoint), 'dataset_index': args.dataset_index,
                   'question': sample['question'], 'solution': sample['solution'],
                   'image_path': str(image_path), 'max_new_tokens': args.max_new_tokens,
                   'temperature': args.temperature, 'top_p': args.top_p, 'records': records},
                  f, ensure_ascii=False, indent=2)
    print('\n' + '=' * 88, flush=True)
    print('A/B test finished. No optimizer step and no checkpoint update occurred.', flush=True)
    print('Results saved to:', result_path, flush=True)
    print('=' * 88, flush=True)


if __name__ == '__main__':
    main()
