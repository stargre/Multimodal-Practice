"""SFT dataset & exact chat token boundaries for user's HybridQwen25Processor.

One logical image, BS=1. Fast tokenizer offset_mapping is REQUIRED: it avoids
accidentally labeling user/header tokens when byte-pair boundaries shift.
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import torch
from PIL import Image, ImageOps

IMAGE = '<|image_pad|>'
VISION = '<|vision_start|>' + IMAGE + '<|vision_end|>'
EOS = '<|im_end|>'


class LLaVASFT:
    def __init__(self, json_file, image_root):
        self.items = json.loads(Path(json_file).read_text(encoding='utf-8'))
        if not isinstance(self.items, list) or not self.items:
            raise ValueError('SFT annotations must be a nonempty JSON list')
        self.root = Path(image_root)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[int(idx)]
        filename = Path(str(item['image'])).name
        if filename in ('', '.', '..'):
            raise ValueError(f'Invalid image path at index {idx}')
        return {'id': str(item.get('id', idx)),
                'image_path': str(self.root / filename),
                'conversations': item['conversations'], 'dataset_index': int(idx)}


def render_chat(conversations, generation_prompt=False):
    """Render exact Qwen ChatML and supervision char spans; NO image expansion yet.

    Only one <image> marker is allowed, or insert it into first user turn.
    All response text plus EOS token is supervised, no user/header/image labels.
    """
    if not isinstance(conversations, list) or not conversations:
        raise ValueError('Nonempty conversations list required')
    content = []
    for i, t in enumerate(conversations):
        role = t.get('from')
        value = t.get('value')
        if role not in ('human', 'gpt') or not isinstance(value, str) or not value.strip():
            raise ValueError(f'Bad turn {i}: {role!r}')
        if role != ('human' if i % 2 == 0 else 'gpt'):
            raise ValueError(f'Expected alternating human/gpt turns, got {role} at {i}')
        content.append(value.strip())
    if len(content) % 2 and not generation_prompt:
        raise ValueError('Training dialogue must end with assistant reply')
    markers = sum(x.count('<image>') for x in content)
    if markers > 1:
        raise ValueError('Only one image reference is supported by this SFT recipe')
    if markers == 0:
        content[0] = '<image>\n' + content[0]
    if any('<image>' in content[i] for i in range(1, len(content))):
        raise ValueError('Image marker must occur in first user turn for this recipe')
    if any(x in c for c in content for x in ('<|im_start|>', '<|im_end|>', '<|image_pad|>')):
        raise ValueError('User-supplied special ChatML token found; reject to avoid label leakage')
    content[0] = content[0].replace('<image>', VISION, 1)
    pieces = []
    spans = []
    pos = 0
    for i, val in enumerate(content):
        role = 'user' if i % 2 == 0 else 'assistant'
        header = f'<|im_start|>{role}\n'
        pieces.append(header)
        pos += len(header)
        if role == 'assistant':
            spans.append((pos, pos + len(val) + len(EOS)))
        part = val + EOS + '\n'
        pieces.append(part)
        pos += len(part)
    if generation_prompt:
        if len(content) % 2 != 1:
            raise ValueError('Generation requires ending user turn')
        pieces.append('<|im_start|>assistant\n')
    raw = ''.join(pieces)
    if raw.count(IMAGE) != 1 or raw.count('<|vision_start|>') != 1:
        raise ValueError('Exactly one logical image expected')
    return raw, spans


def expanded_chat(raw, spans, image_tokens):
    if image_tokens < 1 or raw.count(IMAGE) != 1:
        raise ValueError('Invalid image placeholder expansion')
    start = raw.index(IMAGE)
    added = (image_tokens - 1) * len(IMAGE)
    text = raw.replace(IMAGE, IMAGE * image_tokens, 1)
    # All assistant spans must be AFTER the initial image reference.
    adjusted = []
    for a, b in spans:
        if a <= start:
            raise ValueError('Assistant span occurs before image; unsupported')
        adjusted.append((a + added, b + added))
    return text, adjusted


def tokenize_and_mask(tokenizer, raw, spans, image_tokens):
    if not getattr(tokenizer, 'is_fast', False):
        raise RuntimeError('A fast tokenizer with return_offsets_mapping is required')
    expanded, target_spans = expanded_chat(raw, spans, image_tokens)
    enc = tokenizer(expanded, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc['input_ids']
    offsets = enc['offset_mapping']
    if not ids or len(ids) != len(offsets):
        raise ValueError('Invalid tokenizer IDs/offsets')
    labels = [-100] * len(ids)
    for k, ((start, end), tok) in enumerate(zip(offsets, ids)):
        if start == end:
            continue
        overlapping = [(a, b) for a, b in target_spans if start < b and end > a]
        if overlapping:
            if not any(start >= a and end <= b for a, b in overlapping):
                raise ValueError(f'Token crosses assistant supervision boundary at token {k}: {start}:{end}')
            labels[k] = tok
    end_id = tokenizer.convert_tokens_to_ids(EOS)
    supervised_end = sum(int(tok == end_id) for tok in labels if tok != -100)
    if supervised_end != len(target_spans):
        raise ValueError(f'Expected {len(target_spans)} supervised EOS, got {supervised_end}')
    if sum(x != -100 for x in labels) <= len(target_spans):
        raise ValueError('No assistant content supervised')
    img_id = tokenizer.convert_tokens_to_ids(IMAGE)
    if sum(x == img_id for x in ids) != image_tokens:
        raise ValueError('Visual token mismatch after tokenization')
    if any(lab != -100 for tok, lab in zip(ids, labels) if tok == img_id):
        raise AssertionError('Image token accidentally supervised')
    return ids, labels


def visual_token_count(width, height, max_tiles_per_side=2):
    rows = min(max_tiles_per_side, max(1, math.ceil(height / 448)))
    cols = min(max_tiles_per_side, max(1, math.ceil(width / 448)))
    return 256 * rows * cols


def image_size_exif(path):
    with Image.open(path) as img:
        width, height = img.size
        orientation = img.getexif().get(274, 1)
        if orientation in (5, 6, 7, 8):
            return height, width
        return width, height


class SFTCollator:
    def __init__(self, processor, max_tokens=4096):
        self.processor = processor
        self.max_tokens = int(max_tokens)

    def __call__(self, samples):
        if len(samples) != 1:
            raise ValueError('SFT collator currently supports batch_size=1; use DDP and accumulation')
        sample = samples[0]
        raw, spans = render_chat(sample['conversations'])
        with Image.open(sample['image_path']) as handle:
            image = ImageOps.exif_transpose(handle).convert('RGB')
        batch = self.processor.encode_images([raw], [[image]], padding=False)
        grid = batch['image_grid_thw']
        expected_image_tokens = int((grid.prod(dim=-1) // 4).sum().item())
        ids, labels = tokenize_and_mask(self.processor.tokenizer, raw, spans, expected_image_tokens)
        actual_ids = batch['input_ids'][0].tolist()
        if ids != actual_ids:
            raise ValueError('Fast-tokenizer offset pass disagrees with processor.encode_images()')
        if len(ids) > self.max_tokens:
            raise ValueError(f'SFT item {sample["id"]} has {len(ids)} tokens > max {self.max_tokens}; run prepare_sft.py')
        if (batch['attention_mask'] != 1).any():
            raise ValueError('Unexpected padding with microbatch=1')
        batch['labels'] = torch.tensor([labels], dtype=torch.long)
        if not (batch['labels'] != -100).any():
            raise ValueError('No assistant supervised tokens')
        return batch
