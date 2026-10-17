"""Backward-compatible SFT dataset/collator for image and text-only samples."""
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
        if not isinstance(item, dict):
            raise ValueError(f'Invalid SFT record at index {idx}')
        kind = item.get('sample_type')
        if kind is None:
            kind = 'vision' if item.get('image') else 'text'
        if kind not in ('vision', 'text', 'math'):
            raise ValueError(f'Unsupported sample_type={kind!r} at index {idx}')
        image_path = None
        if kind == 'vision':
            filename = Path(str(item.get('image') or '')).name
            if filename in ('', '.', '..'):
                raise ValueError(f'Invalid image path at index {idx}')
            image_path = str(self.root / filename)
        elif item.get('image'):
            raise ValueError(f'Non-vision sample {idx} unexpectedly contains image field')
        conversations = item.get('conversations')
        if not isinstance(conversations, list) or not conversations:
            raise ValueError(f'Missing conversations at index {idx}')
        return {
            'id': str(item.get('id', idx)), 'sample_type': kind,
            'source': str(item.get('source', kind)), 'image_path': image_path,
            'conversations': conversations, 'dataset_index': int(idx),
        }


def render_chat(conversations, generation_prompt=False, has_image=True):
    """Render Qwen ChatML and assistant character spans; text-only never gets image tokens."""
    if not isinstance(conversations, list) or not conversations:
        raise ValueError('Nonempty conversations list required')
    content = []
    for i, turn in enumerate(conversations):
        role, value = turn.get('from'), turn.get('value')
        if role not in ('human', 'gpt') or not isinstance(value, str) or not value.strip():
            raise ValueError(f'Bad turn {i}: {role!r}')
        expected = 'human' if i % 2 == 0 else 'gpt'
        if role != expected:
            raise ValueError(f'Expected alternating human/gpt turns, got {role} at turn {i}')
        content.append(value.strip())
    if len(content) % 2 and not generation_prompt:
        raise ValueError('Training dialogue must end with assistant reply')
    markers = sum(text.count('<image>') for text in content)
    if has_image:
        if markers > 1:
            raise ValueError('Only one image marker is supported')
        if markers == 0:
            content[0] = '<image>\n' + content[0]
        if any('<image>' in content[i] for i in range(1, len(content))):
            raise ValueError('Image marker must occur in first user turn')
    elif markers:
        raise ValueError('Text-only sample contains an <image> marker')
    forbidden = ('<|im_start|>', '<|im_end|>', '<|image_pad|>',
                 '<|vision_start|>', '<|vision_end|>')
    if any(token in text for text in content for token in forbidden):
        raise ValueError('User-supplied special ChatML/vision token found')
    if has_image:
        content[0] = content[0].replace('<image>', VISION, 1)
    pieces, spans, pos = [], [], 0
    for i, value in enumerate(content):
        role = 'user' if i % 2 == 0 else 'assistant'
        header = f'<|im_start|>{role}\n'
        pieces.append(header)
        pos += len(header)
        if role == 'assistant':
            spans.append((pos, pos + len(value) + len(EOS)))
        part = value + EOS + '\n'
        pieces.append(part)
        pos += len(part)
    if generation_prompt:
        if len(content) % 2 != 1:
            raise ValueError('Generation requires a final user turn')
        pieces.append('<|im_start|>assistant\n')
    raw = ''.join(pieces)
    if has_image and (raw.count(IMAGE) != 1 or raw.count('<|vision_start|>') != 1):
        raise ValueError('Exactly one logical image is expected')
    if not has_image and (IMAGE in raw or '<|vision_start|>' in raw):
        raise ValueError('Text-only rendering unexpectedly contains vision tokens')
    return raw, spans


def expanded_chat(raw, spans, image_tokens):
    if image_tokens < 1 or raw.count(IMAGE) != 1:
        raise ValueError('Invalid image placeholder expansion')
    start = raw.index(IMAGE)
    added = (image_tokens - 1) * len(IMAGE)
    expanded = raw.replace(IMAGE, IMAGE * image_tokens, 1)
    adjusted = []
    for begin, end in spans:
        if begin <= start:
            raise ValueError('Assistant span occurs before image reference')
        adjusted.append((begin + added, end + added))
    return expanded, adjusted


def tokenize_and_mask(tokenizer, raw, spans, image_tokens=None):
    if not getattr(tokenizer, 'is_fast', False):
        raise RuntimeError('A fast tokenizer with offset mapping is required')
    if image_tokens is None:
        if IMAGE in raw or '<|vision_start|>' in raw:
            raise ValueError('Text-only tokenization received vision tokens')
        expanded, target_spans = raw, spans
    else:
        expanded, target_spans = expanded_chat(raw, spans, int(image_tokens))
    encoded = tokenizer(expanded, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded['input_ids'], encoded['offset_mapping']
    if not ids or len(ids) != len(offsets):
        raise ValueError('Invalid tokenizer IDs/offsets')
    labels = [-100] * len(ids)
    for k, ((start, end), token_id) in enumerate(zip(offsets, ids)):
        if start == end:
            continue
        overlapping = [(a, b) for a, b in target_spans if start < b and end > a]
        if overlapping:
            if not any(start >= a and end <= b for a, b in overlapping):
                raise ValueError(f'Token crosses assistant supervision boundary at token {k}')
            labels[k] = token_id
    eos_id = tokenizer.convert_tokens_to_ids(EOS)
    supervised_eos = sum(int(tok == eos_id) for tok, lab in zip(ids, labels) if lab != -100)
    if supervised_eos != len(target_spans):
        raise ValueError(f'Expected {len(target_spans)} supervised EOS, got {supervised_eos}')
    if sum(label != -100 for label in labels) <= len(target_spans):
        raise ValueError('No assistant content is supervised')
    if image_tokens is not None:
        image_id = tokenizer.convert_tokens_to_ids(IMAGE)
        if sum(tok == image_id for tok in ids) != int(image_tokens):
            raise ValueError('Visual token count mismatch after tokenization')
        if any(lab != -100 for tok, lab in zip(ids, labels) if tok == image_id):
            raise AssertionError('Image token accidentally supervised')
    return ids, labels


def visual_token_count(width, height, max_tiles_per_side=2):
    rows = min(max_tiles_per_side, max(1, math.ceil(height / 448)))
    cols = min(max_tiles_per_side, max(1, math.ceil(width / 448)))
    return 256 * rows * cols


def image_size_exif(path):
    with Image.open(path) as image:
        width, height = image.size
        if image.getexif().get(274, 1) in (5, 6, 7, 8):
            return height, width
        return width, height


class SFTCollator:
    def __init__(self, processor, max_tokens=4096):
        self.processor = processor
        self.max_tokens = int(max_tokens)

    def __call__(self, samples):
        if len(samples) != 1:
            raise ValueError('Collator supports batch_size=1; use DDP and accumulation')
        sample = samples[0]
        kind = sample.get('sample_type')
        has_image = kind == 'vision' or (kind is None and bool(sample.get('image_path')))
        if has_image and not sample.get('image_path'):
            raise ValueError(f'Vision sample {sample.get("id")} has no image path')
        raw, spans = render_chat(sample['conversations'], has_image=has_image)
        if has_image:
            with Image.open(sample['image_path']) as handle:
                image = ImageOps.exif_transpose(handle).convert('RGB')
            batch = self.processor.encode_images([raw], [[image]], padding=False)
            grid = batch.get('image_grid_thw')
            if grid is None:
                raise ValueError('Vision processor did not return image_grid_thw')
            n_image_tokens = int((grid.prod(dim=-1) // 4).sum().item())
            ids, labels = tokenize_and_mask(
                self.processor.tokenizer, raw, spans, n_image_tokens
            )
            if ids != batch['input_ids'][0].tolist():
                raise ValueError('Tokenizer offset pass disagrees with processor token IDs')
        else:
            ids, labels = tokenize_and_mask(
                self.processor.tokenizer, raw, spans, image_tokens=None
            )
            input_ids = torch.tensor([ids], dtype=torch.long)
            batch = {'input_ids': input_ids,
                     'attention_mask': torch.ones_like(input_ids, dtype=torch.long)}
        if len(ids) > self.max_tokens:
            raise ValueError(
                f'SFT item {sample["id"]} has {len(ids)} tokens > {self.max_tokens}; '
                'run prepare_sft.py'
            )
        if 'attention_mask' in batch and (batch['attention_mask'] != 1).any():
            raise ValueError('Unexpected padding with microbatch=1')
        batch['labels'] = torch.tensor([labels], dtype=torch.long)
        if not (batch['labels'] != -100).any():
            raise ValueError('No assistant tokens are supervised')
        return batch
