"""Offline screening of mixed image + text-only SFT splits."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from stage1_runtime import load_stage1_components
from stage2lab.sft_data import (
    LLaVASFT, render_chat, tokenize_and_mask, visual_token_count, image_size_exif,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--json', required=True)
    p.add_argument('--image-root', required=True)
    p.add_argument('--split-dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max-tokens', type=int, default=4096)
    p.add_argument('--scan-limit', type=int, default=0, help='Debug only; 0 scans all')
    a = p.parse_args()
    ds = LLaVASFT(a.json, a.image_root)
    _, processor = load_stage1_components()
    tokenizer = processor.tokenizer
    if not getattr(tokenizer, 'is_fast', False):
        raise RuntimeError('Fast tokenizer with offset mapping is required')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    reports = {}
    for part in ('train', 'val'):
        source = Path(a.split_dir) / f'{part}_indices.json'
        if not source.is_file():
            raise FileNotFoundError(f'Missing split index: {source}')
        indices = json.loads(source.read_text(encoding='utf-8'))
        if a.scan_limit:
            indices = indices[:a.scan_limit]
        kept, drops, lengths, type_counts = [], Counter(), [], Counter()
        for at, idx in enumerate(indices, 1):
            try:
                sample = ds[idx]
                has_image = sample['sample_type'] == 'vision'
                raw, spans = render_chat(sample['conversations'], has_image=has_image)
                if has_image:
                    w, h = image_size_exif(sample['image_path'])
                    image_tokens = visual_token_count(w, h, processor.max_tiles_per_side)
                else:
                    image_tokens = None
                ids, labels = tokenize_and_mask(tokenizer, raw, spans, image_tokens)
                if len(ids) > a.max_tokens:
                    raise ValueError('too_long')
                if len(ids) < 2 or not any(token != -100 for token in labels):
                    raise ValueError('empty_targets')
                kept.append(idx)
                lengths.append(len(ids))
                type_counts[sample['sample_type']] += 1
            except Exception as exc:
                drops[str(exc).split(':')[0][:120] or type(exc).__name__] += 1
            if at % 10000 == 0:
                print(f'{part}: processed={at}/{len(indices)} kept={len(kept)}', flush=True)
        (out / f'{part}_indices.json').write_text(json.dumps(kept), encoding='utf-8')
        reports[part] = {
            'checked': len(indices), 'kept': len(kept), 'dropped': sum(drops.values()),
            'drop_reasons': dict(drops), 'min_length': min(lengths, default=0),
            'max_length': max(lengths, default=0), 'sample_type_counts': dict(type_counts),
        }
        print(part, reports[part], flush=True)
    train_ids = set(json.loads((out / 'train_indices.json').read_text()))
    val_ids = set(json.loads((out / 'val_indices.json').read_text()))
    if not train_ids or not val_ids or train_ids & val_ids:
        raise RuntimeError('Prepared train/val indices are empty or overlap')
    report = {
        'max_tokens': a.max_tokens, 'json': str(Path(a.json).resolve()),
        'image_root': str(Path(a.image_root).resolve()),
        'split_dir': str(Path(a.split_dir).resolve()), 'scan_limit': a.scan_limit,
        'tokenizer_len': len(tokenizer), 'train': reports['train'], 'val': reports['val'],
    }
    (out / 'prepare_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print('PASS: prepared mixed SFT indices at', out, flush=True)


if __name__ == '__main__':
    main()
