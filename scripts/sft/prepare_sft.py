"""Offline deterministic length/structure screening of existing image-disjoint SFT split.
Requires the ACTUAL tokenizer from stage1_runtime; uses image headers (no ViT forward).
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from stage1_runtime import load_stage1_components
from stage2lab.sft_data import LLaVASFT, render_chat, tokenize_and_mask, visual_token_count, image_size_exif


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--json', required=True)
    p.add_argument('--image-root', required=True)
    p.add_argument('--split-dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max-tokens', type=int, default=4096)
    p.add_argument('--scan-limit', type=int, default=0, help='Debug only; 0 = scan entire split')
    a = p.parse_args()
    ds = LLaVASFT(a.json, a.image_root)
    _, processor = load_stage1_components()
    tokenizer = processor.tokenizer
    if not getattr(tokenizer, 'is_fast', False):
        raise RuntimeError('Please use the Qwen3 fast tokenizer; offset mapping is required')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kept = {}
    reports = {}
    for part in ('train', 'val'):
        indices = json.loads((Path(a.split_dir) / f'{part}_indices.json').read_text())
        if a.scan_limit:
            indices = indices[:a.scan_limit]
        ok = []
        drops = Counter()
        lengths = []
        multi = []
        singles = []
        for at, idx in enumerate(indices, 1):
            try:
                item = ds[idx]
                raw, spans = render_chat(item['conversations'])
                w, h = image_size_exif(item['image_path'])
                ntok = visual_token_count(w, h, processor.max_tiles_per_side)
                ids, labels = tokenize_and_mask(tokenizer, raw, spans, ntok)
                if len(ids) > a.max_tokens:
                    raise ValueError('too_long')
                if len(ids) < 2 or not any(x != -100 for x in labels):
                    raise ValueError('empty_targets')
                ok.append(idx)
                lengths.append(len(ids))
                (multi if len(spans) > 1 else singles).append(idx)
            except Exception as exc:
                drops[str(exc).split(':')[0][:120]] += 1
            if at % 10000 == 0:
                print(part, 'processed', at, '/', len(indices), 'kept', len(ok), flush=True)
        (out / f'{part}_indices.json').write_text(json.dumps(ok), encoding='utf-8')
        reports[part] = {'checked':len(indices),'kept':len(ok), 'dropped':sum(drops.values()),
                         'drop_reasons':dict(drops),'min_length':min(lengths,default=0),
                         'max_length':max(lengths,default=0),
                         'multi_turn_kept':len(multi)}
        if part == 'train':
            # Repeated 8 records for overfit, balanced when multi-turn exists.
            debug = singles[:4] + multi[:4]
            if len(debug) < 8:
                debug = (debug + [x for x in ok if x not in debug])[:8]
            (out / 'debug8_indices.json').write_text(json.dumps(debug), encoding='utf-8')
        print(part, reports[part], flush=True)
    train_ids = set(json.loads((out / 'train_indices.json').read_text()))
    val_ids = set(json.loads((out / 'val_indices.json').read_text()))
    assert train_ids and val_ids and not (train_ids & val_ids)
    report = {'max_tokens':a.max_tokens,'json':str(Path(a.json).resolve()),
              'image_root':str(Path(a.image_root).resolve()),
              'split_dir':str(Path(a.split_dir).resolve()),'scan_limit':a.scan_limit,
              'tokenizer_len':len(tokenizer),'train':reports['train'],'val':reports['val']}
    (out / 'prepare_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PASS prepared indices:',out)

if __name__=='__main__':
    main()
