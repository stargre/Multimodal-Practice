"""Build mixed SFT JSON from LLaVA, general text, and math JSON/JSONL sources."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
from collections import Counter


def read_records(path):
    text = Path(path).read_text(encoding='utf-8')
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            for key in ('data', 'train', 'records', 'examples'):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
        if not isinstance(data, list):
            raise ValueError(f'{path} must contain a JSON array or JSONL records')
        return data
    except json.JSONDecodeError:
        rows = []
        for line_no, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSONL at {path}:{line_no}: {exc}') from exc
        if not rows:
            raise ValueError(f'No records found in {path}')
        return rows


def as_text(value):
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    return ''


def map_role(role):
    role = str(role or '').strip().lower()
    if role in ('human', 'user', 'question', 'prompter'):
        return 'human'
    if role in ('gpt', 'assistant', 'answer', 'bot'):
        return 'gpt'
    if role == 'system':
        return 'system'
    return None


def normalize_turns(raw_turns):
    mapped, system_parts = [], []
    for turn in raw_turns:
        if not isinstance(turn, dict):
            return None
        role = map_role(turn.get('from', turn.get('role', turn.get('speaker'))))
        value = as_text(turn.get('value', turn.get('content', turn.get('text'))))
        if role is None or not value:
            return None
        if role == 'system':
            system_parts.append(value)
            continue
        if any(token in value for token in (
            '<image>', '<|image_pad|>', '<|vision_start|>', '<|vision_end|>',
            '<|im_start|>', '<|im_end|>'
        )):
            return None
        if mapped and mapped[-1]['from'] == role:
            mapped[-1]['value'] += '\n\n' + value
        else:
            mapped.append({'from': role, 'value': value})
    if not mapped or mapped[0]['from'] != 'human' or mapped[-1]['from'] != 'gpt':
        return None
    if len(mapped) % 2 or any(
        turn['from'] != ('human' if i % 2 == 0 else 'gpt')
        for i, turn in enumerate(mapped)
    ):
        return None
    if system_parts:
        mapped[0]['value'] = (
            'System instruction:\n' + '\n'.join(system_parts) +
            '\n\nUser request:\n' + mapped[0]['value']
        )
    return mapped


def normalize_text_record(record):
    if not isinstance(record, dict):
        return None
    if isinstance(record.get('conversations'), list):
        turns = normalize_turns(record['conversations'])
    elif isinstance(record.get('messages'), list):
        turns = normalize_turns(record['messages'])
    else:
        user = answer = ''
        if record.get('instruction') is not None and record.get('output') is not None:
            user = as_text(record.get('instruction'))
            extra = as_text(record.get('input'))
            if extra:
                user += '\n\nInput:\n' + extra
            answer = as_text(record.get('output'))
        else:
            for uk, ak in (
                ('question', 'answer'), ('query', 'response'), ('problem', 'solution'),
                ('prompt', 'completion'), ('prompt', 'response'),
                ('instruction', 'response'),
            ):
                if record.get(uk) is not None and record.get(ak) is not None:
                    user, answer = as_text(record[uk]), as_text(record[ak])
                    break
        if not user or not answer:
            return None
        turns = normalize_turns([
            {'from': 'human', 'value': user}, {'from': 'gpt', 'value': answer}
        ])
    return turns


def conversation_key(turns):
    payload = json.dumps(turns, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def normalized_pool(path, sample_type, blocked_keys=None):
    records = read_records(path)
    pool, seen, dropped = [], set(), Counter()
    blocked_keys = blocked_keys or set()
    for idx, record in enumerate(records):
        turns = normalize_text_record(record)
        if not turns:
            dropped['unsupported_or_invalid_schema'] += 1
            continue
        key = conversation_key(turns)
        if key in seen or key in blocked_keys:
            dropped['duplicate'] += 1
            continue
        seen.add(key)
        pool.append({
            'id': str(record.get('id', f'{sample_type}-{idx}')),
            'sample_type': sample_type, 'source': str(record.get('source', sample_type)),
            'conversations': turns, '_dedup_key': key,
        })
    return pool, dropped, len(records)


def take_split(pool, train_count, val_count, seed, label):
    needed = train_count + val_count
    if len(pool) < needed:
        raise ValueError(
            f'{label} has only {len(pool)} valid unique records; need {needed} '
            f'(train={train_count}, val={val_count}). Provide more data or lower counts.'
        )
    pool = list(pool)
    random.Random(seed).shuffle(pool)
    train, val = pool[:train_count], pool[train_count:needed]
    for row in train + val:
        row.pop('_dedup_key', None)
    return train, val


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--llava-json', required=True)
    p.add_argument('--llava-split-dir', required=True)
    p.add_argument('--text-json', required=True)
    p.add_argument('--math-json', required=True)
    p.add_argument('--out-json', required=True)
    p.add_argument('--out-split-dir', required=True)
    p.add_argument('--text-train-count', type=int, default=44900)
    p.add_argument('--text-val-count', type=int, default=1000)
    p.add_argument('--math-train-count', type=int, default=22500)
    p.add_argument('--math-val-count', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()

    llava = read_records(a.llava_json)
    split = Path(a.llava_split_dir)
    vision_train_ids = json.loads((split / 'train_indices.json').read_text())
    vision_val_ids = json.loads((split / 'val_indices.json').read_text())
    if set(vision_train_ids) & set(vision_val_ids):
        raise RuntimeError('LLaVA train and validation indices overlap')
    if not vision_train_ids or not vision_val_ids:
        raise RuntimeError('LLaVA train/validation split must be nonempty')
    all_ids = vision_train_ids + vision_val_ids
    if min(all_ids) < 0 or max(all_ids) >= len(llava):
        raise IndexError('LLaVA split index is outside source JSON')

    def vision_rows(indices, part):
        rows = []
        for idx in indices:
            item = llava[idx]
            if not isinstance(item, dict) or not item.get('image'):
                raise ValueError(f'LLaVA {part} record {idx} has no image')
            conv = normalize_turns(item.get('conversations', []))
            if not conv:
                raise ValueError(f'LLaVA {part} record {idx} has invalid conversations')
            rows.append({
                'id': str(item.get('id', f'llava-{idx}')), 'sample_type': 'vision',
                'source': 'llava_instruct_150k', 'image': str(item['image']),
                'conversations': conv,
            })
        return rows

    vtrain = vision_rows(vision_train_ids, 'train')
    vval = vision_rows(vision_val_ids, 'val')
    text_pool, text_dropped, text_total = normalized_pool(a.text_json, 'text')
    text_keys = {row['_dedup_key'] for row in text_pool}
    math_pool, math_dropped, math_total = normalized_pool(
        a.math_json, 'math', blocked_keys=text_keys
    )
    ttrain, tval = take_split(
        text_pool, a.text_train_count, a.text_val_count, a.seed + 1, 'Text source'
    )
    mtrain, mval = take_split(
        math_pool, a.math_train_count, a.math_val_count, a.seed + 2, 'Math source'
    )

    records = vtrain + ttrain + mtrain + vval + tval + mval
    ntrain = len(vtrain) + len(ttrain) + len(mtrain)
    train_indices = list(range(ntrain))
    val_indices = list(range(ntrain, len(records)))
    out_json = Path(a.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(records, ensure_ascii=False, separators=(',', ':')),
                        encoding='utf-8')
    out_split = Path(a.out_split_dir)
    out_split.mkdir(parents=True, exist_ok=True)
    (out_split / 'train_indices.json').write_text(json.dumps(train_indices), encoding='utf-8')
    (out_split / 'val_indices.json').write_text(json.dumps(val_indices), encoding='utf-8')
    report = {
        'seed': a.seed,
        'source_files': {'llava': str(Path(a.llava_json).resolve()),
                         'text': str(Path(a.text_json).resolve()),
                         'math': str(Path(a.math_json).resolve())},
        'source_counts': {
            'llava_records': len(llava), 'text_records_read': text_total,
            'text_records_dropped': dict(text_dropped), 'math_records_read': math_total,
            'math_records_dropped': dict(math_dropped)},
        'counts': {
            'train': {'vision': len(vtrain), 'text': len(ttrain), 'math': len(mtrain),
                      'total': len(train_indices)},
            'val': {'vision': len(vval), 'text': len(tval), 'math': len(mval),
                    'total': len(val_indices)},
            'all_records': len(records)},
        'output_json': str(out_json.resolve()), 'split_dir': str(out_split.resolve()),
        'note': 'Deduplicated within source; math rows duplicating text rows are excluded.',
    }
    (out_json.with_name('build_report.json')).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
