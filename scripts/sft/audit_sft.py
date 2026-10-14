"""Audit LLaVA-Instruct-150K JSON and create image-disjoint SFT train/val index files.

No model loading, no writes to original JSON or COCO image directory.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import statistics


def quantile(values, q):
    return sorted(values)[min(len(values)-1, int(q*(len(values)-1)))] if values else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', required=True)
    parser.add_argument('--image-root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--val-target', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    a = parser.parse_args()
    annotations = json.loads(Path(a.json).read_text(encoding='utf8'))
    if not isinstance(annotations, list) or not annotations:
        raise ValueError('Expected nonempty list of JSON annotations')
    img_root = Path(a.image_root)
    image_to_indices = defaultdict(list)
    turn_counts = Counter()
    role_counts = Counter()
    image_marker_counts = Counter()
    invalid_roles = []
    missing_images = []
    no_answers = []
    blank_turns = []
    lengths = []
    for idx, sample in enumerate(annotations):
        filename = Path(str(sample.get('image', ''))).name
        if not filename or not (img_root / filename).is_file():
            missing_images.append((idx, filename))
        image_to_indices[filename].append(idx)
        convs = sample.get('conversations', [])
        if not isinstance(convs, list):
            invalid_roles.append(idx)
            continue
        turn_counts[len(convs)] += 1
        answers = 0
        chars = 0
        image_marks = 0
        for turn_idx, turn in enumerate(convs):
            role = turn.get('from') if isinstance(turn, dict) else None
            text = turn.get('value', '') if isinstance(turn, dict) else ''
            role_counts[role] += 1
            if role not in ('human', 'gpt'):
                invalid_roles.append((idx, turn_idx, role))
            if role == 'gpt':
                answers += 1
            if not isinstance(text, str) or not text.strip():
                blank_turns.append((idx, turn_idx))
                continue
            image_marks += text.count('<image>')
            chars += len(text)
        lengths.append(chars)
        image_marker_counts[image_marks] += 1
        if not answers:
            no_answers.append(idx)
    print('Total records:', len(annotations))
    print('Unique images:', len(image_to_indices))
    print('Missing image references:', len(missing_images), missing_images[:5])
    print('Dialogue lengths (turns):', dict(sorted(turn_counts.items())))
    print('Role counts:', dict(role_counts))
    print('Image marker count per record:', dict(sorted(image_marker_counts.items())))
    print('Empty turns:', len(blank_turns), 'Invalid roles:', len(invalid_roles),
          'No assistant answer:', len(no_answers))
    print('Chars per record p50/p90/p99:', [quantile(lengths, q) for q in (.50,.90,.99)])

    if missing_images or invalid_roles or no_answers or blank_turns:
        raise RuntimeError('Found invalid records; fix or explicitly filter before making an SFT split')
    if a.val_target < 1 or a.val_target >= len(annotations):
        raise ValueError('--val-target must be between 1 and dataset size-1')
    keys = list(image_to_indices)
    random.Random(a.seed).shuffle(keys)
    val_keys = set()
    count = 0
    for key in keys:
        val_keys.add(key)
        count += len(image_to_indices[key])
        if count >= a.val_target:
            break
    val_indices = sorted(i for key in val_keys for i in image_to_indices[key])
    train_indices = sorted(i for key, values in image_to_indices.items()
                           if key not in val_keys for i in values)
    assert not (set(train_indices) & set(val_indices))
    assert len(train_indices) + len(val_indices) == len(annotations)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'train_indices.json').write_text(json.dumps(train_indices), encoding='utf-8')
    (out / 'val_indices.json').write_text(json.dumps(val_indices), encoding='utf-8')
    manifest = {'input_json': str(Path(a.json).resolve()),
                'image_root': str(img_root.resolve()),
                'seed': a.seed, 'total': len(annotations),
                'unique_images': len(image_to_indices),
                'train_records': len(train_indices),
                'val_records': len(val_indices),
                'train_images': len(image_to_indices) - len(val_keys),
                'val_images': len(val_keys),
                'turn_counts': dict(sorted(turn_counts.items())),
                'roles': dict(role_counts),
                'image_markers': dict(sorted(image_marker_counts.items())),
                'char_len_p50_p90_p99': [quantile(lengths, q) for q in (.5,.9,.99)]}
    (out / 'audit_report.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Image-disjoint split:', len(train_indices), 'train /', len(val_indices), 'val')
    print('PASS: saved audit and image-disjoint indices to', out)

if __name__ == '__main__':
    main()
