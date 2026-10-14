import argparse
import json
import random
from collections import Counter
from pathlib import Path
from stage2lab.dataset import find_image

p = argparse.ArgumentParser(description='LLaVA image-path and decode audit')
p.add_argument('--json', required=True)
p.add_argument('--root', required=True)
p.add_argument('--limit', type=int, default=1000, help='0 means ALL')
p.add_argument('--seed', type=int, default=42)
args = p.parse_args()
with open(args.json, encoding='utf-8') as f:
    rows = json.load(f)
assert isinstance(rows, list)
inds = list(range(len(rows))) if args.limit == 0 else random.Random(args.seed).sample(range(len(rows)), min(args.limit, len(rows)))
missing, found = [], []
for i in inds:
    row = rows[i]
    path = find_image(Path(args.root), row['image'])
    (found if path else missing).append((row['image'], str(path) if path else None))
print('Total annotations:', len(rows), 'Checked:', len(inds), 'Found:', len(found), 'Missing:', len(missing))
print('Found examples:', found[:3])
print('Missing examples:', missing[:10])
if found:
    from PIL import Image
    for name, path in found[:10]:
        with Image.open(path) as img:
            img.verify()
    print('First 10 found images decoded/verified: PASS')
if missing:
    raise SystemExit('ERROR: some images were not found; review archive directory layout')
print('PASS: image location audit')
