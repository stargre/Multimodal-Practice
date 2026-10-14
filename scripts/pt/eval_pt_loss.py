"""Held-out teacher-forced PT loss; no images from validation go into train."""
import argparse
import math

from pt_common import (LLaVAPT, evaluate, fix_seed, load_base,
                       load_merger_weights, split_ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--json', required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--checkpoint', default=None, help='latest_merger.pt / best_merger.pt')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val-size', type=int, default=128)
    p.add_argument('--max-tokens', type=int, default=3072)
    a = p.parse_args()
    fix_seed(a.seed)
    dataset = LLaVAPT(a.json, a.root)
    _, val = split_ids(len(dataset), a.seed, a.val_size)
    model, processor = load_base(train=False)
    if a.checkpoint:
        n = load_merger_weights(model, a.checkpoint)
        print('Loaded trained merger:', a.checkpoint, 'tensors:', n)
    else:
        print('BASELINE: using untrained stage-1 merger')
    loss = evaluate(model, dataset, processor, val, a.max_tokens)
    print(f'PASS: held-out {len(val)} examples; assistant token-weighted NLL={loss:.6f}; '
          f'exp(NLL)={math.exp(min(loss, 30)):.4f}')


if __name__ == '__main__':
    main()
