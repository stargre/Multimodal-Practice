import argparse
import torch
from stage2lab.adapter import load_stage1_model_and_processor
from stage2lab.dataset import LLaVAPT
from stage2lab.collator import PTDebugCollator

p = argparse.ArgumentParser()
p.add_argument('--json', required=True)
p.add_argument('--root', required=True)
p.add_argument('--max-tokens', type=int, default=3072)
args = p.parse_args()
model, processor = load_stage1_model_and_processor()
dataset = LLaVAPT(args.json, args.root, limit=1)
batch = PTDebugCollator(processor, args.max_tokens)([dataset[0]])
print('Sample ID:', dataset[0]['id'])
for k,v in batch.items():
    print(k, tuple(v.shape) if isinstance(v, torch.Tensor) else type(v).__name__)
assert batch['input_ids'].shape == batch['labels'].shape
n_sup = int((batch['labels'] != -100).sum())
n_masked = int((batch['labels'] == -100).sum())
print('Supervised tokens:', n_sup, '| masked tokens:', n_masked)
assert n_sup > 0 and n_masked > 0
assert (batch['labels'][0, :n_masked] == -100).all(), 'Expected prefix masking'
print('PASS: multimodal batch and label mask')
