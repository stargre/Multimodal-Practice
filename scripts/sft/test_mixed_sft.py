"""Single-GPU mixed-data smoke test; no optimizer and no checkpoint writes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from sft_common import load_model, one_loss
from stage2lab.sft_data import LLaVASFT, SFTCollator


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--json', required=True)
    p.add_argument('--image-root', required=True)
    p.add_argument('--indices', required=True)
    p.add_argument('--pt-checkpoint', required=True)
    p.add_argument('--max-tokens', type=int, default=4096)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    torch.cuda.set_device(0)
    ds = LLaVASFT(a.json, a.image_root)
    indices = json.loads(Path(a.indices).read_text(encoding='utf-8'))
    examples = {}
    for idx in indices:
        sample = ds[idx]
        examples.setdefault(sample['sample_type'], idx)
        if all(k in examples for k in ('vision', 'text', 'math')):
            break
    missing = {'vision', 'text', 'math'} - set(examples)
    if missing:
        raise RuntimeError(f'Missing sample types in index list: {sorted(missing)}')
    model, processor = load_model(a.pt_checkpoint, train=True)
    collator = SFTCollator(processor, a.max_tokens)
    model.zero_grad(set_to_none=True)
    for kind in ('vision', 'text', 'math'):
        batch = collator([ds[examples[kind]]])
        loss = one_loss(model, batch)
        if not torch.isfinite(loss).item():
            raise RuntimeError(f'{kind}: non-finite loss')
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not grads or not all(torch.isfinite(g).all().item() for g in grads):
            raise RuntimeError(f'{kind}: missing or non-finite gradients')
        print(f'PASS {kind}: idx={examples[kind]} tokens={batch["input_ids"].shape[-1]} '
              f'loss={loss.detach().item():.5f}', flush=True)
        model.zero_grad(set_to_none=True)
        del batch, loss
        torch.cuda.empty_cache()
    print('PASS mixed SFT forward/backward test; no optimizer/checkpoint writes.')


if __name__ == '__main__':
    main()
