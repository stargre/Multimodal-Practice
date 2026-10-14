"""Use ONE real PT sample to check encoding + loss with the actual Stage-1 model."""
import argparse
import torch

from stage2lab.adapter import load_stage1_model_and_processor
from stage2lab.dataset import LLaVAPT
from stage2lab.collator import PTDebugCollator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--max-tokens', type=int, default=3072)
    args = ap.parse_args()

    model, processor = load_stage1_model_and_processor()
    dataset = LLaVAPT(args.json, args.root, limit=1, seed=42)
    sample = dataset[0]
    print('Sample:', sample['id'], '| image:', sample['image_path'])
    batch = PTDebugCollator(processor, max_tokens=args.max_tokens)([sample])
    for key, value in batch.items():
        print(key, tuple(value.shape) if torch.is_tensor(value) else type(value).__name__)

    supervised = int((batch['labels'] != -100).sum())
    masked = int((batch['labels'] == -100).sum())
    assert supervised > 0 and masked > 0
    assert batch['input_ids'].shape == batch['labels'].shape
    print('Supervised:', supervised, '| Masked:', masked)

    # Always reset the custom Qwen2.5-style cached rope deltas between examples.
    model.model.rope_deltas = None
    model.eval()
    batch = {k: v.to('cuda') if torch.is_tensor(v) else v
             for k, v in batch.items()}
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        outputs = model(**batch, use_cache=False)
    assert outputs.loss is not None and torch.isfinite(outputs.loss)
    print('Loss:', outputs.loss.item())
    print('PASS: real PT image -> tokens -> forward -> finite loss')


if __name__ == '__main__':
    main()
