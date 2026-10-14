"""Stage-1-aware PT debug trainer: merger-only, BS=1, one step/overfit."""
import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from stage2lab.adapter import load_stage1_model_and_processor
from stage2lab.collator import PTDebugCollator
from stage2lab.dataset import LLaVAPT


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--json', required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--mode', choices=('step', 'overfit'), default='step')
    p.add_argument('--num-samples', type=int, default=8)
    p.add_argument('--steps', type=int, default=120)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--max-tokens', type=int, default=3072)
    p.add_argument('--out', default='checkpoints/pt_debug')
    # Kept for compatibility with the previous bundle; only the verified merger
    # is allowed to train at this stage.
    p.add_argument('--train-pattern', default='merger')
    args = p.parse_args()
    if args.train_pattern != 'merger':
        p.error('This verified Stage-1 trainer trains only model.model.visual.merger')
    return args


def main():
    a = parse_args()
    torch.manual_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError('This lab expects CUDA')
    model, proc = load_stage1_model_and_processor()

    model.requires_grad_(False)
    merger = model.model.visual.merger
    # Keep trainable merger weights/AdamW state in FP32. The rest of the
    # (frozen) model remains BF16; autocast handles merger matmuls in forward.
    merger.float()
    merger.requires_grad_(True)
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError('Merger has no trainable parameters')
    print('Trainable parameter count:', sum(p.numel() for _, p in trainable))
    for name, p in trainable:
        print('  ', name, tuple(p.shape))
    assert all('visual.merger.' in n for n, _ in trainable), trainable[:4]

    model = model.to('cuda')
    if hasattr(model, 'config') and hasattr(model.config, 'use_cache'):
        model.config.use_cache = False
    model.train()
    # Do not activate dropout in frozen backbones in tiny-set experiments.
    model.model.visual.vit.eval()
    model.model.language_model.eval()
    model.lm_head.eval()
    model.model.visual.merger.train()

    data = LLaVAPT(a.json, a.root, limit=a.num_samples, seed=42)
    loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=PTDebugCollator(proc, a.max_tokens))
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=a.lr, weight_decay=0.)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / ('step_log.csv' if a.mode == 'step' else 'overfit_log.csv')
    it = iter(loader)
    nsteps = 1 if a.mode == 'step' else a.steps

    with log_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'loss', 'grad_norm', 'max_param_change'])
        for step in range(1, nsteps + 1):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            batch = {k: v.to('cuda') if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            model.model.rope_deltas = None  # As in test-all.py::check_training
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(**batch, use_cache=False)
                loss = out.loss
            if loss is None or loss.ndim != 0 or not torch.isfinite(loss).item():
                raise RuntimeError(f'Non-finite/invalid loss: {loss}')
            loss.backward()
            grads = [(name, p) for name, p in trainable
                     if p.grad is not None and torch.count_nonzero(p.grad).item() > 0]
            if not grads:
                raise RuntimeError('No nonzero merger gradient')
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                [p for _, p in trainable], max_norm=1.0).item())
            if not (grad_norm > 0 and grad_norm < float('inf')):
                raise RuntimeError(f'Invalid merger gradient norm: {grad_norm}')
            # Verify an actively updated merger parameter genuinely changes.
            name0, p0 = grads[0]
            before = p0.detach().float().cpu().clone()
            optimizer.step()
            delta = float((p0.detach().float().cpu() - before).abs().max().item())
            if step == 1 and delta <= 0:
                raise RuntimeError(f'Optimizer did not update {name0}')
            l = float(loss.detach().cpu())
            writer.writerow([step, l, grad_norm, delta])
            if step == 1 or step % 10 == 0 or step == nsteps:
                print(f'step={step:03d} loss={l:.4f} grad_norm={grad_norm:.5f} '
                      f'weight_delta={delta:.3e} gpu_alloc_gb='
                      f'{torch.cuda.max_memory_allocated()/1024**3:.2f}', flush=True)

    print('PASS:', a.mode, 'log:', log_path)
    if a.mode == 'overfit':
        weights = {name: p.detach().cpu().clone() for name, p in trainable}
        torch.save(weights, out_dir / 'trainable_weights_debug.pt')
        print('Trainable-only debug weights:', out_dir / 'trainable_weights_debug.pt')
        print('Check overfit_log.csv for the full loss trend.')


if __name__ == '__main__':
    main()
