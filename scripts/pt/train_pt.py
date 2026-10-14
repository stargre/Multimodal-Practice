"""A40 single-GPU, BS=1/grad-accum PT trainer for the verified hybrid model.

Use unique --out directories for smoke and full runs. Only the Qwen vision
merger trains. Full-state resume starts at a *completed optimizer step*.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR

from pt_common import (LLaVAPT, epoch_order, evaluate, fix_seed, forward_loss,
                       load_base, make_loader, merger_cpu_state, set_mode, split_ids)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--json', required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--grad-accum', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--min-lr-ratio', type=float, default=.1)
    p.add_argument('--warmup-ratio', type=float, default=.03)
    p.add_argument('--max-tokens', type=int, default=3072)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val-size', type=int, default=128)
    p.add_argument('--max-train-samples', type=int, default=0)
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--save-every', type=int, default=2500)
    p.add_argument('--eval-every', type=int, default=5000)
    p.add_argument('--resume', default=None,
                   help='Path to latest_state.pt created by THIS script')
    a = p.parse_args()
    if a.epochs < 1 or a.grad_accum < 1 or a.max_train_samples < 0:
        p.error('epochs and grad-accum must be positive; max-train-samples >= 0')
    if a.lr <= 0 or not (0 < a.min_lr_ratio <= 1) or not (0 <= a.warmup_ratio < 1):
        p.error('invalid LR settings')
    if a.save_every < 1 or a.eval_every < 1 or a.log_every < 1:
        p.error('logging/checkpoint intervals must be positive')
    return a


def atomic_save(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(data, tmp)
    os.replace(tmp, path)


def csv_log(path, row):
    new = not path.exists()
    with path.open('a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['global_step', 'epoch', 'samples_consumed',
                        'train_loss_mean', 'lr', 'grad_norm',
                        'val_token_loss', 'elapsed_s'])
        w.writerow(row)


def main():
    a = parse_args()
    fix_seed(a.seed)
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    dataset = LLaVAPT(a.json, a.root)
    train_ids, val_ids = split_ids(len(dataset), a.seed, a.val_size,
                                   a.max_train_samples)
    n_train = len(train_ids)
    updates_per_epoch = math.ceil(n_train / a.grad_accum)
    total_updates = a.epochs * updates_per_epoch
    warmup = max(1, int(total_updates * a.warmup_ratio)) if a.warmup_ratio else 0
    data_path = str(Path(a.json).resolve())
    meta = dict(json=data_path, json_bytes=Path(a.json).stat().st_size,
                image_root=str(Path(a.root).resolve()), data_length=len(dataset),
                n_train=n_train, val_size=a.val_size, seed=a.seed,
                max_train_samples=a.max_train_samples, epochs=a.epochs,
                grad_accum=a.grad_accum, lr=a.lr,
                warmup_ratio=a.warmup_ratio, min_lr_ratio=a.min_lr_ratio,
                total_updates=total_updates, max_tokens=a.max_tokens)
    meta_file = out / 'run_config.json'
    if meta_file.exists():
        saved_meta = json.loads(meta_file.read_text())
        if saved_meta != meta:
            raise RuntimeError('Output directory already has DIFFERENT run_config.json. '
                               'Use a fresh --out directory or matching resume options.')
        if not a.resume and (out / 'latest_state.pt').exists():
            raise RuntimeError('Existing run checkpoint found. Pass --resume or use a new --out.')
    else:
        if a.resume:
            raise RuntimeError('Resume requires run_config.json in --out')
        meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print('Dataset total:', len(dataset), 'train:', n_train,
          'held-out validation:', len(val_ids), flush=True)
    print('Microbatch: 1; gradient accumulation:', a.grad_accum,
          'optimizer updates:', total_updates, 'warmup:', warmup, flush=True)
    model, processor = load_base(train=True)
    merger = model.model.visual.merger
    params = [p for p in model.parameters() if p.requires_grad]
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    print('Trainable parameter count:', sum(p.numel() for p in params), flush=True)
    for name in trainable_names:
        print('  ', name, flush=True)
    assert trainable_names and all('visual.merger.' in n for n in trainable_names)
    optimizer = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)

    def lr_scale(step):
        if warmup and step < warmup:
            return max((step + 1) / warmup, 1 / warmup)
        progress = (step - warmup) / max(1, total_updates - warmup)
        return a.min_lr_ratio + (1 - a.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_scale)
    start_epoch, consumed_in_epoch, global_step = 0, 0, 0
    if a.resume:
        checkpoint = torch.load(a.resume, map_location='cpu', weights_only=True)
        if checkpoint.get('config') != meta:
            raise RuntimeError('Checkpoint run configuration does not match current CLI')
        merger.load_state_dict(checkpoint['merger'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = int(checkpoint['epoch'])
        consumed_in_epoch = int(checkpoint['consumed_in_epoch'])
        global_step = int(checkpoint['global_step'])
        torch.set_rng_state(checkpoint['rng_cpu'].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in checkpoint['rng_cuda']])
        print('RESUMED: epoch=', start_epoch,
              'offset=', consumed_in_epoch, 'optimizer_step=', global_step, flush=True)
    else:
        # A fixed held-out subset is never passed to the training sampler.
        before = evaluate(model, dataset, processor, val_ids, a.max_tokens)
        print(f'BASELINE validation token loss: {before:.4f}', flush=True)
        set_mode(model, True)
        csv_log(out / 'history.csv', [0, 0, 0, '', optimizer.param_groups[0]['lr'], '', before, 0])
    started = time.perf_counter()
    best_loss_file = out / 'best_metric.json'
    best_val = (json.loads(best_loss_file.read_text())['best_val_loss']
                if best_loss_file.is_file() else float('inf'))
    seen_total = start_epoch * n_train + consumed_in_epoch
    window_loss, window_count = 0.0, 0

    def save_state(epoch, consumed):
        ckpt = {'merger': merger_cpu_state(model),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch, 'consumed_in_epoch': consumed,
                'global_step': global_step, 'config': meta,
                'rng_cpu': torch.get_rng_state(),
                'rng_cuda': torch.cuda.get_rng_state_all()}
        atomic_save(ckpt, out / 'latest_state.pt')
        atomic_save(ckpt['merger'], out / 'latest_merger.pt')
        print(f'SAVED optimizer step {global_step} at {epoch=}, {consumed=}', flush=True)

    for epoch in range(start_epoch, a.epochs):
        ids = epoch_order(train_ids, a.seed, epoch)
        start = consumed_in_epoch if epoch == start_epoch else 0
        if not 0 <= start <= n_train:
            raise RuntimeError('Invalid resume sample offset')
        loader = make_loader(dataset, processor, ids[start:], a.max_tokens)
        set_mode(model, True)
        optimizer.zero_grad(set_to_none=True)
        left_in_window = min(a.grad_accum, n_train - start)
        in_window = 0
        consumed = start
        for batch in loader:
            if left_in_window == 0:
                raise RuntimeError('Invalid accumulation window')
            loss = forward_loss(model, batch)
            (loss / left_in_window).backward()
            window_loss += float(loss.detach())
            window_count += 1
            in_window += 1
            consumed += 1
            seen_total += 1
            if in_window < left_in_window:
                continue
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            if not math.isfinite(grad_norm) or grad_norm <= 0:
                raise RuntimeError(f'Invalid merger grad norm: {grad_norm}')
            lr_used = optimizer.param_groups[0]['lr']
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            train_loss = window_loss / window_count
            window_loss, window_count = 0., 0
            in_window = 0
            left_in_window = min(a.grad_accum, n_train - consumed)
            val_loss = ''
            if global_step % a.eval_every == 0 or consumed == n_train:
                val_loss = evaluate(model, dataset, processor, val_ids, a.max_tokens)
                print(f'VAL step={global_step} token_loss={val_loss:.4f}', flush=True)
                if val_loss < best_val:
                    best_val = val_loss
                    atomic_save(merger_cpu_state(model), out / 'best_merger.pt')
                    best_loss_file.write_text(json.dumps({'best_val_loss': best_val,
                                                          'at_step': global_step}, indent=2))
                set_mode(model, True)
            if global_step == 1 or global_step % a.log_every == 0 or consumed == n_train:
                elapsed = int(time.perf_counter() - started)
                print(f'epoch={epoch+1}/{a.epochs} update={global_step}/{total_updates} '
                      f'samples={consumed}/{n_train} train_loss={train_loss:.4f} '
                      f'lr={lr_used:.2e} grad_norm={grad_norm:.3f} '
                      f'elapsed={elapsed}s', flush=True)
                csv_log(out / 'history.csv', [global_step, epoch + 1, seen_total,
                                               train_loss, lr_used, grad_norm,
                                               val_loss, elapsed])
            if global_step % a.save_every == 0:
                save_state(epoch, consumed)
        # Finished this epoch. Resume from the NEXT epoch, never replay samples.
        consumed_in_epoch = 0
        save_state(epoch + 1, 0)
    print('PASS: full PT training finished. Model-only weights:',
          out / 'latest_merger.pt', flush=True)


if __name__ == '__main__':
    main()
