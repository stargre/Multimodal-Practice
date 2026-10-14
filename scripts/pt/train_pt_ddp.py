"""Single-node, multi-GPU DistributedDataParallel PT trainer for the verified hybrid model.

Torchrun examples:
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
    scripts/pt/train_pt_ddp.py --json ... --root ... --out ... --grad-accum 4

Each worker uses microbatch=1 on ONE GPU. The frozen ViT / Qwen3 are replicated;
only the Qwen vision merger trains. Validation and checkpoints are rank-0-only.
This does not resume checkpoints from the older single-GPU train_pt.py.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
from datetime import timedelta
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR

from pt_common import (
    LLaVAPT, epoch_order, evaluate, fix_seed, load_base, make_loader,
    merger_cpu_state, set_mode, split_ids, to_gpu,
)


def parse_args():
    p = argparse.ArgumentParser(description='DDP, merger-only, multimodal PT')
    p.add_argument('--json', required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--grad-accum', type=int, default=4, help='Per-GPU gradient accumulation')
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
    p.add_argument('--resume', default=None, help='latest_state_ddp.pt from THIS DDP script')
    a = p.parse_args()
    if a.epochs < 1 or a.grad_accum < 1 or a.max_train_samples < 0:
        p.error('epochs and grad-accum must be >=1; max-train-samples >=0')
    if a.lr <= 0 or not (0 < a.min_lr_ratio <= 1) or not (0 <= a.warmup_ratio < 1):
        p.error('invalid learning rate settings')
    if a.save_every < 1 or a.eval_every < 1 or a.log_every < 1:
        p.error('log/eval/save intervals must be >=1')
    return a


def initialize():
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ or 'LOCAL_RANK' not in os.environ:
        raise RuntimeError('Launch with torchrun --standalone --nproc_per_node=N, not python')
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError('LOCAL_RANK exceeds CUDA_VISIBLE_DEVICES GPU count')
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl', timeout=timedelta(minutes=60))
    return rank, local_rank, world


def log0(rank, *args):
    if rank == 0:
        print(*args, flush=True)


def save_atomic(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_row(path: Path, row):
    new = not path.exists()
    with path.open('a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['global_step', 'epoch', 'unique_samples_seen',
                         'train_loss_mean', 'lr', 'grad_norm',
                         'val_token_loss', 'elapsed_s'])
        w.writerow(row)


def rank_shard(epoch_ids, rank, world):
    """Pad to equal per-rank lengths; at most (world-1) repeated rows/epoch.

    Every original item occurs once. Equal shard lengths prevent NCCL hangs
    from uneven worker updates. The padding duplicates are explicitly counted.
    """
    n = len(epoch_ids)
    per_rank = (n + world - 1) // world
    pad = per_rank * world - n
    shuffled = epoch_ids + [epoch_ids[i % n] for i in range(pad)]
    return shuffled[rank::world], pad


def learning_rate_multiplier(step, warmup, total, min_ratio):
    if warmup and step < warmup:
        return max((step + 1) / warmup, 1 / warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return min_ratio + (1 - min_ratio) * .5 * (1 + math.cos(math.pi * min(1., progress)))


def local_rng():
    return {'cpu': torch.get_rng_state().cpu(),
            'cuda': torch.cuda.get_rng_state().cpu()}


def restore_rng(item):
    torch.set_rng_state(item['cpu'].cpu())
    torch.cuda.set_rng_state(item['cuda'].cpu())


def run():
    a = parse_args()
    rank, local_rank, world = initialize()
    try:
        # Identical starting seed for any not-in-checkpoint initialization.
        # DDP broadcasts model weights from rank 0 as an additional safeguard.
        fix_seed(a.seed)
        out = Path(a.out).resolve()
        if rank == 0:
            out.mkdir(parents=True, exist_ok=True)
        dist.barrier()

        dataset = LLaVAPT(a.json, a.root)
        train_ids, val_ids = split_ids(len(dataset), a.seed, a.val_size,
                                       a.max_train_samples)
        n_train = len(train_ids)
        n_local = math.ceil(n_train / world)
        pad = n_local * world - n_train
        steps_per_epoch = math.ceil(n_local / a.grad_accum)
        total_steps = a.epochs * steps_per_epoch
        warmup = max(1, int(total_steps * a.warmup_ratio)) if a.warmup_ratio else 0
        meta = dict(json=str(Path(a.json).resolve()),
                    json_bytes=Path(a.json).stat().st_size,
                    image_root=str(Path(a.root).resolve()), data_length=len(dataset),
                    train_size=n_train, val_size=a.val_size, seed=a.seed,
                    max_train_samples=a.max_train_samples, epochs=a.epochs,
                    grad_accum=a.grad_accum, lr=a.lr,
                    min_lr_ratio=a.min_lr_ratio, warmup_ratio=a.warmup_ratio,
                    max_tokens=a.max_tokens, world_size=world, microbatch_per_gpu=1,
                    effective_batch_size=world * a.grad_accum,
                    per_rank_samples=n_local, padding_duplicates_per_epoch=pad,
                    total_steps=total_steps, scheduler='warmup_cosine',
                    trainer_version='ddp_v1')
        meta_path = out / 'run_config_ddp.json'
        # Same metadata is computed locally, before any rank begins training.
        if rank == 0:
            if meta_path.exists():
                found = json.loads(meta_path.read_text(encoding='utf-8'))
                if found != meta:
                    raise RuntimeError('Run settings changed; use new --out')
                if not a.resume:
                    raise RuntimeError('Run already exists: pass --resume or change --out')
            elif a.resume:
                raise RuntimeError('--resume needs existing run_config_ddp.json')
            else:
                meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
        dist.barrier()
        # Double-check from ALL ranks; prevents a rank-0-only exception from hanging others.
        if json.loads(meta_path.read_text(encoding='utf-8')) != meta:
            raise RuntimeError('Inconsistent run configuration across workers')
        log0(rank, f'Using {world} GPU(s): CUDA_VISIBLE_DEVICES={os.getenv("CUDA_VISIBLE_DEVICES", "<all>")}')
        log0(rank, f'Dataset={len(dataset)} train={n_train} val={len(val_ids)}, padded duplicates/epoch={pad}')
        log0(rank, f'Microbatch/GPU=1; accum/GPU={a.grad_accum}; global effective batch={world*a.grad_accum}')
        log0(rank, f'Optimizer steps={total_steps}, warmup={warmup}; rank samples/epoch={n_local}')

        base, processor = load_base(train=True)
        merger = base.model.visual.merger
        trainable = [p for p in base.parameters() if p.requires_grad]
        names = [n for n, p in base.named_parameters() if p.requires_grad]
        assert names and all('visual.merger.' in n for n in names)
        log0(rank, 'Trainable params:', sum(p.numel() for p in trainable))
        ddp = DDP(base, device_ids=[local_rank], output_device=local_rank,
                  find_unused_parameters=False, broadcast_buffers=False)
        optimizer = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=0.0)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda s: learning_rate_multiplier(
            s, warmup, total_steps, a.min_lr_ratio))
        start_epoch = 0
        offset_local = 0
        global_step = 0
        best_metric_path = out / 'best_metric.json'
        best_loss = (json.loads(best_metric_path.read_text(encoding='utf-8'))['best_val_loss']
                     if rank == 0 and best_metric_path.exists() else float('inf'))

        if a.resume:
            ckpt = torch.load(a.resume, map_location='cpu', weights_only=True)
            if ckpt['config'] != meta:
                raise RuntimeError('Checkpoint config differs; GPU count and accum must stay unchanged')
            merger.load_state_dict(ckpt['merger'], strict=True)
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            start_epoch = int(ckpt['epoch'])
            offset_local = int(ckpt['offset_local'])
            global_step = int(ckpt['global_step'])
            if len(ckpt['rng_by_rank']) != world:
                raise RuntimeError('Checkpoint rank RNG count != WORLD_SIZE')
            restore_rng(ckpt['rng_by_rank'][rank])
            log0(rank, f'RESUMED epoch={start_epoch}, rank_offset={offset_local}, step={global_step}')
        else:
            if rank == 0:
                baseline = evaluate(base, dataset, processor, val_ids, a.max_tokens)
                log0(rank, f'BASELINE validation token loss={baseline:.6f}')
                write_row(out / 'history_ddp.csv',
                          [0, 0, 0, '', optimizer.param_groups[0]['lr'], '', baseline, 0])
            dist.barrier()

        started = time.perf_counter()
        loss_sum = 0.0
        micro_count = 0
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(start_epoch, a.epochs):
            ordered = epoch_order(train_ids, a.seed, epoch)
            shard, verify_pad = rank_shard(ordered, rank, world)
            assert verify_pad == pad and len(shard) == n_local
            begin = offset_local if epoch == start_epoch else 0
            if not 0 <= begin <= n_local:
                raise ValueError('Bad resume per-rank offset')
            loader = make_loader(dataset, processor, shard[begin:], a.max_tokens)
            set_mode(base, True)
            consumed = begin
            window_size = min(a.grad_accum, n_local - begin)
            within_window = 0
            for batch in loader:
                if window_size <= 0:
                    raise RuntimeError('Invalid accumulation window')
                sync = (within_window + 1 == window_size)
                # IMPORTANT: no_sync wraps both DDP forward AND backward.
                context = contextlib.nullcontext() if sync else ddp.no_sync()
                with context:
                    base.model.rope_deltas = None
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        loss = ddp(**to_gpu(batch), use_cache=False).loss
                    if loss is None or loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
                        raise RuntimeError(f'Invalid loss at rank {rank}: {loss}')
                    (loss / window_size).backward()
                loss_sum += float(loss.detach())
                micro_count += 1
                within_window += 1
                consumed += 1
                if not sync:
                    continue

                grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
                if not math.isfinite(grad_norm) or grad_norm <= 0:
                    raise RuntimeError(f'Invalid grad norm at rank {rank}: {grad_norm}')
                current_lr = optimizer.param_groups[0]['lr']
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                within_window = 0
                window_size = min(a.grad_accum, n_local - consumed)

                should_eval = global_step % a.eval_every == 0 or consumed == n_local
                should_save = global_step % a.save_every == 0 or consumed == n_local
                should_log = global_step == 1 or global_step % a.log_every == 0 or consumed == n_local
                val_loss = ''
                if should_log:
                    statistics = torch.tensor([loss_sum, micro_count], dtype=torch.float64,
                                              device=torch.device('cuda', local_rank))
                    dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
                    mean_loss = float(statistics[0] / statistics[1])
                    mean_grad = grad_norm  # already synchronized by DDP optimizer update
                loss_sum = 0.
                micro_count = 0

                if should_eval:
                    if rank == 0:
                        val_loss = evaluate(base, dataset, processor, val_ids, a.max_tokens)
                        log0(rank, f'VAL step={global_step} token_loss={val_loss:.4f}')
                        if val_loss < best_loss:
                            best_loss = val_loss
                            save_atomic(merger_cpu_state(base), out / 'best_merger.pt')
                            best_metric_path.write_text(
                                json.dumps({'best_val_loss': best_loss,
                                            'at_step': global_step}, indent=2), encoding='utf-8')
                    dist.barrier()
                    set_mode(base, True)

                if should_log and rank == 0:
                    elapsed = int(time.perf_counter() - started)
                    unique_seen = epoch * n_train + min(n_train, consumed * world)
                    print(f'epoch={epoch+1}/{a.epochs} step={global_step}/{total_steps} '
                          f'unique_samples={unique_seen} loss={mean_loss:.4f} lr={current_lr:.2e} '
                          f'grad_norm={mean_grad:.4f} elapsed={elapsed}s', flush=True)
                    write_row(out / 'history_ddp.csv', [global_step, epoch + 1,
                              unique_seen, mean_loss, current_lr, mean_grad,
                              val_loss, elapsed])

                if should_save:
                    # Every rank contributes its own RNG state for exact local-rank resume.
                    states = [None] * world
                    dist.all_gather_object(states, local_rng())
                    if rank == 0:
                        next_epoch = epoch + 1 if consumed == n_local else epoch
                        next_offset = 0 if consumed == n_local else consumed
                        checkpoint = {
                            'merger': merger_cpu_state(base),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'config': meta, 'epoch': next_epoch,
                            'offset_local': next_offset,
                            'global_step': global_step,
                            'rng_by_rank': states,
                        }
                        save_atomic(checkpoint, out / 'latest_state_ddp.pt')
                        save_atomic(checkpoint['merger'], out / 'latest_merger.pt')
                        log0(rank, f'SAVED step={global_step} next_epoch={next_epoch} local_offset={next_offset}')
                    dist.barrier()
            offset_local = 0
        log0(rank, f'PASS: DDP PT completed; best={out / "best_merger.pt"}')
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    run()
