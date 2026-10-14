"""Shared, intentionally stage-1-specific PT helpers.

The model and image processor are loaded using the user's proven stage1_runtime.py.
No custom ViT, MRoPE or tiling code is reimplemented here.
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from stage2lab.adapter import load_stage1_model_and_processor
from stage2lab.collator import PTDebugCollator
from stage2lab.dataset import LLaVAPT


def fix_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_base(train: bool):
    model, processor = load_stage1_model_and_processor()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('PT scripts require a CUDA GPU with BF16')
    model.requires_grad_(False)
    # The stage-1 debug training recipe kept merger in FP32 with BF16 autocast.
    merger = model.model.visual.merger
    merger.float()
    merger.requires_grad_(train)
    model.to('cuda')
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False
    set_mode(model, train)
    return model, processor


def set_mode(model, train: bool):
    model.train(train)
    # Keep frozen modules in evaluation mode even while training the merger.
    model.model.visual.vit.eval()
    model.model.language_model.eval()
    model.lm_head.eval()
    model.model.visual.merger.train(train)


def split_ids(dataset_length: int, seed: int, val_size: int, max_train: int = 0):
    if dataset_length <= val_size or val_size < 1:
        raise ValueError(f'Need 1 <= val_size < dataset size; n={dataset_length}, val={val_size}')
    g = torch.Generator().manual_seed(seed)
    all_ids = torch.randperm(dataset_length, generator=g).tolist()
    val_ids = all_ids[:val_size]
    train_ids = all_ids[val_size:]
    if max_train:
        if max_train < 1:
            raise ValueError('max_train_samples must be >=1 (or 0 for unlimited)')
        train_ids = train_ids[:max_train]
    assert not (set(train_ids) & set(val_ids))
    return train_ids, val_ids


def epoch_order(train_ids: list[int], seed: int, epoch: int):
    g = torch.Generator().manual_seed(seed + 100003 * (epoch + 1))
    perm = torch.randperm(len(train_ids), generator=g).tolist()
    return [train_ids[i] for i in perm]


def make_loader(dataset, proc, indices, max_tokens: int):
    # Batch size deliberately remains 1. The custom variable-size image grid
    # and MRoPE batch packing were not yet validated with batch_size > 1.
    return DataLoader(dataset, batch_size=1, sampler=indices,
                      num_workers=0, collate_fn=PTDebugCollator(proc, max_tokens))


def to_gpu(batch):
    return {k: v.to('cuda', non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()}


def forward_loss(model, batch):
    model.model.rope_deltas = None
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = model(**to_gpu(batch), use_cache=False).loss
    if loss is None or loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
        raise RuntimeError(f'Non-finite/invalid loss: {loss}')
    return loss


@torch.no_grad()
def evaluate(model, dataset, processor, indices, max_tokens: int):
    set_mode(model, False)
    weighted_total = 0.0
    supervised_total = 0
    try:
        for batch in make_loader(dataset, processor, indices, max_tokens):
            n_targets = int((batch['labels'] != -100).sum().item())
            loss = forward_loss(model, batch)
            weighted_total += float(loss.detach()) * n_targets
            supervised_total += n_targets
        if supervised_total == 0:
            raise RuntimeError('Validation dataset had no assistant targets')
        return weighted_total / supervised_total
    finally:
        # Only merger enters training mode if model was initialized for train;
        # outer caller will reset its desired mode after evaluation.
        pass


def load_merger_weights(model, filename: str):
    ckpt = torch.load(filename, map_location='cpu', weights_only=True)
    state = ckpt['merger'] if 'merger' in ckpt else ckpt
    model.model.visual.merger.float()
    model.model.visual.merger.load_state_dict(state, strict=True)
    return len(state)


def merger_cpu_state(model):
    return {key: value.detach().cpu().clone()
            for key, value in model.model.visual.merger.state_dict().items()}
