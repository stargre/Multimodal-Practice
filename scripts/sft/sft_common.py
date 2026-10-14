"""User-specific model setup, checkpoint format, evaluation for SFT."""
from __future__ import annotations
import contextlib
import json
import math
import os
from pathlib import Path
import random
import sys

import torch
PROJECT=Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0,str(PROJECT))
from stage1_runtime import load_stage1_components
from stage2lab.sft_data import LLaVASFT, SFTCollator


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(pt_checkpoint, sft_checkpoint=None, train=False, gradient_checkpointing=False):
    base, proc=load_stage1_components()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA BF16 required')
    base.requires_grad_(False)
    # PT checkpoint is a plain merger.state_dict or {'merger':...}.
    pt_state=torch.load(pt_checkpoint,map_location='cpu',weights_only=True)
    if isinstance(pt_state,dict) and 'merger' in pt_state:
        pt_state=pt_state['merger']
    base.model.visual.merger.float()
    base.model.visual.merger.load_state_dict(pt_state,strict=True)
    if train:
        # FP32 master weights for reliable 2e-5 updates; BF16 compute via autocast.
        base.model.language_model.float()
        base.lm_head.float()
        base.model.visual.merger.float()
        base.model.language_model.requires_grad_(True)
        base.lm_head.requires_grad_(True)
        base.model.visual.merger.requires_grad_(True)
        if gradient_checkpointing:
            module=base.model.language_model
            if not hasattr(module, 'gradient_checkpointing_enable'):
                raise RuntimeError('Qwen3 language_model has no gradient_checkpointing_enable')
            module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    if sft_checkpoint:
        # For evaluation load SFT model parameters in the same FP32 shape as training.
        base.model.language_model.float()
        base.lm_head.float()
        weights=torch.load(sft_checkpoint,map_location='cpu',weights_only=True)
        if 'trainable' in weights:
            weights=weights['trainable']
        expected={k for k in base.state_dict() if k.startswith(('model.visual.merger.', 'model.language_model.', 'lm_head.'))}
        if set(weights)!=expected:
            raise RuntimeError(f'Unexpected/incomplete SFT state: missing={list(expected-set(weights))[:8]}, unexpected={list(set(weights)-expected)[:8]}')
        result=base.load_state_dict(weights,strict=False)
        if result.unexpected_keys:
            raise RuntimeError(f'Unexpected SFT keys: {result.unexpected_keys}')
    base.model.rope_deltas=None
    if hasattr(base.config,'use_cache'):
        base.config.use_cache=False
    set_mode(base,train)
    return base,proc


def set_mode(base,train):
    base.train(train)
    base.model.visual.vit.eval()
    base.model.visual.merger.train(train)
    base.model.language_model.train(train)
    base.lm_head.train(train)


def trainable_parameters(base):
    llm,merger=[],[]
    seen=set()
    for n,p in base.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if n.startswith(('model.language_model.','lm_head.')):
            llm.append(p)
        elif n.startswith('model.visual.merger.'):
            merger.append(p)
        else:
            raise RuntimeError(f'Unexpected trainable parameter {n}')
        if p.dtype!=torch.float32:
            raise RuntimeError(f'Trainable parameter must be FP32: {n}={p.dtype}')
    if not llm or not merger:
        raise RuntimeError('Missing LLM or merger trainable parameters')
    return llm,merger


def get_sft_weights(base):
    prefixes=('model.visual.merger.','model.language_model.','lm_head.')
    return {k:v.detach().cpu().clone() for k,v in base.state_dict().items() if k.startswith(prefixes)}


def indices_from(path):
    ids=json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(ids,list) or not ids or any(not isinstance(i,int) for i in ids):
        raise ValueError(f'Invalid indices in {path}')
    if len(set(ids))!=len(ids):
        raise ValueError('Duplicate dataset indices')
    return ids


def epoch_order(ids,seed,epoch):
    g=torch.Generator().manual_seed(seed+100003*(epoch+1))
    perm=torch.randperm(len(ids),generator=g).tolist()
    return [ids[j] for j in perm]


def shard_order(ids,rank,world):
    count=math.ceil(len(ids)/world)
    padded=ids+[ids[i % len(ids)] for i in range(count*world-len(ids))]
    return padded[rank::world]


def to_device(batch,device=None):
    device=device or torch.device('cuda',torch.cuda.current_device())
    return {k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}


def one_loss(model,batch):
    model.model.rope_deltas=None
    with torch.autocast('cuda',dtype=torch.bfloat16):
        loss=model(**to_device(batch),use_cache=False).loss
    if loss is None or loss.ndim!=0 or not torch.isfinite(loss).item():
        raise RuntimeError('Invalid SFT loss')
    return loss


def save_atomic(obj,path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    torch.save(obj,tmp)
    os.replace(tmp,path)


@torch.no_grad()
def evaluate_local(base,proc,ds,indices,max_tokens):
    set_mode(base,False)
    collator=SFTCollator(proc,max_tokens)
    weighted=0.
    tokens=0
    for idx in indices:
        batch=collator([ds[idx]])
        n=int((batch['labels']!=-100).sum().item())
        weighted += one_loss(base,batch).detach().item()*n
        tokens += n
    if not tokens:
        raise RuntimeError('No supervised validation tokens')
    return weighted/tokens,tokens
