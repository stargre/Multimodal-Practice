"""Strict migration helpers; deliberately outside the independently installable package."""
import hashlib
from pathlib import Path
import torch
from transformers import CLIPImageProcessor

PREFIXES=('model.visual.merger.', 'model.language_model.', 'lm_head.')


def add_legacy_args(parser):
    parser.add_argument('--stage',required=True,choices=['pt','sft','dpo'])
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--qwen25',required=True)
    parser.add_argument('--vit',required=True)
    parser.add_argument('--llm',required=True)


def sha256(path):
    digest=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def apply_checkpoint(model, weights, stage):
    if not isinstance(weights,dict): raise ValueError('Checkpoint must be a tensor dictionary')
    wrapper={'pt':'merger','sft':'trainable','dpo':'state_dict'}[stage]
    if wrapper in weights: weights=weights[wrapper]
    full=model.state_dict()
    if stage=='pt':
        model.model.visual.merger.float()
        expected=set(model.model.visual.merger.state_dict())
    else:
        expected={k for k in full if k.startswith(PREFIXES)} if stage=='sft' else set(full)
    if set(weights)!=expected:
        raise ValueError(f'Incomplete/wrong {stage} checkpoint: missing={sorted(expected-set(weights))[:20]}, unexpected={sorted(set(weights)-expected)[:20]}')
    target=model.model.visual.merger if stage=='pt' else model
    target_state=target.state_dict()
    for k,v in weights.items():
        if not torch.is_tensor(v) or v.shape!=target_state[k].shape:
            raise ValueError(f'Wrong tensor type/shape: {k}')
        if v.is_floating_point() and not torch.isfinite(v).all():
            raise ValueError(f'Non-finite checkpoint tensor: {k}')
    if stage=='sft':
        model.model.language_model.float(); model.lm_head.float(); model.model.visual.merger.float()
        if any(v.dtype!=torch.float32 for v in weights.values()):
            raise ValueError('Expected FP32 SFT master weights; refusing implicit precision conversion')
    elif stage=='dpo':
        modules={'model.visual.vit.':model.model.visual.vit,'model.visual.merger.':model.model.visual.merger,
                 'model.language_model.':model.model.language_model,'lm_head.':model.lm_head}
        for prefix,module in modules.items():
            dtypes={v.dtype for k,v in weights.items() if k.startswith(prefix) and v.is_floating_point()}
            if len(dtypes)!=1 or next(iter(dtypes)) not in (torch.float32,torch.bfloat16,torch.float16):
                raise ValueError(f'Unsupported mixed dtypes inside {prefix}: {dtypes}')
            module.to(dtype=next(iter(dtypes)))
    if stage!='pt' and model.model.language_model.embed_tokens.weight is model.lm_head.weight:
        a,b=weights['model.language_model.embed_tokens.weight'],weights['lm_head.weight']
        if a.dtype!=b.dtype or not torch.equal(a,b):
            raise ValueError('Conflicting tied embedding/head values in checkpoint')
    result=target.load_state_dict(weights,strict=(stage!='sft'))
    if stage=='sft':
        allowed=set(full)-expected
        if set(result.missing_keys)!=allowed or result.unexpected_keys:
            raise ValueError(f'Unexpected load result: {result}')
    # No silently rounded checkpoint tensors.
    actual=target.state_dict()
    for k,v in weights.items():
        if v.dtype!=actual[k].dtype or not torch.equal(v,actual[k].cpu()):
            raise ValueError(f'Checkpoint precision/value changed on load: {k}')


def load_legacy(args):
    from models.qwen25_fork.build_model import build_model
    from models.qwen25_fork.processing_hybrid import HybridQwen25Processor
    for path in (args.qwen25,args.vit,args.llm):
        if not Path(path,'config.json').is_file(): raise FileNotFoundError(f'Missing local config: {path}')
    model,tok=build_model(device='cpu',qwen25_path=args.qwen25,vit_path=args.vit,llm_path=args.llm,dtype=torch.bfloat16)
    tok.padding_side='left'  # generation padding; validation uses identical old/new inputs
    processor=HybridQwen25Processor(tok,CLIPImageProcessor.from_pretrained(args.vit,local_files_only=True),max_tiles_per_side=2)
    weights=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    apply_checkpoint(model,weights,args.stage)
    model.model.rope_deltas=None
    return model.eval(),processor
