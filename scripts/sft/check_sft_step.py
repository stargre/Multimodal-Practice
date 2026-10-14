"""One real-image SFT forward/backward/optimizer step on one GPU.
Checks ALL assistant turns supervised; ViT frozen; LLM and merger updated.
"""
import argparse
from pathlib import Path
import torch
from sft_common import (load_model, trainable_parameters, LLaVASFT, SFTCollator, to_device, seed_all)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--json',required=True)
    p.add_argument('--image-root',required=True)
    p.add_argument('--indices',required=True)
    p.add_argument('--pt-checkpoint',required=True)
    p.add_argument('--max-tokens',type=int,default=4096)
    a=p.parse_args()
    seed_all(42)
    model,proc=load_model(a.pt_checkpoint,train=True)
    llm,merger=trainable_parameters(model)
    dataset=LLaVASFT(a.json,a.image_root)
    import json
    ids=json.loads(Path(a.indices).read_text())
    sample=dataset[ids[0]]
    batch=SFTCollator(proc,a.max_tokens)([sample])
    print('SAMPLE',sample['id'],'turns=',len(sample['conversations']))
    print('shape',tuple(batch['input_ids'].shape),'image_tiles',batch['pixel_values'].shape[0])
    print('supervised_tokens',int((batch['labels']!=-100).sum()),'masked_tokens',int((batch['labels']==-100).sum()))
    end_id=proc.tokenizer.convert_tokens_to_ids('<|im_end|>')
    actual=int((batch['labels']==end_id).sum())
    expected=len(sample['conversations'])//2
    assert actual==expected, f'{actual} supervised EOT tokens, expected {expected}'
    opt=torch.optim.AdamW([{'params':llm,'lr':2e-5},{'params':merger,'lr':5e-5}],weight_decay=0.)
    model.model.rope_deltas=None
    with torch.autocast('cuda',dtype=torch.bfloat16):
        loss=model(**to_device(batch),use_cache=False).loss
    assert torch.isfinite(loss).item(),loss
    loss.backward()
    norm_merger=float(sum(p.grad.detach().float().norm().item() for p in merger if p.grad is not None))
    norm_llm=float(sum(p.grad.detach().float().norm().item() for p in llm if p.grad is not None))
    assert norm_merger>0 and norm_llm>0, (norm_merger,norm_llm)
    assert all(p.grad is None for p in model.model.visual.vit.parameters())
    sample_merger=next(p for p in merger if p.grad is not None and p.grad.detach().abs().max().item()>0)
    sample_llm=next(p for p in llm if p.grad is not None and p.grad.detach().abs().max().item()>0)
    before_merger=sample_merger.detach().clone()
    before_llm=sample_llm.detach().clone()
    torch.nn.utils.clip_grad_norm_(llm+merger,1.)
    opt.step()
    delta_merger=(sample_merger.detach()-before_merger).abs().max().item()
    delta_llm=(sample_llm.detach()-before_llm).abs().max().item()
    print('loss',float(loss.detach()),'grad_norm_merger',norm_merger,'grad_norm_llm',norm_llm)
    print('parameter_change_merger',delta_merger,'parameter_change_llm',delta_llm)
    if delta_merger<=0 or delta_llm<=0:
        raise RuntimeError('Optimizer did not update BOTH trainable modules')
    print('PASS: SFT real-image forward/backward/optimizer-step')

if __name__=='__main__':
    main()
