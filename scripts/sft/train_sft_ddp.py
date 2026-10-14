"""4x A40 SFT: full-Qwen3+merger DDP, frozen ViT, image-disjoint held-out set.

Launch with torchrun (or configs/sft_ddp.sh). The offline prepare_sft.py must
run first so no rank silently drops a too-long/unsupported example mid-epoch.
"""
from __future__ import annotations
import argparse
import contextlib
from datetime import timedelta
import csv
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR

from sft_common import (seed_all,load_model,set_mode,trainable_parameters,
                        get_sft_weights,indices_from,epoch_order,shard_order,
                        save_atomic,one_loss,to_device,LLaVASFT,SFTCollator)


def args_parse():
    p=argparse.ArgumentParser(description='SFT Qwen3+Merger on multiple GPUs')
    p.add_argument('--json',required=True)
    p.add_argument('--image-root',required=True)
    p.add_argument('--pt-checkpoint',required=True)
    p.add_argument('--train-indices',required=True)
    p.add_argument('--val-indices',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--max-train-samples',type=int,default=0)
    p.add_argument('--epochs',type=int,default=1)
    p.add_argument('--grad-accum',type=int,default=2)
    p.add_argument('--max-tokens',type=int,default=4096)
    p.add_argument('--lr-llm',type=float,default=2e-5)
    p.add_argument('--lr-merger',type=float,default=5e-5)
    p.add_argument('--warmup-ratio',type=float,default=.03)
    p.add_argument('--min-lr-ratio',type=float,default=.1)
    p.add_argument('--weight-decay',type=float,default=.01)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--log-every',type=int,default=100)
    p.add_argument('--save-every',type=int,default=2000)
    p.add_argument('--eval-every',type=int,default=2000)
    p.add_argument('--resume',default=None)
    p.add_argument('--gradient-checkpointing',action='store_true')
    p.add_argument('--skip-baseline',action='store_true')
    a=p.parse_args()
    if min(a.epochs,a.grad_accum,a.max_tokens,a.log_every,a.save_every,a.eval_every)<1:
        p.error('epochs, grad-accum, max-tokens, log/save/eval frequencies must be positive')
    if a.lr_llm<=0 or a.lr_merger<=0 or not (0<=a.warmup_ratio<1) or not (0<a.min_lr_ratio<=1):
        p.error('Bad lr schedule')
    return a


def init_ddp():
    if any(name not in os.environ for name in ('RANK','WORLD_SIZE','LOCAL_RANK')):
        raise RuntimeError('Launch with torchrun, not python')
    rank=int(os.environ['RANK']); world=int(os.environ['WORLD_SIZE']); local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    dist.init_process_group(backend='nccl',device_id=torch.device('cuda',local),timeout=timedelta(minutes=60))
    return rank,world,local


def gather_rng():
    return {'cpu':torch.get_rng_state().cpu(),'cuda':torch.cuda.get_rng_state().cpu()}


def restore_rng(rng):
    torch.set_rng_state(rng['cpu'].cpu())
    torch.cuda.set_rng_state(rng['cuda'].cpu())


def schedule_factor(step,warmup,total,min_ratio):
    if warmup and step < warmup:
        return float(step+1)/warmup
    progress=min(1.,max(0.,(step-warmup)/max(1,total-warmup)))
    return min_ratio+(1-min_ratio)*.5*(1+math.cos(math.pi*progress))


def write_log(path,row):
    exists=path.exists()
    with path.open('a',encoding='utf-8',newline='') as f:
        w=csv.writer(f)
        if not exists:
            w.writerow(['step','epoch','samples_seen','train_loss','lr_llm','lr_merger','grad_norm','val_token_loss','elapsed_s'])
        w.writerow(row)


@torch.no_grad()
def eval_distributed(base,processor,ds,val_ids,max_tokens,rank,world,device):
    set_mode(base,False)
    col=SFTCollator(processor,max_tokens)
    stats=torch.zeros(2,dtype=torch.float64,device=device)
    for i in val_ids[rank::world]:
        batch=col([ds[i]])
        n=int((batch['labels']!=-100).sum().item())
        l=one_loss(base,batch)
        stats[0] += l.detach().double()*n
        stats[1] += n
    dist.all_reduce(stats,op=dist.ReduceOp.SUM)
    return (stats[0]/stats[1]).item()


def run():
    a=args_parse()
    rank,world,local=init_ddp()
    device=torch.device('cuda',local)
    try:
        seed_all(a.seed)
        out=Path(a.out).resolve()
        ds=LLaVASFT(a.json,a.image_root)
        train_ids=indices_from(a.train_indices)
        val_ids=indices_from(a.val_indices)
        prepare_report=Path(a.train_indices).parent/'prepare_report.json'
        if not prepare_report.is_file():
            raise RuntimeError('Missing prepare_report.json: first run scripts/sft/prepare_sft.py on full splits')
        prepared=json.loads(prepare_report.read_text(encoding='utf-8'))
        if prepared.get('scan_limit')!=0 or prepared.get('max_tokens')!=a.max_tokens:
            raise RuntimeError('SFT prepare report must cover ALL records with same --max-tokens')
        if prepared['train']['kept']!=len(train_ids) and not Path(a.train_indices).name.startswith('debug'):
            raise RuntimeError('Prepared train index count mismatch')
        if prepared['val']['kept']!=len(val_ids):
            raise RuntimeError('Prepared val index count mismatch')
        if set(train_ids) & set(val_ids):
            raise RuntimeError('Train/val index overlap')
        if max(train_ids+val_ids)>=len(ds) or min(train_ids+val_ids)<0:
            raise ValueError('Out-of-range indices')
        if a.max_train_samples:
            train_ids=train_ids[:a.max_train_samples]
        rank_len=math.ceil(len(train_ids)/world)
        steps_epoch=math.ceil(rank_len/a.grad_accum)
        total_steps=steps_epoch*a.epochs
        warmup=max(1,int(a.warmup_ratio*total_steps)) if a.warmup_ratio else 0
        info={k:getattr(a,k) for k in ('json','image_root','pt_checkpoint','train_indices','val_indices',
              'max_train_samples','epochs','grad_accum','max_tokens','lr_llm','lr_merger','warmup_ratio',
              'min_lr_ratio','weight_decay','seed','log_every','save_every','eval_every',
              'gradient_checkpointing','skip_baseline')}
        info.update(world_size=world,train_size=len(train_ids),val_size=len(val_ids),
                    per_rank=rank_len,total_steps=total_steps,trainer_version='sft_ddp_v1')
        meta_path=out/'run_config.json'
        error=None
        if rank==0:
            try:
                out.mkdir(parents=True,exist_ok=True)
                if meta_path.exists():
                    if json.loads(meta_path.read_text(encoding='utf-8'))!=info:
                        raise RuntimeError('Existing output directory has different config')
                    if not a.resume:
                        raise RuntimeError('Output dir exists, use --resume or a new --out')
                else:
                    if a.resume:
                        raise RuntimeError('Resume requested but no run_config.json exists')
                    meta_path.write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
            except Exception as e:
                error=str(e)
        errors=[error]
        dist.broadcast_object_list(errors,src=0,device=device)
        if errors[0]:
            raise RuntimeError(errors[0])
        if rank==0:
            print(f'SFT GPUs={world}, train={len(train_ids)}, val={len(val_ids)}, accum={a.grad_accum}',flush=True)
            print(f'Steps/epoch={steps_epoch}, total={total_steps}, warmup={warmup}',flush=True)
        base,proc=load_model(a.pt_checkpoint,train=True,gradient_checkpointing=a.gradient_checkpointing)
        if not getattr(proc.tokenizer,'is_fast',False):
            raise RuntimeError('Need fast tokenizer for supervision offsets')
        llm_params,merger_params=trainable_parameters(base)
        if rank==0:
            print(f'Trainable LLM+head={sum(p.numel() for p in llm_params):,}, merger={sum(p.numel() for p in merger_params):,}',flush=True)
        ddp=DDP(base,device_ids=[local],output_device=local,find_unused_parameters=False,broadcast_buffers=False)
        groups=[{'params':llm_params,'lr':a.lr_llm,'name':'qwen3'},
                {'params':merger_params,'lr':a.lr_merger,'name':'merger'}]
        opt=torch.optim.AdamW(groups,weight_decay=a.weight_decay)
        sched=LambdaLR(opt,lr_lambda=lambda step:schedule_factor(step,warmup,total_steps,a.min_lr_ratio))
        step=0; start_epoch=0; offset=0
        metric_path=out/'best_metric.json'
        best=float('inf')
        if a.resume:
            ckpt=torch.load(a.resume,map_location='cpu',weights_only=True)
            if ckpt['config']!=info:
                raise RuntimeError('Resume config mismatch')
            weights=ckpt['trainable']
            missing=set(get_sft_weights(base))-set(weights)
            if missing:
                raise RuntimeError(f'Incomplete resume weights {list(missing)[:5]}')
            base.load_state_dict(weights,strict=False)
            opt.load_state_dict(ckpt['optimizer'])
            sched.load_state_dict(ckpt['scheduler'])
            step=int(ckpt['step']); start_epoch=int(ckpt['epoch']); offset=int(ckpt['offset'])
            if len(ckpt['rng_by_rank'])!=world:
                raise RuntimeError('Rank count mismatch')
            restore_rng(ckpt['rng_by_rank'][rank])
            best=float(ckpt['best_loss'])
            if rank==0: print('RESUME',step,'epoch',start_epoch,'offset',offset,flush=True)
        else:
            if not a.skip_baseline:
                val=eval_distributed(base,proc,ds,val_ids,a.max_tokens,rank,world,device)
                if rank==0:
                    print(f'BASELINE validation token loss={val:.5f}',flush=True)
                    write_log(out/'history.csv',[0,0,0,'',opt.param_groups[0]['lr'],opt.param_groups[1]['lr'],'',val,0])
                # Do not save baseline as best; future first evaluation will save a trained ckpt.
            set_mode(base,True)
        col=SFTCollator(proc,a.max_tokens)
        started=time.perf_counter()
        sumloss=0.; countloss=0
        opt.zero_grad(set_to_none=True)
        for epoch in range(start_epoch,a.epochs):
            ordered=epoch_order(train_ids,a.seed,epoch)
            shard=shard_order(ordered,rank,world)
            consumed=offset if epoch==start_epoch else 0
            if consumed<0 or consumed>len(shard):
                raise RuntimeError('Bad resume offset')
            in_window=0
            window=min(a.grad_accum,len(shard)-consumed)
            set_mode(base,True)
            while consumed<len(shard):
                idx=shard[consumed]
                batch=col([ds[idx]])
                sync=(in_window+1==window)
                with (contextlib.nullcontext() if sync else ddp.no_sync()):
                    base.model.rope_deltas=None
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        loss=ddp(**to_device(batch,device),use_cache=False).loss
                    if loss is None or loss.ndim!=0 or not torch.isfinite(loss).item():
                        raise RuntimeError(f'Rank {rank}: non-finite loss sample {idx}')
                    (loss/window).backward()
                sumloss+=float(loss.detach().item());countloss+=1
                consumed+=1;in_window+=1
                if not sync:
                    continue
                grad=float(torch.nn.utils.clip_grad_norm_(llm_params+merger_params,1.0))
                if not math.isfinite(grad):
                    raise RuntimeError(f'Nonfinite gradient on rank {rank}')
                lr0,lr1=(g['lr'] for g in opt.param_groups)
                opt.step();sched.step();opt.zero_grad(set_to_none=True)
                step+=1;in_window=0
                window=min(a.grad_accum,len(shard)-consumed)
                eval_now=(step%a.eval_every==0 or consumed==len(shard))
                save_now=(step%a.save_every==0 or consumed==len(shard))
                log_now=(step==1 or step%a.log_every==0 or consumed==len(shard))
                val=''
                if log_now:
                    stat=torch.tensor([sumloss,countloss],device=device,dtype=torch.float64)
                    dist.all_reduce(stat,op=dist.ReduceOp.SUM)
                    mean=float(stat[0]/stat[1])
                    sumloss=0.;countloss=0
                if eval_now:
                    val=eval_distributed(base,proc,ds,val_ids,a.max_tokens,rank,world,device)
                    if rank==0:
                        print(f'VAL step={step} token_loss={val:.5f}',flush=True)
                        if val < best:
                            best=val
                            save_atomic(get_sft_weights(base),out/'best_model.pt')
                            metric_path.write_text(json.dumps({'best_val_loss':val,'at_step':step},indent=2),encoding='utf-8')
                    set_mode(base,True)
                    # sync best metric for all ranks used by checkpoint
                    bestbuf=torch.tensor(best,device=device,dtype=torch.float64)
                    dist.broadcast(bestbuf,src=0)
                    best=bestbuf.item()
                if log_now and rank==0:
                    elapsed=round(time.perf_counter()-started,1)
                    seen=epoch*len(train_ids)+min(len(train_ids),consumed*world)
                    print(f'epoch={epoch+1}/{a.epochs} step={step}/{total_steps} samples={seen} '
                          f'loss={mean:.4f} lr_llm={lr0:.2e} lr_merger={lr1:.2e} grad={grad:.2f} elapsed={elapsed}s',flush=True)
                    write_log(out/'history.csv',[step,epoch+1,seen,mean,lr0,lr1,grad,val,elapsed])
                if save_now:
                    states=[None]*world
                    dist.all_gather_object(states,gather_rng())
                    if rank==0:
                        next_epoch=epoch+1 if consumed==len(shard) else epoch
                        next_offset=0 if consumed==len(shard) else consumed
                        weights=get_sft_weights(base)
                        save_atomic({'trainable':weights,'optimizer':opt.state_dict(),
                            'scheduler':sched.state_dict(),'epoch':next_epoch,'offset':next_offset,
                            'step':step,'config':info,'best_loss':best,'rng_by_rank':states},
                            out/'latest_state.pt')
                        save_atomic(weights,out/'latest_model.pt')
                        print('SAVED step',step,flush=True)
                    dist.barrier()
            offset=0
        if rank==0:
            print('PASS: SFT DDP completed. Best model:',out/'best_model.pt',flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__=='__main__':
    run()
