"""Compare a real legacy checkpoint with its HF export. Nonzero exit means not accepted."""
import argparse
import contextlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText,AutoProcessor
import hybrid_vlm
from hf_legacy import add_legacy_args,load_legacy,sha256


def cases(proc, image, full):
    small=Image.new('RGB',(448,448),(40,120,210))
    large=Image.new('RGB',(700,700),(210,80,30))
    def case(name,images,text='Describe the image.'):
        return name,[proc.user_prompt(text,num_images=len(images))],dict(images_per_sample=[images])
    yield case('text',[],'Hello.')
    yield case('single_tile',[small])
    yield case('four_tiles',[large])
    if image: yield case('real_image',[Image.open(image).convert('RGB')])
    if full:
        yield case('multiple_images',[small,large])
        yield 'padded_batch',[proc.user_prompt('Describe.',num_images=1),proc.user_prompt('Hello.')],dict(images_per_sample=[[large],[]])
        yield 'video',[proc.user_prompt('Describe.',num_videos=1)],dict(videos_per_sample=[[[small,small.copy()]]],fps_per_sample=[[2.]])
        yield 'mixed',[proc.user_prompt('Describe.',num_images=1,num_videos=1)],dict(images_per_sample=[[small]],videos_per_sample=[[[small,small.copy()]]],fps_per_sample=[[2.]])


def main():
    p=argparse.ArgumentParser(description=__doc__); add_legacy_args(p)
    p.add_argument('--export',required=True); p.add_argument('--report',required=True)
    p.add_argument('--device',default='cuda'); p.add_argument('--precision',choices=['preserve','float32'],default='preserve')
    p.add_argument('--suite',choices=['basic','full'],default='full'); p.add_argument('--image')
    p.add_argument('--max-new-tokens',type=int,default=8)
    p.add_argument('--atol',type=float,default=1e-5); p.add_argument('--rtol',type=float,default=1e-5)
    args=p.parse_args(); report={'status':'FAIL','cases':[],'options':vars(args)}
    report_path=Path(args.report); report_path.parent.mkdir(parents=True,exist_ok=True)
    try:
        meta=json.loads(Path(args.export,'EXPORT_COMPLETE.json').read_text())
        if meta['checkpoint_sha256']!=sha256(args.checkpoint) or meta['stage']!=args.stage:
            raise ValueError('Export provenance differs from reference checkpoint/stage')
        for name in ('qwen25','vit','llm'):
            if meta['base_configs'][name]['sha256']!=sha256(Path(getattr(args,name),'config.json')):
                raise ValueError(f'Base config differs: {name}')
        torch.manual_seed(0)
        if args.device.startswith('cuda'):
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError('CUDA BF16 required')
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
        old,legacy_proc=load_legacy(args)
        new,info=AutoModelForImageTextToText.from_pretrained(args.export,local_files_only=True,output_loading_info=True,attn_implementation='sdpa')
        report['loading_info']=info
        if any(info[k] for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')): raise AssertionError(info)
        new_proc=AutoProcessor.from_pretrained(args.export,local_files_only=True)
        a,b=old.state_dict(),new.state_dict()
        if a.keys()!=b.keys(): raise AssertionError('State dict keys changed')
        for k,v in a.items():
            if v.dtype!=b[k].dtype or not torch.equal(v,b[k]): raise AssertionError(f'Weight changed: {k}')
        del a,b
        report['weights_exact']=True
        if args.precision=='float32': old.float(); new.float()
        old.to(args.device).eval(); new.to(args.device).eval()
        amp=lambda: torch.autocast('cuda',dtype=torch.bfloat16) if args.device.startswith('cuda') and args.precision=='preserve' else contextlib.nullcontext()
        def compare(a,b,name):
            if not torch.isfinite(a).all() or not torch.isfinite(b).all(): raise AssertionError(f'Nonfinite {name}')
            torch.testing.assert_close(a,b,atol=args.atol,rtol=args.rtol,msg=name)
            return float((a.float()-b.float()).abs().max())
        with torch.inference_mode():
            for name,prompts,data in cases(legacy_proc,args.image,args.suite=='full'):
                item={'name':name,'status':'FAIL'}; report['cases'].append(item)
                left=legacy_proc.encode_batch(prompts,**data)
                right=new_proc.encode_batch(prompts,**data)
                if left.keys()!=right.keys(): raise AssertionError('Processor keys changed')
                for key in left:
                    if left[key] is None:
                        if right[key] is not None: raise AssertionError(key)
                    else: torch.testing.assert_close(left[key],right[key],atol=0,rtol=0,msg=key)
                batch={k:v.to(args.device) for k,v in left.items() if v is not None}
                old.model.rope_deltas=None; new.model.rope_deltas=None
                pos_args={k:batch.get(k) for k in ('input_ids','image_grid_thw','video_grid_thw','second_per_grid_ts','attention_mask')}
                for x,y in zip(old.model.get_rope_index(**pos_args),new.model.get_rope_index(**pos_args)):
                    torch.testing.assert_close(x,y,atol=0,rtol=0)
                labels=batch['input_ids'].clone(); labels[:,:-3]=-100
                labels[batch['attention_mask']==0]=-100
                with amp():
                    out1=old(**batch,labels=labels,use_cache=False)
                    out2=new(**batch,labels=labels,use_cache=False)
                item['logits_max_abs']=compare(out1.logits,out2.logits,'logits')
                item['loss_max_abs']=compare(out1.loss,out2.loss,'loss')
                del out1,out2
                for beams in (1,2):
                    opts=dict(max_new_tokens=args.max_new_tokens,do_sample=False,num_beams=beams,use_cache=True)
                    old.model.rope_deltas=None; new.model.rope_deltas=None
                    with amp():
                        ids1=old.generate(**batch,**opts); ids2=new.generate(**batch,**opts)
                    if not torch.equal(ids1,ids2): raise AssertionError(f'Generation changed: beams={beams}')
                    item[f'beam_{beams}_token_ids']=ids2.cpu().tolist()
                if name in ('text','single_tile'):
                    new.model.rope_deltas=None
                    with amp(): uncached=new.generate(**batch,max_new_tokens=args.max_new_tokens,do_sample=False,use_cache=False)
                    new.model.rope_deltas=None
                    with amp(): cached=new.generate(**batch,max_new_tokens=args.max_new_tokens,do_sample=False,use_cache=True)
                    if not torch.equal(cached,uncached): raise AssertionError('Cached vs uncached generation differs')
                item['status']='PASS'
                # Only generated continuation in report, not thousands of repeated visual IDs.
                for key in ('beam_1_token_ids','beam_2_token_ids'):
                    item[key]=[seq[batch['input_ids'].shape[1]:] for seq in item[key]]
                print(json.dumps(item,ensure_ascii=False),flush=True)
        report['status']='PASS'
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
        print(f"{report['status']}: {report_path}",flush=True)

if __name__=='__main__': main()
