"""Export PT/SFT/DPO weights as a self-contained HF checkpoint; never overwrite output."""
import argparse
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import transformers
from hybrid_vlm import HybridVLMConfig,HybridVLMForConditionalGeneration,HybridVLMProcessor
from hf_legacy import add_legacy_args,load_legacy,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    add_legacy_args(p)
    p.add_argument('--output',required=True)
    p.add_argument('--max-shard-size',default='2GB')
    args=p.parse_args()
    out=Path(args.output).resolve()
    if out.exists(): raise FileExistsError(f'Output already exists; choose a new directory: {out}')
    print('Loading original qwen25_fork on CPU...',flush=True)
    old,proc=load_legacy(args)
    config=HybridVLMConfig.from_legacy(old)
    config._attn_implementation='sdpa'
    new=HybridVLMForConditionalGeneration(config).eval()
    new.load_state_dict(old.state_dict(),strict=True)
    for k,v in old.state_dict().items():
        w=new.state_dict()[k]
        if v.dtype!=w.dtype or not torch.equal(v,w): raise AssertionError(f'Conversion changed {k}')
    new.generation_config=copy.deepcopy(old.generation_config)
    out.mkdir(parents=True)
    # A failed partial export intentionally remains without EXPORT_COMPLETE.json.
    new.save_pretrained(out,safe_serialization=True,max_shard_size=args.max_shard_size)
    HybridVLMProcessor.from_legacy(proc).save_pretrained(out)
    metadata={'format_version':1,'source_commit':'ab0fc63fb9019ccabd61017eada584b8f09c4cd9',
        'stage':args.stage,'checkpoint':str(Path(args.checkpoint).resolve()),
        'checkpoint_sha256':sha256(args.checkpoint),'transformers':transformers.__version__,
        'torch':torch.__version__,'component_dtypes':config.component_dtypes,
        'base_configs':{name:{'path':str(Path(path).resolve()),'sha256':sha256(Path(path,'config.json'))}
            for name,path in [('qwen25',args.qwen25),('vit',args.vit),('llm',args.llm)]},
        'state_dict_exact_match':True,'note':'Optimizer/scheduler/RNG state excluded. Numeric forward validation is a separate required step.'}
    (out/'EXPORT_COMPLETE.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n')
    print(f'EXPORT PASS: {out}\nNext run tools/verify_hf_export.py against the original checkpoint.',flush=True)

if __name__=='__main__': main()
