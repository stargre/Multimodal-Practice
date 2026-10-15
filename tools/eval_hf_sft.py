"""Evaluate an HF export with the unchanged repository SFT dataset/collator/loss."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'/'sft'))
import torch
from transformers import AutoModelForImageTextToText,AutoProcessor
import hybrid_vlm
from sft_common import LLaVASFT,indices_from,evaluate_local


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True); p.add_argument('--json',required=True)
    p.add_argument('--image-root',required=True); p.add_argument('--indices',required=True)
    p.add_argument('--max-tokens',type=int,default=4096); p.add_argument('--limit',type=int,default=0)
    p.add_argument('--report',required=True)
    a=p.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError('CUDA BF16 required')
    m=AutoModelForImageTextToText.from_pretrained(a.model,local_files_only=True).cuda().eval()
    proc=AutoProcessor.from_pretrained(a.model,local_files_only=True)
    proc.tokenizer.padding_side='right'
    indices=indices_from(a.indices)
    if a.limit: indices=indices[:a.limit]
    loss,tokens=evaluate_local(m,proc,LLaVASFT(a.json,a.image_root),indices,a.max_tokens)
    result=dict(token_loss=loss,supervised_tokens=tokens,records=len(indices),options=vars(a))
    report=Path(a.report); report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(f'VAL token_loss={loss:.6f} supervised_tokens={tokens} records={len(indices)}')

if __name__=='__main__': main()
