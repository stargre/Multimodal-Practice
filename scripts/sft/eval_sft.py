"""Token-weighted SFT loss on fixed, image-disjoint held-out split."""
import argparse
from sft_common import load_model,LLaVASFT,indices_from,evaluate_local

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--json',required=True)
    p.add_argument('--image-root',required=True)
    p.add_argument('--indices',required=True)
    p.add_argument('--pt-checkpoint',required=True)
    p.add_argument('--sft-checkpoint',default=None)
    p.add_argument('--max-tokens',type=int,default=4096)
    p.add_argument('--limit',type=int,default=0)
    a=p.parse_args()
    model,proc=load_model(a.pt_checkpoint,sft_checkpoint=a.sft_checkpoint,train=False)
    dataset=LLaVASFT(a.json,a.image_root)
    indices=indices_from(a.indices)
    if a.limit: indices=indices[:a.limit]
    loss,tokens=evaluate_local(model,proc,dataset,indices,a.max_tokens)
    print(f'VAL token_loss={loss:.6f} supervised_tokens={tokens} records={len(indices)}')
    print('CHECKPOINT:',a.sft_checkpoint or 'PT-only (not SFT)')

if __name__=='__main__': main()
