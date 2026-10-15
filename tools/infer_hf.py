"""Offline inference using only the installed hybrid_vlm package and an HF export."""
import argparse
import contextlib
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText,AutoProcessor
import hybrid_vlm


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True); p.add_argument('--image')
    p.add_argument('--text',default='Describe this image in detail.')
    p.add_argument('--device',default='cuda'); p.add_argument('--max-new-tokens',type=int,default=128)
    args=p.parse_args()
    model=AutoModelForImageTextToText.from_pretrained(args.model,local_files_only=True).to(args.device).eval()
    proc=AutoProcessor.from_pretrained(args.model,local_files_only=True)
    proc.tokenizer.padding_side='left'
    images=[Image.open(args.image).convert('RGB')] if args.image else []
    batch=proc(text=proc.user_prompt(args.text,num_images=len(images)),images=images).to(args.device)
    amp=torch.autocast('cuda',dtype=torch.bfloat16) if args.device.startswith('cuda') else contextlib.nullcontext()
    with torch.inference_mode(),amp:
        ids=model.generate(**batch,max_new_tokens=args.max_new_tokens,do_sample=False)
    print(proc.tokenizer.decode(ids[0,batch['input_ids'].shape[1]:],skip_special_tokens=True))

if __name__=='__main__': main()
