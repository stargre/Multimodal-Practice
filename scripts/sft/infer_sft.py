"""Visual chat generation with PT-only or SFT checkpoint. Single image, optional followup."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageOps
import torch
from sft_common import load_model,to_device
from stage2lab.sft_data import VISION,EOS,render_chat,LLaVASFT


def generate(model,proc,img,raw,max_tokens):
    encoded=proc.encode_images([raw],[[img]],padding=False)
    length=encoded['input_ids'].shape[1]
    tok=proc.tokenizer
    eos_ids=list({int(x) for x in (tok.eos_token_id,tok.convert_tokens_to_ids(EOS)) if x is not None and x>=0})
    model.model.rope_deltas=None
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        output=model.generate(**to_device(encoded),max_new_tokens=max_tokens,
                              do_sample=False,num_beams=1,use_cache=True,
                              pad_token_id=tok.pad_token_id or tok.eos_token_id,
                              eos_token_id=eos_ids)
    return tok.decode(output[0,length:],skip_special_tokens=True).strip()


def run_one(model,proc,path,question,follow_up,max_new_tokens):
    with Image.open(path) as f:
        img=ImageOps.exif_transpose(f).convert('RGB')
    turns=[{'from':'human','value':'<image>\n'+question}]
    raw,_=render_chat(turns,generation_prompt=True)
    first=generate(model,proc,img,raw,max_new_tokens)
    result={'image':str(path),'question':question,'prediction':first}
    if follow_up:
        turns.extend([{'from':'gpt','value':first or '(No response)'},
                      {'from':'human','value':follow_up}])
        raw2,_=render_chat(turns,generation_prompt=True)
        result['follow_up']=follow_up
        result['follow_up_prediction']=generate(model,proc,img,raw2,max_new_tokens)
    return result


def main():
    p=argparse.ArgumentParser()
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--image')
    source.add_argument('--json')
    p.add_argument('--image-root')
    p.add_argument('--indices',help='Validation index JSON; required with --json')
    p.add_argument('--examples',type=int,default=8)
    p.add_argument('--question',default='Describe this image in detail.')
    p.add_argument('--follow-up',default=None)
    p.add_argument('--pt-checkpoint',required=True)
    p.add_argument('--sft-checkpoint',default=None)
    p.add_argument('--max-new-tokens',type=int,default=128)
    p.add_argument('--out-json',default=None)
    a=p.parse_args()
    model,proc=load_model(a.pt_checkpoint,sft_checkpoint=a.sft_checkpoint,train=False)
    if a.image:
        tests=[{'image_path':a.image,'question':a.question,'id':Path(a.image).stem}]
    else:
        if not a.image_root or not a.indices: p.error('--json requires --image-root and --indices')
        from sft_common import indices_from
        ds=LLaVASFT(a.json,a.image_root)
        tests=[]
        for idx in indices_from(a.indices)[:a.examples]:
            sample=ds[idx]
            q=sample['conversations'][0]['value'].replace('<image>','').strip()
            tests.append({'image_path':sample['image_path'],'id':sample['id'],
                          'question':q,'reference':sample['conversations'][1]['value']})
    all_results=[]
    for n,x in enumerate(tests,1):
        row=run_one(model,proc,x['image_path'],x['question'],a.follow_up,a.max_new_tokens)
        row['id']=x['id'];row['reference']=x.get('reference')
        all_results.append(row)
        print(f'[{n}/{len(tests)}] {row["id"]}\nQ: {row["question"]}\nGT: {row["reference"]}\nPRED: {row["prediction"]}',flush=True)
        if a.follow_up:
            print('FOLLOW-UP:',row['follow_up_prediction'],flush=True)
    if a.out_json:
        dest=Path(a.out_json);dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(all_results,ensure_ascii=False,indent=2),encoding='utf-8')
        print('Wrote',dest)

if __name__=='__main__':main()
