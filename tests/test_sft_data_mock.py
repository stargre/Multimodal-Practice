"""Synthetic tokenizer/processor regression; does not need HF or downloaded weights."""
import re
import sys
from pathlib import Path
import torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage2lab.sft_data import render_chat,tokenize_and_mask,expanded_chat,SFTCollator,EOS,IMAGE,visual_token_count

class MockTokenizer:
    is_fast=True
    specials=['<|im_start|>','<|im_end|>','<|vision_start|>','<|image_pad|>','<|vision_end|>']
    def __init__(self):
        self.vocab={x:100+i for i,x in enumerate(self.specials)}
        self.pattern=re.compile('('+'|'.join(re.escape(x) for x in self.specials)+r'|[^\s<]+|\s|<)')
    def convert_tokens_to_ids(self,x): return self.vocab.get(x,-1)
    def __call__(self,text,add_special_tokens=False,return_offsets_mapping=False,padding=False,return_tensors=None):
        if isinstance(text,list):
            assert len(text)==1
            text=text[0]
        matches=list(self.pattern.finditer(text))
        ids=[self.vocab.setdefault(m.group(),len(self.vocab)+1000) for m in matches]
        result={'input_ids':ids}
        if return_offsets_mapping:
            result['offset_mapping']=[m.span() for m in matches]
        if return_tensors=='pt':
            return {'input_ids':torch.tensor([ids]),'attention_mask':torch.ones((1,len(ids)),dtype=torch.long)}
        return result

class MockProcessor:
    def __init__(self,tok):
        self.tokenizer=tok
    def encode_images(self,prompts,images,padding=False):
        raw=prompts[0]
        count=256
        s,_=expanded_chat(raw,[],count)
        out=self.tokenizer([s],return_tensors='pt',padding=False)
        out['pixel_values']=torch.zeros(1,3,448,448)
        out['image_grid_thw']=torch.tensor([[1,32,32]])
        out['video_grid_thw']=None
        out['second_per_grid_ts']=None
        out['pixel_values_videos']=None
        return out


def main():
    tok=MockTokenizer();proc=MockProcessor(tok)
    conv=[{'from':'human','value':'<image>\nWhat is it?'},
          {'from':'gpt','value':'A brown dog.'},
          {'from':'human','value':'What is it doing?'},
          {'from':'gpt','value':'Running on the grass.'}]
    raw,spans=render_chat(conv)
    assert len(spans)==2
    ids,labels=tokenize_and_mask(tok,raw,spans,256)
    assert labels.count(tok.convert_tokens_to_ids(EOS))==2
    assert all(l==-100 for id,l in zip(ids,labels) if id==tok.convert_tokens_to_ids(IMAGE))
    assert any(l!=-100 for l in labels)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'image.png';Image.new('RGB',(448,448)).save(p)
        batch=SFTCollator(proc,4096)([{'id':'x','image_path':str(p),'conversations':conv}])
        assert (batch['labels']!=-100).sum().item()>0
        assert batch['labels'].shape==batch['input_ids'].shape
    gen,_=render_chat([{'from':'human','value':'What is here?'}],generation_prompt=True)
    assert gen.endswith('<|im_start|>assistant\n')
    assert gen.count(IMAGE)==1
    try:
        render_chat([{'from':'human','value':'<image> and <image>'},{'from':'gpt','value':'OK'}])
        raise AssertionError('Duplicated images not rejected')
    except ValueError:
        pass
    assert visual_token_count(448,448)==256
    assert visual_token_count(700,800)==1024
    print('PASS: 2-turn masking, image expansion, generation template, prefilter sizing')

if __name__=='__main__': main()
