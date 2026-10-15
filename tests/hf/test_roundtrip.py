import copy
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, AutoProcessor, AutoModelForImageTextToText
from hybrid_vlm import HybridVLMConfig, HybridVLMForConditionalGeneration, HybridVLMProcessor
from hybrid_vlm.image_processing_hybrid import HybridImageProcessor

torch.set_num_threads(2)

def tiny():
    cfg = HybridVLMConfig(
        text_config=dict(vocab_size=32, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=16, tie_word_embeddings=True, eos_token_id=2, pad_token_id=0),
        internvit_config=dict(image_size=448, patch_size=14, hidden_size=16,
            intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
            use_flash_attn=False),
        vision_config=dict(spatial_merge_size=2, tokens_per_second=2),
        mrope_sections=[2,3,3], image_token_id=4, video_token_id=5,
        vision_start_token_id=6, vision_end_token_id=7)
    cfg._attn_implementation='sdpa'
    vocab={'[PAD]':0,'[UNK]':1,'[EOS]':2,'hello':3,'<|image_pad|>':4,
        '<|video_pad|>':5,'<|vision_start|>':6,'<|vision_end|>':7,
        '<|im_start|>':8,'<|im_end|>':9,'user':10,'assistant':11}
    tok=Tokenizer(WordLevel(vocab, unk_token='[UNK]')); tok.pre_tokenizer=Whitespace()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=tok, pad_token='[PAD]',
        eos_token='[EOS]', unk_token='[UNK]', additional_special_tokens=list(vocab)[4:10])
    tokenizer.padding_side='left'
    proc=HybridVLMProcessor(HybridImageProcessor(size={'shortest_edge':448},
        crop_size={'height':448,'width':448}),tokenizer)
    return HybridVLMForConditionalGeneration(cfg).eval(),proc


def test_roundtrip(tmp_path):
    model,proc=tiny()
    model.model.visual.vit.bfloat16()
    model.save_pretrained(tmp_path, max_shard_size='50KB')
    proc.save_pretrained(tmp_path)
    restored,info=AutoModelForImageTextToText.from_pretrained(tmp_path,output_loading_info=True)
    assert not any(info[k] for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')),info
    assert next(restored.model.visual.vit.parameters()).dtype==torch.bfloat16
    for k,v in model.state_dict().items():
        assert v.dtype==restored.state_dict()[k].dtype,k
        torch.testing.assert_close(v,restored.state_dict()[k],rtol=0,atol=0)
    p2=AutoProcessor.from_pretrained(tmp_path)
    for images in ([],[Image.new('RGB',(448,448),'red')],[Image.new('RGB',(700,700),'blue')]):
        prompt=proc.user_prompt('hello',num_images=len(images))
        b=proc(text=prompt,images=images); b2=p2(text=prompt,images=images)
        for k in b: torch.testing.assert_close(b[k],b2[k],rtol=0,atol=0)
        with torch.no_grad():
            model.model.rope_deltas=None; restored.model.rope_deltas=None
            a=model(**b,use_cache=False).logits; c=restored(**b2,use_cache=False).logits
        torch.testing.assert_close(a,c,rtol=0,atol=0)
        for beams in (1,2):
            model.model.rope_deltas=None; restored.model.rope_deltas=None
            opts=dict(max_new_tokens=3,do_sample=False,num_beams=beams)
            assert torch.equal(model.generate(**b,**opts),restored.generate(**b2,**opts))
    uniform=AutoModelForImageTextToText.from_pretrained(tmp_path,dtype=torch.float32)
    assert all(p.dtype==torch.float32 for p in uniform.parameters())


def test_legacy_equivalence():
    from models.qwen25_fork.modeling_hybrid import HybridQwen25VLForConditionalGeneration
    from models.qwen25_fork.processing_hybrid import HybridQwen25Processor
    model,proc=tiny()
    old=HybridQwen25VLForConditionalGeneration(copy.deepcopy(model.config),
        copy.deepcopy(model.model.visual),copy.deepcopy(model.model.language_model),copy.deepcopy(model.lm_head)).eval()
    legacy_proc=HybridQwen25Processor(proc.tokenizer,proc.image_processor,max_tiles_per_side=2)
    image=Image.new('RGB',(700,700),'green')
    cases=[dict(images_per_sample=[[image],[]]),
           dict(videos_per_sample=[[[Image.new('RGB',(448,448)),Image.new('RGB',(448,448))]]],fps_per_sample=[[2.]])]
    for data in cases:
        prompts=([proc.user_prompt('hello',num_images=1),proc.user_prompt('hello hello')]
                 if 'images_per_sample' in data else [proc.user_prompt('hello',num_videos=1)])
        a=legacy_proc.encode_batch(prompts,**data); b=proc.encode_batch(prompts,**data)
        for k in a:
            if a[k] is not None: torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
        with torch.no_grad():
            old.model.rope_deltas=None; model.model.rope_deltas=None
            torch.testing.assert_close(old(**a,use_cache=False).logits,model(**b,use_cache=False).logits,rtol=0,atol=0)


def test_fresh_process_offline(tmp_path):
    import os,subprocess,sys
    model,proc=tiny(); export=tmp_path/'export'
    model.save_pretrained(export); proc.save_pretrained(export)
    env=dict(os.environ,HF_HOME=str(tmp_path/'empty-cache'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    env.pop('PYTHONPATH',None)
    code='''
import importlib.util,sys,torch
assert importlib.util.find_spec('models') is None, 'legacy repo is importable'
import hybrid_vlm
from transformers import AutoModelForImageTextToText,AutoProcessor
torch.set_num_threads(2)
m=AutoModelForImageTextToText.from_pretrained(sys.argv[1],local_files_only=True)
p=AutoProcessor.from_pretrained(sys.argv[1],local_files_only=True)
assert p.tokenizer.padding_side=='left'
b=p(text=p.user_prompt('hello'))
m.generate(**b,max_new_tokens=2,do_sample=False)
print('OFFLINE PASS')
'''
    result=subprocess.run([sys.executable,'-c',code,str(export)],cwd=tmp_path,env=env,text=True,capture_output=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'OFFLINE PASS' in result.stdout


def test_checkpoint_guards():
    import sys
    from pathlib import Path
    import pytest
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
    from hf_legacy import apply_checkpoint
    m,_=tiny()
    full={k:v.clone() for k,v in m.state_dict().items()}
    sft={k:v for k,v in full.items() if k.startswith(('model.visual.merger.','model.language_model.','lm_head.'))}
    apply_checkpoint(m,{'trainable':sft},'sft')
    apply_checkpoint(m,{'state_dict':full},'dpo')
    apply_checkpoint(m,{'merger':m.model.visual.merger.state_dict()},'pt')
    broken=dict(sft); broken.pop(next(iter(broken)))
    with pytest.raises(ValueError,match='Incomplete'): apply_checkpoint(m,broken,'sft')
    broken=dict(sft); broken['lm_head.weight']=sft['lm_head.weight']+1
    with pytest.raises(ValueError,match='Conflicting tied'): apply_checkpoint(m,broken,'sft')


def test_conversion_and_full_verifier(tmp_path,monkeypatch):
    import sys,json
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
    import convert_hf_checkpoint as convert
    import verify_hf_export as verify
    from models.qwen25_fork.modeling_hybrid import HybridQwen25VLForConditionalGeneration
    from models.qwen25_fork.processing_hybrid import HybridQwen25Processor
    m,p=tiny()
    old=HybridQwen25VLForConditionalGeneration(copy.deepcopy(m.config),copy.deepcopy(m.model.visual),
        copy.deepcopy(m.model.language_model),copy.deepcopy(m.lm_head)).eval()
    old_proc=HybridQwen25Processor(p.tokenizer,p.image_processor,max_tiles_per_side=2)
    # Legacy build_model leaves these independent despite a tied source config.
    with torch.no_grad(): old.lm_head.weight.add_(0.1)
    assert HybridVLMConfig.from_legacy(old).tie_word_embeddings is False
    torch.save(old.state_dict(),tmp_path/'legacy.pt')
    (tmp_path/'config.json').write_text('{}')
    args=['--stage','dpo','--checkpoint',str(tmp_path/'legacy.pt'),
        '--qwen25',str(tmp_path),'--vit',str(tmp_path),'--llm',str(tmp_path)]
    monkeypatch.setattr(convert,'load_legacy',lambda args:(old,old_proc))
    monkeypatch.setattr(sys,'argv',['convert',*args,'--output',str(tmp_path/'export')])
    convert.main()
    monkeypatch.setattr(verify,'load_legacy',lambda args:(old,old_proc))
    monkeypatch.setattr(sys,'argv',['verify',*args,'--export',str(tmp_path/'export'),
        '--report',str(tmp_path/'report.json'),'--device','cpu','--max-new-tokens','2'])
    verify.main()
    report=json.loads((tmp_path/'report.json').read_text())
    assert report['status']=='PASS'
    assert len(report['cases'])==7
