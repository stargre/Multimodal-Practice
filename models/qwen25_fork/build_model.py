"""Create the modified Qwen2.5-VL model from three existing checkpoints."""
import copy
from pathlib import Path

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig

from .configuration_hybrid import HybridQwen25VLConfig
from .visual_intern import InternViTQwenVisual
from .qwen3_mrope import Qwen3MRoPEModel
from .modeling_hybrid import HybridQwen25VLForConditionalGeneration

QWEN25_PATH = "./checkpoints/Qwen2.5-VL-3B-Instruct"
VIT_PATH = "./checkpoints/InternViT-300M-448px-V2_5"
LLM_PATH = "./checkpoints/Qwen3-0.6B-Base"


def build_model(
    device="cuda", qwen25_path=QWEN25_PATH,
    vit_path=VIT_PATH, llm_path=LLM_PATH,
    dtype=None, chunk_size=4,
):
    if dtype is None:
        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    # Only the outer multimodal CONFIG is required. Original Qwen2.5-VL weights are not loaded.
    q25_config = (
        Qwen2_5_VLConfig.from_pretrained(qwen25_path)
        if Path(qwen25_path).is_dir()
        else Qwen2_5_VLConfig()
    )
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    vit_cfg = AutoConfig.from_pretrained(vit_path, trust_remote_code=True)
    vit_cfg.use_flash_attn = False  # suppress missing flash-attn warnings in old InternViT code

    vit = AutoModel.from_pretrained(
        vit_path, config=vit_cfg, trust_remote_code=True, dtype=dtype
    ).to(device).eval()
    vit.requires_grad_(False)
    qwen3 = AutoModelForCausalLM.from_pretrained(
        llm_path, dtype=dtype, attn_implementation="sdpa"
    ).to(device).eval()
    config = HybridQwen25VLConfig.from_components(q25_config, qwen3.config, vit.config, tokenizer)
    config._attn_implementation = "sdpa"
    visual = InternViTQwenVisual(
        vit, llm_hidden_size=qwen3.config.hidden_size, chunk_size=chunk_size,
    ).to(device)

    # Initialize Qwen3 MRoPE backbone; copy pretrained weights before discarding old holder.
    backbone = Qwen3MRoPEModel(qwen3.config).to(device=device, dtype=dtype)
    backbone.load_state_dict(qwen3.model.state_dict(), strict=True)
    lm_head = qwen3.lm_head
    gen_config = copy.deepcopy(qwen3.generation_config)
    model = HybridQwen25VLForConditionalGeneration(
        config, visual, backbone, lm_head,
    ).to(device)
    model.generation_config = gen_config
    model.eval()
    del qwen3
    return model, tokenizer
