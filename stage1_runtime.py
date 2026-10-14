"""Stage-1 loader for the exact APIs used by the user's test-all.py.

Run from the multimodal_training project with checkpoints/ and models/ present.
No model architecture, patch merger, or MRoPE implementation is changed here.
"""
from pathlib import Path

import torch
import transformers
from transformers import CLIPImageProcessor

from models.qwen25_fork.build_model import build_model
from models.qwen25_fork.processing_hybrid import HybridQwen25Processor


PROJECT_ROOT = Path(__file__).resolve().parent


def load_stage1_components():
    if transformers.__version__ != "4.57.6":
        raise RuntimeError(
            f"This project was tested with transformers==4.57.6; "
            f"found {transformers.__version__}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-1 PT debug loader expects a CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The current GPU does not support BF16")

    qwen25_path = PROJECT_ROOT / "checkpoints" / "Qwen2.5-VL-3B-Instruct"
    vit_path = PROJECT_ROOT / "checkpoints" / "InternViT-300M-448px-V2_5"
    llm_path = PROJECT_ROOT / "checkpoints" / "Qwen3-0.6B-Base"

    for path in (qwen25_path, vit_path, llm_path):
        if not path.is_dir():
            raise FileNotFoundError(
                f"Checkpoint directory not found: {path}\n"
                "If your working test-all.py used different --qwen25/--vit/--llm "
                "paths, change the three paths in stage1_runtime.py accordingly."
            )

    print("Loading stage-1 model with build_model(...) ...", flush=True)
    model, tokenizer = build_model(
        device="cuda",
        qwen25_path=str(qwen25_path),
        vit_path=str(vit_path),
        llm_path=str(llm_path),
        dtype=torch.bfloat16,
    )

    # Inference used left padding. For BS=1 debugging padding is absent; prepare
    # for ordinary right-padded causal LM training without changing tokenizer IDs.
    tokenizer.padding_side = "right"
    image_processor = CLIPImageProcessor.from_pretrained(str(vit_path))
    processor = HybridQwen25Processor(
        tokenizer, image_processor, max_tiles_per_side=2
    )
    
    # The training adapter needs the SAME tokenizer for assistant target tokens.
    # HybridQwen25Processor is the stage-1 custom class, not a generic HF processor.
    processor.tokenizer = tokenizer
    model.model.rope_deltas = None

    print("Loaded stage-1 model:", type(model).__name__, flush=True)
    print("Image processor:", type(processor).__name__, flush=True)
    return model, processor
