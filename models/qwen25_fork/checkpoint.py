"""Explicit project-checkpoint save/reload (not the HuggingFace AutoModel registry)."""
import json
from pathlib import Path

import torch
import transformers

from .build_model import build_model


def save_checkpoint(
    model, tokenizer, image_processor, save_dir,
    qwen25_path, vit_path, llm_path,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    # torch.save handles shared parameter storage if a backbone ties embeddings.
    torch.save(
        {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
        save_dir / "model_state.pt",
    )
    model.config.save_pretrained(save_dir / "outer_config")
    tokenizer.save_pretrained(save_dir / "tokenizer")
    image_processor.save_pretrained(save_dir / "image_processor")
    with (save_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump({
            "format_version": 1,
            "transformers_version": transformers.__version__,
            "qwen25_path": str(qwen25_path),
            "vit_path": str(vit_path),
            "llm_path": str(llm_path),
            "architecture": "HybridQwen25VLForConditionalGeneration",
            "tile_size": 448,
            "mrope_sections": [16, 24, 24],
        }, f, indent=2, ensure_ascii=False)
    return save_dir


def load_checkpoint(checkpoint_dir, device="cuda", dtype=None):
    checkpoint_dir = Path(checkpoint_dir)
    with (checkpoint_dir / "metadata.json").open(encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    model, tokenizer = build_model(
        device=device,
        dtype=dtype,
        qwen25_path=meta["qwen25_path"],
        vit_path=meta["vit_path"],
        llm_path=meta["llm_path"],
    )
    weights = torch.load(checkpoint_dir / "model_state.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    model.eval()
    return model, tokenizer
