#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose Thyme-RL image/prompt inputs without performing any GRPO update."""
from pathlib import Path
import sys
import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
STAGE4_DIR = PROJECT_ROOT / "stage4lab"
for p in (PROJECT_ROOT, STAGE4_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from thyme_dataset import ThymeRLStream
from grpo_runtime import load_dpo_v2_policy
from train_grpo import build_thyme_prompt, move_batch_to_device, get_stop_ids, trim_for_reward, autocast_bf16


def main():
    checkpoint = PROJECT_ROOT / "runs" / "dpo_mmrlhf_v3" / "last_model.pt"
    out_dir = PROJECT_ROOT / "runs" / "thyme_input_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / "sample_000005.png"

    print("=" * 88)
    print("THYME MULTIMODAL INPUT PROBE (NO TRAINING)")
    print("Project root:", PROJECT_ROOT)
    print("Checkpoint:", checkpoint)
    print("CUDA:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError("This custom model probe expects CUDA, matching Stage-4 training.")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    sample = next(iter(ThymeRLStream(split="train", shuffle=False, shuffle_buffer=1)))
    sample["image"].save(image_path, format="PNG")
    print("Dataset index:", sample.get("index"))
    print("Question:", sample["question"])
    print("Gold solution:", repr(sample["solution"]))
    print("Decoded PIL image:", sample["image"].mode, sample["image"].size)
    print("Saved image:", image_path, "bytes=", image_path.stat().st_size)

    model, processor = load_dpo_v2_policy(
        checkpoint_path=checkpoint,
        freeze_visual=True,
        trainable=False,
        fp32_trainable=False,
    )
    model.to(device)
    model.eval()
    model.model.rope_deltas = None

    prompt = build_thyme_prompt(processor, sample, image_path)
    print("\n--- PROMPT TAIL (last 1800 chars) ---")
    print(prompt[-1800:])
    print("\nPrompt literal <|image_pad|> count:", prompt.count("<|image_pad|>"))
    print("Prompt raw <image> count:", prompt.count("<image>"))

    cpu_batch = processor.encode_images(
        [prompt],
        images_per_sample=[[sample["image"]]],
        padding=True,
    )
    print("\n--- ENCODED BATCH ---")
    print("Batch keys:", list(cpu_batch.keys()))
    for key, value in cpu_batch.items():
        if torch.is_tensor(value):
            desc = f"shape={tuple(value.shape)} dtype={value.dtype} device={value.device}"
            if value.numel() > 0 and (value.is_floating_point() or value.is_complex()):
                finite = torch.isfinite(value)
                desc += f" finite={int(finite.sum().item())}/{value.numel()}"
                if finite.any():
                    vf = value[finite].float()
                    desc += f" min={vf.min().item():.6g} max={vf.max().item():.6g} std={vf.std(unbiased=False).item():.6g}"
            elif value.numel() > 0:
                desc += f" min={value.min().item()} max={value.max().item()}"
            print(f"  {key}: {desc}")
        else:
            print(f"  {key}: type={type(value).__name__} repr={repr(value)[:300]}")

    tokenizer = processor.tokenizer
    ids = cpu_batch.get("input_ids")
    if ids is not None:
        print("\ninput_ids shape:", tuple(ids.shape))
        for token in ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>", "<|im_start|>", "<|im_end|>"):
            token_id = tokenizer.convert_tokens_to_ids(token)
            count = int((ids == token_id).sum().item()) if token_id is not None and token_id >= 0 else -1
            print(f"token {token!r}: id={token_id}, count_in_input_ids={count}")
        print("Decoded encoded-input tail:")
        print(tokenizer.decode(ids[0, -240:].tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False))

    batch = move_batch_to_device(cpu_batch, device)
    stop_ids = get_stop_ids(tokenizer)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        eos = tokenizer.eos_token_id
        pad_id = eos[0] if isinstance(eos, (list, tuple)) and eos else eos

    print("\n--- GREEDY GENERATION (NO TRAINING) ---")
    with torch.inference_mode(), autocast_bf16(device):
        output_ids = model.generate(
            **batch,
            max_new_tokens=256,
            do_sample=False,
            num_beams=1,
            eos_token_id=stop_ids if stop_ids else None,
            pad_token_id=pad_id,
            use_cache=True,
        )
    prompt_len = int(batch["input_ids"].shape[1])
    response_ids = output_ids[0, prompt_len:]
    response_ids = trim_for_reward(response_ids, stop_ids, pad_id)
    completion = tokenizer.decode(
        response_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    print(completion)
    print("\nProbe finished. No optimizer or checkpoint update was run.")


if __name__ == "__main__":
    main()
