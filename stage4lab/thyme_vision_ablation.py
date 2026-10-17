#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Thyme-RL visual-input ablation test (inference only).

Conditions:
  A: original Thyme image
  B: blank white image with the same canvas size
  C: image from another Thyme sample, fitted onto the same canvas size

All conditions use the exact same simplified text prompt, DPO-v3 checkpoint,
processor and greedy decoding settings. No gradients, optimizer steps, or
checkpoint writes are performed.

Run from project root:
python -u stage4lab/thyme_vision_ablation.py \\
  --checkpoint runs/dpo_mmrlhf_v3/last_model.pt \\
  --out runs/thyme_vision_ablation \\
  --dataset_index 0 --alternate_index 1 --max_new_tokens 256
"""

import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageChops

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare original, blank, and swapped Thyme images; inference only."
    )
    parser.add_argument("--checkpoint", default="runs/dpo_mmrlhf_v3/last_model.pt")
    parser.add_argument("--out", default="runs/thyme_vision_ablation")
    parser.add_argument("--dataset_index", type=int, default=0)
    parser.add_argument("--alternate_index", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
        if value is not None
    }


def build_prompt(processor, question):
    system_prompt = (
        "You are a careful visual math assistant. Read the image accurately. "
        "Identify all relevant values before calculating. Check the count of values "
        "and arithmetic. Give a concise explanation and clearly state the final numerical answer."
    )
    user_text = (
        "Read every score in the table carefully. List all the scores you can see. "
        "Count the number of scores. Calculate their arithmetic mean. "
        f"Question: {question}"
    )
    system_block = "<|im_start|>system\n" + system_prompt + "<|im_end|>\n"
    return system_block + processor.user_prompt(user_text, num_images=1)


def get_eos_ids(tokenizer):
    eos_ids = []
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, (tuple, list)):
        eos_ids.extend(int(value) for value in eos if value is not None)
    elif eos is not None:
        eos_ids.append(int(eos))
    try:
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        unk = getattr(tokenizer, "unk_token_id", None)
        if im_end is not None and im_end >= 0 and im_end != unk:
            eos_ids.append(int(im_end))
    except Exception:
        pass
    return sorted(set(eos_ids))


def trim_at_stop(ids, stop_ids, pad_id):
    values = ids.tolist() if torch.is_tensor(ids) else list(ids)
    stop_set = set(stop_ids)
    for index, token_id in enumerate(values):
        if token_id in stop_set:
            return values[:index]
        if pad_id is not None and int(token_id) == int(pad_id):
            return values[:index]
    return values


def diagnostic_answer(text):
    """Best-effort diagnostic extraction only; this is not the GRPO reward function."""
    match = re.search(
        r"<answer>\s*(.*?)\s*</answer>", text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    numbers = re.findall(
        r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*/\s*\d+)?",
        text,
    )
    return numbers[-1].strip() if numbers else ""


def get_sample_at(stream, wanted_index):
    for index, sample in enumerate(stream):
        if index == wanted_index:
            return sample
    raise IndexError(f"Thyme stream has no sample at index {wanted_index}")


def image_sha256(image):
    import io
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def fit_image_to_canvas(image, canvas_size):
    """Preserve aspect ratio; center the alternate image on a white canvas."""
    image = image.convert("RGB")
    target_w, target_h = canvas_size
    image.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), color=(255, 255, 255))
    x = (target_w - image.width) // 2
    y = (target_h - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


@torch.inference_mode()
def run_greedy(model, processor, image, prompt, device, max_new_tokens, seed):
    set_seed(seed)
    batch = processor.encode_images(
        [prompt], images_per_sample=[[image]], padding=True
    )
    batch = move_batch_to_device(batch, device)

    if "input_ids" not in batch:
        raise RuntimeError("encode_images() did not return input_ids.")
    if "pixel_values" not in batch:
        raise RuntimeError(
            "encode_images() did not return pixel_values; visual input is missing."
        )

    pixel_values = batch["pixel_values"]
    finite = bool(torch.isfinite(pixel_values).all().item())
    pixel_mean = float(pixel_values.float().mean().item())
    pixel_std = float(pixel_values.float().std().item())
    shapes = {
        key: list(value.shape)
        for key, value in batch.items()
        if torch.is_tensor(value)
    }
    print("  Batch tensor shapes:", shapes, flush=True)
    print("  pixel_values finite:", finite, flush=True)
    print("  pixel_values mean/std:", pixel_mean, pixel_std, flush=True)

    model_core = getattr(model, "model", None)
    if model_core is not None and hasattr(model_core, "rope_deltas"):
        model_core.rope_deltas = None

    tokenizer = processor.tokenizer
    eos_ids = get_eos_ids(tokenizer)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
    }
    if pad_id is not None:
        generation_kwargs["pad_token_id"] = int(pad_id)
    if eos_ids:
        generation_kwargs["eos_token_id"] = eos_ids

    start = time.time()
    output_ids = model.generate(**batch, **generation_kwargs)
    seconds = round(time.time() - start, 3)

    input_length = batch["input_ids"].shape[1]
    response_ids = (
        output_ids[0, input_length:]
        if output_ids.shape[1] >= input_length
        else output_ids[0]
    )
    trimmed = trim_at_stop(response_ids, eos_ids, pad_id)
    text = tokenizer.decode(
        trimmed,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return {
        "completion": text,
        "diagnostic_answer": diagnostic_answer(text),
        "generated_tokens_after_trim": len(trimmed),
        "generation_seconds": seconds,
        "batch_shapes": shapes,
        "pixel_values_finite": finite,
        "pixel_values_mean": pixel_mean,
        "pixel_values_std": pixel_std,
    }


def main():
    args = parse_args()
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print("=" * 88, flush=True)
    print("THYME VISION INPUT ABLATION (INFERENCE ONLY)", flush=True)
    print("Project root:", PROJECT_ROOT, flush=True)
    print("Checkpoint:", checkpoint, flush=True)
    print("CUDA available:", torch.cuda.is_available(), flush=True)
    print("Original sample index:", args.dataset_index, flush=True)
    print("Alternate image sample index:", args.alternate_index, flush=True)
    print("=" * 88, flush=True)

    from stage4lab.thyme_dataset import ThymeRLStream
    from stage4lab.grpo_runtime import load_dpo_v2_policy

    original_sample = get_sample_at(
        ThymeRLStream(split="train"), args.dataset_index
    )
    alternate_sample = get_sample_at(
        ThymeRLStream(split="train"), args.alternate_index
    )

    original_image = original_sample["image"].convert("RGB")
    alternate_source = alternate_sample["image"].convert("RGB")
    alternate_image = fit_image_to_canvas(alternate_source, original_image.size)
    blank_image = Image.new("RGB", original_image.size, color=(255, 255, 255))

    images = {
        "A_original_image": original_image,
        "B_blank_white_image": blank_image,
        "C_other_sample_image": alternate_image,
    }
    image_paths = {}
    for name, image in images.items():
        path = out_dir / f"{name}.png"
        image.save(path)
        image_paths[name] = str(path)

    diff = ImageChops.difference(original_image, blank_image)
    image_checks = {
        "original_vs_blank_difference_bbox": list(diff.getbbox()) if diff.getbbox() else None,
        "original_vs_alternate_sha256_equal": (
            image_sha256(original_image) == image_sha256(alternate_image)
        ),
        "original_sha256": image_sha256(original_image),
        "blank_sha256": image_sha256(blank_image),
        "alternate_sha256": image_sha256(alternate_image),
    }

    print("Question:", original_sample["question"], flush=True)
    print("Gold solution:", repr(original_sample["solution"]), flush=True)
    print("Original image size:", original_image.size, flush=True)
    print("Alternate sample question (not used as prompt):", alternate_sample["question"], flush=True)
    print("Alternate source image size:", alternate_source.size, flush=True)
    print("Test image paths:", json.dumps(image_paths, indent=2), flush=True)
    print("Image identity checks:", json.dumps(image_checks, indent=2), flush=True)

    print("\nLoading DPO-v3 checkpoint...", flush=True)
    model, processor = load_dpo_v2_policy(
        checkpoint_path=str(checkpoint), freeze_visual=True
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = next(model.parameters()).device
    print("Model:", type(model).__name__, flush=True)
    print("Processor:", type(processor).__name__, flush=True)
    print("Device:", device, flush=True)

    # The same prompt string is used for all conditions; only the image changes.
    prompt = build_prompt(processor, original_sample["question"])
    print("Prompt length:", len(prompt), flush=True)
    print("Prompt tail:\n", prompt[-900:], flush=True)

    results = []
    for condition_name, image in images.items():
        print("\n" + "#" * 88, flush=True)
        print("CONDITION:", condition_name, flush=True)
        print("Image:", image_paths[condition_name], flush=True)
        print("Image size:", image.size, "SHA256:", image_sha256(image), flush=True)

        result = run_greedy(
            model=model,
            processor=processor,
            image=image,
            prompt=prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )
        results.append({
            "condition": condition_name,
            "image_path": image_paths[condition_name],
            "image_sha256": image_sha256(image),
            "question": original_sample["question"],
            "gold_solution": original_sample["solution"],
            **result,
        })
        print("Generation seconds:", result["generation_seconds"], flush=True)
        print("Generated tokens:", result["generated_tokens_after_trim"], flush=True)
        print("Diagnostic answer (not a reward):", repr(result["diagnostic_answer"]), flush=True)
        print("Completion:\n" + result["completion"], flush=True)

    result_path = out_dir / "vision_ablation_results.json"
    payload = {
        "checkpoint": str(checkpoint),
        "dataset_index": args.dataset_index,
        "alternate_index": args.alternate_index,
        "question": original_sample["question"],
        "gold_solution": original_sample["solution"],
        "prompt": prompt,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "image_paths": image_paths,
        "image_identity_checks": image_checks,
        "results": results,
        "warning": (
            "Qualitative input-ablation diagnostic, not definitive causal proof. "
            "Greedy outputs can vary across software/hardware versions."
        ),
    }
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print("\n" + "=" * 88, flush=True)
    print("VISION ABLATION FINISHED.", flush=True)
    print("No optimizer step and no checkpoint update occurred.", flush=True)
    print("Results JSON:", result_path, flush=True)
    print("=" * 88, flush=True)


if __name__ == "__main__":
    main()
