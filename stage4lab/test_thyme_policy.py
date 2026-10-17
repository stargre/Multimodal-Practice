#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage-4 smoke test.

Test:

    Thyme-RL sample
        ->
    DPO-v2 checkpoint
        ->
    Thyme prompt construction
        ->
    HybridQwen25Processor
        ->
    Qwen3 + InternViT
        ->
    generation

This is NOT GRPO training yet.
"""


import re
import time
from pathlib import Path

import torch


from thyme_dataset import (
    get_first_sample
)

from grpo_runtime import (
    load_dpo_v2_policy
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)


SMOKE_DIR = (
    PROJECT_ROOT
    /
    "runs"
    /
    "thyme_smoke"
)


IMAGE_DIR = (
    SMOKE_DIR
    /
    "images"
)


# ============================================================
# Prompt construction
# ============================================================

def build_thyme_prompt(
    processor,
    sample,
    image_path,
):

    system_prompt = (
        sample["system_prompt"]
    )

    user_text = (
        sample["user_prompt"]
    )


    # --------------------------------------------------------
    # Thyme dataset contains a textual <image> marker.
    #
    # Our custom processor generates the actual visual
    # placeholder itself.
    # --------------------------------------------------------

    user_text = user_text.lstrip()

    if user_text.startswith(
        "<image>"
    ):

        user_text = (
            user_text[
                len("<image>"):
            ]
            .lstrip("\r\n")
        )


    # --------------------------------------------------------
    # Replace the original author's machine-local path
    # with the actual local path used by this test.
    #
    # This is important because the original path:
    #
    # /mllm_hdd/yfzhang/...
    #
    # does not exist on this machine.
    # --------------------------------------------------------

    path_pattern = (
        r'(### User Image Path:\*\*\s*")'
        r'[^"]*'
        r'(")'
    )


    user_text, path_count = (
        re.subn(
            path_pattern,
            lambda m:
                m.group(1)
                + str(image_path)
                + m.group(2),
            user_text,
            count=1,
        )
    )


    if path_count == 0:

        # This should not happen for current Thyme-RL.
        # Keep the sample usable anyway.

        user_text += (
            "\n\n"
            "### User Image Path:** "
            f"\"{image_path}\"\n"
        )


    # --------------------------------------------------------
    # Update image size metadata.
    # --------------------------------------------------------

    width, height = (
        sample["image"].size
    )


    size_pattern = (
        r'(### User Image Size:\*\*\s*")'
        r'[^"]*'
        r'(")'
    )


    user_text = re.sub(
        size_pattern,
        lambda m:
            m.group(1)
            + f"{width}x{height}"
            + m.group(2),
        user_text,
        count=1,
    )


    # --------------------------------------------------------
    # Construct system message manually.
    # --------------------------------------------------------

    system_block = (
        "<|im_start|>system\n"
        + system_prompt
        + "<|im_end|>\n"
    )


    # --------------------------------------------------------
    # Let the project's processor create the canonical
    # user/image/assistant prefix.
    # --------------------------------------------------------

    user_block = (
        processor.user_prompt(
            user_text,
            num_images=1
        )
    )


    prompt = (
        system_block
        +
        user_block
    )


    # --------------------------------------------------------
    # Safety checks.
    # --------------------------------------------------------

    if prompt.count(
        "<|image_pad|>"
    ) != 1:

        raise RuntimeError(
            "Expected exactly one "
            "<|image_pad|> token, got "
            f"{prompt.count('<|image_pad|>')}"
        )


    if "<image>" in prompt:

        raise RuntimeError(
            "Raw <image> marker remains "
            "after prompt conversion."
        )


    return prompt


# ============================================================
# Move batch
# ============================================================

def move_batch(
    batch,
    device,
):

    result = {}

    for key, value in batch.items():

        if torch.is_tensor(value):

            result[key] = value.to(
                device,
                non_blocking=True
            )

        else:

            result[key] = value


    return result


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "Stage-4 Thyme + DPO-v2 smoke test"
    )

    print(
        "=" * 80
    )


    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is required."
        )


    device = torch.device(
        "cuda"
    )


    # ========================================================
    # Load one Thyme sample
    # ========================================================

    print(
        "\nLoading first Thyme sample...",
        flush=True
    )


    sample = get_first_sample()


    print(
        "Question:",
        sample["question"],
        flush=True
    )

    print(
        "Solution:",
        sample["solution"],
        flush=True
    )

    print(
        "Image:",
        sample["image"].size,
        sample["image"].mode,
        flush=True
    )


    # ========================================================
    # Save local image
    # ========================================================

    IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    image_path = (
        IMAGE_DIR
        /
        "sample_000000.png"
    )


    sample["image"].save(
        image_path,
        format="PNG"
    )


    print(
        "Local image:",
        image_path,
        flush=True
    )


    # ========================================================
    # Load DPO-v2
    # ========================================================

    model, processor = (
        load_dpo_v2_policy()
    )


    model.eval()


    # ========================================================
    # Construct Thyme prompt
    # ========================================================

    prompt = build_thyme_prompt(
        processor,
        sample,
        image_path,
    )


    print(
        "\n========== FINAL PROMPT ==========",
        flush=True
    )

    print(
        prompt,
        flush=True
    )

    print(
        "===================================",
        flush=True
    )


    # ========================================================
    # Encode
    # ========================================================

    print(
        "\nEncoding image + prompt...",
        flush=True
    )


    batch = processor.encode_images(
        [prompt],
        images_per_sample=[
            [sample["image"]]
        ],
        padding=False
    )


    batch = move_batch(
        batch,
        device
    )


    batch = {
        key: value
        for key, value in batch.items()
        if value is not None
    }


    print(
        "input_ids:",
        tuple(
            batch["input_ids"].shape
        ),
        flush=True
    )


    print(
        "attention_mask:",
        tuple(
            batch["attention_mask"].shape
        ),
        flush=True
    )


    print(
        "pixel_values:",
        tuple(
            batch["pixel_values"].shape
        ),
        flush=True
    )


    print(
        "image_grid_thw:",
        batch["image_grid_thw"].tolist(),
        flush=True
    )


    # ========================================================
    # Generate
    # ========================================================

    model.model.rope_deltas = None


    print(
        "\nGenerating...",
        flush=True
    )


    start = time.time()


    with torch.inference_mode():

        output_ids = model.generate(

            **batch,

            max_new_tokens=256,

            do_sample=False,

            num_beams=1,

            eos_token_id=[
                processor.tokenizer.eos_token_id,
                processor.tokenizer.convert_tokens_to_ids(
                    "<|im_end|>"
                ),
            ],

            pad_token_id=(
                processor.tokenizer.pad_token_id
                if processor.tokenizer.pad_token_id
                is not None
                else
                processor.tokenizer.eos_token_id
            ),

            use_cache=True,
        )


    elapsed = (
        time.time()
        - start
    )


    input_length = (
        batch["input_ids"].shape[1]
    )


    generated_ids = (
        output_ids[
            :,
            input_length:
        ]
    )


    output_text = (
        processor.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        .strip()
    )


    # ========================================================
    # Result
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "GENERATED OUTPUT"
    )

    print(
        "=" * 80
    )

    print(
        output_text
    )


    print(
        "\nGenerated tokens:",
        generated_ids.shape[1]
    )

    print(
        "Generation time:",
        f"{elapsed:.3f} sec"
    )


    print(
        "\n" + "=" * 80
    )

    print(
        "SMOKE TEST COMPLETE"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()