#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage-4A: Thyme-RL rollout baseline.

Purpose
-------

This script is NOT GRPO training.

It tests whether the Stage-3 DPO-v2 policy has useful exploration
potential on Thyme-RL.

For each Thyme problem:

    one image + one prompt
            |
            v
       DPO-v2 policy
            |
            +---- rollout 0
            +---- rollout 1
            +---- rollout 2
            +---- rollout 3
            |
            v
      answer / format reward
            |
            v
       group statistics

The most important diagnostic is the reward distribution inside each
group of G generations.

Examples:

    [0, 0, 0, 0]
        -> no useful positive signal

    [1, 1, 1, 1]
        -> already solved

    [0, 1, 0, 1]
        -> very useful GRPO training signal

    [0.2, 0.0, 0.2, 0.0]
        -> format-only diversity

This script intentionally does NOT implement:

    - sandbox execution
    - image crop / resize execution
    - external LLM reward judge
    - GRPO loss
    - optimizer
    - backward
    - checkpoint saving

Those will be added only after this rollout baseline is verified.
"""


# ============================================================
# Standard library
# ============================================================

import argparse
import base64
import io
import json
import math
import random
import re
import sys
import time
import unicodedata

from pathlib import Path
from statistics import mean, pstdev


# ============================================================
# Third-party
# ============================================================

import torch

from datasets import load_dataset
from PIL import Image


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


from grpo_runtime import (
    load_dpo_v2_policy
)


# ============================================================
# Constants
# ============================================================

THYME_DATASET = (
    "Kwai-Keye/Thyme-RL"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    /
    "runs"
    /
    "thyme_rollout_baseline"
)

DEFAULT_NUM_SAMPLES = 8

DEFAULT_NUM_GENERATIONS = 4

DEFAULT_MAX_NEW_TOKENS = 512

DEFAULT_TEMPERATURE = 1.0

DEFAULT_TOP_P = 0.9

DEFAULT_TOP_K = 50

DEFAULT_REPETITION_PENALTY = 1.05

DEFAULT_SEED = 42

DEFAULT_SHUFFLE_BUFFER = 1000

DEFAULT_FORMAT_WEIGHT = 0.2


# ============================================================
# Random seed
# ============================================================

def set_global_seed(
    seed
):

    random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed(
            seed
        )

        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# JSON utility
# ============================================================

def save_json(
    path,
    obj
):

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    tmp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with tmp_path.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2
        )

    tmp_path.replace(
        path
    )


# ============================================================
# Thyme dataset
# ============================================================

def load_thyme_stream(
    seed,
    shuffle=False,
    shuffle_buffer=1000
):

    print(
        "=" * 80,
        flush=True
    )

    print(
        "Loading Thyme-RL streaming dataset...",
        flush=True
    )

    print(
        "Dataset:",
        THYME_DATASET,
        flush=True
    )

    print(
        "Split: train",
        flush=True
    )

    print(
        "Streaming: True",
        flush=True
    )


    dataset = load_dataset(
        THYME_DATASET,
        split="train",
        streaming=True
    )


    if shuffle:

        print(
            "Streaming shuffle enabled.",
            flush=True
        )

        dataset = dataset.shuffle(
            seed=seed,
            buffer_size=shuffle_buffer
        )


    return dataset


# ============================================================
# Dataset sample -> system/user/image
# ============================================================

def parse_thyme_sample(
    raw_sample
):

    messages = (
        raw_sample["messages"]
    )


    if not isinstance(
        messages,
        dict
    ):

        raise RuntimeError(
            "Unexpected Thyme messages type: "
            f"{type(messages)}"
        )


    roles = messages.get(
        "role",
        []
    )

    contents = messages.get(
        "content",
        []
    )


    if len(roles) != len(contents):

        raise RuntimeError(
            "messages.role and messages.content "
            "have different lengths."
        )


    system_prompt = None

    user_prompt = None


    for role, content in zip(
        roles,
        contents
    ):

        if role == "system" and system_prompt is None:

            system_prompt = content

        elif role == "user" and user_prompt is None:

            user_prompt = content


    if system_prompt is None:

        raise RuntimeError(
            "Thyme sample has no system message."
        )


    if user_prompt is None:

        raise RuntimeError(
            "Thyme sample has no user message."
        )


    images = (
        raw_sample.get(
            "images",
            []
        )
    )


    if not images:

        raise RuntimeError(
            "Thyme sample contains no image."
        )


    if len(images) != 1:

        raise RuntimeError(
            "Current Stage-4A rollout expects "
            "exactly one image, but got "
            f"{len(images)}."
        )


    solution = str(
        raw_sample.get(
            "solution",
            ""
        )
    )


    question = str(
        raw_sample.get(
            "question",
            ""
        )
    )


    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "image_b64": images[0],
        "solution": solution,
        "question": question,
    }


# ============================================================
# Base64 image decoding
# ============================================================

def decode_image(
    image_b64
):

    if not isinstance(
        image_b64,
        str
    ):

        raise RuntimeError(
            "Image field is not a string."
        )


    # --------------------------------------------------------
    # Support both:
    #
    # raw base64
    #
    # and:
    #
    # data:image/png;base64,...
    # --------------------------------------------------------

    if image_b64.startswith(
        "data:"
    ):

        if "," not in image_b64:

            raise RuntimeError(
                "Malformed data URI."
            )

        image_b64 = (
            image_b64.split(
                ",",
                1
            )[1]
        )


    try:

        image_bytes = (
            base64.b64decode(
                image_b64
            )
        )

    except Exception as e:

        raise RuntimeError(
            "Base64 decoding failed: "
            f"{e}"
        )


    try:

        image = Image.open(
            io.BytesIO(
                image_bytes
            )
        )

        image = image.convert(
            "RGB"
        )

    except Exception as e:

        raise RuntimeError(
            "PIL image decoding failed: "
            f"{e}"
        )


    return image


# ============================================================
# Prompt normalization
# ============================================================

def build_thyme_prompt(
    processor,
    sample,
    image_path
):

    system_prompt = (
        sample["system_prompt"]
    )

    user_text = (
        sample["user_prompt"]
    )


    # --------------------------------------------------------
    # Thyme stores a textual <image> marker.
    #
    # Our HybridQwen25Processor is responsible for creating:
    #
    # <|vision_start|><|image_pad|><|vision_end|>
    #
    # Therefore remove the textual marker first.
    # --------------------------------------------------------

    user_text = user_text.lstrip()


    if user_text.startswith(
        "<image>"
    ):

        user_text = (
            user_text[
                len("<image>"):
            ]
            .lstrip(
                "\r\n"
            )
        )


    # --------------------------------------------------------
    # Replace original machine-local image path.
    #
    # The Thyme dataset contains paths from the original
    # author's environment.
    #
    # We save the decoded image locally and make the prompt
    # point to that actual local image.
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
            count=1
        )
    )


    if path_count == 0:

        # Current Thyme-RL normally contains the field.
        # Keep the sample usable if a future sample does not.

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
        count=1
    )


    # --------------------------------------------------------
    # Construct system block.
    # --------------------------------------------------------

    system_block = (
        "<|im_start|>system\n"
        + system_prompt
        + "<|im_end|>\n"
    )


    # --------------------------------------------------------
    # Let our processor construct the canonical
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

    image_pad_count = (
        prompt.count(
            "<|image_pad|>"
        )
    )


    if image_pad_count != 1:

        raise RuntimeError(
            "Expected exactly one <|image_pad|>, "
            f"got {image_pad_count}."
        )


    if "<image>" in prompt:

        raise RuntimeError(
            "Raw <image> marker remains in "
            "the final prompt."
        )


    return prompt


# ============================================================
# Move processor output to GPU
# ============================================================

def move_batch_to_device(
    batch,
    device
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
# Token trimming
# ============================================================

def trim_generated_ids(
    generated_ids,
    eos_ids,
    pad_id
):

    ids = (
        generated_ids.tolist()
    )


    stop_positions = []


    for index, token_id in enumerate(
        ids
    ):

        if token_id in eos_ids:

            stop_positions.append(
                index
            )

        elif (
            pad_id is not None
            and token_id == pad_id
        ):

            stop_positions.append(
                index
            )


    if stop_positions:

        end = min(
            stop_positions
        )

        return generated_ids[
            :end
        ]


    return generated_ids


# ============================================================
# Text normalization
# ============================================================

def normalize_text(
    text
):

    if text is None:

        return ""


    text = unicodedata.normalize(
        "NFKC",
        str(text)
    )


    text = (
        text
        .replace(
            "\u00a0",
            " "
        )
    )


    text = re.sub(
        r"\s+",
        " ",
        text
    )


    return text.strip().lower()


# ============================================================
# Answer extraction
# ============================================================

def extract_answer(
    completion
):

    if completion is None:

        return ""


    match = re.search(
        r"<answer>(.*?)</answer>",
        completion,
        flags=re.DOTALL
    )


    if match:

        return match.group(
            1
        ).strip()


    # --------------------------------------------------------
    # Diagnostic fallback.
    #
    # IMPORTANT:
    #
    # This is intentionally more permissive than the final
    # Thyme reward implementation.
    #
    # If the model says:
    #
    #   The answer is 35.
    #
    # without <answer> tags, we still want to know whether
    # the underlying answer is correct.
    #
    # Format reward is recorded separately.
    # --------------------------------------------------------

    return completion.strip()


# ============================================================
# Official-style format check
# ============================================================

def format_check(
    completion
):

    if completion is None:

        return False


    pattern = (
        r"^(?:(?!</think>).)*"
        r"</think>\s"
        r"<answer>"
        r"(?:(?!</answer>).)*"
        r"</answer>\Z"
    )


    match = re.fullmatch(
        pattern,
        completion,
        flags=re.DOTALL
    )


    if match is None:

        return False


    think_count = (
        completion.count(
            "<think>"
        )
    )

    think_end_count = (
        completion.count(
            "</think>"
        )
    )

    answer_count = (
        completion.count(
            "<answer>"
        )
    )

    answer_end_count = (
        completion.count(
            "</answer>"
        )
    )


    return (
        think_count == 1
        and think_end_count == 1
        and answer_count == 1
        and answer_end_count == 1
    )


# ============================================================
# Prefix cleanup
# ============================================================

def clean_answer_prefix(
    answer
):

    prefixes = [

        "The best answer is",

        "The correct answer is",

        "The answer is",

        "The answer:",

        "Answer:",

        "The best option is",

        "The correct option is",

        "Best answer:",

        "Best option:",
    ]


    cleaned = answer.strip()


    changed = True


    while changed:

        changed = False

        for prefix in prefixes:

            if cleaned.lower().startswith(
                prefix.lower()
            ):

                cleaned = (
                    cleaned[
                        len(prefix):
                    ]
                    .strip()
                )

                changed = True


    return cleaned


# ============================================================
# Math verification
# ============================================================

try:

    from math_verify import (
        parse as math_parse,
        verify as math_verify
    )

    HAS_MATH_VERIFY = True

except Exception:

    HAS_MATH_VERIFY = False


def try_math_verify(
    prediction,
    solution
):

    if not HAS_MATH_VERIFY:

        return False


    try:

        parsed_prediction = (
            math_parse(
                prediction
            )
        )

        parsed_solution = (
            math_parse(
                solution
            )
        )


        result = math_verify(
            parsed_prediction,
            parsed_solution
        )


        return float(
            result
        ) > 0.0


    except Exception:

        return False


# ============================================================
# Numeric fallback
# ============================================================

def extract_numbers(
    text
):

    if not text:

        return []


    pattern = (
        r"[-+]?"
        r"(?:"
        r"\d+(?:\.\d*)?"
        r"|"
        r"\.\d+"
        r")"
        r"(?:[eE][-+]?\d+)?"
    )


    return re.findall(
        pattern,
        text
    )


def try_numeric_match(
    prediction,
    solution
):

    pred_numbers = (
        extract_numbers(
            prediction
        )
    )

    sol_numbers = (
        extract_numbers(
            solution
        )
    )


    if not pred_numbers:

        return False


    if not sol_numbers:

        return False


    # --------------------------------------------------------
    # Only use numeric fallback when the reference itself is
    # essentially a numeric answer.
    #
    # This avoids accidentally matching arbitrary numbers
    # appearing in a textual answer.
    # --------------------------------------------------------

    solution_clean = (
        solution
        .strip()
    )


    solution_clean = (
        solution_clean
        .replace(
            ",",
            ""
        )
    )


    numeric_solution_pattern = (
        r"^[-+]?"
        r"(?:"
        r"\d+(?:\.\d*)?"
        r"|"
        r"\.\d+"
        r")"
        r"(?:[eE][-+]?\d+)?$"
    )


    if not re.fullmatch(
        numeric_solution_pattern,
        solution_clean
    ):

        return False


    try:

        prediction_value = float(
            pred_numbers[-1]
        )

        solution_value = float(
            sol_numbers[-1]
        )


        return math.isclose(
            prediction_value,
            solution_value,
            rel_tol=1e-6,
            abs_tol=1e-6
        )


    except Exception:

        return False


# ============================================================
# Accuracy reward
# ============================================================

def compute_accuracy_reward(
    completion,
    solution
):

    answer = extract_answer(
        completion
    )


    if not answer:

        return {
            "reward": 0.0,
            "match": False,
            "method": "empty",
            "extracted_answer": "",
        }


    answer = clean_answer_prefix(
        answer
    )


    solution = str(
        solution
    ).strip()


    # --------------------------------------------------------
    # 1. Math verification
    # --------------------------------------------------------

    if try_math_verify(
        answer,
        solution
    ):

        return {
            "reward": 1.0,
            "match": True,
            "method": "math_verify",
            "extracted_answer": answer,
        }


    # --------------------------------------------------------
    # 2. Exact normalized string match
    # --------------------------------------------------------

    normalized_answer = (
        normalize_text(
            answer
        )
    )

    normalized_solution = (
        normalize_text(
            solution
        )
    )


    if (
        normalized_answer
        == normalized_solution
    ):

        return {
            "reward": 1.0,
            "match": True,
            "method": "exact_string",
            "extracted_answer": answer,
        }


    # --------------------------------------------------------
    # 3. Numeric fallback
    # --------------------------------------------------------

    if try_numeric_match(
        answer,
        solution
    ):

        return {
            "reward": 1.0,
            "match": True,
            "method": "numeric",
            "extracted_answer": answer,
        }


    return {
        "reward": 0.0,
        "match": False,
        "method": "no_match",
        "extracted_answer": answer,
    }


# ============================================================
# Combined diagnostic reward
# ============================================================

def compute_rewards(
    completion,
    solution,
    format_weight
):

    accuracy_info = (
        compute_accuracy_reward(
            completion,
            solution
        )
    )


    format_reward = (
        1.0
        if format_check(
            completion
        )
        else 0.0
    )


    accuracy_reward = (
        accuracy_info["reward"]
    )


    total_reward = (
        accuracy_reward
        +
        format_weight
        * format_reward
    )


    return {
        "accuracy_reward":
            float(
                accuracy_reward
            ),

        "format_reward":
            float(
                format_reward
            ),

        "total_reward":
            float(
                total_reward
            ),

        "accuracy_match":
            bool(
                accuracy_info["match"]
            ),

        "accuracy_match_method":
            accuracy_info["method"],

        "extracted_answer":
            accuracy_info["extracted_answer"],
    }


# ============================================================
# Generation
# ============================================================

@torch.inference_mode()
def generate_group(
    model,
    processor,
    prompt,
    image,
    num_generations,
    max_new_tokens,
    temperature,
    top_p,
    top_k,
    repetition_penalty,
    device
):

    # --------------------------------------------------------
    # Replicate the SAME prompt and SAME image G times.
    #
    # This is important.
    #
    # GRPO compares multiple stochastic completions of the
    # SAME underlying problem.
    # --------------------------------------------------------

    prompts = [
        prompt
        for _ in range(
            num_generations
        )
    ]


    images_per_sample = [

        [image]

        for _ in range(
            num_generations
        )

    ]


    batch = processor.encode_images(

        prompts,

        images_per_sample=(
            images_per_sample
        ),

        padding=True

    )


    batch = move_batch_to_device(
        batch,
        device
    )


    batch = {
        key: value
        for key, value in batch.items()
        if value is not None
    }


    model.model.rope_deltas = None


    tokenizer = (
        processor.tokenizer
    )


    eos_ids = []


    if (
        tokenizer.eos_token_id
        is not None
    ):

        if isinstance(
            tokenizer.eos_token_id,
            (list, tuple)
        ):

            eos_ids.extend(
                int(x)
                for x in tokenizer.eos_token_id
            )

        else:

            eos_ids.append(
                int(
                    tokenizer.eos_token_id
                )
            )


    im_end_id = (
        tokenizer.convert_tokens_to_ids(
            "<|im_end|>"
        )
    )


    if (
        im_end_id is not None
        and im_end_id >= 0
    ):

        eos_ids.append(
            int(im_end_id)
        )


    eos_ids = sorted(
        set(
            eos_ids
        )
    )


    pad_id = (
        tokenizer.pad_token_id
    )


    start_time = time.time()


    output_ids = model.generate(

        **batch,

        max_new_tokens=(
            max_new_tokens
        ),

        do_sample=True,

        temperature=(
            temperature
        ),

        top_p=(
            top_p
        ),

        top_k=(
            top_k
        ),

        repetition_penalty=(
            repetition_penalty
        ),

        num_beams=1,

        eos_token_id=(
            eos_ids
            if eos_ids
            else None
        ),

        pad_token_id=(

            pad_id

            if pad_id is not None

            else
            tokenizer.eos_token_id

        ),

        use_cache=True,

    )


    generation_time = (
        time.time()
        - start_time
    )


    input_length = (
        batch["input_ids"]
        .shape[1]
    )


    generated_batch = (
        output_ids[
            :,
            input_length:
        ]
    )


    results = []


    for index in range(
        generated_batch.shape[0]
    ):

        raw_ids = (
            generated_batch[index]
        )


        trimmed_ids = (
            trim_generated_ids(
                raw_ids,
                eos_ids,
                pad_id
            )
        )


        text = (
            tokenizer.batch_decode(
                trimmed_ids.unsqueeze(0),
                skip_special_tokens=True
            )[0]
        )


        text = text.strip()


        results.append({

            "rollout_index":
                int(index),

            "text":
                text,

            "generated_tokens":
                int(
                    trimmed_ids.shape[0]
                ),

        })


    return (
        results,
        generation_time
    )


# ============================================================
# One sample
# ============================================================

def process_one_sample(
    model,
    processor,
    raw_sample,
    sample_index,
    args,
    output_dir,
    device
):

    sample = parse_thyme_sample(
        raw_sample
    )


    image = decode_image(
        sample["image_b64"]
    )


    # --------------------------------------------------------
    # Save decoded image.
    # --------------------------------------------------------

    image_dir = (
        output_dir
        /
        "images"
    )


    image_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    image_path = (
        image_dir
        /
        f"sample_{sample_index:06d}.png"
    )


    image.save(
        image_path,
        format="PNG"
    )


    # --------------------------------------------------------
    # Build prompt.
    # --------------------------------------------------------

    prompt = build_thyme_prompt(
        processor,
        {
            **sample,
            "image": image
        },
        image_path
    )


    # --------------------------------------------------------
    # Sanity print for first sample.
    # --------------------------------------------------------

    if sample_index == 0:

        print(
            "\n" + "=" * 80,
            flush=True
        )

        print(
            "FIRST SAMPLE PROMPT",
            flush=True
        )

        print(
            "=" * 80,
            flush=True
        )

        print(
            prompt,
            flush=True
        )

        print(
            "=" * 80,
            flush=True
        )


    # --------------------------------------------------------
    # Generate G completions.
    # --------------------------------------------------------

    print(
        "\n" + "-" * 80,
        flush=True
    )

    print(
        f"Sample {sample_index}",
        flush=True
    )

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
        f"Generating G={args.num_generations}...",
        flush=True
    )


    if torch.cuda.is_available():

        torch.cuda.reset_peak_memory_stats(
            device
        )


    rollouts, generation_time = (
        generate_group(

            model=model,

            processor=processor,

            prompt=prompt,

            image=image,

            num_generations=(
                args.num_generations
            ),

            max_new_tokens=(
                args.max_new_tokens
            ),

            temperature=(
                args.temperature
            ),

            top_p=(
                args.top_p
            ),

            top_k=(
                args.top_k
            ),

            repetition_penalty=(
                args.repetition_penalty
            ),

            device=device

        )
    )


    # --------------------------------------------------------
    # Rewards
    # --------------------------------------------------------

    for rollout in rollouts:

        reward_info = (
            compute_rewards(

                rollout["text"],

                sample["solution"],

                args.format_weight

            )
        )


        rollout.update(
            reward_info
        )


    # --------------------------------------------------------
    # Group reward statistics.
    # --------------------------------------------------------

    total_rewards = [

        rollout["total_reward"]

        for rollout in rollouts

    ]


    accuracy_rewards = [

        rollout["accuracy_reward"]

        for rollout in rollouts

    ]


    format_rewards = [

        rollout["format_reward"]

        for rollout in rollouts

    ]


    if len(total_rewards) > 1:

        reward_mean = mean(
            total_rewards
        )

        reward_std = pstdev(
            total_rewards
        )

    else:

        reward_mean = (
            total_rewards[0]
            if total_rewards
            else 0.0
        )

        reward_std = 0.0


    # --------------------------------------------------------
    # GRPO-style normalized group advantage.
    #
    # This is diagnostic only.
    #
    # No gradient is calculated here.
    # --------------------------------------------------------

    advantage_denominator = (
        reward_std
        +
        1e-8
    )


    for rollout in rollouts:

        rollout["group_advantage"] = (

            rollout["total_reward"]
            -
            reward_mean

        ) / advantage_denominator


    correct_count = sum(
        1
        for reward in accuracy_rewards
        if reward >= 1.0
    )


    format_count = sum(
        1
        for reward in format_rewards
        if reward >= 1.0
    )


    unique_text_count = len(
        set(
            normalize_text(
                rollout["text"]
            )
            for rollout in rollouts
        )
    )


    if torch.cuda.is_available():

        peak_memory_gb = (

            torch.cuda.max_memory_allocated(
                device
            )
            /
            (1024 ** 3)

        )

    else:

        peak_memory_gb = 0.0


    group = {

        "sample_index":
            int(sample_index),

        "question":
            sample["question"],

        "solution":
            sample["solution"],

        "image_path":
            str(image_path),

        "image_size":
            [
                int(image.width),
                int(image.height)
            ],

        "prompt":
            prompt,

        "generation_time_sec":
            float(generation_time),

        "num_generations":
            int(args.num_generations),

        "reward_mean":
            float(reward_mean),

        "reward_std":
            float(reward_std),

        "accuracy_reward_mean":
            float(
                mean(
                    accuracy_rewards
                )
            ),

        "format_reward_mean":
            float(
                mean(
                    format_rewards
                )
            ),

        "correct_count":
            int(correct_count),

        "format_count":
            int(format_count),

        "accuracy_pass_rate":
            float(
                correct_count
                /
                args.num_generations
            ),

        "format_pass_rate":
            float(
                format_count
                /
                args.num_generations
            ),

        "group_pass":
            bool(
                correct_count > 0
            ),

        "group_all_correct":
            bool(
                correct_count
                ==
                args.num_generations
            ),

        "group_mixed":
            bool(
                0
                <
                correct_count
                <
                args.num_generations
            ),

        "unique_completion_count":
            int(
                unique_text_count
            ),

        "completion_diversity":
            float(
                unique_text_count
                /
                args.num_generations
            ),

        "peak_memory_gb":
            float(
                peak_memory_gb
            ),

        "rollouts":
            rollouts,

    }


    # --------------------------------------------------------
    # Human-readable result.
    # --------------------------------------------------------

    print(
        "Reward vector:",
        total_rewards,
        flush=True
    )

    print(
        "Accuracy vector:",
        accuracy_rewards,
        flush=True
    )

    print(
        "Format vector:",
        format_rewards,
        flush=True
    )

    print(
        "Reward mean:",
        f"{reward_mean:.6f}",
        flush=True
    )

    print(
        "Reward std:",
        f"{reward_std:.6f}",
        flush=True
    )

    print(
        "Correct:",
        f"{correct_count}/{args.num_generations}",
        flush=True
    )

    print(
        "Unique completions:",
        f"{unique_text_count}/{args.num_generations}",
        flush=True
    )


    for rollout in rollouts:

        print(
            "\n"
            f"[Rollout {rollout['rollout_index']}] "
            f"reward={rollout['total_reward']:.3f} "
            f"acc={rollout['accuracy_reward']:.1f} "
            f"fmt={rollout['format_reward']:.1f} "
            f"tokens={rollout['generated_tokens']}",
            flush=True
        )

        print(
            rollout["text"],
            flush=True
        )


    return group


# ============================================================
# Final summary
# ============================================================

def summarize_groups(
    groups,
    args
):

    if not groups:

        return {}


    all_rollouts = [

        rollout

        for group in groups

        for rollout in group["rollouts"]

    ]


    all_accuracy = [

        rollout["accuracy_reward"]

        for rollout in all_rollouts

    ]


    all_format = [

        rollout["format_reward"]

        for rollout in all_rollouts

    ]


    all_total = [

        rollout["total_reward"]

        for rollout in all_rollouts

    ]


    all_lengths = [

        rollout["generated_tokens"]

        for rollout in all_rollouts

    ]


    mixed_groups = sum(

        1

        for group in groups

        if group["group_mixed"]

    )


    all_correct_groups = sum(

        1

        for group in groups

        if group["group_all_correct"]

    )


    pass_groups = sum(

        1

        for group in groups

        if group["group_pass"]

    )


    zero_accuracy_groups = sum(

        1

        for group in groups

        if group["correct_count"] == 0

    )


    zero_std_groups = sum(

        1

        for group in groups

        if group["reward_std"] < 1e-12

    )


    return {

        "num_samples":
            len(groups),

        "num_rollouts":
            len(all_rollouts),

        "num_generations":
            args.num_generations,

        "temperature":
            args.temperature,

        "top_p":
            args.top_p,

        "top_k":
            args.top_k,

        "repetition_penalty":
            args.repetition_penalty,

        "max_new_tokens":
            args.max_new_tokens,

        "format_weight":
            args.format_weight,

        "math_verify_available":
            HAS_MATH_VERIFY,

        "accuracy_reward_mean":
            float(
                mean(
                    all_accuracy
                )
            ),

        "format_reward_mean":
            float(
                mean(
                    all_format
                )
            ),

        "total_reward_mean":
            float(
                mean(
                    all_total
                )
            ),

        "average_generated_tokens":
            float(
                mean(
                    all_lengths
                )
            ),

        "group_pass_rate":
            float(
                pass_groups
                /
                len(groups)
            ),

        "group_all_correct_rate":
            float(
                all_correct_groups
                /
                len(groups)
            ),

        "group_mixed_rate":
            float(
                mixed_groups
                /
                len(groups)
            ),

        "group_zero_accuracy_rate":
            float(
                zero_accuracy_groups
                /
                len(groups)
            ),

        "group_zero_reward_std_rate":
            float(
                zero_std_groups
                /
                len(groups)
            ),

        "groups_with_useful_grpo_signal":
            int(
                mixed_groups
            ),

        "groups_without_any_correct_rollout":
            int(
                zero_accuracy_groups
            ),

        "groups_all_correct":
            int(
                all_correct_groups
            ),

    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(

        description=(
            "Stage-4A Thyme-RL "
            "multi-rollout baseline"
        )

    )


    parser.add_argument(

        "--num-samples",

        type=int,

        default=(
            DEFAULT_NUM_SAMPLES
        ),

        help=(
            "Number of Thyme samples "
            "to evaluate."
        )

    )


    parser.add_argument(

        "--num-generations",

        type=int,

        default=(
            DEFAULT_NUM_GENERATIONS
        ),

        help=(
            "Number of stochastic "
            "rollouts per prompt."
        )

    )


    parser.add_argument(

        "--max-new-tokens",

        type=int,

        default=(
            DEFAULT_MAX_NEW_TOKENS
        )

    )


    parser.add_argument(

        "--temperature",

        type=float,

        default=(
            DEFAULT_TEMPERATURE
        )

    )


    parser.add_argument(

        "--top-p",

        type=float,

        default=(
            DEFAULT_TOP_P
        )

    )


    parser.add_argument(

        "--top-k",

        type=int,

        default=(
            DEFAULT_TOP_K
        )

    )


    parser.add_argument(

        "--repetition-penalty",

        type=float,

        default=(
            DEFAULT_REPETITION_PENALTY
        )

    )


    parser.add_argument(

        "--format-weight",

        type=float,

        default=(
            DEFAULT_FORMAT_WEIGHT
        ),

        help=(
            "Diagnostic format reward weight. "
            "This is NOT the final training reward."
        )

    )


    parser.add_argument(

        "--seed",

        type=int,

        default=(
            DEFAULT_SEED
        )

    )


    parser.add_argument(

        "--skip",

        type=int,

        default=0,

        help=(
            "Skip this many samples "
            "before collecting rollout samples."
        )

    )


    parser.add_argument(

        "--shuffle",

        action="store_true",

        help=(
            "Enable streaming shuffle."
        )

    )


    parser.add_argument(

        "--shuffle-buffer",

        type=int,

        default=(
            DEFAULT_SHUFFLE_BUFFER
        )

    )


    parser.add_argument(

        "--out-dir",

        type=Path,

        default=(
            DEFAULT_OUTPUT_DIR
        )

    )


    args = parser.parse_args()


    # --------------------------------------------------------
    # Argument checks.
    # --------------------------------------------------------

    if args.num_samples <= 0:

        raise ValueError(
            "--num-samples must be > 0"
        )


    if args.num_generations < 2:

        raise ValueError(
            "--num-generations must be >= 2 "
            "for group-based GRPO diagnostics."
        )


    if args.max_new_tokens <= 0:

        raise ValueError(
            "--max-new-tokens must be > 0"
        )


    if args.temperature <= 0:

        raise ValueError(
            "--temperature must be > 0"
        )


    if not (
        0.0
        <
        args.top_p
        <=
        1.0
    ):

        raise ValueError(
            "--top-p must be in (0, 1]."
        )


    if args.top_k < 0:

        raise ValueError(
            "--top-k must be >= 0"
        )


    if args.repetition_penalty <= 0:

        raise ValueError(
            "--repetition-penalty must be > 0"
        )


    # --------------------------------------------------------
    # Seed.
    # --------------------------------------------------------

    set_global_seed(
        args.seed
    )


    # --------------------------------------------------------
    # Output.
    # --------------------------------------------------------

    output_dir = Path(
        args.out_dir
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    records_file = (
        output_dir
        /
        "rollouts.jsonl"
    )


    summary_file = (
        output_dir
        /
        "summary.json"
    )


    config_file = (
        output_dir
        /
        "config.json"
    )


    # --------------------------------------------------------
    # Configuration snapshot.
    # --------------------------------------------------------

    config = {

        "experiment":
        {
            "name":
                "stage4a_thyme_rollout_baseline",

            "dataset":
                THYME_DATASET,

            "split":
                "train",

            "streaming":
                True,

            "seed":
                args.seed,

            "num_samples":
                args.num_samples,

            "num_generations":
                args.num_generations,

            "max_new_tokens":
                args.max_new_tokens,

            "temperature":
                args.temperature,

            "top_p":
                args.top_p,

            "top_k":
                args.top_k,

            "repetition_penalty":
                args.repetition_penalty,

            "format_weight":
                args.format_weight,

            "shuffle":
                args.shuffle,

            "shuffle_buffer":
                args.shuffle_buffer,

            "skip":
                args.skip,

        },

        "purpose":
            (
                "Diagnostic multi-rollout "
                "baseline before GRPO training."
            ),

        "important":
            (
                "No backward, optimizer, "
                "GRPO loss, sandbox, or "
                "external reward model."
            ),

    }


    save_json(
        config_file,
        config
    )


    # --------------------------------------------------------
    # CUDA.
    # --------------------------------------------------------

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is required for Stage-4 rollout."
        )


    device = torch.device(
        "cuda"
    )


    print(
        "=" * 80,
        flush=True
    )

    print(
        "STAGE-4A THYME ROLLOUT BASELINE",
        flush=True
    )

    print(
        "=" * 80,
        flush=True
    )


    print(
        "Project root:",
        PROJECT_ROOT,
        flush=True
    )

    print(
        "Output directory:",
        output_dir,
        flush=True
    )

    print(
        "Device:",
        device,
        flush=True
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(
            device
        ),
        flush=True
    )

    print(
        "num_samples:",
        args.num_samples,
        flush=True
    )

    print(
        "num_generations:",
        args.num_generations,
        flush=True
    )

    print(
        "temperature:",
        args.temperature,
        flush=True
    )

    print(
        "max_new_tokens:",
        args.max_new_tokens,
        flush=True
    )

    print(
        "math_verify:",
        HAS_MATH_VERIFY,
        flush=True
    )


    # --------------------------------------------------------
    # Load model.
    # --------------------------------------------------------

    print(
        "\nLoading DPO-v2 policy...",
        flush=True
    )


    model, processor = (
        load_dpo_v2_policy()
    )


    model.to(
        device
    )


    model.eval()


    # --------------------------------------------------------
    # Load dataset.
    # --------------------------------------------------------

    dataset = load_thyme_stream(

        seed=args.seed,

        shuffle=args.shuffle,

        shuffle_buffer=(
            args.shuffle_buffer
        )

    )


    # --------------------------------------------------------
    # Open JSONL.
    # --------------------------------------------------------

    groups = []


    with records_file.open(
        "w",
        encoding="utf-8"
    ) as records_fp:


        dataset_index = 0


        collected = 0


        for raw_sample in dataset:


            if dataset_index < args.skip:

                dataset_index += 1

                continue


            if collected >= args.num_samples:

                break


            try:

                group = process_one_sample(

                    model=model,

                    processor=processor,

                    raw_sample=raw_sample,

                    sample_index=collected,

                    args=args,

                    output_dir=output_dir,

                    device=device

                )


                groups.append(
                    group
                )


                records_fp.write(
                    json.dumps(
                        group,
                        ensure_ascii=False
                    )
                    + "\n"
                )


                records_fp.flush()


                collected += 1


            except Exception as e:

                print(
                    "\n" + "=" * 80,
                    flush=True
                )

                print(
                    "SAMPLE FAILED",
                    flush=True
                )

                print(
                    f"dataset_index={dataset_index}",
                    flush=True
                )

                print(
                    repr(e),
                    flush=True
                )

                print(
                    "=" * 80,
                    flush=True
                )

                raise


            finally:

                dataset_index += 1


            # ------------------------------------------------
            # Release temporary GPU cache.
            # ------------------------------------------------

            if torch.cuda.is_available():

                torch.cuda.empty_cache()


    # --------------------------------------------------------
    # Final summary.
    # --------------------------------------------------------

    summary = summarize_groups(
        groups,
        args
    )


    save_json(
        summary_file,
        summary
    )


    # --------------------------------------------------------
    # Final console report.
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80,
        flush=True
    )

    print(
        "STAGE-4A FINAL SUMMARY",
        flush=True
    )

    print(
        "=" * 80,
        flush=True
    )


    for key, value in summary.items():

        print(
            f"{key}: {value}",
            flush=True
        )


    print(
        "\nOutput files:",
        flush=True
    )

    print(
        "  config:",
        config_file,
        flush=True
    )

    print(
        "  rollouts:",
        records_file,
        flush=True
    )

    print(
        "  summary:",
        summary_file,
        flush=True
    )


    print(
        "\n" + "=" * 80,
        flush=True
    )

    print(
        "STAGE-4A ROLLOUT COMPLETE",
        flush=True
    )

    print(
        "=" * 80,
        flush=True
    )


if __name__ == "__main__":

    main()