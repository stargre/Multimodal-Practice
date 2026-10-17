#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage-3 evaluation:

    SFT
    DPO-v1
    DPO-v2

This script evaluates the three checkpoints on the SAME MM-RLHF
validation samples.

It reports:

1. Raw chosen/rejected sequence log-probabilities
2. Raw preference margin
3. Raw preference accuracy
4. DPO implicit rewards relative to the SFT reference
5. DPO reward margin
6. DPO reward accuracy
7. DPO loss
8. Chosen/rejected log-probability displacement
9. Legacy training mask vs corrected answer-only mask
10. Generation outputs

IMPORTANT:

The current project collator computes prompt length by tokenizing the
unexpanded prompt:

    tokenizer(prompt)

while HybridQwen25Processor expands <|image_pad|> into many visual
placeholder tokens.

Therefore this evaluator intentionally reports BOTH:

    legacy mask
    corrected answer-only mask

This is diagnostic and should not be silently changed.

The SFT checkpoint is treated as the DPO reference model for
DPO-v1 and DPO-v2, matching the Stage-3 runtime design.
"""


import argparse
import gc
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path


import torch
import torch.nn.functional as F


from tqdm import tqdm
from PIL import Image


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent


if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


from stage1_runtime import load_stage1_components


# ============================================================
# Default paths
# ============================================================

SFT_CKPT = (
    PROJECT_ROOT
    /
    "runs"
    /
    "sft_150k_ddp_6gpu"
    /
    "best_model.pt"
)


DPO_V1_CKPT = (
    PROJECT_ROOT
    /
    "runs"
    /
    "dpo_mmrlhf_6gpu"
    /
    "best_model.pt"
)


DPO_V2_CKPT = (
    PROJECT_ROOT
    /
    "runs"
    /
    "dpo_mmrlhf_freeze_vit_v2"
    /
    "best_model.pt"
)


VAL_JSON = (
    PROJECT_ROOT
    /
    "data_splits"
    /
    "mmrlhf_dpo"
    /
    "val.json"
)


IMAGE_ROOT = (
    Path(
        "/mnt/data2/zhiyu/first-task/"
        "datasets/MMRLHF_ROOT"
    )
)


DEFAULT_OUT_DIR = (
    PROJECT_ROOT
    /
    "runs"
    /
    "sft_vs_dpo_v1_v2_eval"
)


# ============================================================
# Evaluation defaults
# ============================================================

DEFAULT_LIMIT = 2000

DEFAULT_GENERATION_LIMIT = 100

DEFAULT_SEED = 42

DEFAULT_BETA = 0.1

DEFAULT_MAX_NEW_TOKENS = 256

DEVICE = "cuda"

DTYPE = torch.bfloat16


# ============================================================
# Logging
# ============================================================

LOGGER = logging.getLogger(
    "sft_vs_dpo_v1_v2"
)


def setup_logging(
    log_file
):

    LOGGER.setLevel(
        logging.INFO
    )

    LOGGER.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        log_file,
        mode="w",
        encoding="utf-8"
    )

    file_handler.setFormatter(
        formatter
    )

    stream_handler = logging.StreamHandler(
        sys.stdout
    )

    stream_handler.setFormatter(
        formatter
    )

    LOGGER.addHandler(
        file_handler
    )

    LOGGER.addHandler(
        stream_handler
    )


# ============================================================
# JSON utilities
# ============================================================

def save_json(
    path,
    obj
):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with tmp.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2
        )

    tmp.replace(
        path
    )


# ============================================================
# Dataset loading
# ============================================================

def load_preference_json(
    path
):

    path = Path(path)

    LOGGER.info(
        "Loading validation dataset: %s",
        path
    )

    text = path.read_text(
        encoding="utf-8"
    )

    text = text.strip()

    if not text:

        raise RuntimeError(
            f"Validation file is empty: {path}"
        )


    # --------------------------------------------------------
    # JSON array
    # --------------------------------------------------------

    try:

        obj = json.loads(
            text
        )

        if isinstance(
            obj,
            list
        ):

            LOGGER.info(
                "Detected JSON array format."
            )

            return obj

    except json.JSONDecodeError:

        pass


    # --------------------------------------------------------
    # JSONL
    # --------------------------------------------------------

    items = []

    for line_number, line in enumerate(
        text.splitlines(),
        start=1
    ):

        if not line.strip():

            continue

        try:

            item = json.loads(
                line
            )

        except json.JSONDecodeError as e:

            raise RuntimeError(
                f"Invalid JSON at line {line_number}: {e}"
            )

        items.append(
            item
        )


    LOGGER.info(
        "Detected JSONL format."
    )

    return items


# ============================================================
# Image path
# ============================================================

def resolve_image_path(
    image_path
):

    image_path = Path(
        image_path
    )

    if image_path.is_absolute():

        return image_path


    return (
        IMAGE_ROOT
        /
        image_path
    )


# ============================================================
# Dataset preparation
# ============================================================

def prepare_items(
    items
):

    valid = []

    skipped = 0


    for item in items:

        if not isinstance(
            item,
            dict
        ):

            skipped += 1

            continue


        required = [
            "image",
            "prompt",
            "chosen",
            "rejected"
        ]


        if any(
            key not in item
            for key in required
        ):

            skipped += 1

            continue


        image_path = resolve_image_path(
            item["image"]
        )


        if not image_path.exists():

            skipped += 1

            continue


        item_copy = dict(
            item
        )


        item_copy["_resolved_image"] = str(
            image_path
        )


        valid.append(
            item_copy
        )


    LOGGER.info(
        "Original samples: %d",
        len(items)
    )

    LOGGER.info(
        "Valid samples: %d",
        len(valid)
    )

    LOGGER.info(
        "Skipped samples: %d",
        skipped
    )


    if not valid:

        raise RuntimeError(
            "No valid evaluation samples."
        )


    return valid


# ============================================================
# Sampling
# ============================================================

def select_items(
    items,
    limit,
    seed
):

    if limit is None or limit <= 0:

        selected = list(
            items
        )

    elif limit >= len(items):

        selected = list(
            items
        )

    else:

        rng = random.Random(
            seed
        )

        indices = rng.sample(
            range(len(items)),
            limit
        )

        indices.sort()

        selected = [
            items[i]
            for i in indices
        ]


    LOGGER.info(
        "Evaluation sample count: %d",
        len(selected)
    )

    LOGGER.info(
        "Sampling seed: %d",
        seed
    )


    return selected


# ============================================================
# Checkpoint loading
# ============================================================

def load_checkpoint_state(
    checkpoint_path
):

    LOGGER.info(
        "Loading checkpoint: %s",
        checkpoint_path
    )


    try:

        state = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )

    except TypeError:

        state = torch.load(
            checkpoint_path,
            map_location="cpu"
        )


    # --------------------------------------------------------
    # Support several common checkpoint wrappers
    # --------------------------------------------------------

    if isinstance(
        state,
        dict
    ):

        for key in (
            "state_dict",
            "model_state_dict",
            "model"
        ):

            candidate = state.get(
                key
            )

            if isinstance(
                candidate,
                dict
            ):

                state = candidate

                LOGGER.info(
                    "Unwrapped checkpoint key: %s",
                    key
                )

                break


    if not isinstance(
        state,
        dict
    ):

        raise RuntimeError(
            "Checkpoint is not a state_dict-like dictionary."
        )


    # --------------------------------------------------------
    # Remove DDP "module." prefix if necessary
    # --------------------------------------------------------

    keys = list(
        state.keys()
    )


    if keys and all(
        k.startswith("module.")
        for k in keys
    ):

        LOGGER.info(
            "Removing DDP 'module.' prefix."
        )

        state = {
            k[len("module."):]: v
            for k, v in state.items()
        }


    LOGGER.info(
        "Checkpoint tensors: %d",
        len(state)
    )


    return state


# ============================================================
# Model loading
# ============================================================

def load_model(
    checkpoint_path,
    model_name
):

    LOGGER.info("")
    LOGGER.info(
        "============================================================"
    )

    LOGGER.info(
        "Loading model: %s",
        model_name
    )

    LOGGER.info(
        "Checkpoint: %s",
        checkpoint_path
    )


    if not checkpoint_path.exists():

        raise FileNotFoundError(
            checkpoint_path
        )


    model, processor = (
        load_stage1_components()
    )


    state = load_checkpoint_state(
        checkpoint_path
    )


    # --------------------------------------------------------
    # Strict loading
    #
    # This evaluation script intentionally does NOT silently
    # accept missing/unexpected keys.
    # --------------------------------------------------------

    try:

        incompatible = model.load_state_dict(
            state,
            strict=False
        )

        missing_keys = list(incompatible.missing_keys)
        unexpected_keys = list(incompatible.unexpected_keys)

        allowed_missing_prefixes = (
            "model.visual.vit.",
        )

        allowed_missing_keys = []
        for key in missing_keys:
            if key.startswith(allowed_missing_prefixes):
                allowed_missing_keys.append(key)

        forbidden_missing_keys = [
            key
            for key in missing_keys
            if key not in allowed_missing_keys
        ]

        LOGGER.info(
            "Checkpoint missing keys: %d",
            len(missing_keys)
        )

        LOGGER.info(
            "Allowed missing keys: %d",
            len(allowed_missing_keys)
        )

        LOGGER.info(
            "Forbidden missing keys: %d",
            len(forbidden_missing_keys)
        )

        LOGGER.info(
            "Unexpected keys: %d",
            len(unexpected_keys)
        )

        # ------------------------------------------------------------
        # Print allowed missing keys
        # ------------------------------------------------------------

        if allowed_missing_keys:
            LOGGER.info(
                "Allowed missing keys are InternViT parameters:"
            )

            for key in allowed_missing_keys[:20]:
                LOGGER.info(
                    "  [ALLOWED] %s",
                    key
                )

            if len(allowed_missing_keys) > 20:
                LOGGER.info(
                    "  ... %d more InternViT keys",
                    len(allowed_missing_keys) - 20
                )

        # ------------------------------------------------------------
        # Forbidden missing keys -> hard failure
        # ------------------------------------------------------------

        if forbidden_missing_keys:

            LOGGER.error(
                "Checkpoint loading failed: "
                "non-ViT parameters are missing."
            )

            for key in forbidden_missing_keys:
                LOGGER.error(
                    "  [FORBIDDEN MISSING] %s",
                    key
                )

            raise RuntimeError(
                f"{name}: checkpoint has "
                f"{len(forbidden_missing_keys)} forbidden missing keys."
            )

        # ------------------------------------------------------------
        # Unexpected keys -> hard failure
        # ------------------------------------------------------------

        if unexpected_keys:

            LOGGER.error(
                "Checkpoint loading failed: unexpected keys."
            )

            for key in unexpected_keys:
                LOGGER.error(
                    "  [UNEXPECTED] %s",
                    key
                )

            raise RuntimeError(
                f"{name}: checkpoint has "
                f"{len(unexpected_keys)} unexpected keys."
            )

        LOGGER.info(
            "Checkpoint load: PASS"
        )

    except RuntimeError as e:

        LOGGER.exception(
            "Checkpoint loading failed."
        )

        raise


    missing = getattr(
        incompatible,
        "missing_keys",
        []
    )

    unexpected = getattr(
        incompatible,
        "unexpected_keys",
        []
    )


    LOGGER.info(
        "Missing keys: %d",
        len(missing)
    )

    LOGGER.info(
        "Unexpected keys: %d",
        len(unexpected)
    )


    if missing:

        LOGGER.error(
            "Missing keys:"
        )

        for key in missing[:20]:

            LOGGER.error(
                "  %s",
                key
            )


    if unexpected:

        LOGGER.error(
            "Unexpected keys:"
        )

        for key in unexpected[:20]:

            LOGGER.error(
                "  %s",
                key
            )


    model.eval()


    # Important for multimodal forward/generation.
    model.model.rope_deltas = None


    LOGGER.info(
        "Model class: %s",
        type(model).__name__
    )

    LOGGER.info(
        "Model dtype: %s",
        next(model.parameters()).dtype
    )


    LOGGER.info(
        "Model device: %s",
        next(model.parameters()).device
    )


    return model, processor


# ============================================================
# Model cleanup
# ============================================================

def release_model(
    model
):

    if model is not None:

        try:

            model.cpu()

        except Exception:

            pass


        del model


    gc.collect()


    if torch.cuda.is_available():

        torch.cuda.empty_cache()


        try:

            torch.cuda.synchronize()

        except Exception:

            pass


# ============================================================
# Device movement
# ============================================================

def move_batch_to_device(
    batch
):

    output = {}

    for key, value in batch.items():

        if torch.is_tensor(
            value
        ):

            output[key] = value.to(
                DEVICE,
                non_blocking=True
            )

        else:

            output[key] = value


    return output


# ============================================================
# Prompt length diagnostics
# ============================================================

def get_prompt_length_diagnostics(
    processor,
    prompt,
    images
):

    tokenizer = processor.tokenizer


    raw_ids = tokenizer(
        prompt,
        add_special_tokens=False
    )["input_ids"]


    raw_prompt_len = len(
        raw_ids
    )


    raw_image_placeholder_count = prompt.count("<|image_pad|>")

    if raw_image_placeholder_count != len(images):
        raise RuntimeError(
            f"Mismatch between logical image placeholders and images: "
            f"prompt contains {raw_image_placeholder_count} "
            f"<|image_pad|> placeholders, "
            f"but {len(images)} images were supplied."
        )



    expanded_prompt_len = (
        raw_prompt_len
    )


    visual_token_counts = []


    for image in images:

        _, grid = processor.tile_image(
            image
        )


        t, h, w = grid


        visual_tokens = (
            int(t)
            *
            int(h)
            *
            int(w)
            //
            4
        )


        visual_token_counts.append(
            visual_tokens
        )


        # One raw <|image_pad|> becomes N visual tokens.
        expanded_prompt_len += (
            visual_tokens - 1
        )


    return {

        "raw_prompt_tokens":
            int(raw_prompt_len),

        "expanded_prompt_tokens":
            int(expanded_prompt_len),

        "visual_token_counts":
            visual_token_counts,

        "raw_image_placeholder_count":
            int(raw_image_placeholder_count),
        


        "extra_visual_tokens_added":
            int(
                expanded_prompt_len
                -
                raw_prompt_len
            ),

    }


# ============================================================
# Processor encoding
# ============================================================

def encode_pair(
    processor,
    prompt,
    chosen,
    rejected,
    image
):

        # ------------------------------------------------------------
    # Normalize raw dataset prompt into the project's
    # Qwen2.5-VL multimodal prompt format.
    #
    # val.json stores the raw user question only.
    # Training dataset adds:
    #
    # <|im_start|>user
    # <|vision_start|><|image_pad|><|vision_end|>
    # question
    # <|im_end|>
    # <|im_start|>assistant
    #
    # The evaluator must use exactly the same format.
    # ------------------------------------------------------------

    if "<|image_pad|>" not in prompt:

        prompt = processor.user_prompt(
            prompt,
            num_images=1
        )

    else:

        if prompt.count("<|image_pad|>") != 1:

            raise RuntimeError(
                "Expected exactly one <|image_pad|> "
                f"placeholder for one supplied image, "
                f"but found {prompt.count('<|image_pad|>')}."
            )


    full_prompts = [

        prompt + chosen,

        prompt + rejected,

    ]
    

    images_per_sample = [

        [image],

        [image],

    ]


    diagnostics = (
        get_prompt_length_diagnostics(
            processor,
            prompt,
            [image]
        )
    )


    result = processor.encode_images(

        full_prompts,

        images_per_sample=images_per_sample,

        padding=True

    )


    result = move_batch_to_device(
        result
    )


    return result, diagnostics


# ============================================================
# Masked sequence log probability
# ============================================================

def sequence_logprob(
    logits,
    input_ids,
    attention_mask,
    prompt_lengths
):

    """
    Return:

        average_logprob
        sum_logprob
        token_count

    for each sequence.

    prompt_lengths refers to the number of input tokens BEFORE
    the answer starts.

    For a causal LM:

        logits[:, t] -> token[:, t+1]

    Therefore answer token j is scored by logit j-1.
    """


    shift_logits = logits[:, :-1, :]

    shift_labels = input_ids[:, 1:]

    shift_attention = (
        attention_mask[:, 1:].bool()
    )


    batch_size = input_ids.shape[0]

    shifted_length = shift_labels.shape[1]


    positions = torch.arange(
        1,
        shifted_length + 1,
        device=input_ids.device
    )


    positions = positions.unsqueeze(
        0
    ).expand(
        batch_size,
        -1
    )


    prompt_lengths_tensor = torch.tensor(
        prompt_lengths,
        dtype=torch.long,
        device=input_ids.device
    ).unsqueeze(
        1
    )


    response_mask = (
        positions
        >=
        prompt_lengths_tensor
    )


    response_mask = (
        response_mask
        &
        shift_attention
    )


    safe_labels = shift_labels.clone()


    safe_labels[
        ~response_mask
    ] = 0


    log_probs = F.log_softmax(
        shift_logits,
        dim=-1
    )


    token_log_probs = torch.gather(

        log_probs,

        dim=-1,

        index=safe_labels.unsqueeze(-1)

    ).squeeze(-1)


    token_log_probs = (
        token_log_probs
        *
        response_mask
    )


    token_count = (
        response_mask.sum(
            dim=-1
        )
        .clamp(
            min=1
        )
    )


    sum_logprob = (
        token_log_probs.sum(
            dim=-1
        )
    )


    avg_logprob = (
        sum_logprob
        /
        token_count
    )


    return (
        avg_logprob,
        sum_logprob,
        token_count
    )


# ============================================================
# Pair scoring
# ============================================================

@torch.inference_mode()
def score_pair(
    model,
    processor,
    item
):

    image_path = item[
        "_resolved_image"
    ]


    with Image.open(
        image_path
    ) as opened:

        image = opened.convert(
            "RGB"
        )


    batch, diagnostics = encode_pair(

        processor,

        item["prompt"],
        item["chosen"],
        item["rejected"],

        image

    )


    model.model.rope_deltas = None


    start = time.time()


    outputs = model(

        input_ids=batch["input_ids"],

        attention_mask=batch["attention_mask"],

        pixel_values=batch["pixel_values"],

        image_grid_thw=batch["image_grid_thw"],

    )


    forward_time = (
        time.time()
        -
        start
    )


    logits = outputs.logits


    # --------------------------------------------------------
    # Legacy mask
    #
    # This intentionally reproduces the current collator logic:
    #
    # len(tokenizer(prompt))
    #
    # without accounting for image-token expansion.
    # --------------------------------------------------------

    legacy_prompt_len = diagnostics[
        "raw_prompt_tokens"
    ]


    legacy_avg, legacy_sum, legacy_count = (
        sequence_logprob(

            logits,

            batch["input_ids"],

            batch["attention_mask"],

            [
                legacy_prompt_len,
                legacy_prompt_len
            ]

        )
    )


    # --------------------------------------------------------
    # Correct answer-only mask
    # --------------------------------------------------------

    correct_prompt_len = diagnostics[
        "expanded_prompt_tokens"
    ]


    correct_avg, correct_sum, correct_count = (
        sequence_logprob(

            logits,

            batch["input_ids"],

            batch["attention_mask"],

            [
                correct_prompt_len,
                correct_prompt_len
            ]

        )
    )


    # --------------------------------------------------------
    # Convert to CPU floats
    # --------------------------------------------------------

    legacy_avg = legacy_avg.float().cpu()

    legacy_sum = legacy_sum.float().cpu()

    legacy_count = legacy_count.cpu()


    correct_avg = correct_avg.float().cpu()

    correct_sum = correct_sum.float().cpu()

    correct_count = correct_count.cpu()


    del outputs

    del logits

    del batch

    del image


    return {

        "diagnostics":
            diagnostics,

        "forward_time_sec":
            float(forward_time),

        "legacy":
        {

            "chosen_avg_logprob":
                float(legacy_avg[0]),

            "rejected_avg_logprob":
                float(legacy_avg[1]),

            "chosen_sum_logprob":
                float(legacy_sum[0]),

            "rejected_sum_logprob":
                float(legacy_sum[1]),

            "chosen_token_count":
                int(legacy_count[0]),

            "rejected_token_count":
                int(legacy_count[1]),

            "margin":
                float(
                    legacy_avg[0]
                    -
                    legacy_avg[1]
                ),

            "sum_margin":
                float(
                    legacy_sum[0]
                    -
                    legacy_sum[1]
                ),

            "accuracy":
                float(
                    (
                        legacy_avg[0]
                        >
                        legacy_avg[1]
                    )
                ),

        },

        "correct":
        {

            "chosen_avg_logprob":
                float(correct_avg[0]),

            "rejected_avg_logprob":
                float(correct_avg[1]),

            "chosen_sum_logprob":
                float(correct_sum[0]),

            "rejected_sum_logprob":
                float(correct_sum[1]),

            "chosen_token_count":
                int(correct_count[0]),

            "rejected_token_count":
                int(correct_count[1]),

            "margin":
                float(
                    correct_avg[0]
                    -
                    correct_avg[1]
                ),

            "sum_margin":
                float(
                    correct_sum[0]
                    -
                    correct_sum[1]
                ),

            "accuracy":
                float(
                    (
                        correct_avg[0]
                        >
                        correct_avg[1]
                    )
                ),

        }

    }


# ============================================================
# Statistics
# ============================================================

def mean(
    values
):

    if not values:

        return None

    return float(
        sum(values)
        /
        len(values)
    )


def std(
    values
):

    if len(values) < 2:

        return 0.0

    m = mean(
        values
    )

    variance = sum(
        (x - m) ** 2
        for x in values
    ) / (
        len(values) - 1
    )

    return float(
        math.sqrt(
            variance
        )
    )


def accuracy_stats(
    values
):

    n = len(
        values
    )

    if n == 0:

        return {}

    acc = sum(
        values
    ) / n


    se = math.sqrt(
        max(
            acc * (1.0 - acc),
            0.0
        )
        /
        n
    )


    return {

        "accuracy":
            float(acc),

        "accuracy_percent":
            float(acc * 100.0),

        "n":
            int(n),

        "stderr":
            float(se),

        "ci95_low":
            float(
                max(
                    0.0,
                    acc - 1.96 * se
                )
            ),

        "ci95_high":
            float(
                min(
                    1.0,
                    acc + 1.96 * se
                )
            ),

    }


def summarize_values(
    values
):

    if not values:

        return {}

    m = mean(
        values
    )

    s = std(
        values
    )

    se = (
        s
        /
        math.sqrt(
            len(values)
        )
    )


    return {

        "mean":
            float(m),

        "std":
            float(s),

        "n":
            int(len(values)),

        "ci95_low":
            float(
                m - 1.96 * se
            ),

        "ci95_high":
            float(
                m + 1.96 * se
            ),

    }


# ============================================================
# DPO statistics
# ============================================================

def compute_dpo_metrics(
    policy_records,
    reference_records,
    beta
):

    correct_policy_chosen = []

    correct_policy_rejected = []

    correct_ref_chosen = []

    correct_ref_rejected = []


    legacy_policy_chosen = []

    legacy_policy_rejected = []

    legacy_ref_chosen = []

    legacy_ref_rejected = []


    for item_id, policy in policy_records.items():

        reference = reference_records[
            item_id
        ]


        correct_policy_chosen.append(
            policy["correct"]["chosen_avg_logprob"]
        )

        correct_policy_rejected.append(
            policy["correct"]["rejected_avg_logprob"]
        )

        correct_ref_chosen.append(
            reference["correct"]["chosen_avg_logprob"]
        )

        correct_ref_rejected.append(
            reference["correct"]["rejected_avg_logprob"]
        )


        legacy_policy_chosen.append(
            policy["legacy"]["chosen_avg_logprob"]
        )

        legacy_policy_rejected.append(
            policy["legacy"]["rejected_avg_logprob"]
        )

        legacy_ref_chosen.append(
            reference["legacy"]["chosen_avg_logprob"]
        )

        legacy_ref_rejected.append(
            reference["legacy"]["rejected_avg_logprob"]
        )


    def build_metrics(
        policy_chosen,
        policy_rejected,
        ref_chosen,
        ref_rejected
    ):

        policy_chosen = torch.tensor(
            policy_chosen,
            dtype=torch.float64
        )

        policy_rejected = torch.tensor(
            policy_rejected,
            dtype=torch.float64
        )

        ref_chosen = torch.tensor(
            ref_chosen,
            dtype=torch.float64
        )

        ref_rejected = torch.tensor(
            ref_rejected,
            dtype=torch.float64
        )


        # ----------------------------------------------------
        # Raw policy preference
        # ----------------------------------------------------

        raw_margin = (
            policy_chosen
            -
            policy_rejected
        )


        raw_accuracy = (
            raw_margin
            >
            0
        ).double()


        # ----------------------------------------------------
        # DPO log-ratios
        # ----------------------------------------------------

        chosen_logratio = (
            policy_chosen
            -
            ref_chosen
        )


        rejected_logratio = (
            policy_rejected
            -
            ref_rejected
        )


        reward_margin = (
            beta
            *
            (
                chosen_logratio
                -
                rejected_logratio
            )
        )


        reward_accuracy = (
            reward_margin
            >
            0
        ).double()


        chosen_rewards = (
            beta
            *
            chosen_logratio
        )


        rejected_rewards = (
            beta
            *
            rejected_logratio
        )


        # ----------------------------------------------------
        # DPO loss
        # ----------------------------------------------------

        dpo_loss = (
            -F.logsigmoid(
                reward_margin
            )
        )


        return {

            "policy_chosen_avg_logprob":
                summarize_values(
                    policy_chosen.tolist()
                ),

            "policy_rejected_avg_logprob":
                summarize_values(
                    policy_rejected.tolist()
                ),

            "reference_chosen_avg_logprob":
                summarize_values(
                    ref_chosen.tolist()
                ),

            "reference_rejected_avg_logprob":
                summarize_values(
                    ref_rejected.tolist()
                ),

            "raw_margin":
                summarize_values(
                    raw_margin.tolist()
                ),

            "raw_preference_accuracy":
                accuracy_stats(
                    raw_accuracy.tolist()
                ),

            "chosen_reward":
                summarize_values(
                    chosen_rewards.tolist()
                ),

            "rejected_reward":
                summarize_values(
                    rejected_rewards.tolist()
                ),

            "reward_margin":
                summarize_values(
                    reward_margin.tolist()
                ),

            "reward_accuracy":
                accuracy_stats(
                    reward_accuracy.tolist()
                ),

            "dpo_loss":
                summarize_values(
                    dpo_loss.tolist()
                ),

            "chosen_logprob_delta_vs_reference":
                summarize_values(
                    (
                        policy_chosen
                        -
                        ref_chosen
                    ).tolist()
                ),

            "rejected_logprob_delta_vs_reference":
                summarize_values(
                    (
                        policy_rejected
                        -
                        ref_rejected
                    ).tolist()
                ),

        }


    return {

        "correct":
            build_metrics(

                correct_policy_chosen,

                correct_policy_rejected,

                correct_ref_chosen,

                correct_ref_rejected

            ),

        "legacy":
            build_metrics(

                legacy_policy_chosen,

                legacy_policy_rejected,

                legacy_ref_chosen,

                legacy_ref_rejected

            )

    }


# ============================================================
# Model-wide raw score statistics
# ============================================================

def compute_model_summary(
    records
):

    legacy_margins = []

    legacy_acc = []

    correct_margins = []

    correct_acc = []

    legacy_chosen = []

    legacy_rejected = []

    correct_chosen = []

    correct_rejected = []

    legacy_counts = []

    correct_counts = []

    forward_times = []


    for record in records.values():

        legacy = record["legacy"]

        correct = record["correct"]


        legacy_margins.append(
            legacy["margin"]
        )

        legacy_acc.append(
            legacy["accuracy"]
        )

        correct_margins.append(
            correct["margin"]
        )

        correct_acc.append(
            correct["accuracy"]
        )


        legacy_chosen.append(
            legacy["chosen_avg_logprob"]
        )

        legacy_rejected.append(
            legacy["rejected_avg_logprob"]
        )


        correct_chosen.append(
            correct["chosen_avg_logprob"]
        )

        correct_rejected.append(
            correct["rejected_avg_logprob"]
        )


        legacy_counts.append(
            (
                legacy["chosen_token_count"]
                +
                legacy["rejected_token_count"]
            )
            /
            2.0
        )


        correct_counts.append(
            (
                correct["chosen_token_count"]
                +
                correct["rejected_token_count"]
            )
            /
            2.0
        )


        forward_times.append(
            record["forward_time_sec"]
        )


    return {

        "num_samples":
            len(records),

        "legacy":
        {

            "margin":
                summarize_values(
                    legacy_margins
                ),

            "preference_accuracy":
                accuracy_stats(
                    legacy_acc
                ),

            "chosen_avg_logprob":
                summarize_values(
                    legacy_chosen
                ),

            "rejected_avg_logprob":
                summarize_values(
                    legacy_rejected
                ),

            "average_answer_tokens":
                summarize_values(
                    legacy_counts
                ),

        },

        "correct":
        {

            "margin":
                summarize_values(
                    correct_margins
                ),

            "preference_accuracy":
                accuracy_stats(
                    correct_acc
                ),

            "chosen_avg_logprob":
                summarize_values(
                    correct_chosen
                ),

            "rejected_avg_logprob":
                summarize_values(
                    correct_rejected
                ),

            "average_answer_tokens":
                summarize_values(
                    correct_counts
                ),

        },

        "forward_time_sec":
            summarize_values(
                forward_times
            ),

    }


# ============================================================
# Generation
# ============================================================

@torch.inference_mode()
def generate_answer(
    model,
    processor,
    item,
    max_new_tokens
):

    image_path = item[
        "_resolved_image"
    ]


    with Image.open(
        image_path
    ) as opened:

        image = opened.convert(
            "RGB"
        )


    prompt = processor.user_prompt(
        item["prompt"],
        num_images=1
    )


    batch = processor.encode_images(

        [prompt],

        images_per_sample=[
            [image]
        ],

        padding=False

    )


    batch = move_batch_to_device(
        batch
    )


    batch = {

        key: value

        for key, value in batch.items()

        if value is not None

    }


    model.model.rope_deltas = None


    start = time.time()


    output_ids = model.generate(

        **batch,

        max_new_tokens=max_new_tokens,

        do_sample=False,

        num_beams=1,

        eos_token_id=[

            processor.tokenizer.eos_token_id,

            processor.tokenizer.convert_tokens_to_ids(
                "<|im_end|>"
            )

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


    generation_time = (
        time.time()
        -
        start
    )


    input_length = (
        batch["input_ids"]
        .shape[1]
    )


    generated_ids = (
        output_ids[
            :,
            input_length:
        ]
    )


    text = processor.tokenizer.batch_decode(

        generated_ids,

        skip_special_tokens=True

    )[0]


    text = text.strip()


    generated_length = (
        generated_ids.shape[1]
    )


    return {

        "answer":
            text,

        "generated_tokens":
            int(generated_length),

        "generation_time_sec":
            float(generation_time),

    }


# ============================================================
# Generation result helper
# ============================================================

def add_generation_result(
    generation_records,
    model_name,
    item_id,
    result
):

    if item_id not in generation_records:

        generation_records[
            item_id
        ] = {}


    generation_records[
        item_id
    ][
        model_name
    ] = result


# ============================================================
# Main evaluation
# ============================================================

def load_project_checkpoint(
    model,
    checkpoint_path,
    model_name,
    allowed_missing_prefixes=None,
):
    """
    Load project best_model.pt into an already constructed
    Stage-1 base model.

    Some project checkpoints intentionally omit frozen modules
    such as InternViT. Those missing keys are allowed explicitly.

    Everything else must match exactly.
    """

    if allowed_missing_prefixes is None:
        allowed_missing_prefixes = []

    checkpoint_path = Path(checkpoint_path)

    print("=" * 80, flush=True)
    print(f"[{model_name}] Loading checkpoint", flush=True)
    print(f"[{model_name}] Path: {checkpoint_path}", flush=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    if not isinstance(state_dict, dict):
        raise TypeError(
            f"{model_name}: checkpoint must be a state_dict dict, "
            f"got {type(state_dict)}"
        )

    print(
        f"[{model_name}] Checkpoint tensors: {len(state_dict)}",
        flush=True,
    )

    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict,
        strict=False,
    )
    

    missing_keys = list(missing_keys)
    unexpected_keys = list(unexpected_keys)

    allowed_missing = []
    forbidden_missing = []

    for key in missing_keys:
        allowed = any(
            key.startswith(prefix)
            for prefix in allowed_missing_prefixes
        )

        if allowed:
            allowed_missing.append(key)
        else:
            forbidden_missing.append(key)

    print(
        f"[{model_name}] Missing keys: {len(missing_keys)}",
        flush=True,
    )

    print(
        f"[{model_name}] Allowed missing keys: "
        f"{len(allowed_missing)}",
        flush=True,
    )

    print(
        f"[{model_name}] Forbidden missing keys: "
        f"{len(forbidden_missing)}",
        flush=True,
    )

    print(
        f"[{model_name}] Unexpected keys: "
        f"{len(unexpected_keys)}",
        flush=True,
    )

    # --------------------------------------------------
    # Print missing keys
    # --------------------------------------------------

    if allowed_missing:
        print(
            f"\n[{model_name}] Allowed missing keys:",
            flush=True,
        )

        for key in allowed_missing:
            print(
                f"  [ALLOWED] {key}",
                flush=True,
            )

    # --------------------------------------------------
    # Forbidden missing keys
    # --------------------------------------------------

    if forbidden_missing:
        print(
            f"\n[{model_name}] ERROR: Forbidden missing keys:",
            flush=True,
        )

        for key in forbidden_missing:
            print(
                f"  [ERROR] {key}",
                flush=True,
            )

        raise RuntimeError(
            f"{model_name}: checkpoint is incomplete. "
            f"{len(forbidden_missing)} non-allowed parameters "
            f"are missing."
        )

    # --------------------------------------------------
    # Unexpected keys
    # --------------------------------------------------

    if unexpected_keys:
        print(
            f"\n[{model_name}] ERROR: Unexpected keys:",
            flush=True,
        )

        for key in unexpected_keys:
            print(
                f"  [ERROR] {key}",
                flush=True,
            )

        raise RuntimeError(
            f"{model_name}: checkpoint contains "
            f"{len(unexpected_keys)} unexpected parameters."
        )

    print(
        f"[{model_name}] CHECKPOINT LOAD: PASS",
        flush=True,
    )

    print("=" * 80, flush=True)

    return model


def main():

    global IMAGE_ROOT

    parser = argparse.ArgumentParser(

        description=(
            "Evaluate SFT vs DPO-v1 vs DPO-v2 "
            "on MM-RLHF preference pairs."
        )

    )


    parser.add_argument(

        "--sft-ckpt",

        type=Path,

        default=SFT_CKPT

    )


    parser.add_argument(

        "--dpo-v1-ckpt",

        type=Path,

        default=DPO_V1_CKPT

    )


    parser.add_argument(

        "--dpo-v2-ckpt",

        type=Path,

        default=DPO_V2_CKPT

    )


    parser.add_argument(

        "--val-json",

        type=Path,

        default=VAL_JSON

    )


    parser.add_argument(

        "--image-root",

        type=Path,

        default=IMAGE_ROOT

    )


    parser.add_argument(

        "--out-dir",

        type=Path,

        default=DEFAULT_OUT_DIR

    )


    parser.add_argument(

        "--limit",

        type=int,

        default=DEFAULT_LIMIT,

        help=(
            "Number of validation samples. "
            "0 means all."
        )

    )


    parser.add_argument(

        "--generation-limit",

        type=int,

        default=DEFAULT_GENERATION_LIMIT,

        help=(
            "Number of samples for generation comparison. "
            "0 disables generation."
        )

    )


    parser.add_argument(

        "--max-new-tokens",

        type=int,

        default=DEFAULT_MAX_NEW_TOKENS

    )


    parser.add_argument(

        "--seed",

        type=int,

        default=DEFAULT_SEED

    )


    parser.add_argument(

        "--beta",

        type=float,

        default=DEFAULT_BETA

    )


    args = parser.parse_args()


    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    out_dir = Path(
        args.out_dir
    )


    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    log_file = (
        out_dir
        /
        "test_sft_vs_dpo_v1_v2.log"
    )


    summary_file = (
        out_dir
        /
        "summary.json"
    )


    scores_file = (
        out_dir
        /
        "per_sample_scores.json"
    )


    generation_file = (
        out_dir
        /
        "generation_results.json"
    )


    selection_file = (
        out_dir
        /
        "selected_samples.json"
    )


    setup_logging(
        log_file
    )


    LOGGER.info(
        "============================================================"
    )

    LOGGER.info(
        "SFT vs DPO-v1 vs DPO-v2 evaluation"
    )

    LOGGER.info(
        "============================================================"
    )


    LOGGER.info(
        "Project root: %s",
        PROJECT_ROOT
    )

    LOGGER.info(
        "torch: %s",
        torch.__version__
    )

    LOGGER.info(
        "CUDA available: %s",
        torch.cuda.is_available()
    )

    LOGGER.info(
        "Device: %s",
        DEVICE
    )

    LOGGER.info(
        "dtype: %s",
        DTYPE
    )

    LOGGER.info(
        "beta: %.6f",
        args.beta
    )


    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is required."
        )


    if not torch.cuda.is_bf16_supported():

        raise RuntimeError(
            "Current GPU does not support BF16."
        )


    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    LOGGER.info(
        "SFT checkpoint: %s",
        args.sft_ckpt
    )

    LOGGER.info(
        "DPO-v1 checkpoint: %s",
        args.dpo_v1_ckpt
    )

    LOGGER.info(
        "DPO-v2 checkpoint: %s",
        args.dpo_v2_ckpt
    )

    LOGGER.info(
        "Validation JSON: %s",
        args.val_json
    )

    LOGGER.info(
        "Image root: %s",
        args.image_root
    )

    LOGGER.info(
        "Output directory: %s",
        out_dir
    )


    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    raw_items = load_preference_json(
        args.val_json
    )


    # --------------------------------------------------------
    # Temporarily override global image root
    #
    # resolve_image_path() uses IMAGE_ROOT.
    # Therefore update it here for command-line overrides.
    # --------------------------------------------------------


    IMAGE_ROOT = Path(
        args.image_root
    )


    items = prepare_items(
        raw_items
    )


    selected = select_items(

        items,

        args.limit,

        args.seed

    )


    selection_metadata = {

        "seed":
            args.seed,

        "requested_limit":
            args.limit,

        "selected_count":
            len(selected),

        "sample_ids":
            [
                item.get(
                    "id",
                    str(i)
                )
                for i, item in enumerate(
                    selected
                )
            ]

    }


    save_json(
        selection_file,
        selection_metadata
    )


    # --------------------------------------------------------
    # Main result container
    # --------------------------------------------------------

    all_scores = {}


    generation_records = {}


    model_paths = {

        "SFT":
            Path(args.sft_ckpt),

        "DPO-v1":
            Path(args.dpo_v1_ckpt),

        "DPO-v2":
            Path(args.dpo_v2_ckpt),

    }


    # ========================================================
    # Evaluate models one by one
    # ========================================================

    for model_name, checkpoint_path in model_paths.items():

        LOGGER.info("")
        LOGGER.info(
            "############################################################"
        )

        LOGGER.info(
            "START MODEL: %s",
            model_name
        )

        LOGGER.info(
            "############################################################"
        )


        model = None

        processor = None


        try:

            model, processor = load_model(

                checkpoint_path,

                model_name

            )


            model_records = {}


            generation_enabled = (

                args.generation_limit > 0
            )


            generation_count = min(

                args.generation_limit,

                len(selected)

            )


            LOGGER.info(
                "Generation samples for %s: %d",
                model_name,
                generation_count
            )


            # ------------------------------------------------
            # Score all samples
            # ------------------------------------------------

            progress = tqdm(

                selected,

                desc=f"{model_name} scoring",

                unit="sample"

            )


            for index, item in enumerate(
                progress
            ):

                item_id = item.get(

                    "id",

                    str(index)

                )


                score = score_pair(

                    model,

                    processor,

                    item

                )


                model_records[
                    item_id
                ] = score


                # --------------------------------------------
                # Save generation on the first N samples
                # --------------------------------------------

                if (

                    generation_enabled

                    and

                    index < generation_count

                ):

                    generation_result = (
                        generate_answer(

                            model,

                            processor,

                            item,

                            args.max_new_tokens

                        )
                    )


                    add_generation_result(

                        generation_records,

                        model_name,

                        item_id,

                        generation_result

                    )


                    LOGGER.info("")
                    LOGGER.info(
                        "[%s][GENERATION] sample=%d id=%s",
                        model_name,
                        index,
                        item_id
                    )

                    LOGGER.info(
                        "[%s][GENERATION] %s",
                        model_name,
                        generation_result["answer"]
                    )


                # --------------------------------------------
                # Periodic progress
                # --------------------------------------------

                if (

                    (index + 1) % 50 == 0

                    or

                    index == 0

                ):

                    legacy_margin = (
                        score["legacy"]["margin"]
                    )

                    correct_margin = (
                        score["correct"]["margin"]
                    )

                    LOGGER.info(

                        "[%s] %d/%d | "
                        "legacy_margin=%.6f | "
                        "correct_margin=%.6f | "
                        "raw_prompt=%d | "
                        "expanded_prompt=%d",

                        model_name,

                        index + 1,

                        len(selected),

                        legacy_margin,

                        correct_margin,

                        score[
                            "diagnostics"
                        ][
                            "raw_prompt_tokens"
                        ],

                        score[
                            "diagnostics"
                        ][
                            "expanded_prompt_tokens"
                        ]

                    )


                    # ----------------------------------------
                    # Incremental checkpoint
                    # ----------------------------------------

                    save_json(

                        scores_file,

                        {

                            "status":
                                "running",

                            "current_model":
                                model_name,

                            "completed_samples":
                                index + 1,

                            "model_records":
                                model_records,

                        }

                    )


                    save_json(

                        generation_file,

                        generation_records

                    )


            progress.close()


            all_scores[
                model_name
            ] = model_records


            # ------------------------------------------------
            # Model summary
            # ------------------------------------------------

            model_summary = compute_model_summary(
                model_records
            )


            LOGGER.info("")
            LOGGER.info(
                "================ MODEL RESULT: %s ================",
                model_name
            )


            LOGGER.info(
                "[%s] legacy raw margin: %.6f",
                model_name,
                model_summary[
                    "legacy"
                ][
                    "margin"
                ][
                    "mean"
                ]
            )


            LOGGER.info(
                "[%s] legacy preference accuracy: %.4f",
                model_name,
                model_summary[
                    "legacy"
                ][
                    "preference_accuracy"
                ][
                    "accuracy"
                ]
            )


            LOGGER.info(
                "[%s] correct raw margin: %.6f",
                model_name,
                model_summary[
                    "correct"
                ][
                    "margin"
                ][
                    "mean"
                ]
            )


            LOGGER.info(
                "[%s] correct preference accuracy: %.4f",
                model_name,
                model_summary[
                    "correct"
                ][
                    "preference_accuracy"
                ][
                    "accuracy"
                ]
            )


            LOGGER.info(
                "[%s] correct chosen avg logprob: %.6f",
                model_name,
                model_summary[
                    "correct"
                ][
                    "chosen_avg_logprob"
                ][
                    "mean"
                ]
            )


            LOGGER.info(
                "[%s] correct rejected avg logprob: %.6f",
                model_name,
                model_summary[
                    "correct"
                ][
                    "rejected_avg_logprob"
                ][
                    "mean"
                ]
            )


            LOGGER.info(
                "[%s] average correct answer tokens: %.2f",
                model_name,
                model_summary[
                    "correct"
                ][
                    "average_answer_tokens"
                ][
                    "mean"
                ]
            )


            LOGGER.info(
                "[%s] average forward time: %.4f sec",
                model_name,
                model_summary[
                    "forward_time_sec"
                ][
                    "mean"
                ]
            )


            # ------------------------------------------------
            # Save model summary temporarily
            # ------------------------------------------------

            save_json(

                scores_file,

                {

                    "status":
                        "running",

                    "current_model":
                        model_name,

                    "model_summary":
                        model_summary,

                    "all_scores":
                        all_scores,

                }

            )


        finally:

            LOGGER.info(
                "Releasing model: %s",
                model_name
            )


            release_model(
                model
            )


            model = None

            processor = None


            LOGGER.info(
                "GPU memory released for: %s",
                model_name
            )


    # ========================================================
    # DPO relative-to-SFT analysis
    # ========================================================

    LOGGER.info("")
    LOGGER.info(
        "============================================================"
    )

    LOGGER.info(
        "DPO relative-to-SFT analysis"
    )

    LOGGER.info(
        "============================================================"
    )


    sft_records = all_scores[
        "SFT"
    ]


    dpo_v1_records = all_scores[
        "DPO-v1"
    ]


    dpo_v2_records = all_scores[
        "DPO-v2"
    ]


    dpo_v1_metrics = compute_dpo_metrics(

        dpo_v1_records,

        sft_records,

        args.beta

    )


    dpo_v2_metrics = compute_dpo_metrics(

        dpo_v2_records,

        sft_records,

        args.beta

    )


    LOGGER.info("")
    LOGGER.info(
        "================ DPO-v1 / CORRECT MASK ================"
    )


    LOGGER.info(
        "DPO-v1 chosen reward: %.6f",
        dpo_v1_metrics[
            "correct"
        ][
            "chosen_reward"
        ][
            "mean"
        ]
    )


    LOGGER.info(
        "DPO-v1 rejected reward: %.6f",
        dpo_v1_metrics[
            "correct"
        ][
            "rejected_reward"
        ][
            "mean"
        ]
    )


    LOGGER.info(
        "DPO-v1 reward margin: %.6f",
        dpo_v1_metrics[
            "correct"
        ][
            "reward_margin"
        ][
            "mean"
        ]
    )


    LOGGER.info(
        "DPO-v1 reward accuracy: %.4f",
        dpo_v1_metrics[
            "correct"
        ][
            "reward_accuracy"
        ][
            "accuracy"
        ]
    )


    LOGGER.info(
        "DPO-v1 DPO loss: %.6f",
        dpo_v1_metrics[
            "correct"
        ][
            "dpo_loss"
        ][
            "mean"
        ]
    )


    LOGGER.info(
        "DPO-v1 chosen logprob delta: %.6f",
        dpo_v1_metrics[
            "correct"
        ][
            "chosen_logprob_delta_vs_reference"
        ][
            "mean"
        ]
    )


    LOGGER.info(
        "DPO-v1 rejected logprob delta: %.6f",
        dpo_v1_metrics[
            "correct"
        ][
            "rejected_logprob_delta_vs_reference"
        ][
            "mean"
        ]
    )


    LOGGER.info("")
    LOGGER.info(
        "================ DPO-v2 / CORRECT MASK ================"
    )


    LOGGER.info(
        "DPO-v2 chosen reward: %.6f",
        dpo_v2_metrics[
            "correct"
        ][
            "chosen_reward"
        ][
            "mean"
        ]
    )


    LOGGER.info(
        "DPO-v2 rejected reward: %.6f",
        dpo_v2_metrics[
            "correct"
        ][
            "rejected_reward"
        ][
            "mean"
        ]
    )


    LOGGER.info(
        "DPO-v2 reward margin: %.6f",
        dpo_v2_metrics[
            "correct"
        ][
            "reward_margin"
        ][
            "mean"
        ]
    )


    LOGGER.info(
        "DPO-v2 reward accuracy: %.4f",
        dpo_v2_metrics[
            "correct"
        ][
            "reward_accuracy"
        ][
            "accuracy"
        ]
    )


    LOGGER.info(
        "DPO-v2 DPO loss: %.6f",
        dpo_v2_metrics[
            "correct"
        ][
            "dpo_loss"
        ][
            "mean"
        ]
    )


    LOGGER.info(
        "DPO-v2 chosen logprob delta: %.6f",
        dpo_v2_metrics[
            "correct"
        ][
            "chosen_logprob_delta_vs_reference"
        ][
            "mean"
        ]
    )


    LOGGER.info(
        "DPO-v2 rejected logprob delta: %.6f",
        dpo_v2_metrics[
            "correct"
        ][
            "rejected_logprob_delta_vs_reference"
        ][
            "mean"
        ]
    )


    # ========================================================
    # Compare DPO-v1 vs DPO-v2
    # ========================================================

    def get_mean(
        obj,
        *keys
    ):

        x = obj

        for key in keys:

            x = x[key]

        return x["mean"]


    comparison = {

        "correct_mask":

        {

            "SFT_raw_margin":
                get_mean(
                    compute_model_summary(
                        sft_records
                    ),
                    "correct",
                    "margin"
                ),

            "DPO_v1_raw_margin":
                get_mean(
                    compute_model_summary(
                        dpo_v1_records
                    ),
                    "correct",
                    "margin"
                ),

            "DPO_v2_raw_margin":
                get_mean(
                    compute_model_summary(
                        dpo_v2_records
                    ),
                    "correct",
                    "margin"
                ),

            "DPO_v1_reward_margin":
                get_mean(
                    dpo_v1_metrics,
                    "correct",
                    "reward_margin"
                ),

            "DPO_v2_reward_margin":
                get_mean(
                    dpo_v2_metrics,
                    "correct",
                    "reward_margin"
                ),

            "DPO_v1_reward_accuracy":
                dpo_v1_metrics[
                    "correct"
                ][
                    "reward_accuracy"
                ][
                    "accuracy"
                ],

            "DPO_v2_reward_accuracy":
                dpo_v2_metrics[
                    "correct"
                ][
                    "reward_accuracy"
                ][
                    "accuracy"
                ],

            "DPO_v1_loss":
                get_mean(
                    dpo_v1_metrics,
                    "correct",
                    "dpo_loss"
                ),

            "DPO_v2_loss":
                get_mean(
                    dpo_v2_metrics,
                    "correct",
                    "dpo_loss"
                ),

            "DPO_v1_chosen_delta":
                get_mean(
                    dpo_v1_metrics,
                    "correct",
                    "chosen_logprob_delta_vs_reference"
                ),

            "DPO_v1_rejected_delta":
                get_mean(
                    dpo_v1_metrics,
                    "correct",
                    "rejected_logprob_delta_vs_reference"
                ),

            "DPO_v2_chosen_delta":
                get_mean(
                    dpo_v2_metrics,
                    "correct",
                    "chosen_logprob_delta_vs_reference"
                ),

            "DPO_v2_rejected_delta":
                get_mean(
                    dpo_v2_metrics,
                    "correct",
                    "rejected_logprob_delta_vs_reference"
                ),

        },

        "legacy_mask":

        {

            "DPO_v1_reward_margin":
                get_mean(
                    dpo_v1_metrics,
                    "legacy",
                    "reward_margin"
                ),

            "DPO_v2_reward_margin":
                get_mean(
                    dpo_v2_metrics,
                    "legacy",
                    "reward_margin"
                ),

            "DPO_v1_reward_accuracy":
                dpo_v1_metrics[
                    "legacy"
                ][
                    "reward_accuracy"
                ][
                    "accuracy"
                ],

            "DPO_v2_reward_accuracy":
                dpo_v2_metrics[
                    "legacy"
                ][
                    "reward_accuracy"
                ][
                    "accuracy"
                ],

        }

    }


    # ========================================================
    # Prompt expansion statistics
    # ========================================================

    expansion_values = []

    raw_prompt_values = []

    expanded_prompt_values = []


    for record in sft_records.values():

        diagnostics = record[
            "diagnostics"
        ]


        raw_prompt_values.append(
            diagnostics[
                "raw_prompt_tokens"
            ]
        )


        expanded_prompt_values.append(
            diagnostics[
                "expanded_prompt_tokens"
            ]
        )


        expansion_values.append(
            diagnostics[
                "extra_visual_tokens_added"
            ]
        )


    mask_diagnostics = {

        "raw_prompt_tokens":
            summarize_values(
                raw_prompt_values
            ),

        "expanded_prompt_tokens":
            summarize_values(
                expanded_prompt_values
            ),

        "extra_visual_tokens_added":
            summarize_values(
                expansion_values
            ),

        "num_samples_where_lengths_differ":
            int(
                sum(
                    1
                    for raw, expanded
                    in zip(
                        raw_prompt_values,
                        expanded_prompt_values
                    )
                    if raw != expanded
                )
            ),

        "fraction_where_lengths_differ":
            float(
                sum(
                    1
                    for raw, expanded
                    in zip(
                        raw_prompt_values,
                        expanded_prompt_values
                    )
                    if raw != expanded
                )
                /
                max(
                    len(raw_prompt_values),
                    1
                )
            ),

    }


    LOGGER.info("")
    LOGGER.info(
        "============================================================"
    )

    LOGGER.info(
        "PROMPT / VISUAL TOKEN MASK DIAGNOSTICS"
    )

    LOGGER.info(
        "============================================================"
    )


    LOGGER.info(
        "Average raw prompt tokens: %.2f",
        mask_diagnostics[
            "raw_prompt_tokens"
        ][
            "mean"
        ]
    )


    LOGGER.info(
        "Average expanded prompt tokens: %.2f",
        mask_diagnostics[
            "expanded_prompt_tokens"
        ][
            "mean"
        ]
    )


    LOGGER.info(
        "Average extra visual tokens: %.2f",
        mask_diagnostics[
            "extra_visual_tokens_added"
        ][
            "mean"
        ]
    )


    LOGGER.info(
        "Samples with different raw/expanded prompt length: %d / %d",
        mask_diagnostics[
            "num_samples_where_lengths_differ"
        ],
        len(
            sft_records
        )
    )


    LOGGER.info(
        "Fraction with different lengths: %.4f",
        mask_diagnostics[
            "fraction_where_lengths_differ"
        ]
    )


    # ========================================================
    # Generation summary
    # ========================================================

    generation_summary = {

        "num_samples":
            len(generation_records),

        "models":
            {}

    }


    for model_name in (
        "SFT",
        "DPO-v1",
        "DPO-v2"
    ):

        answers = []

        lengths = []

        times = []


        for item_id, result in generation_records.items():

            if model_name not in result:

                continue


            x = result[
                model_name
            ]


            answers.append(
                x["answer"]
            )

            lengths.append(
                x["generated_tokens"]
            )

            times.append(
                x["generation_time_sec"]
            )


        generation_summary[
            "models"
        ][
            model_name
        ] = {

            "num_samples":
                len(answers),

            "average_generated_tokens":
                mean(
                    lengths
                ),

            "average_generation_time_sec":
                mean(
                    times
                ),

        }


    # ========================================================
    # Final summary
    # ========================================================

    final_summary = {

        "experiment":
        {

            "name":
                "SFT vs DPO-v1 vs DPO-v2",

            "beta":
                args.beta,

            "limit":
                args.limit,

            "selected_samples":
                len(selected),

            "seed":
                args.seed,

            "generation_limit":
                args.generation_limit,

            "max_new_tokens":
                args.max_new_tokens,

        },

        "checkpoints":
        {

            "SFT":
                str(args.sft_ckpt),

            "DPO-v1":
                str(args.dpo_v1_ckpt),

            "DPO-v2":
                str(args.dpo_v2_ckpt),

        },

        "dataset":
        {

            "validation_json":
                str(args.val_json),

            "image_root":
                str(args.image_root),

            "original_samples":
                len(raw_items),

            "valid_samples":
                len(items),

            "evaluated_samples":
                len(selected),

        },

        "model_summaries":
        {

            model_name:
                compute_model_summary(
                    records
                )

            for model_name, records
            in all_scores.items()

        },

        "dpo_relative_to_sft":
        {

            "DPO-v1":
                dpo_v1_metrics,

            "DPO-v2":
                dpo_v2_metrics,

        },

        "comparison":
            comparison,

        "mask_diagnostics":
            mask_diagnostics,

        "generation_summary":
            generation_summary,

        "output_files":
        {

            "log":
                str(log_file),

            "summary":
                str(summary_file),

            "per_sample_scores":
                str(scores_file),

            "generation":
                str(generation_file),

            "selection":
                str(selection_file),

        }

    }


    # --------------------------------------------------------
    # Save final files
    # --------------------------------------------------------

    save_json(
        summary_file,
        final_summary
    )


    save_json(

        scores_file,

        {

            "status":
                "finished",

            "all_scores":
                all_scores,

        }

    )


    save_json(
        generation_file,
        generation_records
    )


    # ========================================================
    # Human-readable final output
    # ========================================================

    LOGGER.info("")
    LOGGER.info(
        "############################################################"
    )

    LOGGER.info(
        "FINAL RESULT"
    )

    LOGGER.info(
        "############################################################"
    )


    for model_name in (
        "SFT",
        "DPO-v1",
        "DPO-v2"
    ):

        summary = final_summary[
            "model_summaries"
        ][
            model_name
        ]


        LOGGER.info("")
        LOGGER.info(
            "[%s]",
            model_name
        )


        LOGGER.info(
            "Correct raw margin = %.6f",
            summary[
                "correct"
            ][
                "margin"
            ][
                "mean"
            ]
        )


        LOGGER.info(
            "Correct preference accuracy = %.4f",
            summary[
                "correct"
            ][
                "preference_accuracy"
            ][
                "accuracy"
            ]
        )


        LOGGER.info(
            "Correct chosen logprob = %.6f",
            summary[
                "correct"
            ][
                "chosen_avg_logprob"
            ][
                "mean"
            ]
        )


        LOGGER.info(
            "Correct rejected logprob = %.6f",
            summary[
                "correct"
            ][
                "rejected_avg_logprob"
            ][
                "mean"
            ]
        )


    LOGGER.info("")
    LOGGER.info(
        "---------------- DPO-v1 relative to SFT ----------------"
    )


    LOGGER.info(
        "Reward margin = %.6f",
        comparison[
            "correct_mask"
        ][
            "DPO_v1_reward_margin"
        ]
    )


    LOGGER.info(
        "Reward accuracy = %.4f",
        comparison[
            "correct_mask"
        ][
            "DPO_v1_reward_accuracy"
        ]
    )


    LOGGER.info(
        "Chosen delta = %.6f",
        comparison[
            "correct_mask"
        ][
            "DPO_v1_chosen_delta"
        ]
    )


    LOGGER.info(
        "Rejected delta = %.6f",
        comparison[
            "correct_mask"
        ][
            "DPO_v1_rejected_delta"
        ]
    )


    LOGGER.info("")
    LOGGER.info(
        "---------------- DPO-v2 relative to SFT ----------------"
    )


    LOGGER.info(
        "Reward margin = %.6f",
        comparison[
            "correct_mask"
        ][
            "DPO_v2_reward_margin"
        ]
    )


    LOGGER.info(
        "Reward accuracy = %.4f",
        comparison[
            "correct_mask"
        ][
            "DPO_v2_reward_accuracy"
        ]
    )


    LOGGER.info(
        "Chosen delta = %.6f",
        comparison[
            "correct_mask"
        ][
            "DPO_v2_chosen_delta"
        ]
    )


    LOGGER.info(
        "Rejected delta = %.6f",
        comparison[
            "correct_mask"
        ][
            "DPO_v2_rejected_delta"
        ]
    )


    LOGGER.info("")
    LOGGER.info(
        "Prompt expansion:"
    )


    LOGGER.info(
        "raw prompt tokens = %.2f",
        mask_diagnostics[
            "raw_prompt_tokens"
        ][
            "mean"
        ]
    )


    LOGGER.info(
        "expanded prompt tokens = %.2f",
        mask_diagnostics[
            "expanded_prompt_tokens"
        ][
            "mean"
        ]
    )


    LOGGER.info(
        "extra visual tokens = %.2f",
        mask_diagnostics[
            "extra_visual_tokens_added"
        ][
            "mean"
        ]
    )


    LOGGER.info("")
    LOGGER.info(
        "Saved log: %s",
        log_file
    )

    LOGGER.info(
        "Saved summary: %s",
        summary_file
    )

    LOGGER.info(
        "Saved per-sample scores: %s",
        scores_file
    )

    LOGGER.info(
        "Saved generation results: %s",
        generation_file
    )

    LOGGER.info(
        "Saved sample selection: %s",
        selection_file
    )


    LOGGER.info(
        "Evaluation finished successfully."
    )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    main()