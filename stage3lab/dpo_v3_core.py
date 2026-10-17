#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DPO-v3 core components.

Contains:

1. MMRefDPODatasetV3
2. MMRefDPOCollatorV3
3. multimodal response-only masking
4. sequence log-probability computation
5. DPO-v3 loss
6. chosen reward anchor
7. chosen/rejected concatenation helper
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from PIL import Image, ImageOps


IMAGE_TOKEN = "<|image_pad|>"
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


# ============================================================
# Dataset
# ============================================================

class MMRefDPODatasetV3(Dataset):

    def __init__(
        self,
        json_path: str,
        image_root: str,
        verbose: bool = True,
    ):

        self.json_path = Path(json_path)
        self.image_root = Path(image_root)

        if not self.json_path.exists():
            raise FileNotFoundError(self.json_path)

        self.items = []

        total = 0
        skipped = 0

        # ----------------------------------------------------
        # JSON array
        # ----------------------------------------------------

        with open(
            self.json_path,
            "r",
            encoding="utf-8",
        ) as f:

            first_char = f.read(1)

        if first_char == "[":

            with open(
                self.json_path,
                "r",
                encoding="utf-8",
            ) as f:

                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError(
                    "JSON array expected."
                )

            for x in data:

                total += 1

                if self._validate_item(x):
                    self.items.append(x)
                else:
                    skipped += 1

        # ----------------------------------------------------
        # JSONL
        # ----------------------------------------------------

        else:

            with open(
                self.json_path,
                "r",
                encoding="utf-8",
            ) as f:

                for line in f:

                    if not line.strip():
                        continue

                    total += 1

                    try:
                        x = json.loads(line)
                    except Exception:
                        skipped += 1
                        continue

                    if self._validate_item(x):
                        self.items.append(x)
                    else:
                        skipped += 1

        if verbose:

            print(
                f"Original samples: {total}",
                flush=True,
            )

            print(
                f"Valid image DPO samples: {len(self.items)}",
                flush=True,
            )

            print(
                f"Skipped samples: {skipped}",
                flush=True,
            )

    # --------------------------------------------------------
    # validation
    # --------------------------------------------------------

    def _validate_item(self, x):

        if not isinstance(x, dict):
            return False

        if "image" not in x:
            return False

        if "prompt" not in x:
            return False

        if "chosen" not in x:
            return False

        if "rejected" not in x:
            return False

        image_path = self.image_root / str(
            x["image"]
        )

        if not image_path.exists():
            return False

        if not str(x["prompt"]).strip():
            return False

        if not str(x["chosen"]).strip():
            return False

        if not str(x["rejected"]).strip():
            return False

        return True

    # --------------------------------------------------------
    # length
    # --------------------------------------------------------

    def __len__(self):

        return len(self.items)

    # --------------------------------------------------------
    # prompt
    # --------------------------------------------------------

    @staticmethod
    def build_prompt(text: str):

        text = str(text)

        if any(
            token in text
            for token in (
                IM_START,
                IM_END,
                IMAGE_TOKEN,
            )
        ):

            raise ValueError(
                "DPO prompt already contains ChatML/image "
                "special tokens. Expected plain text prompt."
            )

        return (
            f"{IM_START}user\n"
            f"{VISION_START}"
            f"{IMAGE_TOKEN}"
            f"{VISION_END}\n"
            f"{text}"
            f"{IM_END}\n"
            f"{IM_START}assistant\n"
        )

    # --------------------------------------------------------
    # get item
    # --------------------------------------------------------

    def __getitem__(self, index):

        x = self.items[index]

        image_path = (
            self.image_root
            /
            str(x["image"])
        )

        with Image.open(image_path) as img:

            image = ImageOps.exif_transpose(
                img
            ).convert("RGB")

        return {

            "id":
                str(x.get("id", index)),

            "image":
                image,

            "prompt":
                self.build_prompt(
                    x["prompt"]
                ),

            "chosen":
                str(x["chosen"]),

            "rejected":
                str(x["rejected"]),
        }


# ============================================================
# Collator
# ============================================================

class MMRefDPOCollatorV3:

    def __init__(
        self,
        processor,
        max_tokens: int = 0,
    ):

        self.processor = processor
        self.max_tokens = int(max_tokens)

    # --------------------------------------------------------
    # answer normalization
    # --------------------------------------------------------

    @staticmethod
    def normalize_answer(answer: str):

        answer = str(answer).strip()

        # If source already has EOS, remove it first.
        while answer.endswith(IM_END):

            answer = answer[
                :-len(IM_END)
            ].rstrip()

        # We do not allow raw ChatML/image special tokens
        # inside the answer.
        for token in (
            IM_START,
            VISION_START,
            VISION_END,
            IMAGE_TOKEN,
        ):

            if token in answer:

                raise ValueError(
                    f"Unexpected special token {token!r} "
                    "inside DPO answer."
                )

        # Always train EOS.
        return answer + IM_END

    # --------------------------------------------------------
    # expanded text + supervision span
    # --------------------------------------------------------

    @staticmethod
    def expand_response_span(
        raw_text: str,
        response_start: int,
        response_end: int,
        image_tokens: int,
    ):

        if raw_text.count(IMAGE_TOKEN) != 1:

            raise ValueError(
                "Expected exactly one image placeholder."
            )

        if image_tokens < 1:

            raise ValueError(
                "image_tokens must be >= 1."
            )

        image_start = raw_text.index(
            IMAGE_TOKEN
        )

        if response_start <= image_start:

            raise ValueError(
                "Response must occur after image placeholder."
            )

        added_chars = (
            image_tokens - 1
        ) * len(IMAGE_TOKEN)

        expanded = raw_text.replace(
            IMAGE_TOKEN,
            IMAGE_TOKEN * image_tokens,
            1,
        )

        expanded_response_start = (
            response_start
            +
            added_chars
        )

        expanded_response_end = (
            response_end
            +
            added_chars
        )

        return (
            expanded,
            (
                expanded_response_start,
                expanded_response_end,
            ),
        )

    # --------------------------------------------------------
    # exact response-only token mask
    # --------------------------------------------------------

    def tokenize_and_mask(
        self,
        raw_text: str,
        response_start: int,
        response_end: int,
        image_tokens: int,
    ):

        tokenizer = self.processor.tokenizer

        if not getattr(
            tokenizer,
            "is_fast",
            False,
        ):

            raise RuntimeError(
                "DPO-v3 requires a fast tokenizer "
                "because return_offsets_mapping is required."
            )

        (
            expanded_text,
            response_span,
        ) = self.expand_response_span(
            raw_text,
            response_start,
            response_end,
            image_tokens,
        )

        encoded = tokenizer(
            expanded_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )

        ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]

        if len(ids) != len(offsets):

            raise RuntimeError(
                "Tokenizer IDs/offsets length mismatch."
            )

        labels = [
            -100
            for _ in ids
        ]

        start, end = response_span

        for token_index, (
            (offset_start, offset_end),
            token_id,
        ) in enumerate(
            zip(offsets, ids)
        ):

            # Special tokens can have zero-width offsets.
            if offset_start == offset_end:
                continue

            overlaps = (
                offset_start < end
                and
                offset_end > start
            )

            if not overlaps:
                continue

            # Do not allow a token to cross the
            # response boundary.
            if not (
                offset_start >= start
                and
                offset_end <= end
            ):

                raise RuntimeError(
                    "Tokenizer token crosses response "
                    f"boundary at token {token_index}: "
                    f"{offset_start}:{offset_end}, "
                    f"response={start}:{end}"
                )

            labels[token_index] = token_id

        supervised = sum(
            x != -100
            for x in labels
        )

        if supervised <= 0:

            raise RuntimeError(
                "No response tokens were supervised."
            )

        image_token_id = (
            tokenizer.convert_tokens_to_ids(
                IMAGE_TOKEN
            )
        )

        actual_image_tokens = sum(
            token_id == image_token_id
            for token_id in ids
        )

        if actual_image_tokens != image_tokens:

            raise RuntimeError(
                "Visual token mismatch: "
                f"expected={image_tokens}, "
                f"tokenizer={actual_image_tokens}"
            )

        for token_id, label in zip(
            ids,
            labels,
        ):

            if (
                token_id == image_token_id
                and
                label != -100
            ):

                raise AssertionError(
                    "Image token was accidentally supervised."
                )

        return ids, labels

    # --------------------------------------------------------
    # encode one branch
    # --------------------------------------------------------

    def encode(
        self,
        prompts,
        answers,
        images,
    ):

        full_texts = []

        response_starts = []
        response_ends = []

        normalized_answers = []

        for prompt, answer in zip(
            prompts,
            answers,
        ):

            normalized = (
                self.normalize_answer(
                    answer
                )
            )

            full = (
                prompt
                +
                normalized
            )

            response_start = len(prompt)
            response_end = len(full)

            full_texts.append(full)
            normalized_answers.append(normalized)

            response_starts.append(
                response_start
            )

            response_ends.append(
                response_end
            )

        # ----------------------------------------------------
        # Processor performs actual visual-token expansion.
        # ----------------------------------------------------

        result = self.processor.encode_images(
            full_texts,
            images_per_sample=images,
            padding=True,
        )

        input_ids = result["input_ids"]
        attention_mask = result["attention_mask"]

        grid = result["image_grid_thw"]

        if grid is None:

            raise RuntimeError(
                "image_grid_thw is missing."
            )

        if grid.ndim != 2:

            raise RuntimeError(
                f"Unexpected image_grid_thw shape: {grid.shape}"
            )

        if grid.shape[0] != len(full_texts):

            raise RuntimeError(
                "Number of image grids does not match "
                "batch size."
            )

        labels = torch.full_like(
            input_ids,
            -100,
        )

        response_token_counts = []
        prompt_token_counts = []
        image_token_counts = []

        for i in range(
            len(full_texts)
        ):

            image_tokens = int(
                (
                    grid[i].prod()
                    //
                    4
                ).item()
            )

            (
                tokenizer_ids,
                local_labels,
            ) = self.tokenize_and_mask(
                full_texts[i],
                response_starts[i],
                response_ends[i],
                image_tokens,
            )

            valid_positions = (
                torch.nonzero(
                    attention_mask[i],
                    as_tuple=False,
                )
                .flatten()
            )

            actual_ids = (
                input_ids[i][valid_positions]
                .tolist()
            )

            if actual_ids != tokenizer_ids:

                raise RuntimeError(
                    "Fast-tokenizer masking pass "
                    "disagrees with processor.encode_images(). "
                    f"sample={i}, "
                    f"processor_len={len(actual_ids)}, "
                    f"tokenizer_len={len(tokenizer_ids)}"
                )

            local_labels_tensor = torch.tensor(
                local_labels,
                dtype=torch.long,
                device=input_ids.device,
            )

            labels[
                i,
                valid_positions,
            ] = local_labels_tensor

            response_count = int(
                (local_labels_tensor != -100)
                .sum()
                .item()
            )

            prompt_count = (
                len(tokenizer_ids)
                -
                response_count
            )

            response_token_counts.append(
                response_count
            )

            prompt_token_counts.append(
                prompt_count
            )

            image_token_counts.append(
                image_tokens
            )

        # Explicitly mask padding.
        labels = labels.masked_fill(
            attention_mask == 0,
            -100,
        )

        if self.max_tokens > 0:

            if input_ids.shape[1] > self.max_tokens:

                raise RuntimeError(
                    "Sequence length exceeds max_tokens: "
                    f"{input_ids.shape[1]} > "
                    f"{self.max_tokens}"
                )

        result["labels"] = labels

        result["response_token_counts"] = torch.tensor(
            response_token_counts,
            dtype=torch.long,
        )

        result["prompt_token_counts"] = torch.tensor(
            prompt_token_counts,
            dtype=torch.long,
        )

        result["image_token_counts"] = torch.tensor(
            image_token_counts,
            dtype=torch.long,
        )

        return result

    # --------------------------------------------------------
    # call
    # --------------------------------------------------------

    def __call__(
        self,
        batch,
    ):

        images = [
            [item["image"]]
            for item in batch
        ]

        prompts = [
            item["prompt"]
            for item in batch
        ]

        chosen_answers = [
            item["chosen"]
            for item in batch
        ]

        rejected_answers = [
            item["rejected"]
            for item in batch
        ]

        chosen = self.encode(
            prompts,
            chosen_answers,
            images,
        )

        rejected = self.encode(
            prompts,
            rejected_answers,
            images,
        )

        return {

            "ids": [
                item["id"]
                for item in batch
            ],

            "chosen_input_ids":
                chosen["input_ids"],

            "chosen_attention_mask":
                chosen["attention_mask"],

            "chosen_labels":
                chosen["labels"],

            "chosen_pixel_values":
                chosen["pixel_values"],

            "chosen_image_grid_thw":
                chosen["image_grid_thw"],

            "chosen_response_token_counts":
                chosen["response_token_counts"],

            "chosen_prompt_token_counts":
                chosen["prompt_token_counts"],

            "chosen_image_token_counts":
                chosen["image_token_counts"],

            "rejected_input_ids":
                rejected["input_ids"],

            "rejected_attention_mask":
                rejected["attention_mask"],

            "rejected_labels":
                rejected["labels"],

            "rejected_pixel_values":
                rejected["pixel_values"],

            "rejected_image_grid_thw":
                rejected["image_grid_thw"],

            "rejected_response_token_counts":
                rejected["response_token_counts"],

            "rejected_prompt_token_counts":
                rejected["prompt_token_counts"],

            "rejected_image_token_counts":
                rejected["image_token_counts"],
        }


# ============================================================
# Pair concatenation
# ============================================================

def concat_pair_inputs(batch):
    """
    Concatenate chosen/rejected inputs into one forward batch.

    chosen and rejected may have different sequence lengths.
    Therefore all sequence-aligned tensors must first be right-padded
    to the same sequence length.

    Padding policy:
        input_ids      -> 0
        attention_mask -> 0
        labels         -> -100

    The padding tokens are ignored by attention_mask / label mask and
    therefore do not participate in the DPO log-probability calculation.
    """

    # ============================================================
    # Original tensors
    # ============================================================

    chosen_input_ids = batch["chosen_input_ids"]
    rejected_input_ids = batch["rejected_input_ids"]

    chosen_attention_mask = batch["chosen_attention_mask"]
    rejected_attention_mask = batch["rejected_attention_mask"]

    chosen_labels = batch["chosen_labels"]
    rejected_labels = batch["rejected_labels"]

    # ============================================================
    # Determine common sequence length
    # ============================================================

    chosen_len = chosen_input_ids.shape[1]
    rejected_len = rejected_input_ids.shape[1]

    target_len = max(
        chosen_len,
        rejected_len
    )

    # ============================================================
    # Right-padding helper
    # ============================================================

    def right_pad(
        tensor,
        target_length,
        pad_value
    ):
        current_length = tensor.shape[1]

        if current_length == target_length:
            return tensor

        pad_length = (
            target_length
            - current_length
        )

        padding = tensor.new_full(
            (
                tensor.shape[0],
                pad_length
            ),
            pad_value
        )

        return torch.cat(
            [
                tensor,
                padding
            ],
            dim=1
        )

    # ============================================================
    # Pad input_ids
    # ============================================================
    #
    # Use a neutral token id here.
    #
    # The corresponding attention_mask is 0, so these positions
    # are not visible to the model's valid-token computation.
    #
    # Do NOT use -100 for input_ids.
    #
    # ============================================================

    chosen_input_ids = right_pad(
        chosen_input_ids,
        target_len,
        0
    )

    rejected_input_ids = right_pad(
        rejected_input_ids,
        target_len,
        0
    )

    # ============================================================
    # Pad attention masks
    # ============================================================

    chosen_attention_mask = right_pad(
        chosen_attention_mask,
        target_len,
        0
    )

    rejected_attention_mask = right_pad(
        rejected_attention_mask,
        target_len,
        0
    )

    # ============================================================
    # Pad labels
    # ============================================================
    #
    # -100 means "ignore this position" in our DPO log-probability
    # computation.
    #
    # ============================================================

    chosen_labels = right_pad(
        chosen_labels,
        target_len,
        -100
    )

    rejected_labels = right_pad(
        rejected_labels,
        target_len,
        -100
    )

    # ============================================================
    # Concatenate chosen + rejected along batch dimension
    # ============================================================

    chosen_bsz = chosen_input_ids.shape[0]

    input_ids = torch.cat(
        [
            chosen_input_ids,
            rejected_input_ids
        ],
        dim=0
    )

    attention_mask = torch.cat(
        [
            chosen_attention_mask,
            rejected_attention_mask
        ],
        dim=0
    )

    labels = torch.cat(
        [
            chosen_labels,
            rejected_labels
        ],
        dim=0
    )

    # ============================================================
    # Multimodal tensors
    # ============================================================
    #
    # These are already represented independently from the text
    # sequence dimension, so they are concatenated directly.
    #
    # ============================================================

    pixel_values = torch.cat(
        [
            batch["chosen_pixel_values"],
            batch["rejected_pixel_values"]
        ],
        dim=0
    )

    image_grid_thw = torch.cat(
        [
            batch["chosen_image_grid_thw"],
            batch["rejected_image_grid_thw"]
        ],
        dim=0
    )

    # ============================================================
    # Safety checks
    # ============================================================

    assert (
        input_ids.shape[1]
        ==
        attention_mask.shape[1]
        ==
        labels.shape[1]
    ), (
        "Sequence-aligned tensors have inconsistent lengths: "
        f"input_ids={input_ids.shape}, "
        f"attention_mask={attention_mask.shape}, "
        f"labels={labels.shape}"
    )

    assert (
        input_ids.shape[0]
        ==
        chosen_bsz * 2
    ), (
        "Unexpected concatenated batch size: "
        f"{input_ids.shape[0]} "
        f"vs expected {chosen_bsz * 2}"
    )

    # ============================================================
    # Model inputs
    # ============================================================

    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }

    return (
        model_inputs,
        labels,
        chosen_bsz
    )

# ============================================================
# Log probability
# ============================================================

def get_selected_logprob_stats(
    logits: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 64,
):
    """
    Compute log-probabilities for an already aligned set of logits/labels.

    ``logits[:, i, :]`` predicts ``labels[:, i]`` directly. Positions with
    label == -100 are ignored.

    Only supervised rows are materialized before the vocabulary reduction.
    This avoids applying FP32 logsumexp to the full [B, T, V] tensor.
    The valid rows are further processed in chunks to bound temporary
    vocabulary-reduction memory.
    """

    if logits.ndim != 3:
        raise ValueError(
            f"logits must be [B,T,V], got {logits.shape}"
        )

    if labels.ndim != 2:
        raise ValueError(
            f"labels must be [B,T], got {labels.shape}"
        )

    if (
        logits.shape[0] != labels.shape[0]
        or logits.shape[1] != labels.shape[1]
    ):
        raise ValueError(
            "Logits/labels shape mismatch: "
            f"logits={logits.shape}, "
            f"labels={labels.shape}"
        )

    chunk_size = max(1, int(chunk_size))

    mask = labels != -100
    token_counts = mask.sum(dim=-1).clamp(min=1)

    if not bool(mask.any().item()):
        raise RuntimeError(
            "No supervised response tokens found."
        )

    # This is at most (#response tokens) x vocab, instead of
    # B x T x vocab.  For DPO this is the important memory reduction.
    valid_logits = logits[mask]
    valid_labels = labels[mask]

    token_log_probs = torch.empty(
        valid_labels.shape,
        dtype=torch.float32,
        device=logits.device,
    )

    for start_idx in range(
        0,
        valid_logits.shape[0],
        chunk_size,
    ):
        end_idx = min(
            start_idx + chunk_size,
            valid_logits.shape[0],
        )

        chunk_logits = valid_logits[start_idx:end_idx]

        # Keep the numerically sensitive vocabulary reduction in FP32,
        # but only for a small supervised chunk.
        log_normalizer = torch.logsumexp(
            chunk_logits.float(),
            dim=-1,
        )

        target_logits = torch.gather(
            chunk_logits,
            dim=-1,
            index=valid_labels[start_idx:end_idx].unsqueeze(-1),
        ).squeeze(-1).float()

        token_log_probs[start_idx:end_idx] = (
            target_logits - log_normalizer
        )

    # Map flat valid-token log-probabilities back to samples.
    sum_log_probs = torch.zeros(
        labels.shape[0],
        dtype=torch.float32,
        device=logits.device,
    )

    flat_sample_ids = torch.nonzero(
        mask,
        as_tuple=False,
    )[:, 0]

    sum_log_probs.scatter_add_(
        0,
        flat_sample_ids,
        token_log_probs,
    )

    avg_log_probs = (
        sum_log_probs
        /
        token_counts.float()
    )

    return (
        sum_log_probs,
        avg_log_probs,
        token_counts,
    )


def get_masked_logprob_stats(
    logits: torch.Tensor,
    labels: torch.Tensor,
):
    """
    Backward-compatible causal-LM log-probability helper.

    ``logits[:, t]`` predicts ``labels[:, t+1]``.
    """

    if logits.ndim != 3:
        raise ValueError(
            f"logits must be [B,T,V], got {logits.shape}"
        )

    if labels.ndim != 2:
        raise ValueError(
            f"labels must be [B,T], got {labels.shape}"
        )

    if (
        logits.shape[0] != labels.shape[0]
        or logits.shape[1] != labels.shape[1]
    ):
        raise ValueError(
            "Logits/labels shape mismatch: "
            f"logits={logits.shape}, "
            f"labels={labels.shape}"
        )

    return get_selected_logprob_stats(
        logits[:, :-1, :],
        labels[:, 1:],
    )

# ============================================================
# DPO-v3 loss
# ============================================================

def dpo_v3_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta: float = 0.1,
    anchor_weight: float = 0.25,
    anchor_delta: float = 0.0,
):

    # --------------------------------------------------------
    # Standard DPO
    # --------------------------------------------------------

    policy_logratio = (
        policy_chosen_logps
        -
        policy_rejected_logps
    )

    reference_logratio = (
        reference_chosen_logps
        -
        reference_rejected_logps
    )

    raw_margin = (
        policy_logratio
        -
        reference_logratio
    )

    dpo_margin = (
        beta
        *
        raw_margin
    )

    dpo_loss = (
        -F.logsigmoid(
            dpo_margin
        )
    ).mean()

    # --------------------------------------------------------
    # Implicit rewards
    # --------------------------------------------------------

    chosen_rewards = (
        beta
        *
        (
            policy_chosen_logps
            -
            reference_chosen_logps
        )
    )

    rejected_rewards = (
        beta
        *
        (
            policy_rejected_logps
            -
            reference_rejected_logps
        )
    )

    # --------------------------------------------------------
    # Chosen reward anchor
    #
    # Force:
    #
    # chosen_reward > anchor_delta
    #
    # Default anchor_delta = 0.
    # --------------------------------------------------------

    anchor_loss = (
        -F.logsigmoid(
            chosen_rewards
            -
            anchor_delta
        )
    ).mean()

    total_loss = (
        dpo_loss
        +
        anchor_weight
        *
        anchor_loss
    )

    accuracy = (
        dpo_margin > 0
    ).float().mean()

    return {

        "loss":
            total_loss,

        "dpo_loss":
            dpo_loss,

        "anchor_loss":
            anchor_loss,

        "chosen_rewards":
            chosen_rewards.detach(),

        "rejected_rewards":
            rejected_rewards.detach(),

        "margin":
            dpo_margin.detach(),

        "raw_margin":
            raw_margin.detach(),

        "accuracy":
            accuracy.detach(),
    }