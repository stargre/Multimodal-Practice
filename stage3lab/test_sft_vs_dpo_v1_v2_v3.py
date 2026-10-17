#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SFT vs DPO-v1 vs DPO-v2 vs DPO-v3 evaluation.

Metrics:

1. Sum-logp preference accuracy
2. Average-logp preference accuracy
3. DPO reward accuracy relative to SFT
4. DPO reward margin relative to SFT
5. Chosen log-probability delta
6. Rejected log-probability delta
7. Response lengths
8. Generation samples

The same exact validation samples are used for all models.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from stage1_runtime import (
    load_stage1_components,
)

from dpo_v3_core import (
    MMRefDPODatasetV3,
    MMRefDPOCollatorV3,
    concat_pair_inputs,
    get_masked_logprob_stats,
)


# ============================================================
# Logging
# ============================================================

logger = logging.getLogger(
    "sft_vs_dpo_v3"
)

logger.setLevel(
    logging.INFO
)

if not logger.handlers:

    handler = logging.StreamHandler(
        sys.stdout
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    handler.setFormatter(
        formatter
    )

    logger.addHandler(
        handler
    )


# ============================================================
# Checkpoints
# ============================================================

SFT_CHECKPOINT = (
    PROJECT_ROOT
    /
    "runs"
    /
    "sft_150k_ddp_6gpu"
    /
    "best_model.pt"
)

DPO_V1_CHECKPOINT = (
    PROJECT_ROOT
    /
    "runs"
    /
    "dpo_mmrlhf_6gpu"
    /
    "best_model.pt"
)

DPO_V2_CHECKPOINT = (
    PROJECT_ROOT
    /
    "runs"
    /
    "dpo_mmrlhf_freeze_vit_v2"
    /
    "best_model.pt"
)

DPO_V3_CHECKPOINT = (
    PROJECT_ROOT
    /
    "runs"
    /
    "dpo_mmrlhf_v3"
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


# ============================================================
# Model loader
# ============================================================

def load_model(
    model_name,
    checkpoint,
    device,
):

    logger.info(
        "=" * 60
    )

    logger.info(
        "Loading model: %s",
        model_name,
    )

    logger.info(
        "Checkpoint: %s",
        checkpoint,
    )

    model, processor = (
        load_stage1_components()
    )

    state_dict = torch.load(
        checkpoint,
        map_location="cpu",
    )

    if (
        isinstance(state_dict, dict)
        and
        "state_dict" in state_dict
        and
        isinstance(
            state_dict["state_dict"],
            dict,
        )
    ):

        state_dict = (
            state_dict["state_dict"]
        )

    # Handle accidental DDP prefix.
    keys = list(
        state_dict.keys()
    )

    if keys and all(
        key.startswith("module.")
        for key in keys
    ):

        state_dict = {
            key[len("module."):]:
                value
            for key, value in (
                state_dict.items()
            )
        }

    missing, unexpected = (
        model.load_state_dict(
            state_dict,
            strict=False,
        )
    )

    allowed_missing = [
        key
        for key in missing
        if (
            "model.visual.vit."
            in key
        )
    ]

    forbidden_missing = [
        key
        for key in missing
        if (
            "model.visual.vit."
            not in key
        )
    ]

    logger.info(
        "Checkpoint tensors: %d",
        len(state_dict),
    )

    logger.info(
        "Missing keys: %d",
        len(missing),
    )

    logger.info(
        "Allowed missing ViT keys: %d",
        len(allowed_missing),
    )

    logger.info(
        "Unexpected keys: %d",
        len(unexpected),
    )

    if forbidden_missing:

        logger.error(
            "Forbidden missing keys:"
        )

        for key in forbidden_missing[:30]:

            logger.error(
                "  %s",
                key,
            )

        raise RuntimeError(
            "Checkpoint has forbidden missing keys."
        )

    if unexpected:

        logger.error(
            "Unexpected keys:"
        )

        for key in unexpected[:30]:

            logger.error(
                "  %s",
                key,
            )

        raise RuntimeError(
            "Checkpoint has unexpected keys."
        )

    model.to(device)
    model.eval()

    logger.info(
        "Checkpoint load: PASS"
    )

    logger.info(
        "Model class: %s",
        model.__class__.__name__,
    )

    logger.info(
        "Model dtype: %s",
        next(model.parameters()).dtype,
    )

    logger.info(
        "Model device: %s",
        next(model.parameters()).device,
    )

    return (
        model,
        processor,
    )


# ============================================================
# Move batch
# ============================================================

def move_batch(
    batch,
    device,
):

    output = {}

    for key, value in batch.items():

        if torch.is_tensor(value):

            output[key] = value.to(
                device,
                non_blocking=True,
            )

        else:

            output[key] = value

    return output


# ============================================================
# Pair score
# ============================================================

@torch.no_grad()
def score_pair(
    model,
    processor,
    item,
    device,
):

    collator = (
        MMRefDPOCollatorV3(
            processor
        )
    )

    batch = collator(
        [item]
    )

    batch = move_batch(
        batch,
        device,
    )

    (
        model_inputs,
        labels,
        batch_size,
    ) = concat_pair_inputs(
        batch
    )

    outputs = model(
        **model_inputs,
        use_cache=False,
    )

    logits = outputs.logits

    chosen_logits = (
        logits[:batch_size]
    )

    rejected_logits = (
        logits[batch_size:]
    )

    chosen_labels = (
        labels[:batch_size]
    )

    rejected_labels = (
        labels[batch_size:]
    )

    chosen_sum, chosen_avg, chosen_count = (
        get_masked_logprob_stats(
            chosen_logits,
            chosen_labels,
        )
    )

    rejected_sum, rejected_avg, rejected_count = (
        get_masked_logprob_stats(
            rejected_logits,
            rejected_labels,
        )
    )

    return {

        "chosen_sum":
            float(
                chosen_sum[0]
                .detach()
                .cpu()
                .item()
            ),

        "rejected_sum":
            float(
                rejected_sum[0]
                .detach()
                .cpu()
                .item()
            ),

        "chosen_avg":
            float(
                chosen_avg[0]
                .detach()
                .cpu()
                .item()
            ),

        "rejected_avg":
            float(
                rejected_avg[0]
                .detach()
                .cpu()
                .item()
            ),

        "chosen_len":
            int(
                chosen_count[0]
                .detach()
                .cpu()
                .item()
            ),

        "rejected_len":
            int(
                rejected_count[0]
                .detach()
                .cpu()
                .item()
            ),
    }


# ============================================================
# Score complete model
# ============================================================

def score_model(
    model_name,
    checkpoint,
    dataset,
    indices,
    device,
    beta,
    baseline=None,
    save_per_sample=None,
):

    model, processor = load_model(
        model_name,
        checkpoint,
        device,
    )

    results = []

    if save_per_sample is not None:

        per_file = open(
            save_per_sample,
            "w",
            encoding="utf-8",
        )

    iterator = tqdm(
        indices,
        desc=f"{model_name} scoring",
        unit="sample",
    )

    for position, dataset_index in enumerate(
        iterator
    ):

        item = dataset[
            dataset_index
        ]

        score = score_pair(
            model,
            processor,
            item,
            device,
        )

        record = {

            "dataset_index":
                int(dataset_index),

            "id":
                item["id"],

            **score,
        }

        # ----------------------------------------------------
        # Raw preference
        # ----------------------------------------------------

        record["sum_margin"] = (
            score["chosen_sum"]
            -
            score["rejected_sum"]
        )

        record["avg_margin"] = (
            score["chosen_avg"]
            -
            score["rejected_avg"]
        )

        record["sum_preference"] = (
            1
            if
            record["sum_margin"] > 0
            else
            0
        )

        record["avg_preference"] = (
            1
            if
            record["avg_margin"] > 0
            else
            0
        )

        # ----------------------------------------------------
        # Relative DPO reward vs SFT
        # ----------------------------------------------------

        if baseline is not None:

            base = baseline[
                position
            ]

            chosen_delta = (
                score["chosen_sum"]
                -
                base["chosen_sum"]
            )

            rejected_delta = (
                score["rejected_sum"]
                -
                base["rejected_sum"]
            )

            reward_chosen = (
                beta
                *
                chosen_delta
            )

            reward_rejected = (
                beta
                *
                rejected_delta
            )

            reward_margin = (
                reward_chosen
                -
                reward_rejected
            )

            record[
                "chosen_delta_sum"
            ] = chosen_delta

            record[
                "rejected_delta_sum"
            ] = rejected_delta

            record[
                "chosen_reward"
            ] = reward_chosen

            record[
                "rejected_reward"
            ] = reward_rejected

            record[
                "reward_margin"
            ] = reward_margin

            record[
                "reward_accuracy"
            ] = (
                1
                if reward_margin > 0
                else 0
            )

            # Average-logp diagnostic.
            chosen_avg_delta = (
                score["chosen_avg"]
                -
                base["chosen_avg"]
            )

            rejected_avg_delta = (
                score["rejected_avg"]
                -
                base["rejected_avg"]
            )

            record[
                "chosen_delta_avg"
            ] = chosen_avg_delta

            record[
                "rejected_delta_avg"
            ] = rejected_avg_delta

            record[
                "avg_reward_margin"
            ] = (
                beta
                *
                (
                    chosen_avg_delta
                    -
                    rejected_avg_delta
                )
            )

        results.append(
            record
        )

        if save_per_sample is not None:

            per_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                +
                "\n"
            )

    if save_per_sample is not None:

        per_file.close()

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    n = len(results)

    def mean(
        key
    ):

        return (
            sum(
                float(x[key])
                for x in results
            )
            /
            max(
                1,
                n,
            )
        )

    summary = {

        "model":
            model_name,

        "checkpoint":
            str(checkpoint),

        "samples":
            n,

        "sum_preference_accuracy":
            mean(
                "sum_preference"
            ),

        "avg_preference_accuracy":
            mean(
                "avg_preference"
            ),

        "mean_sum_margin":
            mean(
                "sum_margin"
            ),

        "mean_avg_margin":
            mean(
                "avg_margin"
            ),

        "mean_chosen_sum_logp":
            mean(
                "chosen_sum"
            ),

        "mean_rejected_sum_logp":
            mean(
                "rejected_sum"
            ),

        "mean_chosen_avg_logp":
            mean(
                "chosen_avg"
            ),

        "mean_rejected_avg_logp":
            mean(
                "rejected_avg"
            ),

        "mean_chosen_response_tokens":
            mean(
                "chosen_len"
            ),

        "mean_rejected_response_tokens":
            mean(
                "rejected_len"
            ),
    }

    if baseline is not None:

        summary[
            "dpo_reward_accuracy"
        ] = mean(
            "reward_accuracy"
        )

        summary[
            "mean_reward_margin"
        ] = mean(
            "reward_margin"
        )

        summary[
            "mean_chosen_delta_sum"
        ] = mean(
            "chosen_delta_sum"
        )

        summary[
            "mean_rejected_delta_sum"
        ] = mean(
            "rejected_delta_sum"
        )

        summary[
            "mean_chosen_delta_avg"
        ] = mean(
            "chosen_delta_avg"
        )

        summary[
            "mean_rejected_delta_avg"
        ] = mean(
            "rejected_delta_avg"
        )

        summary[
            "mean_avg_reward_margin"
        ] = mean(
            "avg_reward_margin"
        )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    del model
    del processor

    torch.cuda.empty_cache()

    return (
        summary,
        results,
    )


# ============================================================
# Generation
# ============================================================

@torch.no_grad()
def generate_one(
    model,
    processor,
    item,
    device,
    max_new_tokens,
):

    inputs = processor.encode_images(
        [item["prompt"]],
        [[item["image"]]],
        padding=False,
    )

    model_inputs = {}

    for key, value in inputs.items():

        if torch.is_tensor(value):

            model_inputs[key] = (
                value.to(
                    device
                )
            )

    old_use_cache = getattr(
        model.config,
        "use_cache",
        None,
    )

    if hasattr(
        model.config,
        "use_cache",
    ):

        model.config.use_cache = True

    output_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    input_length = (
        model_inputs["input_ids"]
        .shape[1]
    )

    generated_ids = (
        output_ids[
            0,
            input_length:
        ]
    )

    text = (
        processor.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
    )

    if old_use_cache is not None:

        model.config.use_cache = (
            old_use_cache
        )

    return text


def generate_samples(
    model_name,
    checkpoint,
    dataset,
    generation_indices,
    device,
    output_file,
    max_new_tokens,
):

    model, processor = load_model(
        model_name,
        checkpoint,
        device,
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:

        for dataset_index in tqdm(
            generation_indices,
            desc=f"{model_name} generation",
            unit="sample",
        ):

            item = dataset[
                dataset_index
            ]

            generated = generate_one(
                model,
                processor,
                item,
                device,
                max_new_tokens,
            )

            record = {

                "dataset_index":
                    int(dataset_index),

                "id":
                    item["id"],

                "prompt":
                    item["prompt"],

                "chosen":
                    item["chosen"],

                "rejected":
                    item["rejected"],

                "generated":
                    generated,
            }

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                +
                "\n"
            )

    del model
    del processor

    torch.cuda.empty_cache()


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--generation-limit",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--generation-max-new-tokens",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--val-json",
        type=str,
        default=str(VAL_JSON),
    )

    parser.add_argument(
        "--image-root",
        type=str,
        default=str(IMAGE_ROOT),
    )

    parser.add_argument(
        "--sft-checkpoint",
        type=str,
        default=str(SFT_CHECKPOINT),
    )

    parser.add_argument(
        "--dpo-v1-checkpoint",
        type=str,
        default=str(DPO_V1_CHECKPOINT),
    )

    parser.add_argument(
        "--dpo-v2-checkpoint",
        type=str,
        default=str(DPO_V2_CHECKPOINT),
    )

    parser.add_argument(
        "--dpo-v3-checkpoint",
        type=str,
        default=str(DPO_V3_CHECKPOINT),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(
            PROJECT_ROOT
            /
            "runs"
            /
            "sft_vs_dpo_v1_v2_v3_eval"
        ),
    )

    args = parser.parse_args()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # File logging
    # --------------------------------------------------------

    file_handler = logging.FileHandler(
        output_dir
        /
        "console.log",
        encoding="utf-8",
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler.setFormatter(
        formatter
    )

    logger.addHandler(
        file_handler
    )

    logger.info(
        "=" * 60
    )

    logger.info(
        "SFT vs DPO-v1 vs DPO-v2 vs DPO-v3 evaluation"
    )

    logger.info(
        "=" * 60
    )

    logger.info(
        "Project root: %s",
        PROJECT_ROOT,
    )

    logger.info(
        "torch: %s",
        torch.__version__,
    )

    logger.info(
        "CUDA available: %s",
        torch.cuda.is_available(),
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    logger.info(
        "Device: %s",
        device,
    )

    if device.type == "cuda":

        logger.info(
            "GPU: %s",
            torch.cuda.get_device_name(
                0
            ),
        )

    logger.info(
        "dtype: torch.bfloat16"
    )

    logger.info(
        "beta: %.6f",
        args.beta,
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = (
        MMRefDPODatasetV3(
            json_path=args.val_json,
            image_root=args.image_root,
            verbose=True,
        )
    )

    all_indices = list(
        range(
            len(dataset)
        )
    )

    rng = random.Random(
        args.seed
    )

    rng.shuffle(
        all_indices
    )

    if args.limit > 0:

        indices = (
            all_indices[
                :min(
                    args.limit,
                    len(all_indices),
                )
            ]
        )

    else:

        indices = all_indices

    logger.info(
        "Evaluation sample count: %d",
        len(indices),
    )

    logger.info(
        "Sampling seed: %d",
        args.seed,
    )

    # --------------------------------------------------------
    # SFT
    # --------------------------------------------------------

    logger.info(
        "\n"
        +
        "#" * 60
    )

    logger.info(
        "START MODEL: SFT"
    )

    logger.info(
        "#" * 60
    )

    sft_summary, sft_results = score_model(
        "SFT",
        args.sft_checkpoint,
        dataset,
        indices,
        device,
        args.beta,
        baseline=None,
        save_per_sample=(
            output_dir
            /
            "SFT_per_sample.jsonl"
        ),
    )

    logger.info(
        "SFT summary:\n%s",
        json.dumps(
            sft_summary,
            indent=2,
            ensure_ascii=False,
        ),
    )

    # --------------------------------------------------------
    # DPO-v1
    # --------------------------------------------------------

    dpo_v1_summary, _ = score_model(
        "DPO-v1",
        args.dpo_v1_checkpoint,
        dataset,
        indices,
        device,
        args.beta,
        baseline=sft_results,
        save_per_sample=(
            output_dir
            /
            "DPO_v1_per_sample.jsonl"
        ),
    )

    # --------------------------------------------------------
    # DPO-v2
    # --------------------------------------------------------

    dpo_v2_summary, _ = score_model(
        "DPO-v2",
        args.dpo_v2_checkpoint,
        dataset,
        indices,
        device,
        args.beta,
        baseline=sft_results,
        save_per_sample=(
            output_dir
            /
            "DPO_v2_per_sample.jsonl"
        ),
    )

    # --------------------------------------------------------
    # DPO-v3
    # --------------------------------------------------------

    dpo_v3_summary, _ = score_model(
        "DPO-v3",
        args.dpo_v3_checkpoint,
        dataset,
        indices,
        device,
        args.beta,
        baseline=sft_results,
        save_per_sample=(
            output_dir
            /
            "DPO_v3_per_sample.jsonl"
        ),
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summaries = [

        sft_summary,
        dpo_v1_summary,
        dpo_v2_summary,
        dpo_v3_summary,
    ]

    summary_path = (
        output_dir
        /
        "summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summaries,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Human-readable table
    # --------------------------------------------------------

    logger.info(
        "\n"
        +
        "=" * 110
    )

    logger.info(
        "FINAL PREFERENCE COMPARISON"
    )

    logger.info(
        "=" * 110
    )

    header = (
        f"{'Model':<10}"
        f"{'SumAcc':>10}"
        f"{'AvgAcc':>10}"
        f"{'RewardAcc':>12}"
        f"{'RewardMargin':>15}"
        f"{'ChosenΔ':>15}"
        f"{'RejectedΔ':>15}"
    )

    logger.info(
        header
    )

    logger.info(
        "-" * 110
    )

    for summary in summaries:

        name = summary[
            "model"
        ]

        sum_acc = summary[
            "sum_preference_accuracy"
        ]

        avg_acc = summary[
            "avg_preference_accuracy"
        ]

        reward_acc = summary.get(
            "dpo_reward_accuracy",
            float("nan"),
        )

        reward_margin = summary.get(
            "mean_reward_margin",
            float("nan"),
        )

        chosen_delta = summary.get(
            "mean_chosen_delta_sum",
            float("nan"),
        )

        rejected_delta = summary.get(
            "mean_rejected_delta_sum",
            float("nan"),
        )

        logger.info(
            f"{name:<10}"
            f"{sum_acc:>10.4f}"
            f"{avg_acc:>10.4f}"
            f"{reward_acc:>12.4f}"
            f"{reward_margin:>15.6f}"
            f"{chosen_delta:>15.6f}"
            f"{rejected_delta:>15.6f}"
        )

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    gen_rng = random.Random(
        args.seed
        +
        10086
    )

    generation_count = min(
        args.generation_limit,
        len(indices),
    )

    generation_indices = (
        gen_rng.sample(
            indices,
            generation_count,
        )
    )

    checkpoints = [

        (
            "SFT",
            args.sft_checkpoint,
        ),

        (
            "DPO-v1",
            args.dpo_v1_checkpoint,
        ),

        (
            "DPO-v2",
            args.dpo_v2_checkpoint,
        ),

        (
            "DPO-v3",
            args.dpo_v3_checkpoint,
        ),
    ]

    for model_name, checkpoint in checkpoints:

        generation_path = (
            output_dir
            /
            f"{model_name.replace('-', '_')}_generation.jsonl"
        )

        logger.info(
            "Generating samples for %s: %d",
            model_name,
            generation_count,
        )

        generate_samples(
            model_name,
            checkpoint,
            dataset,
            generation_indices,
            device,
            generation_path,
            args.generation_max_new_tokens,
        )

    logger.info(
        "\nEvaluation finished."
    )

    logger.info(
        "Summary: %s",
        summary_path,
    )


if __name__ == "__main__":

    main()