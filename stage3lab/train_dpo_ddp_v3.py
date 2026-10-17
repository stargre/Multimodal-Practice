#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DPO-v3 distributed training.

Key changes over v1/v2:

1. Exact multimodal response-only masking.
2. EOS is included in chosen/rejected responses.
3. Sequence log-probability SUM is used for DPO.
4. FP32 log-probability computation.
5. Chosen reward anchor.
6. Separate merger learning rate.
7. Cosine scheduler + warmup.
8. One-epoch controlled training.
9. Validation-based best checkpoint.
10. Aggregated training diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist

from torch.nn.parallel import (
    DistributedDataParallel as DDP,
)

from torch.utils.data import (
    DataLoader,
    Sampler,
)

from torch.utils.data.distributed import (
    DistributedSampler,
)


from runtime_v3 import (
    build_dpo_v3_runtime,
)

from dpo_v3_core import (
    MMRefDPODatasetV3,
    MMRefDPOCollatorV3,
    concat_pair_inputs,
    get_selected_logprob_stats,
    dpo_v3_loss,
)


# ============================================================
# DDP
# ============================================================

def setup_ddp():

    local_rank = int(
        os.environ["LOCAL_RANK"]
    )

    rank = int(
        os.environ["RANK"]
    )

    world_size = int(
        os.environ["WORLD_SIZE"]
    )

    torch.cuda.set_device(
        local_rank
    )

    # Explicitly bind the NCCL process to its local CUDA device.
    # This removes the "Guessing device ID based on global rank" warning.
    dist.init_process_group(
        backend="nccl",
        device_id=local_rank,
    )

    return (
        local_rank,
        rank,
        world_size,
    )


def cleanup_ddp():

    if dist.is_initialized():

        dist.barrier()

        dist.destroy_process_group()


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(
    seed: int,
    rank: int,
):

    final_seed = (
        int(seed)
        +
        int(rank)
    )

    random.seed(
        final_seed
    )

    torch.manual_seed(
        final_seed
    )

    torch.cuda.manual_seed_all(
        final_seed
    )


# ============================================================
# Batch utilities
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


def _get_logit_window(labels):
    """
    Return the smallest contiguous logit-position window covering all
    supervised response tokens.

    If labels[p] is supervised, the causal LM logit at p-1 predicts it.
    Therefore the returned logits correspond exactly to
    labels[:, label_start:label_end].
    """

    if labels.ndim != 2:
        raise ValueError(
            f"labels must be [B,T], got {labels.shape}"
        )

    mask = labels != -100

    if not bool(mask.any().item()):
        raise RuntimeError(
            "No supervised response tokens found in DPO batch."
        )

    positions = torch.nonzero(
        mask,
        as_tuple=False,
    )

    min_label_pos = int(
        positions[:, 1].min().item()
    )

    max_label_pos = int(
        positions[:, 1].max().item()
    )

    logit_start = max(
        0,
        min_label_pos - 1,
    )

    logit_end = max_label_pos

    logit_positions = torch.arange(
        logit_start,
        logit_end,
        device=labels.device,
        dtype=torch.long,
    )

    label_start = logit_start + 1
    label_end = logit_end + 1

    return (
        logit_positions,
        label_start,
        label_end,
    )


def forward_single_branch(
    model,
    batch,
    prefix,
    require_grad=True,
):
    model_inputs = {
        "input_ids": batch[f"{prefix}_input_ids"],
        "attention_mask": batch[f"{prefix}_attention_mask"],
        "pixel_values": batch[f"{prefix}_pixel_values"],
        "image_grid_thw": batch[f"{prefix}_image_grid_thw"],
    }

    labels = batch[f"{prefix}_labels"]

    (
        logit_positions,
        label_start,
        label_end,
    ) = _get_logit_window(labels)

    aligned_labels = labels[
        :,
        label_start:label_end,
    ]

    if require_grad:
        outputs = model(
            **model_inputs,
            use_cache=False,
            logits_to_keep=logit_positions,
        )

        stats = get_selected_logprob_stats(
            outputs.logits,
            aligned_labels,
        )

    else:
        with torch.no_grad():
            outputs = model(
                **model_inputs,
                use_cache=False,
                logits_to_keep=logit_positions,
            )

            stats = get_selected_logprob_stats(
                outputs.logits,
                aligned_labels,
            )

    del outputs
    del model_inputs
    del aligned_labels
    del logit_positions

    return stats


# ============================================================
# Pair forward
# ============================================================

def forward_pair(
    model,
    batch,
):
    (
        model_inputs,
        labels,
        batch_size,
    ) = concat_pair_inputs(
        batch
    )

    (
        logit_positions,
        label_start,
        label_end,
    ) = _get_logit_window(labels)

    aligned_labels = labels[
        :,
        label_start:label_end,
    ]

    outputs = model(
        **model_inputs,
        use_cache=False,
        logits_to_keep=logit_positions,
    )

    # The returned logits are aligned to the contiguous selected positions,
    # while aligned_labels are shifted to the corresponding target-token
    # positions.  No additional causal shift is performed below.
    chosen_logits = (
        outputs.logits[:batch_size]
    )

    rejected_logits = (
        outputs.logits[batch_size:]
    )

    chosen_labels = (
        aligned_labels[:batch_size]
    )

    rejected_labels = (
        aligned_labels[batch_size:]
    )

    chosen_stats = (
        get_selected_logprob_stats(
            chosen_logits,
            chosen_labels,
        )
    )

    rejected_stats = (
        get_selected_logprob_stats(
            rejected_logits,
            rejected_labels,
        )
    )

    del outputs
    del model_inputs
    del labels
    del aligned_labels
    del chosen_logits
    del rejected_logits
    del chosen_labels
    del rejected_labels
    del logit_positions

    return (
        chosen_stats,
        rejected_stats,
    )

# ============================================================
# Optimizer
# ============================================================

def build_optimizer(
    model,
    args,
):

    merger_params = []
    base_params = []

    for name, param in (
        model.named_parameters()
    ):

        if not param.requires_grad:
            continue

        if (
            "model.visual.merger."
            in name
        ):

            merger_params.append(
                param
            )

        else:

            base_params.append(
                param
            )

    if not merger_params:

        raise RuntimeError(
            "No trainable visual merger parameters found."
        )

    if not base_params:

        raise RuntimeError(
            "No trainable base/language parameters found."
        )

    param_groups = [

        {
            "params":
                base_params,

            "lr":
                args.lr,

            "weight_decay":
                args.weight_decay,

            "group_name":
                "language_and_other",
        },

        {
            "params":
                merger_params,

            "lr":
                args.merger_lr,

            "weight_decay":
                args.weight_decay,

            "group_name":
                "visual_merger",
        },

    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    return optimizer


# ============================================================
# Scheduler
# ============================================================

def build_scheduler(
    optimizer,
    total_steps,
    warmup_ratio,
):

    warmup_steps = max(
        1,
        int(
            total_steps
            *
            warmup_ratio
        ),
    )

    def lr_lambda(step):

        if step < warmup_steps:

            return (
                float(step + 1)
                /
                float(warmup_steps)
            )

        progress = (
            float(
                step
                -
                warmup_steps
            )
            /
            float(
                max(
                    1,
                    total_steps
                    -
                    warmup_steps,
                )
            )
        )

        progress = min(
            max(
                progress,
                0.0,
            ),
            1.0,
        )

        return (
            0.5
            *
            (
                1.0
                +
                math.cos(
                    math.pi
                    *
                    progress
                )
            )
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda,
    )

    return scheduler


# ============================================================
# Gradient diagnostics
# ============================================================

def gradient_norm(
    model,
    keyword=None,
):

    total_sq = 0.0

    for name, param in (
        model.named_parameters()
    ):

        if keyword is not None:

            if keyword not in name:
                continue

        if param.grad is None:
            continue

        value = (
            param.grad
            .detach()
            .float()
            .norm(2)
            .item()
        )

        total_sq += (
            value
            *
            value
        )

    return math.sqrt(
        total_sq
    )


def count_nonzero_vit_grads(
    model,
):

    count = 0

    for name, param in (
        model.named_parameters()
    ):

        if "model.visual.vit." not in name:
            continue

        if param.grad is not None:

            count += 1

    return count


# ============================================================
# Parameter equality diagnostic
# ============================================================

@torch.no_grad()
def compare_policy_reference(
    policy,
    reference,
    warm_start=False,
):

    max_diff = 0.0

    for p1, p2 in zip(
        policy.parameters(),
        reference.parameters(),
    ):

        diff = (
            p1.detach()
            .float()
            -
            p2.detach()
            .float()
        ).abs().max().item()

        max_diff = max(
            max_diff,
            diff,
        )

    
    if warm_start:
        print(
            "Warm-start policy detected; "
            "policy/reference are expected to differ.",
            flush=True,
        )
    else:
        if max_diff != 0:
            raise RuntimeError(
                "Policy and reference must be identical "
                "when starting from SFT."
            )

    return max_diff


# ============================================================
# Save
# ============================================================
# ============================================================
# Save / Resume
# ============================================================

def unwrap_model(model):
    if isinstance(model, DDP):
        return model.module
    return model


def _to_cpu_tree(obj):
    """
    Recursively move checkpoint tensors to CPU.

    This prevents checkpoint serialization from holding large optimizer/model
    state on the GPU while torch.save is running.
    """
    if torch.is_tensor(obj):
        return obj.detach().cpu()

    if isinstance(obj, dict):
        return {
            key: _to_cpu_tree(value)
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [
            _to_cpu_tree(value)
            for value in obj
        ]

    if isinstance(obj, tuple):
        return tuple(
            _to_cpu_tree(value)
            for value in obj
        )

    return obj


def capture_rng_state():
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(),
    }


def restore_rng_state(
    rng_states,
    rank,
):
    if not rng_states:
        print(
            "Resume checkpoint has no per-rank RNG state; "
            "continuing from deterministic seed only.",
            flush=True,
        )
        return

    if rank >= len(rng_states):
        raise RuntimeError(
            "Resume checkpoint does not contain RNG state "
            f"for rank={rank}."
        )

    state = rng_states[rank]

    random.setstate(
        state["python"]
    )

    torch.set_rng_state(
        state["torch_cpu"]
    )

    torch.cuda.set_rng_state(
        state["torch_cuda"]
    )


def save_checkpoint(
    model,
    path,
):
    """
    Model-only checkpoint.
    Used for best_model.pt / last_model.pt.
    """
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    state_dict = _to_cpu_tree(
        unwrap_model(model).state_dict()
    )

    tmp_path = Path(
        str(path) + ".tmp"
    )

    torch.save(
        state_dict,
        tmp_path,
    )

    os.replace(
        tmp_path,
        path,
    )


def save_training_checkpoint(
    model,
    optimizer,
    scheduler,
    path,
    epoch,
    next_batch_index,
    optimizer_step,
    micro_step,
    best_eval_margin,
    total_steps,
    micro_batches_per_epoch,
    world_size,
    args,
    rng_states=None,
):
    """
    Full DPO-v3 training-state checkpoint.

    The checkpoint is CPU-offloaded before serialization and contains
    per-rank RNG states so that a resumed DDP run can continue from the
    same stochastic state.
    """

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "format_version": 2,

        "model_state_dict":
            _to_cpu_tree(
                unwrap_model(model).state_dict()
            ),

        "optimizer_state_dict":
            _to_cpu_tree(
                optimizer.state_dict()
            ),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "rng_states":
            rng_states,

        "training_state": {
            "epoch":
                int(epoch),

            "next_batch_index":
                int(next_batch_index),

            "optimizer_step":
                int(optimizer_step),

            "micro_step":
                int(micro_step),

            "best_eval_margin":
                float(best_eval_margin),

            "total_steps":
                int(total_steps),

            "micro_batches_per_epoch":
                int(micro_batches_per_epoch),
        },

        "meta": {
            "world_size":
                int(world_size),

            "batch_size":
                int(args.batch_size),

            "grad_accum":
                int(args.grad_accum),

            "lr":
                float(args.lr),

            "merger_lr":
                float(args.merger_lr),

            "beta":
                float(args.beta),

            "anchor_weight":
                float(args.anchor_weight),

            "anchor_delta":
                float(args.anchor_delta),

            "weight_decay":
                float(args.weight_decay),

            "warmup_ratio":
                float(args.warmup_ratio),

            "max_grad_norm":
                float(args.max_grad_norm),

            "seed":
                int(args.seed),

            "data":
                str(args.data),

            "val_data":
                str(args.val_data),

            "image_root":
                str(args.image_root),

            "max_tokens":
                int(args.max_tokens),
        },
    }

    tmp_path = Path(
        str(path) + ".tmp"
    )

    torch.save(
        checkpoint,
        tmp_path,
    )

    os.replace(
        tmp_path,
        path,
    )


def load_training_checkpoint(
    model,
    optimizer,
    scheduler,
    path,
    world_size,
    args,
    expected_total_steps,
    rank,
):
    """
    Restore complete training state.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Resume checkpoint not found: {path}"
        )

    print(
        f"Loading DPO-v3 resume checkpoint: {path}",
        flush=True,
    )

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    version = int(
        checkpoint.get(
            "format_version",
            1,
        )
    )

    if version not in (1, 2):
        raise RuntimeError(
            "Unsupported DPO-v3 checkpoint format: "
            f"{version}"
        )

    meta = checkpoint["meta"]
    state = checkpoint["training_state"]

    if int(meta["world_size"]) != int(world_size):
        raise RuntimeError(
            "Resume world_size mismatch: "
            f"checkpoint={meta['world_size']} "
            f"current={world_size}"
        )

    if int(meta["batch_size"]) != int(args.batch_size):
        raise RuntimeError(
            "Resume batch_size mismatch: "
            f"checkpoint={meta['batch_size']} "
            f"current={args.batch_size}"
        )

    if int(meta["grad_accum"]) != int(args.grad_accum):
        raise RuntimeError(
            "Resume grad_accum mismatch: "
            f"checkpoint={meta['grad_accum']} "
            f"current={args.grad_accum}"
        )

    if int(state["total_steps"]) != int(expected_total_steps):
        raise RuntimeError(
            "Resume total_steps mismatch: "
            f"checkpoint={state['total_steps']} "
            f"current={expected_total_steps}"
        )

    if int(meta.get("seed", args.seed)) != int(args.seed):
        raise RuntimeError(
            "Resume seed mismatch: "
            f"checkpoint={meta.get('seed')} "
            f"current={args.seed}"
        )

    if int(meta.get("max_tokens", args.max_tokens)) != int(args.max_tokens):
        raise RuntimeError(
            "Resume max_tokens mismatch: "
            f"checkpoint={meta.get('max_tokens')} "
            f"current={args.max_tokens}"
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    unwrap_model(model).load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    if version >= 2:
        restore_rng_state(
            checkpoint.get("rng_states"),
            rank,
        )

    print(
        "Resume checkpoint loaded.",
        flush=True,
    )

    print(
        "Resume epoch:",
        state["epoch"],
        flush=True,
    )

    print(
        "Resume next batch:",
        state["next_batch_index"],
        flush=True,
    )

    print(
        "Resume optimizer step:",
        state["optimizer_step"],
        flush=True,
    )

    print(
        "Resume best eval margin:",
        state["best_eval_margin"],
        flush=True,
    )

    return state

# ============================================================
# Distributed no-padding sampler
# ============================================================

class DistributedNoPadSampler(
    Sampler
):

    def __init__(
        self,
        dataset_size,
        rank,
        world_size,
        limit=0,
    ):

        self.rank = rank
        self.world_size = world_size

        total = int(
            dataset_size
        )

        if limit > 0:

            total = min(
                total,
                int(limit),
            )

        indices = list(
            range(total)
        )

        self.indices = (
            indices[rank::world_size]
        )

    def __iter__(self):

        return iter(
            self.indices
        )

    def __len__(self):

        return len(
            self.indices
        )


# ============================================================
# Evaluation
# ============================================================

def load_policy_checkpoint(
    model,
    checkpoint_path,
):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Policy checkpoint not found: {checkpoint_path}"
        )

    print(
        f"\n========== LOADING POLICY CHECKPOINT ==========",
        flush=True,
    )

    print(
        f"Checkpoint: {checkpoint_path}",
        flush=True,
    )

    state = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    if isinstance(state, dict) and "model_state_dict" in state:
        state_dict = state["model_state_dict"]

    elif isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]

    else:
        state_dict = state

    if not isinstance(state_dict, dict):
        raise TypeError(
            "Checkpoint does not contain a valid state_dict."
        )

    keys = list(state_dict.keys())

    if keys and all(
        key.startswith("module.")
        for key in keys
    ):
        state_dict = {
            key[len("module."):]: value
            for key, value in state_dict.items()
        }

    missing_keys, unexpected_keys = (
        model.load_state_dict(
            state_dict,
            strict=False,
        )
    )

    allowed_missing = [
        key
        for key in missing_keys
        if key.startswith(
            "model.visual.vit."
        )
    ]

    forbidden_missing = [
        key
        for key in missing_keys
        if key not in allowed_missing
    ]

    print(
        f"Checkpoint tensors: {len(state_dict)}",
        flush=True,
    )

    print(
        f"Missing keys: {len(missing_keys)}",
        flush=True,
    )

    print(
        f"Allowed missing keys: {len(allowed_missing)}",
        flush=True,
    )

    print(
        f"Forbidden missing keys: {len(forbidden_missing)}",
        flush=True,
    )

    print(
        f"Unexpected keys: {len(unexpected_keys)}",
        flush=True,
    )

    if forbidden_missing:
        raise RuntimeError(
            "Policy checkpoint has forbidden missing keys:\n"
            + "\n".join(
                f"  {key}"
                for key in forbidden_missing
            )
        )

    if unexpected_keys:
        raise RuntimeError(
            "Policy checkpoint has unexpected keys:\n"
            + "\n".join(
                f"  {key}"
                for key in unexpected_keys
            )
        )

    print(
        "Policy checkpoint load: PASS",
        flush=True,
    )

    print("=" * 60, flush=True)


@torch.no_grad()
def evaluate(
    policy,
    reference,
    loader,
    device,
    beta,
    anchor_weight,
    anchor_delta,
):

    policy.eval()
    reference.eval()

    # Sum:
    #
    # 0 dpo_loss
    # 1 total_loss
    # 2 margin
    # 3 accuracy
    # 4 avg_margin
    # 5 avg_accuracy
    # 6 chosen_reward
    # 7 rejected_reward
    # 8 count

    stats = torch.zeros(
        9,
        dtype=torch.float64,
        device=device,
    )

    for batch in loader:

        batch = move_batch(
            batch,
            device,
        )

        

        (
            chosen_stats,
            rejected_stats,
        ) = forward_pair(
            policy,
            batch,
        )

        (
            policy_chosen_sum,
            policy_chosen_avg,
            _,
        ) = chosen_stats

        (
            policy_rejected_sum,
            policy_rejected_avg,
            _,
        ) = rejected_stats

        (
            ref_chosen_stats,
            ref_rejected_stats,
        ) = forward_pair(
            reference,
            batch,
        )

        (
            reference_chosen_sum,
            _,
            _,
        ) = ref_chosen_stats

        (
            reference_rejected_sum,
            _,
            _,
        ) = ref_rejected_stats

        result = dpo_v3_loss(
            policy_chosen_sum,
            policy_rejected_sum,
            reference_chosen_sum,
            reference_rejected_sum,
            beta=beta,
            anchor_weight=anchor_weight,
            anchor_delta=anchor_delta,
        )

        raw_avg_margin = (
            policy_chosen_avg
            -
            policy_rejected_avg
        )

        avg_accuracy = (
            raw_avg_margin > 0
        ).float().mean()

        count = (
            policy_chosen_sum.shape[0]
        )

        stats[0] += (
            result["dpo_loss"]
            .detach()
            .float()
            .mean()
            .double()
            *
            count
        )

        stats[1] += (
            result["loss"]
            .detach()
            .float()
            .mean()
            .double()
            *
            count
        )

        stats[2] += (
            result["margin"]
            .detach()
            .float()
            .mean()
            .double()
            *
            count
        )

        stats[3] += (
            result["accuracy"]
            .detach()
            .float()
            .mean()
            .double()
            *
            count
        )

        stats[4] += (
            raw_avg_margin
            .detach()
            .float()
            .mean()
            .double()
            *
            count
        )

        stats[5] += (
            avg_accuracy
            .detach()
            .float()
            .mean()
            .double()
            *
            count
        )

        stats[6] += (
            result["chosen_rewards"]
            .detach()
            .float()
            .mean()
            .double()
            *
            count
        )

        stats[7] += (
            result["rejected_rewards"]
            .detach()
            .float()
            .mean()
            .double()
            *
            count
        )

        stats[8] += count

    dist.all_reduce(
        stats,
        op=dist.ReduceOp.SUM,
    )

    count = max(
        1.0,
        stats[8].item(),
    )

    output = {

        "eval_dpo_loss":
            stats[0].item()
            /
            count,

        "eval_total_loss":
            stats[1].item()
            /
            count,

        "eval_margin":
            stats[2].item()
            /
            count,

        "eval_accuracy":
            stats[3].item()
            /
            count,

        "eval_avg_margin":
            stats[4].item()
            /
            count,

        "eval_avg_accuracy":
            stats[5].item()
            /
            count,

        "eval_chosen_reward":
            stats[6].item()
            /
            count,

        "eval_rejected_reward":
            stats[7].item()
            /
            count,

        "eval_count":
            int(stats[8].item()),
    }

    policy.train()

    try:

        policy.module.model.visual.vit.eval()

    except Exception:

        pass

    return output


# ============================================================
# Main training
# ============================================================


def enable_gradient_checkpointing(model):
    """
    Enable activation checkpointing before DDP construction.

    The previous v3 code enabled this only after the first validation run,
    which left the first ~500 optimizer steps at the highest memory usage.
    """

    if not hasattr(
        model,
        "gradient_checkpointing_enable",
    ):
        raise RuntimeError(
            "Policy does not support gradient checkpointing."
        )

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        }
    )

    try:
        model.config.use_cache = False
    except Exception:
        pass

    enabled = getattr(
        model,
        "is_gradient_checkpointing",
        False,
    )

    print(
        "Gradient checkpointing:",
        enabled,
        flush=True,
    )

    if not enabled:
        raise RuntimeError(
            "Gradient checkpointing was requested but "
            "the model reports it as disabled."
        )


# ============================================================
# Main training
# ============================================================

def train(args):

    (
        local_rank,
        rank,
        world_size,
    ) = setup_ddp()

    seed_everything(
        args.seed,
        rank,
    )

    device = torch.device(
        f"cuda:{local_rank}"
    )

    torch.backends.cuda.matmul.allow_tf32 = True

    if rank == 0:

        os.makedirs(
            args.out,
            exist_ok=True,
        )

    dist.barrier()

    # --------------------------------------------------------
    # Runtime
    # --------------------------------------------------------

    policy, reference, processor = build_dpo_v3_runtime()

    # --------------------------------------------------------
    # Use memory-efficient SDPA attention for Qwen3 language model
    # --------------------------------------------------------

    def enable_sdpa(model):

        language_model = (
            model.model.language_model
        )

        if hasattr(
            language_model,
            "set_attn_implementation",
        ):

            language_model.set_attn_implementation(
                "sdpa"
            )

        else:

            language_model.config._attn_implementation = (
                "sdpa"
            )

        print(
            "Qwen3 attention backend:",
            getattr(
                language_model.config,
                "_attn_implementation",
                "unknown",
            ),
            flush=True,
        )


    enable_sdpa(policy)
    enable_sdpa(reference)


    if args.init_policy and args.resume:
        raise RuntimeError(
            "--init-policy and --resume cannot be used together."
        )

    if args.init_policy:
        load_policy_checkpoint(
            policy,
            args.init_policy,
        )
        

    policy.to(device)
    reference.to(device)

    reference.eval()

    for p in reference.parameters():

        p.requires_grad = False

    # --------------------------------------------------------
    # Enable policy activation checkpointing BEFORE DDP.
    # --------------------------------------------------------

    enable_gradient_checkpointing(
        policy
    )

    # --------------------------------------------------------
    # Exact initial policy/reference equality
    # --------------------------------------------------------

    if not args.resume:

        initial_diff = (
            compare_policy_reference(
                policy,
                reference,
                warm_start=bool(args.init_policy),
            )
        )

        if rank == 0:

            print(
                "\n========== INITIAL POLICY / REFERENCE CHECK ==========",
                flush=True,
            )

            print(
                "Max parameter absolute difference:",
                initial_diff,
                flush=True,
            )

            if args.init_policy:

                print(
                    "Model-only warm-start detected; "
                    "policy/reference difference is expected.",
                    flush=True,
                )

            else:

                if initial_diff > 1e-6:

                    raise RuntimeError(
                        "Policy and reference are not identical "
                        "at fresh DPO initialization."
                    )

    else:

        if rank == 0:

            print(
                "\n========== RESUME MODE ==========",
                flush=True,
            )

            print(
                "Skipping fresh policy/reference equality check.",
                flush=True,
            )
            

    # --------------------------------------------------------
    # DDP
    # --------------------------------------------------------

    policy = DDP(
        policy,
        device_ids=[
            local_rank
        ],
        find_unused_parameters=False,
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = (
        MMRefDPODatasetV3(
            json_path=args.data,
            image_root=args.image_root,
            verbose=(rank == 0),
        )
    )

    if len(train_dataset) == 0:

        raise RuntimeError(
            "Training dataset is empty."
        )

    sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False,
        seed=args.seed,
    )

    collator = (
        MMRefDPOCollatorV3(
            processor,
            max_tokens=args.max_tokens,
        )
    )

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(
            args.num_workers > 0
        ),
        collate_fn=collator,
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_loader = None

    if args.eval_every > 0:

        val_dataset = (
            MMRefDPODatasetV3(
                json_path=args.val_data,
                image_root=args.image_root,
                verbose=(rank == 0),
            )
        )

        val_sampler = (
            DistributedNoPadSampler(
                len(val_dataset),
                rank,
                world_size,
                limit=args.eval_limit,
            )
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(
                args.num_workers > 0
            ),
            collate_fn=collator,
        )

    # --------------------------------------------------------
    # Steps
    # --------------------------------------------------------

    micro_batches_per_epoch = len(loader)

    estimated_steps_per_epoch = (
        math.ceil(
            micro_batches_per_epoch
            /
            args.grad_accum
        )
    )

    if args.max_steps > 0:

        total_steps = (
            args.max_steps
        )

    else:

        total_steps = (
            estimated_steps_per_epoch
            *
            args.epochs
        )

    if rank == 0:

        print(
            "\n========== TRAINING PLAN ==========",
            flush=True,
        )

        print(
            "World size:",
            world_size,
            flush=True,
        )

        print(
            "Micro batches / epoch:",
            micro_batches_per_epoch,
            flush=True,
        )

        print(
            "Estimated optimizer steps / epoch:",
            estimated_steps_per_epoch,
            flush=True,
        )

        print(
            "Total optimizer steps:",
            total_steps,
            flush=True,
        )

    # --------------------------------------------------------
    # Optimizer / scheduler
    # --------------------------------------------------------

    optimizer = build_optimizer(
        policy,
        args,
    )

    scheduler = build_scheduler(
        optimizer,
        total_steps,
        args.warmup_ratio,
    )

    # --------------------------------------------------------
    # Training state
    # --------------------------------------------------------

    optimizer_step = 0
    micro_step = 0

    start_epoch = 0
    resume_batch_index = 0

    best_eval_margin = (
        -float("inf")
    )

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    if args.resume:

        resume_state = load_training_checkpoint(
            policy,
            optimizer,
            scheduler,
            args.resume,
            world_size,
            args,
            expected_total_steps=total_steps,
            rank=rank,
        )

        optimizer_step = int(
            resume_state["optimizer_step"]
        )

        micro_step = int(
            resume_state["micro_step"]
        )

        start_epoch = int(
            resume_state["epoch"]
        )

        resume_batch_index = int(
            resume_state["next_batch_index"]
        )

        best_eval_margin = float(
            resume_state["best_eval_margin"]
        )

    else:

        resume_state = None

    optimizer.zero_grad(
        set_to_none=True
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metric_file = None

    if rank == 0:

        metric_file = open(
            Path(args.out)
            /
            "dpo_v3_metrics.jsonl",
            "a",
            encoding="utf-8",
        )

        config_path = (
            Path(args.out)
            /
            "config.json"
        )

        with open(
            config_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                vars(args),
                f,
                indent=2,
                ensure_ascii=False,
            )

    # --------------------------------------------------------
    # Training state
    # --------------------------------------------------------


    # Window metrics.
    window = {
        "dpo_loss": 0.0,
        "total_loss": 0.0,
        "anchor_loss": 0.0,
        "chosen_reward": 0.0,
        "rejected_reward": 0.0,
        "margin": 0.0,
        "accuracy": 0.0,
        "avg_margin": 0.0,
        "avg_accuracy": 0.0,
        "chosen_len": 0.0,
        "rejected_len": 0.0,
        "count": 0.0,
    }

    # --------------------------------------------------------
    # Epoch loop
    # --------------------------------------------------------

    stop_training = False

    for epoch in range(
        start_epoch,
        args.epochs,
    ):

        sampler.set_epoch(
            epoch
        )

        micro_in_accum = 0

        current_resume_batch = (
            resume_batch_index
            if epoch == start_epoch
            else 0
        )

        for batch_index, batch in enumerate(
            loader
        ):

            # ------------------------------------------------
            # Skip batches already completed before checkpoint
            # ------------------------------------------------

            if (
                epoch == start_epoch
                and
                batch_index < current_resume_batch
            ):

                continue


            if (
                optimizer_step
                >=
                total_steps
            ):

                stop_training = True
                break

            batch = move_batch(
                batch,
                device,
            )
            chosen_seq_len = (
                batch["chosen_input_ids"].shape[1]
            )

            rejected_seq_len = (
                batch["rejected_input_ids"].shape[1]
            )

            pair_seq_len = max(
                chosen_seq_len,
                rejected_seq_len,
            )

            if (
                rank == 0
                and
                pair_seq_len >= 900
            ):

                print(
                    "[LONG SEQ]",
                    f"epoch={epoch}",
                    f"step={optimizer_step}",
                    f"batch={batch_index}",
                    f"chosen_len={chosen_seq_len}",
                    f"rejected_len={rejected_seq_len}",
                    f"gpu_mem={torch.cuda.memory_allocated(device) / 1024**3:.2f}GB",
                    flush=True,
                )

            # ------------------------------------------------
            # Policy
            # ------------------------------------------------

            (
                policy_chosen_stats,
                policy_rejected_stats,
            ) = forward_pair(
                policy,
                batch,
            )

            (
                policy_chosen_sum,
                policy_chosen_avg,
                chosen_counts,
            ) = policy_chosen_stats

            (
                policy_rejected_sum,
                policy_rejected_avg,
                rejected_counts,
            ) = policy_rejected_stats

            # ------------------------------------------------
            # Reference
            # ------------------------------------------------

            with torch.no_grad():

                (
                    ref_chosen_stats,
                    ref_rejected_stats,
                ) = forward_pair(
                    reference,
                    batch,
                )

            (
                reference_chosen_sum,
                _,
                _,
            ) = ref_chosen_stats

            (
                reference_rejected_sum,
                _,
                _,
            ) = ref_rejected_stats

            # ------------------------------------------------
            # DPO-v3
            # ------------------------------------------------

            result = dpo_v3_loss(

                policy_chosen_sum,

                policy_rejected_sum,

                reference_chosen_sum,

                reference_rejected_sum,

                beta=args.beta,

                anchor_weight=args.anchor_weight,

                anchor_delta=args.anchor_delta,
            )

            loss = result["loss"]

            if not torch.isfinite(loss):

                raise RuntimeError(
                    "Non-finite DPO-v3 loss detected."
                )

            # ------------------------------------------------
            # Accumulate diagnostics
            # ------------------------------------------------

            avg_margin = (
                policy_chosen_avg
                -
                policy_rejected_avg
            )

            avg_accuracy = (
                avg_margin > 0
            ).float()

            bsz = (
                policy_chosen_sum.shape[0]
            )

            window["dpo_loss"] += float(
                result["dpo_loss"]
                .detach()
                .float()
                .mean()
                .item()
            ) * bsz

            window["total_loss"] += float(
                result["loss"]
                .detach()
                .float()
                .mean()
                .item()
            ) * bsz

            window["anchor_loss"] += float(
                result["anchor_loss"]
                .detach()
                .float()
                .mean()
                .item()
            ) * bsz

            window["chosen_reward"] += float(
                result["chosen_rewards"]
                .detach()
                .float()
                .mean()
                .item()
            ) * bsz

            window["rejected_reward"] += float(
                result["rejected_rewards"]
                .detach()
                .float()
                .mean()
                .item()
            ) * bsz

            window["margin"] += float(
                result["margin"]
                .detach()
                .float()
                .mean()
                .item()
            ) * bsz

            window["accuracy"] += float(
                result["accuracy"]
                .detach()
                .float()
                .mean()
                .item()
            ) * bsz

            window["avg_margin"] += float(
                avg_margin
                .detach()
                .float()
                .mean()
                .item()
            ) * bsz

            window["avg_accuracy"] += float(
                avg_accuracy
                .detach()
                .float()
                .mean()
                .item()
            ) * bsz

            window["chosen_len"] += float(
                chosen_counts
                .float()
                .mean()
                .item()
            ) * bsz

            window["rejected_len"] += float(
                rejected_counts
                .float()
                .mean()
                .item()
            ) * bsz

            window["count"] += bsz

            # ------------------------------------------------
            # Backward
            # ------------------------------------------------

            loss_for_backward = (
                loss
                /
                args.grad_accum
            )

            loss_for_backward.backward()

            micro_step += 1
            micro_in_accum += 1

            is_last_batch = (
                batch_index
                ==
                len(loader) - 1
            )

            should_step = (
                micro_in_accum
                >=
                args.grad_accum
            )

            if (
                is_last_batch
                and
                micro_in_accum > 0
            ):

                should_step = True

            if not should_step:

                continue

            # ------------------------------------------------
            # If final accumulation is incomplete,
            # rescale gradients to obtain mean gradient
            # over actual micro-batches.
            # ------------------------------------------------

            if (
                micro_in_accum
                <
                args.grad_accum
            ):

                scale = (
                    args.grad_accum
                    /
                    micro_in_accum
                )

                for param in (
                    policy.parameters()
                ):

                    if param.grad is not None:

                        param.grad.mul_(
                            scale
                        )

            # ------------------------------------------------
            # Gradient diagnostics before clipping
            # ------------------------------------------------

            total_grad_norm = (
                gradient_norm(
                    policy
                )
            )

            merger_grad_norm = (
                gradient_norm(
                    policy,
                    "model.visual.merger.",
                )
            )

            language_grad_norm = (
                gradient_norm(
                    policy,
                    "model.language_model.",
                )
            )

            vit_grad_tensors = (
                count_nonzero_vit_grads(
                    policy
                )
            )

            if not math.isfinite(
                total_grad_norm
            ):

                raise RuntimeError(
                    "Non-finite gradient norm."
                )

            # ------------------------------------------------
            # Gradient clipping
            # ------------------------------------------------

            clipped_norm = (
                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(),
                    args.max_grad_norm,
                )
            )

            clipped_norm = float(
                clipped_norm
            )

            optimizer.step()

            scheduler.step()

            optimizer.zero_grad(
                set_to_none=True
            )

            optimizer_step += 1
            micro_in_accum = 0

            # ------------------------------------------------
            # Current LR
            # ------------------------------------------------

            current_lr = (
                optimizer.param_groups[0]["lr"]
            )

            current_merger_lr = (
                optimizer.param_groups[1]["lr"]
            )

            # ------------------------------------------------
            # Logging
            # ------------------------------------------------

            if (
                optimizer_step
                %
                args.log_every
                ==
                0
            ):

                values = torch.tensor(
                    [

                        window["dpo_loss"],
                        window["total_loss"],
                        window["anchor_loss"],

                        window["chosen_reward"],
                        window["rejected_reward"],
                        window["margin"],
                        window["accuracy"],

                        window["avg_margin"],
                        window["avg_accuracy"],

                        window["chosen_len"],
                        window["rejected_len"],

                        window["count"],

                    ],
                    dtype=torch.float64,
                    device=device,
                )

                dist.all_reduce(
                    values,
                    op=dist.ReduceOp.SUM,
                )

                global_count = max(
                    1.0,
                    values[11].item(),
                )

                if rank == 0:

                    record = {

                        "type":
                            "train",

                        "epoch":
                            epoch,

                        "step":
                            optimizer_step,

                        "dpo_loss":
                            values[0].item()
                            /
                            global_count,

                        "total_loss":
                            values[1].item()
                            /
                            global_count,

                        "anchor_loss":
                            values[2].item()
                            /
                            global_count,

                        "chosen_reward":
                            values[3].item()
                            /
                            global_count,

                        "rejected_reward":
                            values[4].item()
                            /
                            global_count,

                        "margin":
                            values[5].item()
                            /
                            global_count,

                        "accuracy":
                            values[6].item()
                            /
                            global_count,

                        "avg_margin":
                            values[7].item()
                            /
                            global_count,

                        "avg_accuracy":
                            values[8].item()
                            /
                            global_count,

                        "chosen_response_tokens":
                            values[9].item()
                            /
                            global_count,

                        "rejected_response_tokens":
                            values[10].item()
                            /
                            global_count,

                        "lr":
                            current_lr,

                        "merger_lr":
                            current_merger_lr,

                        "grad_norm":
                            total_grad_norm,

                        "merger_grad_norm":
                            merger_grad_norm,

                        "language_grad_norm":
                            language_grad_norm,

                        "vit_nonzero_grad_tensors":
                            vit_grad_tensors,

                        "clipped_grad_norm":
                            clipped_norm,
                    }

                    print(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

                    metric_file.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                        )
                        +
                        "\n"
                    )

                    metric_file.flush()

                window = {
                    "dpo_loss": 0.0,
                    "total_loss": 0.0,
                    "anchor_loss": 0.0,
                    "chosen_reward": 0.0,
                    "rejected_reward": 0.0,
                    "margin": 0.0,
                    "accuracy": 0.0,
                    "avg_margin": 0.0,
                    "avg_accuracy": 0.0,
                    "chosen_len": 0.0,
                    "rejected_len": 0.0,
                    "count": 0.0,
                }

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            if (
                args.eval_every > 0
                and
                optimizer_step > 0
                and
                optimizer_step
                %
                args.eval_every
                ==
                0
                and
                val_loader is not None
            ):

                dist.barrier()

                eval_metrics = evaluate(
                    policy,
                    reference,
                    val_loader,
                    device,
                    args.beta,
                    args.anchor_weight,
                    args.anchor_delta,
                )

                dist.barrier()

                if rank == 0:

                    record = {

                        "type":
                            "eval",

                        "epoch":
                            epoch,

                        "step":
                            optimizer_step,

                        **eval_metrics,
                    }

                    print(
                        "\n"
                        +
                        json.dumps(
                            record,
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

                    metric_file.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                        )
                        +
                        "\n"
                    )

                    metric_file.flush()

                    # ----------------------------------------
                    # Best checkpoint:
                    # higher reward margin = better
                    # ----------------------------------------

                    if (
                        eval_metrics["eval_margin"]
                        >
                        best_eval_margin
                    ):

                        best_eval_margin = (
                            eval_metrics["eval_margin"]
                        )

                        best_path = (
                            Path(args.out)
                            /
                            "best_model.pt"
                        )

                        save_checkpoint(
                            policy,
                            best_path,
                        )

                        print(
                            "Saved BEST:",
                            best_path,
                            flush=True,
                        )

                dist.barrier()

                # Return policy to training mode.
                policy.train()

                try:

                    policy.module.model.visual.vit.eval()

                except Exception:

                    pass
    
            # ------------------------------------------------
            # Full training-state checkpoint
            # ------------------------------------------------

            if (
                args.save_every > 0
                and
                optimizer_step > 0
                and
                optimizer_step
                %
                args.save_every
                ==
                0
            ):

                next_epoch = epoch
                next_batch_index = (
                    batch_index + 1
                )

                if (
                    next_batch_index
                    >=
                    len(loader)
                ):

                    next_epoch = (
                        epoch + 1
                    )

                    next_batch_index = 0

                # Every rank contributes its own RNG state.  Rank 0
                # writes the single shared checkpoint file.
                local_rng_state = capture_rng_state()
                rng_states = [None for _ in range(world_size)]

                dist.all_gather_object(
                    rng_states,
                    local_rng_state,
                )

                dist.barrier()

                if rank == 0:

                    resume_path = (
                        Path(args.out)
                        /
                        "resume_latest.pt"
                    )

                    save_training_checkpoint(
                        policy,
                        optimizer,
                        scheduler,
                        resume_path,
                        epoch=next_epoch,
                        next_batch_index=next_batch_index,
                        optimizer_step=optimizer_step,
                        micro_step=micro_step,
                        best_eval_margin=best_eval_margin,
                        total_steps=total_steps,
                        micro_batches_per_epoch=(
                            micro_batches_per_epoch
                        ),
                        world_size=world_size,
                        args=args,
                        rng_states=rng_states,
                    )

                    print(
                        "Saved RESUME:",
                        resume_path,
                        flush=True,
                    )

                dist.barrier()
                

        if stop_training:

            break

    # --------------------------------------------------------
    # Final model
    # --------------------------------------------------------

    dist.barrier()

    # Keep a final full training-state checkpoint as well.  This is not
    # needed for ordinary completion, but makes the run restartable even
    # when total_steps is not divisible by save_every.
    final_rng_state = capture_rng_state()
    final_rng_states = [None for _ in range(world_size)]

    dist.all_gather_object(
        final_rng_states,
        final_rng_state,
    )

    dist.barrier()

    if rank == 0:
        final_resume_path = (
            Path(args.out)
            /
            "resume_latest.pt"
        )

        save_training_checkpoint(
            policy,
            optimizer,
            scheduler,
            final_resume_path,
            epoch=args.epochs,
            next_batch_index=0,
            optimizer_step=optimizer_step,
            micro_step=micro_step,
            best_eval_margin=best_eval_margin,
            total_steps=total_steps,
            micro_batches_per_epoch=micro_batches_per_epoch,
            world_size=world_size,
            args=args,
            rng_states=final_rng_states,
        )

        print(
            "Saved FINAL RESUME:",
            final_resume_path,
            flush=True,
        )

    dist.barrier()

    if rank == 0:

        last_path = (
            Path(args.out)
            /
            "last_model.pt"
        )

        save_checkpoint(
            policy,
            last_path,
        )

        print(
            "\nSaved LAST:",
            last_path,
            flush=True,
        )

        # If no evaluation happened,
        # still create best_model.pt.
        best_path = (
            Path(args.out)
            /
            "best_model.pt"
        )

        if not best_path.exists():

            save_checkpoint(
                policy,
                best_path,
            )

            print(
                "No validation checkpoint existed; "
                "copied final model to BEST.",
                flush=True,
            )

    dist.barrier()

    if metric_file is not None:

        metric_file.close()

    cleanup_ddp()


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=str,
        default=(
            "/mnt/data2/zhiyu/first-task/"
            "qwen2.5/multimodal_training/"
            "data_splits/mmrlhf_dpo/train.json"
        ),
    )

    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help=(
            "Full DPO-v3 training-state checkpoint used to resume "
            "model/optimizer/scheduler/training position."
        ),
    )

    parser.add_argument(
        "--init-policy",
        type=str,
        default="",
        help=(
            "Optional model-only checkpoint used to initialize "
            "the DPO policy. The DPO reference remains the SFT model."
        ),
    )

    parser.add_argument(
        "--val_data",
        type=str,
        default=(
            "/mnt/data2/zhiyu/first-task/"
            "qwen2.5/multimodal_training/"
            "data_splits/mmrlhf_dpo/val.json"
        ),
    )

    parser.add_argument(
        "--image_root",
        type=str,
        default=(
            "/mnt/data2/zhiyu/first-task/"
            "datasets/MMRLHF_ROOT"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--grad_accum",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=5e-7,
    )

    parser.add_argument(
        "--merger_lr",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--anchor_weight",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--anchor_delta",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=2232,
    )

    parser.add_argument(
        "--log_every",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--eval_every",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--eval_limit",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--save_every",
        type=int,
        default=50,
        help=(
            "Save a full CPU-offloaded resume checkpoint every N "
            "optimizer steps. 0 disables periodic checkpoints."
        ),
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=0,
        help="0 means unlimited.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--out",
        type=str,
        default=(
            "runs/"
            "dpo_mmrlhf_v3"
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    args = parse_args()

    train(args)