#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DPO-v3 runtime.

Policy:
    SFT checkpoint
    InternViT frozen
    visual merger trainable
    Qwen3 language model trainable

Reference:
    exact copy of SFT policy
    completely frozen
"""

from pathlib import Path
import copy
import sys

import torch


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from stage1_runtime import (
    load_stage1_components
)


SFT_CHECKPOINT = (
    PROJECT_ROOT
    /
    "runs"
    /
    "sft_150k_ddp_6gpu"
    /
    "best_model.pt"
)


def _is_vit_parameter(name: str):

    return (
        "model.visual.vit." in name
    )


def _is_merger_parameter(name: str):

    return (
        "model.visual.merger." in name
    )


def load_sft_components():

    print(
        "Loading Stage1 model...",
        flush=True,
    )

    model, processor = (
        load_stage1_components()
    )

    print(
        "Loading SFT checkpoint:",
        SFT_CHECKPOINT,
        flush=True,
    )

    if not SFT_CHECKPOINT.exists():

        raise FileNotFoundError(
            SFT_CHECKPOINT
        )

    state_dict = torch.load(
        SFT_CHECKPOINT,
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

    missing, unexpected = (
        model.load_state_dict(
            state_dict,
            strict=False,
        )
    )

    forbidden_missing = [
        key
        for key in missing
        if not _is_vit_parameter(key)
    ]

    if unexpected:

        raise RuntimeError(
            "Unexpected keys in SFT checkpoint:\n"
            +
            "\n".join(
                unexpected[:20]
            )
        )

    if forbidden_missing:

        raise RuntimeError(
            "Forbidden missing keys in SFT checkpoint:\n"
            +
            "\n".join(
                forbidden_missing[:50]
            )
        )

    print(
        "SFT checkpoint loaded.",
        flush=True,
    )

    print(
        "Missing keys:",
        len(missing),
        flush=True,
    )

    print(
        "Allowed missing keys:",
        len(
            [
                x
                for x in missing
                if _is_vit_parameter(x)
            ]
        ),
        flush=True,
    )

    # Training configuration.
    if hasattr(
        model.config,
        "use_cache",
    ):

        model.config.use_cache = False

    model.train()

    return model, processor


def build_dpo_v3_runtime():

    policy_model, processor = (
        load_sft_components()
    )

    trainable = 0
    total = 0

    merger_trainable = 0
    vit_frozen = 0

    print(
        "\n========== DPO-v3 PARAMETER POLICY ==========",
        flush=True,
    )

    for name, param in (
        policy_model.named_parameters()
    ):

        total += param.numel()

        # ----------------------------------------------------
        # InternViT frozen
        # ----------------------------------------------------

        if _is_vit_parameter(name):

            param.requires_grad = False

            vit_frozen += param.numel()

        # ----------------------------------------------------
        # Everything else trainable.
        # In particular:
        #
        # visual.merger
        # language_model
        # lm_head
        # ----------------------------------------------------

        else:

            param.requires_grad = True

            trainable += param.numel()

            if _is_merger_parameter(name):

                merger_trainable += param.numel()

    # --------------------------------------------------------
    # Frozen ViT must remain in eval mode.
    # --------------------------------------------------------

    policy_model.train()

    try:

        policy_model.model.visual.vit.eval()

    except Exception:

        print(
            "WARNING: could not explicitly set "
            "visual.vit.eval()",
            flush=True,
        )

    print(
        "Total parameters:",
        total,
        flush=True,
    )

    print(
        "Trainable parameters:",
        trainable,
        flush=True,
    )

    print(
        "Frozen ViT parameters:",
        vit_frozen,
        flush=True,
    )

    print(
        "Trainable merger parameters:",
        merger_trainable,
        flush=True,
    )

    print(
        "Trainable ratio:",
        trainable / total,
        flush=True,
    )

    # --------------------------------------------------------
    # Reference = exact SFT policy before DPO update.
    # --------------------------------------------------------

    print(
        "\nCreating reference model...",
        flush=True,
    )

    reference_model = copy.deepcopy(
        policy_model
    )

    reference_model.eval()

    for param in (
        reference_model.parameters()
    ):

        param.requires_grad = False

    if hasattr(
        reference_model.config,
        "use_cache",
    ):

        reference_model.config.use_cache = False

    print(
        "DPO-v3 runtime ready.",
        flush=True,
    )

    return (
        policy_model,
        reference_model,
        processor,
    )