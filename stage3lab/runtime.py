#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage3 DPO runtime

Load:

Stage1 verified HybridQwen25VL model
+
Stage2 SFT checkpoint

Return:

policy_model:
    trainable DPO model

reference_model:
    frozen reference model

processor:
    HybridQwen25Processor
"""


from pathlib import Path
import copy
import sys

import torch


# --------------------------------------------------
# add project root
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


from stage1_runtime import load_stage1_components



SFT_CHECKPOINT = (
    PROJECT_ROOT
    /
    "runs"
    /
    "sft_150k_ddp_6gpu"
    /
    "best_model.pt"
)



def load_sft_components():

    print(
        "Loading Stage1 base model...",
        flush=True
    )


    model, processor = (
        load_stage1_components()
    )


    print(
        "Loading SFT checkpoint:",
        SFT_CHECKPOINT,
        flush=True
    )


    if not SFT_CHECKPOINT.exists():

        raise FileNotFoundError(
            SFT_CHECKPOINT
        )


    state_dict = torch.load(
        SFT_CHECKPOINT,
        map_location="cpu"
    )


    print(
        "Checkpoint keys:",
        len(state_dict)
    )


    # 保留原始key
    # 因为模型和checkpoint都包含model.xxx

    missing, unexpected = (
        model.load_state_dict(
            state_dict,
            strict=False
        )
    )


    print(
        "SFT checkpoint loaded"
    )


    print(
        "Missing keys:",
        len(missing)
    )


    print(
        "Unexpected keys:",
        len(unexpected)
    )


    trainable=0
    total=0

    for n,p in model.named_parameters():

        total+=p.numel()

        if p.requires_grad:
            trainable+=p.numel()
            print(
                "TRAIN:",
                n,
                p.numel()
            )

    print(
        "trainable:",
        trainable,
        "/",
        total
    )

    # policy保持训练状态

    model.train()


    return model, processor




def build_dpo_runtime(
    freeze_visual=True
):


    policy_model, processor = (
        load_sft_components()
    )

    if freeze_visual:

        print(
            "Freezing visual modules..."
        )

        for name, param in policy_model.named_parameters():
            if "visual.vit" in name:
                param.requires_grad=False

                print(
                    "Frozen:",
                    name
                )

            elif "visual.merger" in name:

                param.requires_grad=True

                print(
                    "Train merger:",
                    name
                )



    print(
        "Creating reference model...",
        flush=True
    )


    reference_model = copy.deepcopy(
        policy_model
    )


    reference_model.eval()


    # reference完全冻结

    for p in reference_model.parameters():

        p.requires_grad = False



    print(
        "DPO runtime ready",
        flush=True
    )


    return (
        policy_model,
        reference_model,
        processor
    )