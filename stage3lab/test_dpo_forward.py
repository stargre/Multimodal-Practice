#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch

from torch.utils.data import DataLoader


from runtime import build_dpo_runtime

from mmrlhf_dpo_dataset import MMRefDPODataset

from dpo_collator import MMRefDPOCollator


from dpo_utils import (
    move_batch_to_device,
    build_model_inputs,
    forward_policy,
    forward_reference,
)



DATA_PATH = (
    "/mnt/data2/zhiyu/first-task/"
    "datasets/MMRLHF_ROOT/"
    "dpo_pairs.jsonl"
)


IMAGE_ROOT = (
    "/mnt/data2/zhiyu/first-task/"
    "datasets/MMRLHF_ROOT"
)



device = "cuda"



if __name__ == "__main__":


    print(
        "Loading runtime..."
    )


    policy, reference, processor = (
        build_dpo_runtime()
    )


    policy.to(device)

    reference.to(device)



    print(
        "Loading dataset..."
    )


    dataset = MMRefDPODataset(
        json_path=DATA_PATH,
        image_root=IMAGE_ROOT
    )


    collator = MMRefDPOCollator(
        processor
    )


    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collator
    )



    batch = next(
        iter(loader)
    )


    batch = move_batch_to_device(
        batch,
        device
    )


    print("\n========== INPUT ==========")


    for k,v in batch.items():

        if torch.is_tensor(v):

            print(
                k,
                v.shape,
                v.dtype,
                v.device
            )



    print("\n========== POLICY CHOSEN ==========")


    chosen_inputs = build_model_inputs(
        batch,
        "chosen"
    )


    policy_chosen = forward_policy(
        policy,
        chosen_inputs
    )


    print(
        type(policy_chosen)
    )


    print(
        "logits:",
        policy_chosen.logits.shape
    )



    print("\n========== POLICY REJECTED ==========")


    rejected_inputs = build_model_inputs(
        batch,
        "rejected"
    )


    policy_rejected = forward_policy(
        policy,
        rejected_inputs
    )


    print(
        "logits:",
        policy_rejected.logits.shape
    )



    print("\n========== REFERENCE CHOSEN ==========")


    ref_chosen = forward_reference(
        reference,
        chosen_inputs
    )


    print(
        "logits:",
        ref_chosen.logits.shape
    )



    print("\n========== REFERENCE REJECTED ==========")


    ref_rejected = forward_reference(
        reference,
        rejected_inputs
    )


    print(
        "logits:",
        ref_rejected.logits.shape
    )



    print("\n========== SUCCESS ==========")