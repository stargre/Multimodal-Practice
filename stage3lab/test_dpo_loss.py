#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
from torch.utils.data import DataLoader


from runtime import build_dpo_runtime

from mmrlhf_dpo_dataset import MMRefDPODataset

from dpo_collator import MMRefDPOCollator


from dpo_loss import (
    get_masked_logprob,
    dpo_loss
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



def move_to_cuda(batch):

    output = {}

    for k, v in batch.items():

        if torch.is_tensor(v):

            output[k] = v.cuda()

        else:

            output[k] = v


    return output




def build_inputs(
    batch,
    prefix
):

    return {

        "input_ids":
            batch[f"{prefix}_input_ids"],


        "attention_mask":
            batch[f"{prefix}_attention_mask"],


        "pixel_values":
            batch[f"{prefix}_pixel_values"],


        "image_grid_thw":
            batch[f"{prefix}_image_grid_thw"],

    }





if __name__ == "__main__":


    torch.cuda.empty_cache()



    print(
        "Loading runtime..."
    )


    policy, reference, processor = (
        build_dpo_runtime()
    )


    policy.cuda()

    reference.cuda()



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


    batch = move_to_cuda(
        batch
    )



    print("\n========== LABEL CHECK ==========")


    print(
        "chosen labels:",
        batch["chosen_labels"].shape
    )


    print(
        "rejected labels:",
        batch["rejected_labels"].shape
    )


    print(
        "chosen valid tokens:",
        (batch["chosen_labels"] != -100).sum()
    )


    print(
        "rejected valid tokens:",
        (batch["rejected_labels"] != -100).sum()
    )



    print(
        "\n========== POLICY FORWARD =========="
    )


    policy_chosen = policy(
        **build_inputs(
            batch,
            "chosen"
        )
    )



    policy_rejected = policy(
        **build_inputs(
            batch,
            "rejected"
        )
    )



    print(
        "\n========== REFERENCE FORWARD =========="
    )


    with torch.no_grad():


        reference_chosen = reference(
            **build_inputs(
                batch,
                "chosen"
            )
        )


        reference_rejected = reference(
            **build_inputs(
                batch,
                "rejected"
            )
        )



    print(
        "\n========== LOG PROB =========="
    )



    policy_chosen_logps = (
        get_masked_logprob(
            policy_chosen.logits,
            batch["chosen_labels"]
        )
    )


    policy_rejected_logps = (
        get_masked_logprob(
            policy_rejected.logits,
            batch["rejected_labels"]
        )
    )



    reference_chosen_logps = (
        get_masked_logprob(
            reference_chosen.logits,
            batch["chosen_labels"]
        )
    )


    reference_rejected_logps = (
        get_masked_logprob(
            reference_rejected.logits,
            batch["rejected_labels"]
        )
    )



    print(
        "policy chosen:",
        policy_chosen_logps
    )


    print(
        "policy rejected:",
        policy_rejected_logps
    )


    print(
        "reference chosen:",
        reference_chosen_logps
    )


    print(
        "reference rejected:",
        reference_rejected_logps
    )



    print(
        "\n========== DPO LOSS =========="
    )



    result = dpo_loss(

        policy_chosen_logps,

        policy_rejected_logps,

        reference_chosen_logps,

        reference_rejected_logps,

        beta=0.1

    )



    for k, v in result.items():

        print(
            k,
            v
        )



    print(
        "\n========== BACKWARD =========="
    )


    result["loss"].backward()



    grad_params = 0


    grad_norm = 0.0


    for p in policy.parameters():

        if p.grad is not None:

            grad_params += 1

            grad_norm += (
                p.grad.detach()
                .float()
                .norm()
                .item()
            )



    print(
        "params with grad:",
        grad_params
    )


    print(
        "grad norm sum:",
        grad_norm
    )


    print(
        "\nDPO LOSS TEST PASS"
    )