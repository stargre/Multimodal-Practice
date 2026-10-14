#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(
    str(PROJECT_ROOT)
)

import torch

from torch.utils.data import DataLoader, Subset

from tqdm import tqdm


from stage1_runtime import load_stage1_components

from mmrlhf_dpo_dataset import MMRefDPODataset

from dpo_collator import MMRefDPOCollator

from dpo_loss import get_masked_logprob



DEVICE = "cuda"



# ==========================
# checkpoint
# ==========================

DPO_CKPT = (
    "/mnt/data2/zhiyu/first-task/"
    "qwen2.5/multimodal_training/"
    "runs/dpo_mmrlhf_6gpu/"
    "best_model.pt"
)


DATA_JSON = (
    "/mnt/data2/zhiyu/first-task/"
    "datasets/MMRLHF_ROOT/"
    "dpo_pairs.jsonl"
)


IMAGE_ROOT = (
    "/mnt/data2/zhiyu/first-task/"
    "datasets/MMRLHF_ROOT"
)



# ==========================
# eval
# ==========================


@torch.no_grad()
def evaluate(
    model,
    loader,
    device
):

    model.eval()


    total_margin = 0.0

    total_acc = 0.0

    count = 0



    for batch in tqdm(loader):


        for k,v in batch.items():

            if torch.is_tensor(v):

                batch[k] = v.to(
                    device,
                    non_blocking=True
                )



        # =====================
        # chosen
        # =====================

        chosen_logits = model(

            input_ids=batch["chosen_input_ids"],

            attention_mask=batch["chosen_attention_mask"],

            pixel_values=batch["chosen_pixel_values"],

            image_grid_thw=batch["chosen_image_grid_thw"],

        ).logits



        chosen_logp = get_masked_logprob(

            chosen_logits,

            batch["chosen_labels"]

        )



        del chosen_logits



        # =====================
        # rejected
        # =====================


        rejected_logits = model(

            input_ids=batch["rejected_input_ids"],

            attention_mask=batch["rejected_attention_mask"],

            pixel_values=batch["rejected_pixel_values"],

            image_grid_thw=batch["rejected_image_grid_thw"],

        ).logits



        rejected_logp = get_masked_logprob(

            rejected_logits,

            batch["rejected_labels"]

        )



        del rejected_logits



        margin = (
            chosen_logp
            -
            rejected_logp
        )



        total_margin += (
            margin.float()
            .mean()
            .item()
        )


        total_acc += (
            (margin > 0)
            .float()
            .mean()
            .item()
        )


        count += 1



    print("\n========== RESULT ==========")

    print(
        "avg margin:",
        total_margin / count
    )


    print(
        "preference accuracy:",
        total_acc / count
    )





def main():


    print(
        "Loading base model..."
    )


    model, processor = (
        load_stage1_components()
    )



    print(
        "Loading DPO checkpoint:"
    )

    print(
        DPO_CKPT
    )



    ckpt = torch.load(

        DPO_CKPT,

        map_location="cpu"

    )


    missing, unexpected = (
        model.load_state_dict(

            ckpt,

            strict=False

        )
    )


    print(
        "Missing keys:",
        len(missing)
    )


    print(
        "Unexpected keys:",
        len(unexpected)
    )



    model.to(
        DEVICE
    )


    model.eval()



    print(
        "Loading dataset..."
    )


    dataset = MMRefDPODataset(

        json_path=DATA_JSON,

        image_root=IMAGE_ROOT

    )


    print(
        "Full valid dataset:",
        len(dataset)
    )



    # ==========================
    # only evaluate subset
    # ==========================


    eval_num = min(
        2000,
        len(dataset)
    )


    dataset = Subset(

        dataset,

        range(eval_num)

    )


    print(
        "Evaluation samples:",
        len(dataset)
    )



    collator = MMRefDPOCollator(

        processor

    )



    loader = DataLoader(

        dataset,

        batch_size=1,

        shuffle=False,

        num_workers=4,

        pin_memory=True,

        collate_fn=collator

    )



    evaluate(

        model,

        loader,

        DEVICE

    )





if __name__ == "__main__":

    main()