#!/usr/bin/env python3


import torch


from torch.utils.data import DataLoader



from stage3lab.runtime import build_dpo_runtime


from stage3lab.dpo_dataset import MMRefDPODataset

from stage3lab.dpo_collator import DPOCollator

from stage3lab.dpo_utils import (
    get_policy_logp,
    get_reference_logp
)

from stage3lab.dpo_loss import dpo_loss



DATA="data_splits/mmrlhf_dpo/train.json"


CKPT="runs/sft_150k_ddp_6gpu/best_model.pt"



device="cuda"



def main():


    print(
        "Loading model..."
    )


    model,processor=build_dpo_runtime()



    state=torch.load(
        CKPT,
        map_location="cpu"
    )


    model.load_state_dict(
        state,
        strict=False
    )


    model.to(device)


    model.eval()



    print(
        "Loading dataset..."
    )


    dataset=MMRefDPODataset(
        DATA
    )



    sample=[
        dataset[0]
    ]



    collator=DPOCollator(
        processor
    )


    batch=collator(
        sample
    )



    print("===================")

    print(
        "chosen ids:",
        batch["chosen"]["input_ids"].shape
    )


    print(
        "rejected ids:",
        batch["rejected"]["input_ids"].shape
    )


    print(
        "pixel:",
        batch["chosen"]["pixel_values"].shape
    )


    print("===================")



    chosen = {
        k: v.to(device)
        for k, v in batch["chosen"].items()
        if v is not None
    }



    rejected = {
        k: v.to(device)
        for k, v in batch["rejected"].items()
        if v is not None
    }




    with torch.no_grad():


        chosen_logp=get_policy_logp(
            model,
            chosen
        )


        rejected_logp=get_policy_logp(
            model,
            rejected
        )



    print(
        "chosen logp:",
        chosen_logp
    )


    print(
        "rejected logp:",
        rejected_logp
    )



    loss=dpo_loss(
        chosen_logp,
        rejected_logp,
        torch.zeros_like(chosen_logp),
        torch.zeros_like(rejected_logp)
    )


    print(
        "DPO loss:",
        loss.item()
    )



    assert torch.isfinite(loss)


    print(
        "PASS: DPO forward"
    )



if __name__=="__main__":

    main()