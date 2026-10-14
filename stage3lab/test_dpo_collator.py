#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from torch.utils.data import DataLoader


from mmrlhf_dpo_dataset import MMRefDPODataset

from runtime import build_dpo_runtime

from dpo_collator import MMRefDPOCollator




DATA_PATH = (
    "/mnt/data2/zhiyu/first-task/"
    "datasets/MMRLHF_ROOT/"
    "dpo_pairs.jsonl"
)


IMAGE_ROOT = (
    "/mnt/data2/zhiyu/first-task/"
    "datasets/MMRLHF_ROOT"
)



if __name__ == "__main__":



    print(
        "Loading runtime..."
    )


    _, _, processor = (
        build_dpo_runtime()
    )



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



    print(
        "\n========== OUTPUT KEYS =========="
    )


    for k in batch:

        print(k)



    print(
        "\n========== SHAPES =========="
    )


    for k,v in batch.items():

        if v is None:

            print(
                k,
                None
            )

        else:

            print(
                k,
                v.shape
            )



    print(
        "\n========== DONE =========="
    )