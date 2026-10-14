import torch

from mmrlhf_dpo_dataset import MMRefDPODataset
from dpo_collator import MMRefDPOCollator
from runtime import build_dpo_runtime


DATA = "/mnt/data2/zhiyu/first-task/datasets/MMRLHF_ROOT/dpo_pairs.jsonl"

IMAGE_ROOT = "/mnt/data2/zhiyu/first-task/datasets/MMRLHF_ROOT"



def main():

    print("loading processor")

    _, _, processor = build_dpo_runtime()


    dataset = MMRefDPODataset(
        DATA,
        IMAGE_ROOT
    )


    collator = MMRefDPOCollator(
        processor
    )


    sample=[
        dataset[0]
    ]


    batch = collator(sample)



    print("\n========== SHAPE ==========")

    for k,v in batch.items():

        if torch.is_tensor(v):

            print(
                k,
                v.shape
            )


    print("\n========== LABEL CHECK ==========")


    labels = batch["chosen_labels"][0]

    input_ids = batch["chosen_input_ids"][0]


    total = labels.shape[0]


    ignore = (
        labels == -100
    ).sum().item()


    answer_tokens = (
        labels != -100
    ).sum().item()



    print(
        "total tokens:",
        total
    )

    print(
        "masked tokens:",
        ignore
    )

    print(
        "answer tokens:",
        answer_tokens
    )



    print("\nfirst 100 labels:")

    print(
        labels[:100].tolist()
    )



    print("\nfirst 100 input ids:")

    print(
        input_ids[:100].tolist()
    )



if __name__=="__main__":

    main()