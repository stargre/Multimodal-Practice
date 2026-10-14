from mmrlhf_dpo_dataset import MMRefDPODataset



dataset = MMRefDPODataset(

    json_path=
    "/mnt/data2/zhiyu/first-task/datasets/MMRLHF_ROOT/dpo_pairs.jsonl",

    image_root=
    "/mnt/data2/zhiyu/first-task/datasets/MMRLHF_ROOT"
)



print("\nlength:")
print(len(dataset))


sample = dataset[0]


print("\nkeys:")
print(sample.keys())


print("\nimage:")
print(sample["image"])


print("\nimage path:")
print(sample["image_path"])


print("\nraw prompt:")
print(sample["raw_prompt"])


print("\nchosen:")
print(sample["chosen"][:300])


print("\nrejected:")
print(sample["rejected"][:300])