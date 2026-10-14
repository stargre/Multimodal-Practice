import json
from torch.utils.data import Dataset
from PIL import Image


class MMRefDPODataset(Dataset):

    def __init__(self,json_path):

        with open(json_path) as f:
            self.items=json.load(f)


    def __len__(self):
        return len(self.items)



    def build_prompt(self,text):

        return (
            "<|im_start|>user\n"
            "<|vision_start|>"
            "<|image_pad|>"
            "<|vision_end|>\n"
            f"{text}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )



    def __getitem__(self,index):

        x=self.items[index]


        image=Image.open(
            x["image"]
        ).convert("RGB")


        prompt=self.build_prompt(
            x["prompt"]
        )


        return {

            "image":image,

            "chosen":
                prompt+x["chosen"],

            "rejected":
                prompt+x["rejected"]

        }