#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
import os

from torch.utils.data import Dataset

from PIL import Image



class MMRefDPODataset(Dataset):


    def __init__(
        self,
        json_path,
        image_root
    ):

        self.image_root = image_root


        self.items=[]


        total=0
        skipped=0


        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as f:


            for line in f:


                if not line.strip():
                    continue


                total += 1


                x=json.loads(line)



                # ==========================
                # only keep image samples
                # ==========================

                if "image" not in x:

                    skipped += 1

                    continue



                image_path=os.path.join(
                    self.image_root,
                    x["image"]
                )



                if not os.path.exists(
                    image_path
                ):

                    skipped += 1

                    continue



                self.items.append(
                    x
                )



        print(
            f"Original samples: {total}"
        )

        print(
            f"Valid image DPO samples: {len(self.items)}"
        )

        print(
            f"Skipped samples: {skipped}"
        )





    def __len__(self):

        return len(self.items)




    def build_prompt(
        self,
        text
    ):


        return (

            "<|im_start|>user\n"

            "<|vision_start|>"

            "<|image_pad|>"

            "<|vision_end|>\n"

            f"{text}"

            "<|im_end|>\n"

            "<|im_start|>assistant\n"

        )





    def __getitem__(
        self,
        index
    ):


        x=self.items[index]


        image_path=os.path.join(
            self.image_root,
            x["image"]
        )


        image=Image.open(
            image_path
        ).convert("RGB")



        prompt=self.build_prompt(
            x["prompt"]
        )



        return {


            "image":
                image,


            "prompt":
                prompt,


            "chosen":
                x["chosen"],


            "rejected":
                x["rejected"],


        }