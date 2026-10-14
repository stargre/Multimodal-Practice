#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch



class MMRefDPOCollator:


    def __init__(
        self,
        processor
    ):

        self.processor=processor



    def make_labels(
        self,
        input_ids,
        prompt_lengths
    ):


        labels=input_ids.clone()


        for i,length in enumerate(prompt_lengths):

            labels[i,:length]=-100

        print(
            "input length:",
            input_ids.shape[1],
            "prompt length:",
            length,
            "answer tokens:",
            (labels!=-100).sum().item()
        )

        return labels



    def encode(
        self,
        prompts,
        answers,
        images
    ):


        full_prompts=[]

        prompt_lengths=[]


        for p,a in zip(
            prompts,
            answers
        ):

            full=p+a


            full_prompts.append(
                full
            )


            prompt_ids=self.processor.tokenizer(
                p,
                add_special_tokens=False
            )["input_ids"]


            prompt_lengths.append(
                len(prompt_ids)
            )



        result=self.processor.encode_images(
            full_prompts,
            images_per_sample=images,
            padding=True
        )


        labels=self.make_labels(
            result["input_ids"],
            prompt_lengths
        )


        result["labels"]=labels


        return result



    def __call__(
        self,
        batch
    ):


        chosen_prompts=[]
        rejected_prompts=[]

        chosen_answers=[]
        rejected_answers=[]

        images=[]



        for item in batch:


            images.append(
                [item["image"]]
            )


            chosen_prompts.append(
                item["prompt"]
            )


            rejected_prompts.append(
                item["prompt"]
            )


            chosen_answers.append(
                item["chosen"]
            )


            rejected_answers.append(
                item["rejected"]
            )



        chosen=self.encode(
            chosen_prompts,
            chosen_answers,
            images
        )


        rejected=self.encode(
            rejected_prompts,
            rejected_answers,
            images
        )



        return {


            "chosen_input_ids":
                chosen["input_ids"],

            "chosen_attention_mask":
                chosen["attention_mask"],

            "chosen_labels":
                chosen["labels"],

            "chosen_pixel_values":
                chosen["pixel_values"],

            "chosen_image_grid_thw":
                chosen["image_grid_thw"],



            "rejected_input_ids":
                rejected["input_ids"],

            "rejected_attention_mask":
                rejected["attention_mask"],

            "rejected_labels":
                rejected["labels"],

            "rejected_pixel_values":
                rejected["pixel_values"],

            "rejected_image_grid_thw":
                rejected["image_grid_thw"],

        }