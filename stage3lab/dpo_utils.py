#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch



def move_batch_to_device(
    batch,
    device
):

    new_batch = {}

    for k, v in batch.items():

        if v is None:
            new_batch[k] = None

        elif torch.is_tensor(v):

            new_batch[k] = v.to(device)

        else:

            new_batch[k] = v


    return new_batch



def build_model_inputs(
    batch,
    prefix
):

    """
    prefix:

    chosen
    or
    rejected

    """

    inputs = {

        "input_ids":
            batch[f"{prefix}_input_ids"],


        "attention_mask":
            batch[f"{prefix}_attention_mask"],


        "pixel_values":
            batch[f"{prefix}_pixel_values"],


        "image_grid_thw":
            batch[f"{prefix}_image_grid_thw"],

    }


    return inputs



@torch.no_grad()
def forward_reference(
    model,
    inputs
):

    outputs = model(
        **inputs
    )

    return outputs



def forward_policy(
    model,
    inputs
):

    outputs = model(
        **inputs
    )

    return outputs