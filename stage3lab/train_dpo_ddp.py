#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import json
import argparse
from pathlib import Path


import torch
import torch.distributed as dist


from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


from torch.nn.parallel import DistributedDataParallel as DDP



from runtime import build_dpo_runtime


from mmrlhf_dpo_dataset import MMRefDPODataset


from dpo_collator import MMRefDPOCollator


from dpo_loss import (
    get_masked_logprob,
    dpo_loss,
)



# =====================================================
# DDP
# =====================================================


def setup_ddp():

    dist.init_process_group(
        backend="nccl"
    )


    local_rank = int(
        os.environ["LOCAL_RANK"]
    )


    torch.cuda.set_device(
        local_rank
    )


    return local_rank



def cleanup():

    dist.destroy_process_group()



# =====================================================
# batch utils
# =====================================================


def move_batch(
    batch,
    device
):

    result = {}


    for k,v in batch.items():

        if torch.is_tensor(v):

            result[k] = v.to(
                device,
                non_blocking=True
            )

        else:

            result[k] = v


    return result




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




# =====================================================
# checkpoint
# =====================================================


def save_checkpoint(
    model,
    path
):


    state_dict = (

        model.module.state_dict()

        if isinstance(model, DDP)

        else model.state_dict()

    )


    torch.save(
        state_dict,
        path
    )




# =====================================================
# trainable check
# =====================================================


def print_trainable_parameters(
    model
):


    trainable = 0

    total = 0


    print(
        "\n========== TRAINABLE PARAMETERS =========="
    )


    for name,param in model.named_parameters():


        total += param.numel()


        if param.requires_grad:


            trainable += param.numel()


            if (
                "visual" in name
                or
                "merger" in name
            ):


                print(
                    "[TRAIN]",
                    name,
                    param.numel()
                )


        else:


            if (
                "visual.vit" in name
            ):


                print(
                    "[FROZEN]",
                    name
                )



    print(
        "Trainable params:",
        trainable
    )


    print(
        "Total params:",
        total
    )


    print(
        "Trainable ratio:",
        trainable / total
    )


    return trainable




def count_optimizer_parameters(
    optimizer
):


    total = 0


    for group in optimizer.param_groups:


        for p in group["params"]:

            total += p.numel()


    return total




# =====================================================
# training
# =====================================================


def train(args):


    rank = setup_ddp()


    device = torch.device(
        f"cuda:{rank}"
    )



    if rank == 0:


        os.makedirs(
            args.out,
            exist_ok=True
        )



    # -------------------------------------------------
    # load models
    # -------------------------------------------------


    policy, reference, processor = (
        build_dpo_runtime()
    )


    policy.to(device)


    reference.to(device)



    reference.eval()


    for p in reference.parameters():

        p.requires_grad = False



    policy.train()



    if rank == 0:


        print_trainable_parameters(
            policy
        )



    policy = DDP(

        policy,

        device_ids=[
            rank
        ],

        find_unused_parameters=False

    )



    # -------------------------------------------------
    # dataset
    # -------------------------------------------------


    dataset = MMRefDPODataset(

        json_path=args.data,

        image_root=args.image_root

    )



    sampler = DistributedSampler(

        dataset,

        shuffle=True

    )



    collator = MMRefDPOCollator(

        processor

    )



    loader = DataLoader(

        dataset,

        batch_size=args.batch_size,

        sampler=sampler,

        num_workers=args.num_workers,

        pin_memory=True,

        collate_fn=collator

    )



    # -------------------------------------------------
    # optimizer
    # DPO-v2 modification
    # only optimize trainable params
    # -------------------------------------------------


    optimizer = torch.optim.AdamW(

        [

            p for p in policy.parameters()

            if p.requires_grad

        ],

        lr=args.lr,

        weight_decay=0.01

    )



    if rank == 0:


        optimizer_params = (
            count_optimizer_parameters(
                optimizer
            )
        )


        print(
            "Optimizer params:",
            optimizer_params
        )


    optimizer.zero_grad()



    global_step = 0

    optimizer_step = 0



    # -------------------------------------------------
    # metrics file
    # -------------------------------------------------


    metric_file = None


    if rank == 0:


        metric_path = (

            Path(args.out)

            /
            "dpo_metrics.jsonl"

        )


        metric_file = open(

            metric_path,

            "a",

            encoding="utf-8"

        )



    # Part 2 continues...

# =====================================================
# continue train()
# =====================================================


    best_loss = float("inf")



    for epoch in range(args.epochs):


        sampler.set_epoch(
            epoch
        )


        for batch in loader:


            if optimizer_step >= args.max_steps:

                break



            batch = move_batch(
                batch,
                device
            )


            #
            # ==========================
            # policy chosen
            # ==========================
            #


            chosen_logits = policy(

                **build_inputs(
                    batch,
                    "chosen"
                )

            ).logits



            policy_chosen_logps = (

                get_masked_logprob(

                    chosen_logits,

                    batch["chosen_labels"]

                )

            )



            del chosen_logits



            #
            # ==========================
            # policy rejected
            # ==========================
            #


            rejected_logits = policy(

                **build_inputs(
                    batch,
                    "rejected"
                )

            ).logits



            policy_rejected_logps = (

                get_masked_logprob(

                    rejected_logits,

                    batch["rejected_labels"]

                )

            )



            del rejected_logits



            #
            # ==========================
            # reference
            # ==========================
            #


            with torch.no_grad():


                ref_chosen_logits = reference(

                    **build_inputs(
                        batch,
                        "chosen"
                    )

                ).logits



                reference_chosen_logps = (

                    get_masked_logprob(

                        ref_chosen_logits,

                        batch["chosen_labels"]

                    )

                )



                del ref_chosen_logits



                ref_rejected_logits = reference(

                    **build_inputs(
                        batch,
                        "rejected"
                    )

                ).logits



                reference_rejected_logps = (

                    get_masked_logprob(

                        ref_rejected_logits,

                        batch["rejected_labels"]

                    )

                )


                del ref_rejected_logits



            #
            # ==========================
            # DPO loss
            # ==========================
            #


            result = dpo_loss(

                policy_chosen_logps,

                policy_rejected_logps,

                reference_chosen_logps,

                reference_rejected_logps,

                beta=args.beta

            )



            loss = result["loss"]



            if torch.isnan(loss):

                raise RuntimeError(
                    "NaN loss detected"
                )



            loss = (

                loss

                /

                args.grad_accum

            )



            loss.backward()



            global_step += 1



            #
            # ==========================
            # optimizer step
            # ==========================
            #


            if global_step % args.grad_accum == 0:



                #
                # gradient monitor
                #
                if (
                    rank == 0
                    and
                    optimizer_step % 20 == 0
                ):


                    print(
                        "\n========== GRAD CHECK =========="
                    )


                    module_check = [

                        "visual.merger",

                        "visual.vit",

                        "language_model.layers.0"

                    ]


                    for name,param in policy.module.named_parameters():


                        if param.grad is None:

                            continue



                        if any(
                            x in name
                            for x in module_check
                        ):


                            print(

                                name,

                                "grad_norm=",

                                float(
                                    param.grad.norm()
                                )

                            )



                torch.nn.utils.clip_grad_norm_(

                    policy.parameters(),

                    1.0

                )



                optimizer.step()



                optimizer.zero_grad()



                optimizer_step += 1



                torch.cuda.empty_cache()



                #
                # ==========================
                # logging
                # ==========================
                #


                if rank == 0 and optimizer_step % args.log_every == 0:



                    chosen_reward = (

                        result["chosen_rewards"]

                        .detach()

                        .float()

                        .mean()

                        .item()

                    )



                    rejected_reward = (

                        result["rejected_rewards"]

                        .detach()

                        .float()

                        .mean()

                        .item()

                    )



                    margin = (

                        result["margin"]

                        .detach()

                        .float()

                        .mean()

                        .item()

                    )



                    acc = (

                        (
                            result["margin"] > 0
                        )

                        .float()

                        .mean()

                        .item()

                    )



                    loss_value = (

                        result["loss"]

                        .detach()

                        .float()

                        .item()

                    )



                    print(

                        f"[epoch {epoch}] "

                        f"step={optimizer_step} "

                        f"loss={loss_value:.6f} "

                        f"chosen_reward={chosen_reward:.6f} "

                        f"rejected_reward={rejected_reward:.6f} "

                        f"margin={margin:.6f} "

                        f"acc={acc:.4f}",

                        flush=True

                    )



                    metric_file.write(

                        json.dumps(

                            {

                                "epoch": epoch,

                                "step": optimizer_step,

                                "loss": loss_value,

                                "chosen_reward": chosen_reward,

                                "rejected_reward": rejected_reward,

                                "margin": margin,

                                "accuracy": acc

                            },

                            ensure_ascii=False

                        )

                        +

                        "\n"

                    )


                    metric_file.flush()



                    if loss_value < best_loss:


                        best_loss = loss_value



                        save_path = (

                            Path(args.out)

                            /

                            "best_model.pt"

                        )


                        save_checkpoint(

                            policy,

                            save_path

                        )


                        print(

                            "Saved best:",

                            save_path

                        )




        if optimizer_step >= args.max_steps:

            break



    if metric_file is not None:

        metric_file.close()



    cleanup()





# =====================================================
# args
# =====================================================


def parse_args():


    parser = argparse.ArgumentParser()



    parser.add_argument(

        "--data",

        type=str,

        default="/mnt/data2/zhiyu/first-task/datasets/MMRLHF_ROOT/dpo_pairs.jsonl"

    )


    parser.add_argument(

        "--image_root",

        type=str,

        default="/mnt/data2/zhiyu/first-task/datasets/MMRLHF_ROOT"

    )


    parser.add_argument(

        "--epochs",

        type=int,

        default=2

    )


    parser.add_argument(

        "--batch_size",

        type=int,

        default=1

    )


    parser.add_argument(

        "--grad_accum",

        type=int,

        default=4

    )


    parser.add_argument(

        "--lr",

        type=float,

        default=5e-7

    )


    parser.add_argument(

        "--beta",

        type=float,

        default=0.1

    )


    parser.add_argument(

        "--log_every",

        type=int,

        default=20

    )


    parser.add_argument(

        "--save_every",

        type=int,

        default=500

    )


    parser.add_argument(

        "--max_steps",

        type=int,

        default=5200

    )


    parser.add_argument(

        "--num_workers",

        type=int,

        default=4

    )


    parser.add_argument(

        "--out",

        type=str,

        default="runs/dpo_mmrlhf_v2_freeze_vit"

    )


    return parser.parse_args()




# =====================================================
# main
# =====================================================


if __name__ == "__main__":


    args = parse_args()


    train(args)