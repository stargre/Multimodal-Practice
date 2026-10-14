#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse

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
    dpo_loss
)



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



def move_batch(
    batch,
    device
):

    result={}

    for k,v in batch.items():

        if torch.is_tensor(v):

            result[k]=v.to(
                device,
                non_blocking=True
            )

        else:

            result[k]=v


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



def train(args):


    rank = setup_ddp()


    device=torch.device(
        f"cuda:{rank}"
    )


    if rank==0:

        os.makedirs(
            args.out,
            exist_ok=True
        )


    #
    # load model
    #

    policy, reference, processor = (
        build_dpo_runtime()
    )


    policy.to(device)

    reference.to(device)



    reference.eval()


    for p in reference.parameters():

        p.requires_grad=False



    policy.train()



    policy = DDP(

        policy,

        device_ids=[rank],

        find_unused_parameters=False

    )



    #
    # dataset
    #

    dataset=MMRefDPODataset(

        json_path=args.data,

        image_root=args.image_root

    )


    sampler=DistributedSampler(

        dataset,

        shuffle=True

    )


    collator=MMRefDPOCollator(
        processor
    )


    loader=DataLoader(

        dataset,

        batch_size=args.batch_size,

        sampler=sampler,

        num_workers=args.num_workers,

        pin_memory=True,

        collate_fn=collator

    )



    #
    # optimizer
    #

    optimizer=torch.optim.AdamW(

        policy.parameters(),

        lr=args.lr,

        weight_decay=0.01

    )


    optimizer.zero_grad()



    global_step=0

    optimizer_step=0



    for epoch in range(args.epochs):


        sampler.set_epoch(
            epoch
        )


        for batch in loader:


            batch=move_batch(
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
            # DPO
            #

            result=dpo_loss(

                policy_chosen_logps,

                policy_rejected_logps,

                reference_chosen_logps,

                reference_rejected_logps,

                beta=args.beta

            )


            loss=result["loss"]


            loss = (
                loss /
                args.grad_accum
            )


            loss.backward()



            global_step += 1



            if global_step % args.grad_accum == 0:


                torch.nn.utils.clip_grad_norm_(

                    policy.parameters(),

                    1.0

                )


                optimizer.step()


                optimizer.zero_grad()



                optimizer_step += 1



                torch.cuda.empty_cache()



                if rank==0 and optimizer_step % args.log_every==0:


                    print(

                        f"epoch={epoch} "

                        f"step={optimizer_step} "

                        f"loss={loss.item()*args.grad_accum:.4f}",

                        flush=True

                    )



                if (

                    rank==0
                    and
                    optimizer_step % args.save_every==0

                ):


                    save_checkpoint(

                        policy,

                        os.path.join(

                            args.out,

                            f"step_{optimizer_step}.pt"

                        )

                    )



            if optimizer_step >= args.max_steps:

                break



        if optimizer_step >= args.max_steps:

            break



    if rank==0:


        save_checkpoint(

            policy,

            os.path.join(
                args.out,
                "best_model.pt"
            )

        )


        print(
            "Training finished",
            flush=True
        )



    cleanup()





if __name__=="__main__":


    parser=argparse.ArgumentParser()



    parser.add_argument(
        "--data",
        default="/mnt/data2/zhiyu/first-task/datasets/MMRLHF_ROOT/dpo_pairs.jsonl"
    )


    parser.add_argument(
        "--image_root",
        default="/mnt/data2/zhiyu/first-task/datasets/MMRLHF_ROOT"
    )


    parser.add_argument(
        "--out",
        default="runs/dpo_mmrlhf_6gpu"
    )


    parser.add_argument(
        "--epochs",
        type=int,
        default=1
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
        "--num_workers",
        type=int,
        default=4
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
        default=2500
    )



    args=parser.parse_args()


    train(args)