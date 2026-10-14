#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch


from runtime import build_dpo_runtime



def count_parameters(model):

    total = sum(
        p.numel()
        for p in model.parameters()
    )


    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


    return total, trainable




if __name__ == "__main__":


    (
        policy,
        reference,
        processor
    ) = build_dpo_runtime()



    print("\n========== MODEL ==========")

    print(
        "policy:",
        type(policy)
    )

    print(
        "reference:",
        type(reference)
    )

    print(
        "processor:",
        type(processor)
    )



    print("\n========== PARAMETER CHECK ==========")


    policy_total, policy_trainable = (
        count_parameters(policy)
    )


    ref_total, ref_trainable = (
        count_parameters(reference)
    )


    print(
        "policy total params:",
        policy_total
    )


    print(
        "policy trainable params:",
        policy_trainable
    )


    print(
        "policy trainable ratio:",
        policy_trainable / policy_total
    )


    print(
        "reference total params:",
        ref_total
    )


    print(
        "reference trainable params:",
        ref_trainable
    )



    print("\n========== WEIGHT CHECK ==========")


    p1 = next(
        policy.parameters()
    )

    r1 = next(
        reference.parameters()
    )


    print(
        "first parameter equal:",
        torch.equal(
            p1.detach().cpu(),
            r1.detach().cpu()
        )
    )


    print(
        "policy first requires_grad:",
        p1.requires_grad
    )


    print(
        "reference first requires_grad:",
        r1.requires_grad
    )