#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn.functional as F



def get_masked_logprob(
    logits,
    labels
):
    """
    Compute answer-only average log probability.

    logits:
        [B, T, vocab]

    labels:
        [B, T]

    labels:
        -100 tokens are ignored

    Return:
        [B]

    """



    # shift for causal LM

    logits = logits[:, :-1, :]

    labels = labels[:, 1:]



    mask = (
        labels != -100
    )



    safe_labels = labels.clone()

    safe_labels[
        ~mask
    ] = 0



    log_probs = F.log_softmax(
        logits.float(),
        dim=-1
    )



    token_log_probs = torch.gather(

        log_probs,

        dim=-1,

        index=safe_labels.unsqueeze(-1)

    ).squeeze(-1)





    # remove ignored tokens

    token_log_probs = (
        token_log_probs * mask
    )



    token_count = (
        mask.sum(dim=-1)
        .clamp(min=1)
    )



    # IMPORTANT:
    # use average token log probability
    #
    # instead of:
    #
    # token_log_probs.sum()
    #
    # to avoid length bias


    seq_logprob = (
        token_log_probs * mask
    ).sum(dim=-1)


    return seq_logprob





def dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta=0.1
):
    """
    Standard DPO objective.

    r_chosen =
        beta *
        (
        policy_chosen - reference_chosen
        )


    r_rejected =
        beta *
        (
        policy_rejected - reference_rejected
        )


    loss =
        -log sigmoid(
            r_chosen-r_rejected
        )

    """



    chosen_rewards = (

        beta
        *
        (
            policy_chosen_logps
            -
            reference_chosen_logps
        )

    )



    rejected_rewards = (

        beta
        *
        (
            policy_rejected_logps
            -
            reference_rejected_logps
        )

    )



    margin = (
        chosen_rewards
        -
        rejected_rewards
    )



    loss = -F.logsigmoid(
        margin
    ).mean()



    return {

        "loss":
            loss,


        "chosen_rewards":
            chosen_rewards.detach(),


        "rejected_rewards":
            rejected_rewards.detach(),


        "margin":
            margin.detach(),

    }