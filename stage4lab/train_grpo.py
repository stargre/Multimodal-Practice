#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-4 GRPO trainer for Thyme-RL and the custom HybridQwen25VL model.

This is a self-contained replacement trainer. It deliberately keeps the
project's custom processor/model API and does not assume AutoProcessor support.

Key safeguards:
- reward text is trimmed at the first stop token, while the policy loss mask
  includes the first sampled stop token and masks tokens after it;
- the exact same prompt builder is used for every rollout in a group;
- per-rollout completion, extracted answer and reward components are logged;
- trainable weights are FP32 master parameters (the runtime casts them before
  loading); forward passes use BF16 autocast;
- an optimizer step is skipped when neither group-reward variation nor a
  non-zero KL term provides a training signal;
- checkpoint files store the model under the `model` key, which the runtime
  loader understands.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
import unicodedata
from pathlib import Path
from statistics import mean
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from thyme_dataset import ThymeRLStream  # noqa: E402
from grpo_runtime import load_dpo_v2_policy  # noqa: E402


# ---------------------------------------------------------------------------
# Distributed setup / utilities
# ---------------------------------------------------------------------------


def setup_distributed() -> tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-4 GRPO requires CUDA")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    return rank, local_rank, world_size, device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def core_model(model):
    return model.module if isinstance(model, DDP) else model


def set_seed(seed: int, rank: int = 0) -> None:
    actual_seed = int(seed) + int(rank)
    random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    torch.cuda.manual_seed_all(actual_seed)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if value is None:
            continue
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def autocast_bf16(device: torch.device):
    return torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=(device.type == "cuda"),
    )


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def append_jsonl(fp, obj: Any) -> None:
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
    fp.flush()


# ---------------------------------------------------------------------------
# Thyme prompt construction
# ---------------------------------------------------------------------------


def build_thyme_prompt(processor, sample: dict[str, Any], image_path: Path) -> str:
    """Match the prompt construction used by the Stage-4 smoke/rollout scripts."""
    system_prompt = sample["system_prompt"]
    user_text = sample["user_prompt"].lstrip()

    # Thyme has a textual marker. The custom processor inserts the actual
    # <|vision_start|><|image_pad|><|vision_end|> sequence.
    if user_text.startswith("<image>"):
        user_text = user_text[len("<image>"):].lstrip("\r\n")

    # Replace the original author's machine-local path with this sample's path.
    path_pattern = r'(### User Image Path:\*\*\s*")[^"]*(")'
    user_text, path_count = re.subn(
        path_pattern,
        lambda m: m.group(1) + str(image_path) + m.group(2),
        user_text,
        count=1,
    )
    if path_count == 0:
        user_text += f'\n\n### User Image Path:** "{image_path}"\n'

    width, height = sample["image"].size
    size_pattern = r'(### User Image Size:\*\*\s*")[^"]*(")'
    user_text, size_count = re.subn(
        size_pattern,
        lambda m: m.group(1) + f"{width}x{height}" + m.group(2),
        user_text,
        count=1,
    )
    if size_count == 0:
        user_text += f'\n### User Image Size:** "{width}x{height}"\n'

    system_block = "<|im_start|>system\n" + system_prompt + "<|im_end|>\n"
    user_block = processor.user_prompt(user_text, num_images=1)
    prompt = system_block + user_block

    image_pad_count = prompt.count("<|image_pad|>")
    if image_pad_count != 1:
        raise RuntimeError(
            f"Expected exactly one logical image placeholder before encoding; got {image_pad_count}"
        )
    if "<image>" in prompt:
        raise RuntimeError("Raw <image> marker remains in the final prompt")
    return prompt


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip().lower()


def extract_answer(completion: str) -> str:
    if completion is None:
        return ""
    match = re.search(r"<answer>(.*?)</answer>", completion, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return completion.strip()


def clean_answer_prefix(answer: str) -> str:
    prefixes = (
        "The best answer is", "The correct answer is", "The answer is",
        "The answer:", "Answer:", "The best option is", "The correct option is",
        "Best answer:", "Best option:",
    )
    cleaned = (answer or "").strip()
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
                changed = True
    return cleaned


def format_check(completion: str) -> bool:
    if not completion:
        return False
    # Accept any whitespace between the two required blocks, but require exactly
    # one pair of each tag and the answer block to be the final content.
    pattern = (
        r"^(?:(?!</think>).)*</think>\s*<answer>"
        r"(?:(?!</answer>).)*</answer>\s*$"
    )
    if re.fullmatch(pattern, completion, flags=re.DOTALL) is None:
        return False
    return (
        completion.count("<think>") == 1
        and completion.count("</think>") == 1
        and completion.count("<answer>") == 1
        and completion.count("</answer>") == 1
    )


try:
    from math_verify import parse as math_parse, verify as math_verify
    HAS_MATH_VERIFY = True
except Exception:
    HAS_MATH_VERIFY = False
    math_parse = None
    math_verify = None


NUMBER_PATTERN = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)


def is_plain_numeric_answer(text: str) -> bool:
    normalized = str(text).strip().replace(",", "").replace(" ", "")
    return re.fullmatch(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
        normalized,
    ) is not None


def try_math_verify(prediction: str, solution: str) -> bool:
    if not HAS_MATH_VERIFY:
        return False
    try:
        pred = math_parse(prediction)
        gold = math_parse(solution)
        return float(math_verify(pred, gold)) > 0.0
    except Exception:
        return False


def try_numeric_match(prediction: str, solution: str) -> bool:
    # Numeric fallback is deliberately restricted to numeric references, so a
    # number mentioned in an arbitrary textual answer cannot create a false hit.
    if not is_plain_numeric_answer(solution):
        return False
    pred_numbers = NUMBER_PATTERN.findall(prediction or "")
    gold_text = str(solution).strip().replace(",", "").replace(" ", "")
    if not pred_numbers:
        return False
    try:
        pred_value = float(pred_numbers[-1])  # final numeric candidate, not first mention
        gold_value = float(gold_text)
        return math.isclose(pred_value, gold_value, rel_tol=1e-6, abs_tol=1e-6)
    except Exception:
        return False


def compute_accuracy_reward(completion: str, solution: str) -> dict[str, Any]:
    answer = clean_answer_prefix(extract_answer(completion))
    solution = str(solution).strip()
    if not answer:
        return {"reward": 0.0, "match": False, "method": "empty", "extracted_answer": ""}

    if try_math_verify(answer, solution):
        return {"reward": 1.0, "match": True, "method": "math_verify", "extracted_answer": answer}

    if normalize_text(answer) == normalize_text(solution):
        return {"reward": 1.0, "match": True, "method": "exact_string", "extracted_answer": answer}

    if try_numeric_match(answer, solution):
        return {"reward": 1.0, "match": True, "method": "numeric_last_value", "extracted_answer": answer}

    return {"reward": 0.0, "match": False, "method": "no_match", "extracted_answer": answer}


def compute_rewards(completion: str, solution: str, format_weight: float) -> dict[str, Any]:
    accuracy = compute_accuracy_reward(completion, solution)
    format_reward = 1.0 if format_check(completion) else 0.0
    accuracy_reward = float(accuracy["reward"])
    return {
        "accuracy_reward": accuracy_reward,
        "format_reward": float(format_reward),
        "total_reward": float(accuracy_reward + format_weight * format_reward),
        "accuracy_match": bool(accuracy["match"]),
        "accuracy_match_method": accuracy["method"],
        "extracted_answer": accuracy["extracted_answer"],
    }


# ---------------------------------------------------------------------------
# Response stop handling and log probabilities
# ---------------------------------------------------------------------------


def get_stop_ids(tokenizer) -> list[int]:
    result: list[int] = []
    eos = tokenizer.eos_token_id
    if isinstance(eos, (tuple, list)):
        result.extend(int(x) for x in eos if x is not None)
    elif eos is not None:
        result.append(int(eos))

    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and int(im_end) >= 0:
        result.append(int(im_end))
    return sorted(set(result))


def trim_for_reward(row: torch.Tensor, stop_ids: list[int], pad_id: Optional[int]) -> torch.Tensor:
    """Return tokens before the first stop token/padding for text scoring only."""
    ids = row.tolist()
    stops = set(stop_ids)
    end = len(ids)
    for i, token_id in enumerate(ids):
        if token_id in stops:
            end = i  # do not include EOS / im_end in the reward text
            break
        if pad_id is not None and int(token_id) == int(pad_id):
            end = i
            break
    return row[:end]


def make_response_mask(
    response_ids: torch.Tensor,
    stop_ids: list[int],
    pad_id: Optional[int],
) -> torch.Tensor:
    """Mask generated action tokens, including the first EOS/im_end but not padding after it."""
    mask = torch.zeros_like(response_ids, dtype=torch.bool)
    stop_set = set(stop_ids)
    for row_idx in range(response_ids.shape[0]):
        ids = response_ids[row_idx].tolist()
        end = len(ids)
        include_end = False
        for pos, token_id in enumerate(ids):
            if token_id in stop_set:
                end = pos
                include_end = True
                break
            if pad_id is not None and int(token_id) == int(pad_id):
                end = pos
                include_end = False
                break
        if end > 0:
            mask[row_idx, :end] = True
        if include_end and end < len(ids):
            mask[row_idx, end] = True
        elif end == len(ids):
            mask[row_idx, :] = True
    return mask


def make_full_batch(
    prompt_batch: dict[str, Any],
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[dict[str, Any], int]:
    prompt_ids = prompt_batch["input_ids"]
    prompt_mask = prompt_batch["attention_mask"]
    prompt_length = int(prompt_ids.shape[1])
    full_ids = torch.cat([prompt_ids, response_ids], dim=1)
    full_attention = torch.cat([prompt_mask, response_mask.long()], dim=1)

    full_batch = dict(prompt_batch)
    full_batch["input_ids"] = full_ids
    full_batch["attention_mask"] = full_attention
    return full_batch, prompt_length


def compute_response_logps(
    model,
    full_batch: dict[str, Any],
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    prompt_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute selected-token log probabilities for generated response tokens only."""
    core_model(model).model.rope_deltas = None
    with autocast_bf16(device):
        outputs = model(**full_batch, use_cache=False, return_dict=True)
        logits = outputs.logits

        response_length = int(response_ids.shape[1])
        # logits at position (prompt_length - 1 + j) predicts response token j.
        selected_logits = logits[
            :, prompt_length - 1 : prompt_length + response_length - 1, :
        ].float()
        log_probs = F.log_softmax(selected_logits, dim=-1)
        token_logps = log_probs.gather(
            dim=-1,
            index=response_ids.unsqueeze(-1),
        ).squeeze(-1)
        token_logps = token_logps * response_mask.to(token_logps.dtype)

    del outputs, logits, selected_logits, log_probs
    return token_logps


def generate_group(
    policy,
    processor,
    prompt: str,
    image,
    num_generations: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    device: torch.device,
) -> tuple[dict[str, Any], list[str], float]:
    prompts = [prompt] * num_generations
    images_per_sample = [[image] for _ in range(num_generations)]
    prompt_batch = processor.encode_images(
        prompts,
        images_per_sample=images_per_sample,
        padding=True,
    )
    prompt_batch = move_batch_to_device(prompt_batch, device)
    if "input_ids" not in prompt_batch or "attention_mask" not in prompt_batch:
        raise RuntimeError("Processor output must include input_ids and attention_mask")

    tokenizer = processor.tokenizer
    stop_ids = get_stop_ids(tokenizer)
    pad_id = tokenizer.pad_token_id
    generation_pad_id = pad_id
    if generation_pad_id is None:
        generation_pad_id = tokenizer.eos_token_id
        if isinstance(generation_pad_id, (list, tuple)):
            generation_pad_id = generation_pad_id[0]

    policy.model.rope_deltas = None
    start_time = time.time()
    with torch.inference_mode(), autocast_bf16(device):
        output_ids = policy.generate(
            **prompt_batch,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            num_beams=1,
            num_return_sequences=1,
            eos_token_id=stop_ids if stop_ids else None,
            pad_token_id=generation_pad_id,
            use_cache=True,
        )
    generation_seconds = time.time() - start_time

    prompt_length = int(prompt_batch["input_ids"].shape[1])
    response_ids = output_ids[:, prompt_length:].contiguous()
    if response_ids.shape[0] != num_generations:
        raise RuntimeError(
            f"Expected {num_generations} completions, got {response_ids.shape[0]}"
        )

    completions = []
    for row in response_ids:
        text = tokenizer.decode(
            trim_for_reward(row, stop_ids, pad_id),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ).strip()
        completions.append(text)

    response_mask = make_response_mask(response_ids, stop_ids, pad_id)
    full_batch, actual_prompt_length = make_full_batch(
        prompt_batch, response_ids, response_mask
    )
    if actual_prompt_length != prompt_length:
        raise RuntimeError("Prompt length changed during response batch construction")

    return {
        "prompt_batch": prompt_batch,
        "response_ids": response_ids,
        "response_mask": response_mask,
        "full_batch": full_batch,
        "prompt_length": prompt_length,
        "stop_ids": stop_ids,
        "pad_id": pad_id,
    }, completions, generation_seconds


# ---------------------------------------------------------------------------
# One GRPO update
# ---------------------------------------------------------------------------


def reduce_any_signal(local_signal: bool, device: torch.device, world_size: int) -> bool:
    signal = torch.tensor(1 if local_signal else 0, dtype=torch.int32, device=device)
    if world_size > 1:
        dist.all_reduce(signal, op=dist.ReduceOp.MAX)
    return bool(signal.item())


def process_sample(
    *,
    sample: dict[str, Any],
    step: int,
    policy,
    train_forward_model,
    reference_model,
    processor,
    optimizer,
    trainable_params: list[torch.nn.Parameter],
    args,
    out_dir: Path,
    device: torch.device,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    image_dir = out_dir / "images" / f"rank_{rank:03d}"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"sample_{step:06d}.png"
    sample["image"].save(image_path, format="PNG")

    prompt = build_thyme_prompt(processor, sample, image_path)
    group_batch, completions, generation_seconds = generate_group(
        policy=policy,
        processor=processor,
        prompt=prompt,
        image=sample["image"],
        num_generations=args.num_generations,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        device=device,
    )

    reward_rows = [
        compute_rewards(text, sample["solution"], args.format_weight)
        for text in completions
    ]
    rewards = torch.tensor(
        [item["total_reward"] for item in reward_rows],
        dtype=torch.float32,
        device=device,
    )
    reward_mean = rewards.mean()
    reward_std = rewards.std(unbiased=False)
    if float(reward_std.detach().item()) > args.reward_std_epsilon:
        advantages = (rewards - reward_mean) / (reward_std + args.advantage_epsilon)
    else:
        advantages = torch.zeros_like(rewards)
    advantages = advantages.detach()

    full_batch = group_batch["full_batch"]
    response_ids = group_batch["response_ids"]
    response_mask = group_batch["response_mask"]
    prompt_length = group_batch["prompt_length"]

    # Recompute policy log-probabilities with gradients. The policy has not
    # changed since rollout generation, so detached current logps are the
    # on-policy old logps for this single update.
    current_logps = compute_response_logps(
        train_forward_model,
        full_batch,
        response_ids,
        response_mask,
        prompt_length,
        device,
    )
    old_logps = current_logps.detach()
    log_ratio = (current_logps - old_logps).clamp(-20.0, 20.0)
    ratio = torch.exp(log_ratio)
    clipped_ratio = ratio.clamp(1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon)
    adv_expanded = advantages.unsqueeze(1)
    surrogate_1 = ratio * adv_expanded
    surrogate_2 = clipped_ratio * adv_expanded
    surrogate = torch.minimum(surrogate_1, surrogate_2)
    mask_float = response_mask.to(surrogate.dtype)
    token_denominator = mask_float.sum().clamp(min=1.0)
    policy_loss = -(surrogate * mask_float).sum() / token_denominator

    if reference_model is not None and args.kl_beta > 0.0:
        with torch.no_grad():
            reference_logps = compute_response_logps(
                reference_model,
                full_batch,
                response_ids,
                response_mask,
                prompt_length,
                device,
            )
        # Non-negative k3-style estimate on sampled response tokens.
        log_ratio_ref = (reference_logps - current_logps).clamp(-20.0, 20.0)
        kl_per_token = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0
        kl_loss = (kl_per_token * mask_float).sum() / token_denominator
    else:
        reference_logps = None
        kl_loss = current_logps.new_zeros(())

    loss = policy_loss + args.kl_beta * kl_loss
    if not torch.isfinite(loss.detach()):
        raise RuntimeError(f"Non-finite GRPO loss at step {step}: {loss.detach().item()}")

    group_has_reward_signal = float(reward_std.detach().item()) > args.reward_std_epsilon
    group_has_kl_signal = args.kl_beta > 0.0 and float(kl_loss.detach().item()) > args.kl_signal_epsilon
    global_has_signal = reduce_any_signal(
        group_has_reward_signal or group_has_kl_signal,
        device,
        world_size,
    )

    grad_norm_value = 0.0
    max_param_delta = 0.0
    update_applied = False
    optimizer.zero_grad(set_to_none=True)

    # Use a small view for update diagnostics; do not clone a huge embedding matrix.
    probe_param = trainable_params[0]
    probe_before = probe_param.detach().reshape(-1)[: min(1024, probe_param.numel())].float().clone()

    if global_has_signal:
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_params,
            args.max_grad_norm,
        )
        grad_norm_value = float(grad_norm.detach().float().item())
        if not math.isfinite(grad_norm_value):
            raise RuntimeError(f"Non-finite gradient norm at step {step}: {grad_norm_value}")
        if grad_norm_value > 0.0:
            optimizer.step()
            update_applied = True
            probe_after = probe_param.detach().reshape(-1)[: probe_before.numel()].float()
            max_param_delta = float((probe_after - probe_before).abs().max().item())
        else:
            # Do not apply AdamW weight decay when backward produced no gradient.
            optimizer.zero_grad(set_to_none=True)
    else:
        # DDP must complete a backward pass for every DDP forward. Backpropagate
        # the graph-connected zero loss on all ranks, but do not step the optimizer.
        if world_size > 1:
            loss.backward()
        # Crucial: do not let AdamW weight decay change parameters when the
        # group has no reward variation and KL contributes no meaningful signal.
        optimizer.zero_grad(set_to_none=True)

    accuracy_rewards = [row["accuracy_reward"] for row in reward_rows]
    format_rewards = [row["format_reward"] for row in reward_rows]
    record = {
        "step": int(step),
        "rank": int(rank),
        "question": sample["question"],
        "solution": sample["solution"],
        "image_path": str(image_path),
        "image_size": [int(sample["image"].width), int(sample["image"].height)],
        "prompt": prompt,
        "completions": completions,
        "extracted_answers": [row["extracted_answer"] for row in reward_rows],
        "accuracy_rewards": accuracy_rewards,
        "format_rewards": format_rewards,
        "rewards": [float(x) for x in rewards.detach().cpu().tolist()],
        "reward_mean": float(reward_mean.detach().item()),
        "group_std": float(reward_std.detach().item()),
        "advantages": [float(x) for x in advantages.detach().cpu().tolist()],
        "accuracy_match_methods": [row["accuracy_match_method"] for row in reward_rows],
        "accuracy_matches": [row["accuracy_match"] for row in reward_rows],
        "policy_loss": float(policy_loss.detach().item()),
        "kl_loss": float(kl_loss.detach().item()),
        "loss": float(loss.detach().item()),
        "grad_norm": grad_norm_value,
        "max_param_delta_probe": max_param_delta,
        "update_applied": bool(update_applied),
        "generation_seconds": float(generation_seconds),
        "generated_tokens": [int(x) for x in group_batch["response_mask"].sum(dim=1).tolist()],
        "unique_completions": len(set(normalize_text(x) for x in completions)),
        "zero_var": bool(float(reward_std.detach().item()) <= args.reward_std_epsilon),
    }

    if is_main_process(rank):
        print("\n" + "=" * 80, flush=True)
        print(f"STEP {step} | question={sample['question']}", flush=True)
        print("solution:", sample["solution"], flush=True)
        print("reward vector:", record["rewards"], flush=True)
        print("accuracy vector:", accuracy_rewards, flush=True)
        print("format vector:", format_rewards, flush=True)
        print(
            "metrics: "
            f"reward={record['reward_mean']:.4f} "
            f"group_std={record['group_std']:.6f} "
            f"loss={record['loss']:.6f} "
            f"policy_loss={record['policy_loss']:.6f} "
            f"kl={record['kl_loss']:.6f} "
            f"grad={record['grad_norm']:.6f} "
            f"param_delta={record['max_param_delta_probe']:.3e} "
            f"zero_var={int(record['zero_var'])} "
            f"update={int(record['update_applied'])}",
            flush=True,
        )
        for index, (completion, reward_row) in enumerate(zip(completions, reward_rows)):
            print(
                f"\n[rollout {index}] total={reward_row['total_reward']:.3f} "
                f"acc={reward_row['accuracy_reward']:.1f} "
                f"fmt={reward_row['format_reward']:.1f} "
                f"method={reward_row['accuracy_match_method']} "
                f"answer={reward_row['extracted_answer']!r}",
                flush=True,
            )
            print(repr(completion[:2000]), flush=True)
        print("=" * 80, flush=True)

    # Drop references to large forward tensors before the next sample.
    del current_logps, old_logps, ratio, clipped_ratio, surrogate_1, surrogate_2, surrogate
    if reference_logps is not None:
        del reference_logps
    del full_batch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


# ---------------------------------------------------------------------------
# Checkpointing and CLI
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    policy,
    optimizer,
    step: int,
    optimizer_step: int,
    args,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_state = core_model(policy).state_dict()
    payload = {
        # Runtime loader unwraps this key and loads only the model parameters.
        # Do not embed AdamW moments here: they can add several GB to the file.
        "model": model_state,
        "global_step": int(step),
        "optimizer_step": int(optimizer_step),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage-4 Thyme-RL GRPO trainer")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/dpo_mmrlhf_v3/last_model.pt"))
    parser.add_argument("--out", type=Path, default=Path("runs/thyme_grpo_v1"))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--kl_beta", type=float, default=0.02)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--format_weight", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--shuffle_buffer", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--reward_std_epsilon", type=float, default=1e-6)
    parser.add_argument("--advantage_epsilon", type=float, default=1e-4)
    parser.add_argument("--kl_signal_epsilon", type=float, default=1e-8)
    args = parser.parse_args()

    if args.max_steps < 1:
        parser.error("--max_steps must be >= 1")
    if args.num_generations < 2:
        parser.error("--num_generations must be >= 2 for GRPO")
    if args.max_new_tokens < 1:
        parser.error("--max_new_tokens must be >= 1")
    if args.learning_rate <= 0:
        parser.error("--learning_rate must be > 0")
    if args.weight_decay < 0:
        parser.error("--weight_decay must be >= 0")
    if args.kl_beta < 0:
        parser.error("--kl_beta must be >= 0")
    if not (0.0 < args.top_p <= 1.0):
        parser.error("--top_p must be in (0, 1]")
    if args.top_k < 0:
        parser.error("--top_k must be >= 0")
    if args.temperature <= 0:
        parser.error("--temperature must be > 0")
    if args.repetition_penalty <= 0:
        parser.error("--repetition_penalty must be > 0")
    if args.save_every < 1 or args.log_every < 1:
        parser.error("--save_every and --log_every must be >= 1")
    if args.shuffle_buffer < 1:
        parser.error("--shuffle_buffer must be >= 1")
    return args


def main():
    args = parse_args()
    rank, local_rank, world_size, device = setup_distributed()
    set_seed(args.seed, rank)

    out_dir = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else PROJECT_ROOT / args.checkpoint
    out_dir.mkdir(parents=True, exist_ok=True)

    if is_main_process(rank):
        print("=" * 80, flush=True)
        print("STAGE-4 THYME-RL GRPO", flush=True)
        print("Project root:", PROJECT_ROOT, flush=True)
        print("Checkpoint:", checkpoint_path, flush=True)
        print("Output:", out_dir, flush=True)
        print("Device:", device, "| world_size:", world_size, flush=True)
        print("GPU:", torch.cuda.get_device_name(device), flush=True)
        print("max_steps:", args.max_steps, flush=True)
        print("num_generations:", args.num_generations, flush=True)
        print("max_new_tokens:", args.max_new_tokens, flush=True)
        print("learning_rate:", args.learning_rate, flush=True)
        print("kl_beta:", args.kl_beta, flush=True)
        print("math_verify:", HAS_MATH_VERIFY, flush=True)
        print("=" * 80, flush=True)

    # Trainable model has FP32 master weights for non-ViT parameters.
    policy, processor = load_dpo_v2_policy(
        checkpoint_path=checkpoint_path,
        freeze_visual=True,
        trainable=True,
        fp32_trainable=True,
    )
    policy.to(device)
    policy.eval()

    # Optional frozen DPO reference policy for KL regularization.
    reference_model = None
    if args.kl_beta > 0.0:
        if is_main_process(rank):
            print("Loading frozen DPO reference policy for KL...", flush=True)
        reference_model, _ = load_dpo_v2_policy(
            checkpoint_path=checkpoint_path,
            freeze_visual=True,
            trainable=False,
            fp32_trainable=False,
        )
        reference_model.to(device)
        reference_model.eval()
        reference_model.requires_grad_(False)

    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    if world_size > 1:
        train_forward_model = DDP(
            policy,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    else:
        train_forward_model = policy

    dataset = iter(
        ThymeRLStream(
            split=args.split,
            revision=args.revision,
            shuffle=args.shuffle,
            seed=args.seed,
            shuffle_buffer=args.shuffle_buffer,
        )
    )

    # Deterministically shard the same stream across ranks. Each rank skips its
    # offset and then consumes one item out of every world_size items.
    for _ in range(args.skip + rank):
        try:
            next(dataset)
        except StopIteration:
            raise RuntimeError("Thyme stream ended before the requested --skip offset")

    records_path = out_dir / "rollouts_rank0.jsonl"
    metrics_path = out_dir / "metrics_rank0.jsonl"
    config_path = out_dir / "config.json"
    if is_main_process(rank):
        save_json(config_path, {
            "dataset": "Kwai-Keye/Thyme-RL",
            "split": args.split,
            "checkpoint": str(checkpoint_path),
            "world_size": world_size,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "num_generations": args.num_generations,
            "max_new_tokens": args.max_new_tokens,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "kl_beta": args.kl_beta,
            "clip_epsilon": args.clip_epsilon,
            "format_weight": args.format_weight,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "fp32_trainable_master_weights": True,
            "math_verify_available": HAS_MATH_VERIFY,
        })

    records_fp = records_path.open("w", encoding="utf-8") if is_main_process(rank) else None
    metrics_fp = metrics_path.open("w", encoding="utf-8") if is_main_process(rank) else None
    reward_history: list[float] = []
    std_history: list[float] = []
    update_count = 0
    processed_steps = 0

    try:
        for step in range(1, args.max_steps + 1):
            try:
                sample = next(dataset)
            except StopIteration:
                if is_main_process(rank):
                    print("Thyme stream exhausted; stopping early.", flush=True)
                break

            # Consume the other ranks' records so ranks do not train on the same sample.
            exhausted = False
            for _ in range(max(0, world_size - 1)):
                try:
                    next(dataset)
                except StopIteration:
                    exhausted = True
                    break

            policy.eval()  # no dropout: rollout and differentiable logps must agree
            record = process_sample(
                sample=sample,
                step=step,
                policy=policy,
                train_forward_model=train_forward_model,
                reference_model=reference_model,
                processor=processor,
                optimizer=optimizer,
                trainable_params=trainable_params,
                args=args,
                out_dir=out_dir,
                device=device,
                rank=rank,
                world_size=world_size,
            )
            processed_steps = step
            update_count += int(record["update_applied"])
            reward_history.append(float(record["reward_mean"]))
            std_history.append(float(record["group_std"]))

            if is_main_process(rank):
                append_jsonl(records_fp, record)
                append_jsonl(metrics_fp, {
                    "step": step,
                    "reward": record["reward_mean"],
                    "group_std": record["group_std"],
                    "zero_var": record["zero_var"],
                    "loss": record["loss"],
                    "policy_loss": record["policy_loss"],
                    "kl_loss": record["kl_loss"],
                    "grad": record["grad_norm"],
                    "param_delta": record["max_param_delta_probe"],
                    "update_applied": record["update_applied"],
                    "accuracy_reward_mean": mean(record["accuracy_rewards"]),
                    "format_reward_mean": mean(record["format_rewards"]),
                })

            if step % args.save_every == 0 or step == args.max_steps or exhausted:
                if is_main_process(rank):
                    save_checkpoint(
                        out_dir / "last_model.pt",
                        policy,
                        optimizer,
                        step,
                        update_count,
                        args,
                    )
                    print(f"Saved last_model.pt at step {step}; optimizer updates={update_count}", flush=True)
                if world_size > 1:
                    dist.barrier()

            if is_main_process(rank) and step % args.log_every == 0:
                print(
                    f"[SUMMARY] step={step} reward={record['reward_mean']:.4f} "
                    f"group_std={record['group_std']:.6f} "
                    f"zero_var={int(record['zero_var'])} "
                    f"loss={record['loss']:.6f} grad={record['grad_norm']:.6f} "
                    f"updates={update_count}",
                    flush=True,
                )
            if exhausted:
                break

        if is_main_process(rank):
            summary = {
                "processed_steps": processed_steps,
                "optimizer_updates": update_count,
                "mean_reward_rank0": mean(reward_history) if reward_history else 0.0,
                "mean_group_std_rank0": mean(std_history) if std_history else 0.0,
                "zero_variance_steps_rank0": sum(x <= args.reward_std_epsilon for x in std_history),
                "math_verify_available": HAS_MATH_VERIFY,
                "checkpoint": str(out_dir / "last_model.pt"),
            }
            save_json(out_dir / "summary.json", summary)
            print("=" * 80, flush=True)
            print("STAGE-4 GRPO FINISHED", flush=True)
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
            print("=" * 80, flush=True)
    finally:
        if records_fp is not None:
            records_fp.close()
        if metrics_fp is not None:
            metrics_fp.close()
        cleanup_distributed()


if __name__ == "__main__":
    main()
