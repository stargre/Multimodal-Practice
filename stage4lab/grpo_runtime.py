#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime loader for the custom Stage-1 HybridQwen25VL + Stage-3 DPO policy."""

from pathlib import Path
import sys
from typing import Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stage1_runtime import load_stage1_components  # noqa: E402

DEFAULT_DPO_CHECKPOINT = PROJECT_ROOT / "runs" / "dpo_mmrlhf_v3" / "last_model.pt"
VISUAL_VIT_PREFIXES = ("model.visual.vit.", "visual.vit.")


def load_checkpoint_state(checkpoint_path: Path) -> dict:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"DPO checkpoint not found: {checkpoint_path}")

    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint must contain a state-dict-like dictionary")

    # Support plain state_dict and common training checkpoint wrappers.
    for key in ("state_dict", "model_state_dict", "model", "policy_model"):
        candidate = state.get(key)
        if isinstance(candidate, dict) and candidate and all(
            isinstance(k, str) for k in candidate.keys()
        ):
            state = candidate
            print(f"Unwrapped checkpoint key: {key}", flush=True)
            break

    if not isinstance(state, dict) or not state:
        raise RuntimeError("Checkpoint does not contain model parameters")

    keys = list(state.keys())
    if keys and all(key.startswith("module.") for key in keys):
        state = {key[len("module."):]: value for key, value in state.items()}
        print("Removed DDP 'module.' prefix", flush=True)

    print(f"Checkpoint tensors: {len(state)}", flush=True)
    return state


def _is_visual_vit(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in VISUAL_VIT_PREFIXES) or (
        "visual.vit." in name
    )


def load_dpo_policy(
    checkpoint_path: Optional[str | Path] = None,
    *,
    freeze_visual: bool = True,
    trainable: bool = False,
    fp32_trainable: bool = False,
):
    """Load Stage-1 architecture plus DPO weights.

    If ``trainable`` is true, every non-frozen-ViT parameter remains trainable.
    When ``fp32_trainable`` is true, trainable parameters are FP32 master weights;
    the training script should use BF16 autocast for forward passes.
    """
    if checkpoint_path is None:
        checkpoint_path = DEFAULT_DPO_CHECKPOINT
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    print("=" * 80, flush=True)
    print("Loading Stage-1 base model...", flush=True)
    model, processor = load_stage1_components()

    # Set requires_grad and optional FP32 master weights before loading checkpoint
    # so loading preserves FP32 trainable parameter dtype.
    for name, param in model.named_parameters():
        is_vit = _is_visual_vit(name)
        should_train = bool(trainable and not (freeze_visual and is_vit))
        param.requires_grad_(should_train)
        if should_train and fp32_trainable and param.dtype != torch.float32:
            param.data = param.data.float()

    print("=" * 80, flush=True)
    print("Loading DPO checkpoint...", flush=True)
    print("Checkpoint:", checkpoint_path, flush=True)
    state = load_checkpoint_state(checkpoint_path)
    incompatible = model.load_state_dict(state, strict=False)

    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)
    allowed_missing = [key for key in missing_keys if _is_visual_vit(key)]
    forbidden_missing = [key for key in missing_keys if key not in allowed_missing]

    print(f"Missing keys: {len(missing_keys)}", flush=True)
    print(f"Allowed missing keys (frozen InternViT): {len(allowed_missing)}", flush=True)
    print(f"Forbidden missing keys: {len(forbidden_missing)}", flush=True)
    print(f"Unexpected keys: {len(unexpected_keys)}", flush=True)

    if forbidden_missing:
        for key in forbidden_missing[:50]:
            print("[MISSING]", key, flush=True)
        raise RuntimeError(
            f"DPO checkpoint is missing {len(forbidden_missing)} non-ViT parameters"
        )
    if unexpected_keys:
        for key in unexpected_keys[:50]:
            print("[UNEXPECTED]", key, flush=True)
        raise RuntimeError(
            f"DPO checkpoint contains {len(unexpected_keys)} unexpected parameters"
        )

    # Re-assert trainability after loading and report parameter dtype/counts.
    total_params = 0
    trainable_params = 0
    fp32_trainable_params = 0
    for name, param in model.named_parameters():
        is_vit = _is_visual_vit(name)
        should_train = bool(trainable and not (freeze_visual and is_vit))
        param.requires_grad_(should_train)
        total_params += param.numel()
        if should_train:
            trainable_params += param.numel()
            if param.dtype == torch.float32:
                fp32_trainable_params += param.numel()

    model.model.rope_deltas = None
    model.eval()  # eval() does not disable gradients; it avoids dropout mismatch.

    print("DPO checkpoint load: PASS", flush=True)
    print("Trainable params:", f"{trainable_params:,}", flush=True)
    print("Total params:", f"{total_params:,}", flush=True)
    print("Trainable ratio:", f"{trainable_params / max(total_params, 1):.6f}", flush=True)
    print("FP32 trainable params:", f"{fp32_trainable_params:,}", flush=True)
    print("Policy ready.", flush=True)
    print("=" * 80, flush=True)
    return model, processor


# Backward-compatible name used by the existing Stage-4 smoke/rollout scripts.
def load_dpo_v2_policy(
    checkpoint_path: Optional[str | Path] = None,
    freeze_visual: bool = True,
    trainable: bool = False,
    fp32_trainable: bool = False,
):
    return load_dpo_policy(
        checkpoint_path=checkpoint_path,
        freeze_visual=freeze_visual,
        trainable=trainable,
        fp32_trainable=fp32_trainable,
    )
