#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Streaming Thyme-RL adapter for the custom Qwen3 + InternViT model."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from datasets import load_dataset
from PIL import Image


THYME_DATASET = "Kwai-Keye/Thyme-RL"


def decode_base64_image(image_string: str) -> Image.Image:
    """Decode a Thyme base64 image and return a fully materialized RGB PIL image."""
    if not isinstance(image_string, str):
        raise TypeError(f"Expected image string, got {type(image_string)!r}")

    if image_string.startswith("data:"):
        if "," not in image_string:
            raise ValueError("Malformed data URI: missing comma separator")
        image_string = image_string.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(image_string, validate=True)
    except Exception as exc:
        raise ValueError(f"Failed to decode base64 image: {exc}") from exc

    try:
        with Image.open(io.BytesIO(image_bytes)) as opened:
            opened.load()
            image = opened.convert("RGB")
    except Exception as exc:
        raise ValueError(f"Failed to decode image bytes: {exc}") from exc

    return image


def parse_messages(messages: Dict[str, Any]) -> list[dict[str, Any]]:
    """Convert Thyme's columnar messages object into a list of role/content dicts."""
    if not isinstance(messages, dict):
        raise TypeError(f"Expected messages dict, got {type(messages)!r}")

    roles = messages.get("role")
    contents = messages.get("content")
    if not isinstance(roles, list):
        raise TypeError("messages['role'] must be a list")
    if not isinstance(contents, list):
        raise TypeError("messages['content'] must be a list")
    if len(roles) != len(contents):
        raise ValueError(
            f"messages role/content length mismatch: {len(roles)} vs {len(contents)}"
        )

    return [{"role": role, "content": content} for role, content in zip(roles, contents)]


def extract_system_and_user(messages: Dict[str, Any]) -> tuple[str, str]:
    parsed = parse_messages(messages)
    system_prompt = None
    user_prompt = None

    for message in parsed:
        role = message["role"]
        content = message["content"]
        if role == "system":
            system_prompt = content
        elif role == "user":
            user_prompt = content

    if system_prompt is None:
        raise ValueError("Thyme sample does not contain a system message")
    if user_prompt is None:
        raise ValueError("Thyme sample does not contain a user message")
    if not isinstance(system_prompt, str) or not isinstance(user_prompt, str):
        raise TypeError("Thyme system/user message content must be strings")

    return system_prompt, user_prompt


def load_thyme_stream(
    split: str = "train",
    revision: Optional[str] = None,
    shuffle: bool = False,
    seed: int = 42,
    shuffle_buffer: int = 1000,
):
    """Load Thyme-RL in streaming mode, optionally using HF iterable shuffle."""
    kwargs: dict[str, Any] = {"split": split, "streaming": True}
    if revision is not None:
        kwargs["revision"] = revision

    dataset = load_dataset(THYME_DATASET, **kwargs)
    if shuffle:
        if shuffle_buffer < 1:
            raise ValueError("shuffle_buffer must be >= 1")
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
    return dataset


class ThymeRLStream:
    """Lightweight wrapper that converts raw streamed samples into training samples."""

    def __init__(
        self,
        split: str = "train",
        revision: Optional[str] = None,
        shuffle: bool = False,
        seed: int = 42,
        shuffle_buffer: int = 1000,
    ):
        self.dataset = load_thyme_stream(
            split=split,
            revision=revision,
            shuffle=shuffle,
            seed=seed,
            shuffle_buffer=shuffle_buffer,
        )

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for index, sample in enumerate(self.dataset):
            yield self.convert_sample(sample, index=index)

    @staticmethod
    def convert_sample(
        sample: Dict[str, Any],
        index: Optional[int] = None,
    ) -> Dict[str, Any]:
        required = ("messages", "images", "solution", "question")
        missing = [key for key in required if key not in sample]
        if missing:
            raise KeyError(f"Missing required Thyme fields: {missing}")

        parsed_messages = parse_messages(sample["messages"])
        system_prompt = None
        user_prompt = None
        for message in parsed_messages:
            if message["role"] == "system":
                system_prompt = message["content"]
            elif message["role"] == "user":
                user_prompt = message["content"]

        if system_prompt is None or user_prompt is None:
            raise ValueError("Thyme sample must contain system and user messages")
        if not isinstance(system_prompt, str) or not isinstance(user_prompt, str):
            raise TypeError("Thyme system/user message content must be strings")

        images = sample["images"]
        if not isinstance(images, list):
            raise TypeError(f"Expected images list, got {type(images)!r}")
        if not images:
            raise ValueError("Thyme sample contains no image")
        if len(images) != 1:
            raise ValueError(
                f"This Stage-4 trainer expects one image per sample; got {len(images)}"
            )

        image = decode_base64_image(images[0])
        return {
            "index": index,
            "messages": parsed_messages,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "question": str(sample["question"]),
            "solution": str(sample["solution"]),
            "image": image,
        }


def get_first_sample() -> Dict[str, Any]:
    return next(iter(ThymeRLStream(split="train")))


if __name__ == "__main__":
    sample = get_first_sample()
    print("=" * 80)
    print("Thyme-RL adapter test")
    print("index:", sample["index"])
    print("question:", repr(sample["question"]))
    print("solution:", repr(sample["solution"]))
    print("system_prompt length:", len(sample["system_prompt"]))
    print("user_prompt length:", len(sample["user_prompt"]))
    print("image type:", type(sample["image"]))
    print("image mode:", sample["image"].mode)
    print("image size:", sample["image"].size)
    print("messages:")
    for message in sample["messages"]:
        print(f"  [{message['role']}] {message['content'][:300]!r}")
    print("PASS")
