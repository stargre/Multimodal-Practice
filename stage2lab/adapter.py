"""PT adapter using the exact image-processing and tokenization path in test-all.py.

Stage 1 verified:
  prompt = processor.user_prompt(question, num_images=1)
  one = processor.encode_images([prompt], [[image]])
  answer_ids = tokenizer(answer + '<|im_end|>', add_special_tokens=False)
  train_ids = cat([one['input_ids'], answer_ids], dim=1)
  loss labels = -100 for prefix, target IDs for assistant suffix

Using this route avoids applying an incompatible generic HF chat processor.
"""

from PIL import Image, ImageOps
import torch


def load_stage1_model_and_processor():
    from stage1_runtime import load_stage1_components
    model, processor = load_stage1_components()
    assert hasattr(model.model.visual, "merger"), "Expected model.model.visual.merger"
    assert hasattr(processor, "user_prompt")
    assert hasattr(processor, "encode_images")
    assert hasattr(processor, "tokenizer")
    return model, processor


def encode_pt_example(processor, sample):
    """Encode one real LLaVA PT row; BS=1; answer-only supervised labels."""
    with Image.open(sample["image_path"]) as f:
        image = ImageOps.exif_transpose(f).convert("RGB")

    tokenizer = processor.tokenizer
    question = sample["question"].strip()
    answer = sample["answer"].strip()
    if not question or not answer:
        raise ValueError(f"Empty question/answer for {sample.get('id')}")

    # Exact Stage-1 interface that already passed single-image MRoPE and grad tests.
    prompt = processor.user_prompt(question, num_images=1)
    prefix = processor.encode_images([prompt], [[image]])
    needed = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
    for name in needed:
        if name not in prefix:
            raise KeyError(f"HybridQwen25Processor returned no {name}")

    prefix_ids = prefix["input_ids"]
    prefix_mask = prefix["attention_mask"]
    if prefix_ids.ndim != 2 or prefix_ids.shape[0] != 1:
        raise ValueError(f"Expected [1, seq] token IDs, got {prefix_ids.shape}")
    if prefix_mask.shape != prefix_ids.shape:
        raise ValueError("Prefix attention mask shape differs from input IDs")
    if not torch.all(prefix_mask == 1):
        raise ValueError("BS=1 PT debug input unexpectedly contains padding")

    # As in test-all.py::check_training, append assistant tokens only after
    # multimodal image expansion is complete; never tokenize the image placeholders
    # as ordinary text again.
    end_token = "<|im_end|>"
    end_id = tokenizer.convert_tokens_to_ids(end_token)
    if end_id is None or end_id < 0 or end_id == tokenizer.unk_token_id:
        raise ValueError(f"Tokenizer has no valid {end_token} token")
    answer_ids = tokenizer(
        answer + end_token,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"]
    if not answer_ids.numel() or answer_ids[0, -1].item() != end_id:
        raise ValueError("Answer target was not terminated with <|im_end|>")

    input_ids = torch.cat([prefix_ids, answer_ids], dim=1)
    attention_mask = torch.cat(
        [prefix_mask, torch.ones_like(answer_ids)], dim=1
    )
    labels = torch.full_like(input_ids, -100)
    labels[:, prefix_ids.shape[1]:] = answer_ids

    image_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    grid = prefix["image_grid_thw"]
    # Stage 1 uses 2x2 merger: every 2x2 image patch grid gives one LLM token.
    expected_vis_tokens = int((grid.prod(dim=-1) // 4).sum().item())
    actual_vis_tokens = int((prefix_ids == image_id).sum().item())
    if not expected_vis_tokens or actual_vis_tokens != expected_vis_tokens:
        raise ValueError(
            f"Image token mismatch: placeholders={actual_vis_tokens}, "
            f"merged features={expected_vis_tokens}, grid={grid.tolist()}"
        )
    if not torch.equal(labels[:, :prefix_ids.shape[1]],
                       torch.full_like(prefix_ids, -100)):
        raise AssertionError("Prompt/image loss masking failed")
    if (labels != -100).sum().item() != answer_ids.numel():
        raise AssertionError("Expected one loss target for each assistant token")

    result = dict(prefix)
    result.update(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return result
