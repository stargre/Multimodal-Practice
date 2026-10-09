import torch

from PIL import Image

from transformers import CLIPImageProcessor

from models.qwen25_fork.build_model import (
    build_model,
    VIT_PATH,
)


device = "cuda"

model, tokenizer = build_model(device)

processor = CLIPImageProcessor.from_pretrained(
    VIT_PATH
)

image = Image.open(
    "./assets/test.jpg"
).convert("RGB")

pixel_values = processor(
    images=image,
    return_tensors="pt",
).pixel_values.to(
    device=device,
    dtype=torch.bfloat16,
)

# 保留 Qwen2.5-VL 视觉网格表示
image_grid_thw = torch.tensor(
    [[1, 32, 32]],
    dtype=torch.long,
    device=device,
)

prompt = (
    "<|im_start|>user\n"
    "<|vision_start|>"
    + "<|image_pad|>" * 256
    + "<|vision_end|>\n"
    + "请描述这张图片。"
    + "<|im_end|>\n"
    + "<|im_start|>assistant\n"
)

inputs = tokenizer(
    prompt,
    return_tensors="pt",
    add_special_tokens=False,
).to(device)

# ========== 检查 Qwen2.5-VL MRoPE ==========
position_ids, rope_deltas = model.model.get_rope_index(
    input_ids=inputs["input_ids"],
    image_grid_thw=image_grid_thw,
    attention_mask=inputs["attention_mask"],
)

print("position_ids:", position_ids.shape)
print("rope_deltas:", rope_deltas.shape)

assert position_ids.shape == (
    3, 1, inputs["input_ids"].shape[1]
)

# ========== 完整 Forward ==========
with torch.no_grad():

    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        use_cache=False,
    )

print("logits:", outputs.logits.shape)

assert outputs.logits.shape[-1] == 151936
assert torch.isfinite(outputs.logits).all().item()

print("HYBRID QWEN25 FORWARD PASS")