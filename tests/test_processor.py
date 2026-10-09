from PIL import Image

from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
)

from models.qwen25_fork.processing_hybrid import (
    HybridQwen25Processor,
)


tokenizer = AutoTokenizer.from_pretrained(
    "./checkpoints/Qwen3-0.6B-Base"
)

image_processor = CLIPImageProcessor.from_pretrained(
    "./checkpoints/InternViT-300M-448px-V2_5"
)

processor = HybridQwen25Processor(
    tokenizer=tokenizer,
    image_processor=image_processor,
)

img1 = Image.open("./assets/test.jpg").convert("RGB")

# 第二张模拟高分辨率输入
img2 = img1.resize((896, 896))

prompt = (
    "<|im_start|>user\n"
    "比较这两张图片："
    "<|vision_start|><|image_pad|><|vision_end|>"
    "和"
    "<|vision_start|><|image_pad|><|vision_end|>"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)

inputs = processor.encode_images(
    prompts=[prompt],
    images_per_sample=[[img1, img2]],
)

for name, tensor in inputs.items():
    if tensor is not None:
        print(name, tensor.shape)

image_token_id = tokenizer.convert_tokens_to_ids(
    "<|image_pad|>"
)

expected_count = (
    inputs["image_grid_thw"].prod(-1) // 4
).sum().item()

actual_count = (
    inputs["input_ids"] == image_token_id
).sum().item()

assert actual_count == expected_count

print("MULTI IMAGE PROCESSOR PASS")

import torch

from models.qwen25_fork.build_model import build_model

model, _ = build_model("cuda")

model_inputs = {
    k: (
        v.to("cuda") if v is not None else None
    )
    for k, v in inputs.items()
}

with torch.no_grad():
    outputs = model(
        **model_inputs,
        use_cache=False,
    )

assert torch.isfinite(outputs.logits).all().item()

print("MULTI IMAGE FORWARD PASS")