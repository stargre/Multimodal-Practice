import torch

from PIL import Image

from transformers import (
    AutoModel,
    CLIPImageProcessor,
)

from models.qwen25_fork.visual_intern import (
    InternViTQwenVisual,
)


VIT_PATH = (
    "./checkpoints/InternViT-300M-448px-V2_5"
)

device = "cuda"

vit = AutoModel.from_pretrained(
    VIT_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).to(device).eval()

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

visual = InternViTQwenVisual(
    internvit=vit,
    llm_hidden_size=1024,
).to(device).eval()

image_grid_thw = torch.tensor(
    [[1, 32, 32]],
    device=device,
    dtype=torch.long,
)

with torch.no_grad():
    image_embeds = visual(
        pixel_values,
        grid_thw=image_grid_thw,
    )

print("image_embeds:", image_embeds.shape)

assert image_embeds.shape == (256, 1024)
assert torch.isfinite(image_embeds).all().item()

print("QWEN VISUAL ADAPTER PASS")