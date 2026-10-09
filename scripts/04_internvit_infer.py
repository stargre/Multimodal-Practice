import torch
from PIL import Image

from transformers import (
    AutoModel,
    CLIPImageProcessor,
)


MODEL_PATH = (
    "./checkpoints/"
    "InternViT-300M-448px-V2_5"
)

IMAGE_PATH = "./assets/test.jpg"


vision_model = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).cuda().eval()

image_processor = CLIPImageProcessor.from_pretrained(
    MODEL_PATH
)

image = Image.open(
    IMAGE_PATH
).convert("RGB")

pixel_values = image_processor(
    images=image,
    return_tensors="pt",
).pixel_values

pixel_values = pixel_values.to(
    device="cuda",
    dtype=torch.bfloat16,
)

with torch.inference_mode():
    outputs = vision_model(pixel_values)

print(outputs.last_hidden_state.shape)