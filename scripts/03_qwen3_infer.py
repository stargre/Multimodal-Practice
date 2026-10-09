# scripts/03_qwen3_infer.py

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


MODEL_PATH = "./checkpoints/Qwen3-0.6B-Base"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model.eval()

text = "The capital of China is"

inputs = tokenizer(
    text,
    return_tensors="pt"
).to(model.device)

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=32,
    )

print(tokenizer.decode(
    output[0],
    skip_special_tokens=True
))