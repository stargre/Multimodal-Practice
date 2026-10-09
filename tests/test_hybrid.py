
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from hybrid_vlm import HybridVLM
import torch

from transformers import AutoModel, CLIPImageProcessor
from PIL import Image


VIT_PATH = "./checkpoints/InternViT-300M-448px-V2_5"

device = "cuda"

# 加载你已经跑通过的 InternViT
vision_model = AutoModel.from_pretrained(
    VIT_PATH,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to(device)

vision_model.eval()

# 加载对应的图像预处理器
image_processor = CLIPImageProcessor.from_pretrained(
    VIT_PATH
)

# 读取图片
image = Image.open(
    "./assets/test.jpg"
).convert("RGB")

pixel_values = image_processor(
    images=image,
    return_tensors="pt",
).pixel_values

pixel_values = pixel_values.to(
    device=device,
    dtype=torch.bfloat16,
)

# 提取视觉特征
with torch.no_grad():
    outputs = vision_model(
        pixel_values=pixel_values,
        return_dict=True,
    )

features = outputs.last_hidden_state

print("pixel_values:", pixel_values.shape)
print("vision_features:", features.shape)

from hybrid_vlm import VisionProjector


# 去掉 CLS token
patch_features = features[:, 1:, :]

print("patch_features:", patch_features.shape)

# 创建 Projector
projector = VisionProjector(
    vit_dim=1024,
    llm_dim=1024,
).to(device=device, dtype=torch.float32)

# Projector 暂时使用 FP32
projected_features = projector(
    patch_features.float()
)

print("projected_features:", projected_features.shape)

assert projected_features.shape == (1, 256, 1024)

print("PROJECTOR PASS")

LLM_PATH = "./checkpoints/Qwen3-0.6B-Base"

tokenizer = AutoTokenizer.from_pretrained(
    LLM_PATH
)

language_model = AutoModelForCausalLM.from_pretrained(
    LLM_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
).to(device)

language_model.eval()

image_token_id = tokenizer.convert_tokens_to_ids(
    "<|image_pad|>"
)

print("image_token_id:", image_token_id)

print(
    "Qwen3 hidden_size:",
    language_model.config.hidden_size
)

model = HybridVLM(
    vision_model=vision_model,
    language_model=language_model,
    image_token_id=image_token_id,
).to(device)

model.eval()

# 先冻结两侧预训练模型
model.vision_model.requires_grad_(False)
model.language_model.requires_grad_(False)

# 只保留 Projector 可训练
model.projector.requires_grad_(True)

print(model)


num_image_tokens = 256

prompt = (
    "<|im_start|>user\n"
    "<|vision_start|>"
    + "<|image_pad|>" * num_image_tokens
    + "<|vision_end|>\n"
    + "请描述这张图片。\n"
    + "<|im_end|>\n"
    + "<|im_start|>assistant\n"
)

inputs = tokenizer(
    prompt,
    return_tensors="pt",
    add_special_tokens=False,
)

input_ids = inputs["input_ids"].to(device)

attention_mask = inputs["attention_mask"].to(device)

print("input_ids:", input_ids.shape)

mask = input_ids == image_token_id

print("图像占位 Token 数量:", mask.sum().item())

assert mask.sum().item() == 256

text_embeds = model.language_model.get_input_embeddings()(
    input_ids
)

print("text_embeds:", text_embeds.shape)

print(model.language_model.model.embed_tokens)

fused_embeds = model.fuse_embeddings(
    input_ids=input_ids,
    pixel_values=pixel_values,
)

print("fused_embeds:", fused_embeds.shape)

assert fused_embeds.shape == text_embeds.shape

# 验证文本位置没被改变
text_mask = ~mask

assert torch.allclose(
    fused_embeds[text_mask],
    text_embeds[text_mask],
)

# 验证图像占位符位置确实发生了变化
assert not torch.allclose(
    fused_embeds[mask],
    text_embeds[mask],
)

print("FUSION PASS")

with torch.no_grad():

    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
    )

print("logits:", outputs.logits.shape)

assert outputs.logits.shape[0] == 1
assert outputs.logits.shape[1] == input_ids.shape[1]

print("FORWARD PASS")

print("\n========== LOSS & BACKWARD TEST ==========")

# 1. 构造一个测试答案
answer = "这是一张图片。<|im_end|>"

answer_ids = tokenizer(
    answer,
    return_tensors="pt",
    add_special_tokens=False,
)["input_ids"].to(device)

# 2. 拼接 Prompt + Answer
train_input_ids = torch.cat(
    [input_ids, answer_ids],
    dim=1,
)

# 3. 构造 Labels
labels = torch.full_like(
    train_input_ids,
    fill_value=-100,
)

# 只监督答案部分
labels[:, input_ids.shape[1]:] = answer_ids

# 4. Attention Mask
train_attention_mask = torch.ones_like(
    train_input_ids
)

# 5. 保持冻结模块的参数不更新
model.vision_model.requires_grad_(False)
model.language_model.requires_grad_(False)
model.projector.requires_grad_(True)

model.vision_model.eval()
model.language_model.eval()
model.projector.train()

# 6. 清空已有梯度
model.zero_grad(set_to_none=True)

# 7. Forward
# 注意：这里不能使用 torch.no_grad()
outputs = model(
    input_ids=train_input_ids,
    pixel_values=pixel_values,
    attention_mask=train_attention_mask,
    labels=labels,
)

loss = outputs.loss

print("Loss:", loss.item())

assert torch.isfinite(loss).item()

# 8. Backward
loss.backward()

# 9. 检查 Projector 梯度
total_grad = 0.0

for name, param in model.projector.named_parameters():

    if param.grad is not None:
        grad_norm = param.grad.norm().item()

        print(name, "grad_norm =", grad_norm)

        total_grad += grad_norm

assert total_grad > 0, "Projector 没有接收到梯度"

# 10. 检查两侧冻结模型
assert all(
    p.grad is None
    for p in model.vision_model.parameters()
)

assert all(
    p.grad is None
    for p in model.language_model.parameters()
)

print("BACKWARD PASS")

print("\n========== GENERATE TEST ==========")

model.eval()

with torch.inference_mode():

    generated_ids = model.generate(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        max_new_tokens=64,
        eos_token_id=[
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|im_end|>"),
        ],
        pad_token_id=tokenizer.eos_token_id,
    )

print("Generated IDs Shape:", generated_ids.shape)

print(
    "Generated Text:",
    tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    )
)

print("GENERATE PASS")