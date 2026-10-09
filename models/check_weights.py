import torch

from pathlib import Path
from safetensors import safe_open
from transformers import AutoModel


VIT_PATH = Path(
    "./checkpoints/InternViT-300M-448px-V2_5"
)

# 不使用任何 Monkey Patch
model = AutoModel.from_pretrained(
    str(VIT_PATH),
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
)

# 选择 ViT 的 Patch Embedding 权重进行核对
key = "embeddings.patch_embedding.weight"

with safe_open(
    str(VIT_PATH / "model.safetensors"),
    framework="pt",
    device="cpu",
) as f:
    checkpoint_weight = f.get_tensor(key)

loaded_weight = (
    model.state_dict()[key].detach().cpu()
)

# checkpoint 可能原本是不同 dtype
checkpoint_weight = checkpoint_weight.to(
    loaded_weight.dtype
)

print("Checkpoint:", checkpoint_weight.shape)
print("Loaded:", loaded_weight.shape)

assert torch.equal(
    checkpoint_weight,
    loaded_weight,
), "InternViT 权重与原始 checkpoint 不一致"

print("WEIGHT LOAD PASS")