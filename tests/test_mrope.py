import torch

from transformers import AutoModelForCausalLM

from models.qwen25_fork.qwen3_mrope import (
    Qwen3MRoPEModel,
)


PATH = "./checkpoints/Qwen3-0.6B-Base"

qwen3 = AutoModelForCausalLM.from_pretrained(
    PATH,
    dtype=torch.float32,
    attn_implementation="sdpa",
).eval()

# 重新创建 Qwen3 Model，但使用新的 MRoPE
new_model = Qwen3MRoPEModel(
    qwen3.config
).eval()

# 将原始 Qwen3 backbone 权重严格加载
new_model.load_state_dict(
    qwen3.model.state_dict(),
    strict=True,
)

# 构造纯文本输入
input_ids = torch.tensor([
    [100, 200, 300, 400, 500]
])

position_ids = torch.arange(5).view(1, 5)

mrope_position_ids = position_ids.unsqueeze(0).expand(
    3, -1, -1
)

with torch.no_grad():

    original_outputs = qwen3.model(
        input_ids=input_ids,
        use_cache=False,
    )

    new_outputs = new_model(
        input_ids=input_ids,
        position_ids=mrope_position_ids,
        use_cache=False,
    )

torch.testing.assert_close(
    original_outputs.last_hidden_state,
    new_outputs.last_hidden_state,
    rtol=1e-5,
    atol=1e-5,
)

position_ids_3d = torch.stack([
    torch.tensor([[0, 0, 0, 0, 0]]),
    torch.tensor([[0, 0, 1, 1, 2]]),
    torch.tensor([[0, 1, 0, 1, 0]]),
])

with torch.no_grad():
    outputs = new_model(
        input_ids=input_ids,
        position_ids=position_ids_3d,
        use_cache=False,
    )

assert outputs.last_hidden_state.shape == (1, 5, 1024)

print("QWEN3 MROPE FORWARD PASS")

print("QWEN3 BACKBONE EQUIVALENCE PASS")