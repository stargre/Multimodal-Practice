import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionProjector(nn.Module):

    def __init__(self, vit_dim=1024, llm_dim=1024):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.LayerNorm(vit_dim * 4),
            nn.Linear(vit_dim * 4, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, x):
        # x: [B, 1024, 1024]

        B, N, C = x.shape

        H = W = math.isqrt(N)

        assert H * W == N, "视觉 Token 无法构成正方形"
        assert H % 2 == 0, "空间尺寸必须能被 2 整除"

        # [B, 1024, 1024]
        # -> [B, 32, 32, 1024]
        x = x.reshape(B, H, W, C)

        # -> [B, 1024, 32, 32]
        x = x.permute(0, 3, 1, 2).contiguous()

        # -> [B, 4096, 16, 16]
        x = F.pixel_unshuffle(x, downscale_factor=2)

        # -> [B, 256, 4096]
        x = x.flatten(2).transpose(1, 2)

        # -> [B, 256, 1024]
        x = self.mlp(x)

        return x
    
class HybridVLM(nn.Module):

    def __init__(
        self,
        vision_model,
        language_model,
        image_token_id,
    ):
        super().__init__()

        self.vision_model = vision_model

        self.language_model = language_model

        self.image_token_id = image_token_id

        self.projector = VisionProjector(
            vit_dim=vision_model.config.hidden_size,
            llm_dim=language_model.config.hidden_size,
        )
    
    def encode_image(self, pixel_values):

        # 1. InternViT 提取特征
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            return_dict=True,
        )

        image_features = vision_outputs.last_hidden_state

        # 2. 去掉 CLS token
        image_features = image_features[:, 1:, :]

        # 3. 转成 Projector 所需的 dtype
        projector_param = next(self.projector.parameters())

        image_features = image_features.to(
            device=projector_param.device,
            dtype=projector_param.dtype,
        )

        # 4. Pixel Unshuffle + Projector
        image_features = self.projector(image_features)

        return image_features

    def fuse_embeddings(
        self,
        input_ids,
        pixel_values,
    ):
        # 1. 文本 Token 转成 Embedding
        text_embeds = self.language_model.get_input_embeddings()(
            input_ids
        )

        # 2. 提取图像特征
        image_embeds = self.encode_image(pixel_values)

        # 3. 图像与文本对齐 dtype/device
        image_embeds = image_embeds.to(
            device=text_embeds.device,
            dtype=text_embeds.dtype,
        )

        # 4. 找到图像占位符
        image_mask = input_ids == self.image_token_id

        # 5. 检查 Batch 和视觉 Token 数量
        B, N, C = image_embeds.shape

        assert input_ids.shape[0] == B, (
            "文本 Batch 与图像 Batch 不一致"
        )

        assert torch.all(image_mask.sum(dim=1) == N), (
            "图像占位符数量与视觉 Token 数量不一致"
        )

        # 6. 将 mask 扩展至 hidden_size
        image_mask = image_mask.unsqueeze(-1)
        image_mask = image_mask.expand_as(text_embeds)

        # 7. 用视觉特征替换占位符 Embedding
        fused_embeds = text_embeds.masked_scatter(
            image_mask,
            image_embeds.reshape(-1),
        )

        return fused_embeds

    def forward(
        self,
        input_ids,
        pixel_values,
        attention_mask=None,
        labels=None,
    ):

        # 1. 多模态 Embedding 融合
        inputs_embeds = self.fuse_embeddings(
            input_ids=input_ids,
            pixel_values=pixel_values,
        )

        # 2. 送入 Qwen3
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )

        return outputs

    
    @torch.no_grad()
    def generate(
        self,
        input_ids,
        pixel_values,
        attention_mask=None,
        max_new_tokens=64,
        eos_token_id=None,
        pad_token_id=None,
    ):
        # 1. 提取图片并融合图文 Embedding
        inputs_embeds = self.fuse_embeddings(
            input_ids=input_ids,
            pixel_values=pixel_values,
        )

        # 2. 调用 Qwen3 的生成接口
        generated_ids = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )

        return generated_ids
    