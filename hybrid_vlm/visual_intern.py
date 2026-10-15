"""InternViT-300M -> Qwen2.5-VL PatchMerger adapter.

Pixel protocol: [N_tiles,3,448,448] (already CLIP-preprocessed).
Grid protocol: one [T,H,W] per *logical* image/video; H/W in 14px patches.
The returned tensor is [sum(T*H*W//4), qwen3_hidden_size], like Qwen2.5-VL visual.
"""
import torch
from torch import nn
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLPatchMerger


class InternViTQwenVisual(nn.Module):
    def __init__(self, internvit: nn.Module, llm_hidden_size: int = 1024, chunk_size: int = 4):
        super().__init__()
        self.vit = internvit
        self.spatial_merge_size = 2
        self.patch_size = 14
        self.tile_patch_side = 32
        self.chunk_size = int(chunk_size)
        self.vision_hidden_size = int(internvit.config.hidden_size)
        self.merger = Qwen2_5_VLPatchMerger(
            dim=llm_hidden_size,
            context_dim=self.vision_hidden_size,
            spatial_merge_size=self.spatial_merge_size,
        )

    @property
    def dtype(self):
        return next(self.vit.parameters()).dtype

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 4 or tuple(pixel_values.shape[1:]) != (3, 448, 448):
            raise ValueError(f"Expected [N_tiles,3,448,448], got {tuple(pixel_values.shape)}")
        if grid_thw is None or grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
            raise ValueError("grid_thw must have shape [N_images_or_videos,3]")
        grids = grid_thw.detach().cpu().tolist()
        tile_counts = []
        for t, h, w in grids:
            if t < 1 or h < 32 or w < 32 or h % 32 or w % 32:
                raise ValueError(f"Unsupported tile grid: {(t,h,w)}")
            tile_counts.append(t * (h // 32) * (w // 32))
        if sum(tile_counts) != len(pixel_values):
            raise ValueError(f"Grid expects {sum(tile_counts)} tiles, received {len(pixel_values)}")

        vit_param = next(self.vit.parameters())
        tile_embeddings = []
        for chunk in pixel_values.split(self.chunk_size, dim=0):
            features = self.vit(
                pixel_values=chunk.to(device=vit_param.device, dtype=vit_param.dtype),
                return_dict=True,
            ).last_hidden_state
            if features.shape[1:] != (1025, self.vision_hidden_size):
                raise ValueError(f"Unexpected InternViT features: {tuple(features.shape)}")
            features = features[:, 1:, :]
            n, _, c = features.shape
            # Group every contiguous 2x2 patch into four consecutive rows for the *original* merger.
            features = features.reshape(n, 16, 2, 16, 2, c)
            features = features.permute(0, 1, 3, 2, 4, 5).contiguous()
            features = features.reshape(n * 1024, c)
            merger_param = next(self.merger.parameters())
            merged = self.merger(features.to(device=merger_param.device, dtype=merger_param.dtype))
            tile_embeddings.append(merged.reshape(n, 16, 16, -1))

        # [N_tiles,16,16,H_llm]
        tiles = torch.cat(tile_embeddings, dim=0)
        offset = 0
        outputs = []
        for (t, h, w), count in zip(grids, tile_counts):
            rows, cols = h // 32, w // 32
            group = tiles[offset: offset + count]
            # Tile order expected by Processor: time, tile-row, tile-col.
            # Convert to global raster scan: time, tile-row, local-row, tile-col, local-col.
            group = group.reshape(t, rows, cols, 16, 16, -1)
            group = group.permute(0, 1, 3, 2, 4, 5).contiguous()
            outputs.append(group.reshape(t * (h // 2) * (w // 2), -1))
            offset += count
        return torch.cat(outputs, dim=0)
