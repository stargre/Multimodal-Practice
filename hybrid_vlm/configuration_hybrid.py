"""Serializable configuration for the qwen25_fork architecture (format v1)."""
import copy

from transformers import Qwen3Config
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig, Qwen2_5_VLVisionConfig,
)
from .configuration_intern_vit import InternVisionConfig


class HybridVLMConfig(Qwen2_5_VLConfig):
    model_type = "qwen25_internvit_qwen3"
    sub_configs = {"text_config": Qwen3Config,
                   "vision_config": Qwen2_5_VLVisionConfig,
                   "internvit_config": InternVisionConfig}

    def __init__(self, text_config=None, vision_config=None, internvit_config=None,
                 mrope_sections=(16, 24, 24), vision_chunk_size=4,
                 component_dtypes=None, format_version=1, **kwargs):
        text = text_config.to_dict() if hasattr(text_config, "to_dict") else copy.deepcopy(text_config or {})
        vision = vision_config.to_dict() if hasattr(vision_config, "to_dict") else copy.deepcopy(vision_config or {})
        intern = internvit_config.to_dict() if hasattr(internvit_config, "to_dict") else copy.deepcopy(internvit_config or {
            "image_size": 448, "patch_size": 14, "hidden_size": 1024,
            "num_hidden_layers": 24, "num_attention_heads": 16,
            "intermediate_size": 4096, "use_flash_attn": False,
        })
        # Qwen2.5's outer methods only need the grid/merger fields in vision_config.
        # The actual decoder must retain ALL Qwen3 fields, including head_dim and RoPE.
        super().__init__(text_config=text, vision_config=vision, **kwargs)
        self.text_config = Qwen3Config(**text)
        self.internvit_config = InternVisionConfig(**intern)
        self.mrope_sections = list(mrope_sections)
        self.vision_chunk_size = int(vision_chunk_size)
        self.format_version = int(format_version)
        self.component_dtypes = dict(component_dtypes or {
            "vit": "float32", "merger": "float32", "language_model": "float32", "lm_head": "float32"})
        self.architectures = ["HybridVLMForConditionalGeneration"]
        self.vocab_size = self.text_config.vocab_size
        self.tie_word_embeddings = self.text_config.tie_word_embeddings
        if self.format_version != 1:
            raise ValueError("Unsupported hybrid format_version")
        if self.vision_chunk_size < 1:
            raise ValueError("vision_chunk_size must be positive")
        if self.internvit_config.image_size != 448 or self.internvit_config.patch_size != 14:
            raise ValueError("Phase 1 preserves the original 448px / 14px visual protocol")
        if self.vision_config.spatial_merge_size != 2:
            raise ValueError("Phase 1 requires spatial_merge_size=2")
        if 2 * sum(self.mrope_sections) != self.text_config.head_dim:
            raise ValueError("MRoPE sections do not match Qwen3 head_dim")
        if set(self.component_dtypes) != {"vit", "merger", "language_model", "lm_head"}:
            raise ValueError("component_dtypes must specify all four components")
        if any(x not in ("float32", "bfloat16", "float16") for x in self.component_dtypes.values()):
            raise ValueError("Unsupported component dtype")
        if self.tie_word_embeddings and self.component_dtypes["lm_head"] != self.component_dtypes["language_model"]:
            raise ValueError("Tied embedding and head must have identical dtypes")
        if self.internvit_config.use_flash_attn:
            raise ValueError("This phase-1 implementation preserves the non-FlashAttention InternViT path")

    @classmethod
    def from_legacy(cls, model):
        data = copy.deepcopy(model.config.to_dict())
        for key in ("model_type", "architectures", "auto_map", "_name_or_path"):
            data.pop(key, None)
        data["text_config"] = model.model.language_model.config.to_dict()
        # The legacy builder copies Qwen3's embedding into a NEW backbone while
        # reusing its original lm_head. They are independent even when the
        # source Qwen3 config says tie_word_embeddings=True. Preserve reality.
        actually_tied = (model.model.language_model.embed_tokens.weight is model.lm_head.weight)
        data["text_config"]["tie_word_embeddings"] = actually_tied
        data["tie_word_embeddings"] = actually_tied
        data["internvit_config"] = model.model.visual.vit.config.to_dict()
        data["internvit_config"]["use_flash_attn"] = False
        data["mrope_sections"] = list(model.model.language_model.rotary_emb.sections)
        data["vision_chunk_size"] = model.model.visual.chunk_size
        components = {"vit": model.model.visual.vit, "merger": model.model.visual.merger,
                      "language_model": model.model.language_model, "lm_head": model.lm_head}
        data["component_dtypes"] = {k: str(next(v.parameters()).dtype).split(".")[-1]
                                    for k, v in components.items()}
        return cls(**data)

    def to_diff_dict(self):
        # Full nested configuration, never rely on changing upstream defaults.
        return self.to_dict()
