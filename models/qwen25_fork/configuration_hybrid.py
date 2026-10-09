"""Qwen2.5-VL outer config adapted to real Qwen3/InternViT shapes.
This is NOT registered with AutoConfig. The custom loader lives in checkpoint.py.
"""
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig


class HybridQwen25VLConfig(Qwen2_5_VLConfig):
    model_type = "hybrid_qwen25_vl"

    @classmethod
    def from_components(cls, qwen25_config, qwen3_config, vit_config, tokenizer):
        # Inherit Qwen2.5's multimodal token and grid configuration, not its decoder weights.
        config = cls(**qwen25_config.to_dict())
        config.architectures = ["HybridQwen25VLForConditionalGeneration"]
        config.text_config.hidden_size = int(qwen3_config.hidden_size)
        config.text_config.vocab_size = int(qwen3_config.vocab_size)
        config.text_config.num_hidden_layers = int(qwen3_config.num_hidden_layers)
        config.text_config.num_attention_heads = int(qwen3_config.num_attention_heads)
        config.text_config.num_key_value_heads = int(qwen3_config.num_key_value_heads)
        config.text_config.intermediate_size = int(qwen3_config.intermediate_size)
        config.vision_config.hidden_size = int(vit_config.hidden_size)
        config.vision_config.out_hidden_size = int(qwen3_config.hidden_size)
        config.vision_config.patch_size = 14
        config.vision_config.spatial_merge_size = 2
        # One visual temporal grid cell represents one individually encoded frame.
        config.vision_config.temporal_patch_size = 1
        for name, token in (
            ("image_token_id", "<|image_pad|>"),
            ("video_token_id", "<|video_pad|>"),
            ("vision_start_token_id", "<|vision_start|>"),
        ):
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is None or token_id == tokenizer.unk_token_id:
                raise ValueError(f"Missing required Qwen multimodal token: {token}")
            setattr(config, name, int(token_id))
        config.vocab_size = int(qwen3_config.vocab_size)
        config.tie_word_embeddings = bool(qwen3_config.tie_word_embeddings)
        config.is_encoder_decoder = False
        config.use_cache = True
        return config
