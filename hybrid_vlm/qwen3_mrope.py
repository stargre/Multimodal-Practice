"""Qwen3 backbone with Qwen2.5-VL-style T/H/W MRoPE positions.
Pin transformers==4.57.6: imports below use that version's internal APIs.
"""
import torch
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model, Qwen3RotaryEmbedding


class Qwen3MRoPE(Qwen3RotaryEmbedding):
    def __init__(self, config, sections=(16, 24, 24)):
        super().__init__(config)
        self.sections = tuple(int(x) for x in sections)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        if 2 * sum(self.sections) != head_dim:
            raise ValueError(f"mrope sections {self.sections} incompatible with head_dim={head_dim}")

    def forward(self, x, position_ids):
        if position_ids.ndim == 2:
            return super().forward(x, position_ids)
        if position_ids.ndim != 3 or position_ids.shape[0] != 3:
            raise ValueError(f"Expected positions [B,L] or [3,B,L], got {tuple(position_ids.shape)}")
        components = [super(Qwen3MRoPE, self).forward(x, position_ids[i]) for i in range(3)]
        # Qwen2.5-VL sections alternate T/H/W in the two halves of head_dim.
        lengths = self.sections * 2
        cos_chunks = [torch.split(cs[0], lengths, dim=-1) for cs in components]
        sin_chunks = [torch.split(cs[1], lengths, dim=-1) for cs in components]
        cos = torch.cat([cos_chunks[i % 3][i] for i in range(6)], dim=-1)
        sin = torch.cat([sin_chunks[i % 3][i] for i in range(6)], dim=-1)
        return cos, sin


class Qwen3MRoPEModel(Qwen3Model):
    def __init__(self, config, sections=(16, 24, 24)):
        super().__init__(config)
        self.rotary_emb = Qwen3MRoPE(config, sections)

    def forward(
        self, input_ids=None, attention_mask=None, position_ids=None,
        past_key_values=None, inputs_embeds=None, use_cache=None,
        cache_position=None, output_attentions=None, output_hidden_states=None,
        return_dict=True, **kwargs,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Exactly one of input_ids and inputs_embeds must be supplied")
        if output_attentions:
            raise NotImplementedError("This adapter does not expose attention weights")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        bsz, length, _ = inputs_embeds.shape
        if use_cache is None:
            use_cache = self.config.use_cache
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if cache_position is None:
            past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_length, past_length + length, device=inputs_embeds.device)

        if position_ids is None:
            spatial_positions = cache_position.unsqueeze(0).expand(bsz, -1)
            mask_positions = spatial_positions
        elif position_ids.ndim == 2:
            spatial_positions = mask_positions = position_ids
        elif position_ids.ndim == 3 and position_ids.shape[0] == 3:

            spatial_positions = position_ids

            mask_positions = (
                cache_position
                .unsqueeze(0)
                .expand(bsz, -1)
            )
        elif position_ids.ndim == 3 and position_ids.shape[0] == 4:
            mask_positions = position_ids[0]
            spatial_positions = position_ids[1:]
        else:
            raise ValueError(f"Unexpected position ids: {tuple(position_ids.shape)}")
        if spatial_positions.shape[-1] != length:
            raise ValueError("RoPE position length does not match current input")

        if isinstance(attention_mask, dict):
            masks = attention_mask
        else:
            mask_kwargs = dict(
                config=self.config, input_embeds=inputs_embeds,
                attention_mask=attention_mask, cache_position=cache_position,
                past_key_values=past_key_values, position_ids=mask_positions,
            )
            masks = {"full_attention": create_causal_mask(**mask_kwargs)}
            if self.has_sliding_layers:
                masks["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden = inputs_embeds
        position_embeddings = self.rotary_emb(hidden, spatial_positions)
        hidden_history = [] if output_hidden_states else None
        for layer in self.layers[:self.config.num_hidden_layers]:
            if hidden_history is not None:
                hidden_history.append(hidden)
            hidden = layer(
                hidden, attention_mask=masks[layer.attention_type],
                position_ids=mask_positions,
                past_key_values=past_key_values, use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings, **kwargs,
            )
        hidden = self.norm(hidden)
        if hidden_history is not None:
            hidden_history.append(hidden)
        result = BaseModelOutputWithPast(
            last_hidden_state=hidden,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=tuple(hidden_history) if hidden_history is not None else None,
        )
        return result if return_dict else result.to_tuple()
