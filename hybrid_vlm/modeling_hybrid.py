"""Qwen2.5-VL-model structure with substituted ViT and Qwen3 backbone.
Base multimodal functions and prepare_inputs_for_generation remain Qwen2.5-VL.
"""
import copy
import torch
from torch import nn
from .configuration_hybrid import HybridVLMConfig
from .modeling_intern_vit import InternVisionModel
from .qwen3_mrope import Qwen3MRoPEModel
from .visual_intern import InternViTQwenVisual
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLPreTrainedModel,
    Qwen2_5_VLModel,
    Qwen2_5_VLForConditionalGeneration,
)


class HybridVLMModel(Qwen2_5_VLModel):
    def __init__(self, config, visual, language_model):
        # Do NOT call Qwen2_5_VLModel.__init__ (creates the old visual/decoder).
        Qwen2_5_VLPreTrainedModel.__init__(self, config)
        self.visual = visual
        self.language_model = language_model
        self.rope_deltas = None

    def get_rope_index(
        self,
        input_ids=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        attention_mask=None,
    ):

            spatial_merge_size = self.config.vision_config.spatial_merge_size
            image_token_id = self.config.image_token_id
            video_token_id = self.config.video_token_id
            vision_start_token_id = self.config.vision_start_token_id
            mrope_position_deltas = []
            if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
                total_input_ids = input_ids
                if attention_mask is not None:
                    attention_mask = attention_mask == 1
                position_ids = torch.ones(
                    3,
                    input_ids.shape[0],
                    input_ids.shape[1],
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                image_index, video_index = 0, 0
                for i, input_ids in enumerate(total_input_ids):
                    if attention_mask is not None:
                        input_ids = input_ids[attention_mask[i]]
                    image_nums, video_nums = 0, 0
                    vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                    vision_tokens = input_ids[vision_start_indices + 1]
                    image_nums = (vision_tokens == image_token_id).sum()
                    video_nums = (vision_tokens == video_token_id).sum()
                    input_tokens = input_ids.tolist()
                    llm_pos_ids_list: list = []
                    st = 0
                    remain_images, remain_videos = image_nums, video_nums
                    for _ in range(image_nums + video_nums):
                        if image_token_id in input_tokens and remain_images > 0:
                            ed_image = input_tokens.index(image_token_id, st)
                        else:
                            ed_image = len(input_tokens) + 1
                        if video_token_id in input_tokens and remain_videos > 0:
                            ed_video = input_tokens.index(video_token_id, st)
                        else:
                            ed_video = len(input_tokens) + 1
                        if ed_image < ed_video:
                            t, h, w = (
                                image_grid_thw[image_index][0],
                                image_grid_thw[image_index][1],
                                image_grid_thw[image_index][2],
                            )
                            second_per_grid_t = 0
                            image_index += 1
                            remain_images -= 1
                            ed = ed_image

                        else:
                            t, h, w = (
                                video_grid_thw[video_index][0],
                                video_grid_thw[video_index][1],
                                video_grid_thw[video_index][2],
                            )
                            if second_per_grid_ts is not None:
                                second_per_grid_t = second_per_grid_ts[video_index]
                            else:
                                second_per_grid_t = 1.0
                            video_index += 1
                            remain_videos -= 1
                            ed = ed_video
                        llm_grid_t, llm_grid_h, llm_grid_w = (
                            t.item(),
                            h.item() // spatial_merge_size,
                            w.item() // spatial_merge_size,
                        )
                        text_len = ed - st

                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                        range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                        expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                        ## normalize type, send to device.
                        second_per_grid_t = torch.as_tensor(
                            second_per_grid_t, dtype=torch.float32, device=range_tensor.device
                        )

                        time_tensor = expanded_range * second_per_grid_t * self.config.vision_config.tokens_per_second

                        time_tensor_long = time_tensor.long()
                        t_index = time_tensor_long.flatten()

                        h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                        w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                        llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                        st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                    if st < len(input_tokens):
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        text_len = len(input_tokens) - st
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                    if attention_mask is not None:
                        position_ids[..., i, attention_mask[i]] = llm_positions.to(position_ids.device)
                    else:
                        position_ids[..., i, :] = llm_positions.to(position_ids.device)
                    mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
                mrope_position_deltas = torch.tensor(mrope_position_deltas).unsqueeze(1).to(device=input_ids.device)
                return position_ids, mrope_position_deltas
            else:
                if attention_mask is not None:
                    position_ids = attention_mask.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attention_mask == 0, 1)
                    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                    max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                    mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
                else:
                    position_ids = (
                        torch.arange(input_ids.shape[1], device=input_ids.device)
                        .view(1, 1, -1)
                        .expand(3, input_ids.shape[0], -1)
                    )
                    mrope_position_deltas = torch.zeros(
                        [input_ids.shape[0], 1],
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )

                return position_ids, mrope_position_deltas



class HybridVLMForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    config_class = HybridVLMConfig
    _no_split_modules = ["Qwen3DecoderLayer", "InternVisionEncoderLayer", "Qwen2_5_VLPatchMerger"]

    def __init__(self, config):
        Qwen2_5_VLPreTrainedModel.__init__(self, config)
        text_config = copy.deepcopy(config.text_config)
        text_config._attn_implementation = config._attn_implementation
        internvit_config = copy.deepcopy(config.internvit_config)
        internvit_config._attn_implementation = "eager"
        visual = InternViTQwenVisual(
            InternVisionModel(internvit_config),
            llm_hidden_size=text_config.hidden_size,
            chunk_size=config.vision_chunk_size,
        )
        backbone = Qwen3MRoPEModel(text_config, sections=config.mrope_sections)
        self.model = HybridVLMModel(config, visual, backbone)
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)
        self.vocab_size = text_config.vocab_size
        self.post_init()
        self._apply_component_dtypes()

    def _apply_component_dtypes(self):
        components = {"vit": self.model.visual.vit, "merger": self.model.visual.merger,
                      "language_model": self.model.language_model, "lm_head": self.lm_head}
        for name, module in components.items():
            module.to(dtype=getattr(torch, self.config.component_dtypes[name]))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # With no override, preserve mixed component precision saved during conversion.
        # An explicit floating dtype intentionally selects uniform inference precision.
        config = kwargs.pop("config", None)
        if config is None:
            config_keys = ("cache_dir", "force_download", "local_files_only", "token", "revision", "subfolder")
            config = cls.config_class.from_pretrained(
                pretrained_model_name_or_path,
                **{k: kwargs[k] for k in config_keys if k in kwargs},
            )
        else:
            config = copy.deepcopy(config)
        requested = kwargs.get("dtype", kwargs.get("torch_dtype"))
        if requested is not None and requested != "auto":
            name = str(requested).removeprefix("torch.")
            if name not in ("float32", "bfloat16", "float16"):
                raise ValueError(f"Unsupported dtype override: {requested}")
            config.component_dtypes = dict.fromkeys(config.component_dtypes, name)
        # Initialization/loading must respect each destination parameter's dtype.
        kwargs.pop("torch_dtype", None)
        kwargs["dtype"] = torch.float32
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)

    def save_pretrained(self, save_directory, **kwargs):
        modules = {"vit": self.model.visual.vit, "merger": self.model.visual.merger,
                   "language_model": self.model.language_model, "lm_head": self.lm_head}
        self.config.component_dtypes = {
            k: str(next(v.parameters()).dtype).removeprefix("torch.") for k, v in modules.items()
        }
        return super().save_pretrained(save_directory, **kwargs)

    def _expand_inputs_for_generation(
        self, expand_size=1, is_encoder_decoder=False, input_ids=None, **model_kwargs
    ):
        if expand_size == 1:
            return input_ids, model_kwargs
        if is_encoder_decoder:
            raise NotImplementedError("Decoder-only model")
        if input_ids is None:
            raise ValueError("input_ids required for multimodal expansion")
        batch = input_ids.shape[0]
        image_nums, video_nums = self._get_image_nums_and_video_nums(
            input_ids, inputs_embeds=model_kwargs.get("inputs_embeds"),
        )
        image_nums, video_nums = image_nums.tolist(), video_nums.tolist()
        image_grid = model_kwargs.get("image_grid_thw")
        video_grid = model_kwargs.get("video_grid_thw")

        def tile_lengths(grids, counts):
            if sum(counts) == 0:
                return [0] * batch
            if grids is None or len(grids) != sum(counts):
                raise ValueError("Missing or inconsistent multimodal grid")
            parts = torch.split(grids, counts, dim=0)
            result = []
            for part in parts:
                n_tiles = 0
                for t, h, w in part.detach().cpu().tolist():
                    if h % 32 or w % 32:
                        raise ValueError("Grid not divisible by 32 patches per tile")
                    n_tiles += t * (h // 32) * (w // 32)
                result.append(n_tiles)
            return result

        image_tiles = tile_lengths(image_grid, image_nums)
        video_tiles = tile_lengths(video_grid, video_nums)
        split_keys = dict(
            pixel_values=image_tiles,
            image_grid_thw=image_nums,
            pixel_values_videos=video_tiles,
            video_grid_thw=video_nums,
            second_per_grid_ts=video_nums,
        )
        for key, lengths in split_keys.items():
            tensor = model_kwargs.get(key)
            if tensor is None:
                continue
            if tensor.shape[0] != sum(lengths):
                raise ValueError(f"{key}: length mismatch ({tensor.shape[0]} vs {sum(lengths)})")
            chunks = torch.split(tensor, lengths, dim=0)
            model_kwargs[key] = torch.cat(
                [chunk for chunk in chunks for _ in range(expand_size)], dim=0
            )

        for key, val in list(model_kwargs.items()):
            if key in split_keys or key == "cache_position" or not isinstance(val, torch.Tensor):
                continue
            if key == "position_ids" and val.ndim == 3 and val.shape[0] in (3, 4):
                model_kwargs[key] = val.repeat_interleave(expand_size, dim=1)
            else:
                model_kwargs[key] = val.repeat_interleave(expand_size, dim=0)
        return input_ids.repeat_interleave(expand_size, dim=0), model_kwargs
