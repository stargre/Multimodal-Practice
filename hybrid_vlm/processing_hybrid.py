"""InternViT image/video tiling with Qwen2.5-VL's token/grid field names.

Input prompt: one <|image_pad|> per logical image, one <|video_pad|>
per logical video, each immediately after <|vision_start|>.
Video inputs are lists of already-decoded PIL frames (not file paths).
"""
import math
import re

import torch
from PIL import Image, ImageOps
from transformers import ProcessorMixin
from transformers.feature_extraction_utils import BatchFeature
from .image_processing_hybrid import HybridImageProcessor


class HybridVLMProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"
    IMAGE = "<|image_pad|>"
    VIDEO = "<|video_pad|>"

    def __init__(self, image_processor, tokenizer, tile_size=None, max_tiles_per_side=None, chat_template=None):
        tile_size = tile_size if tile_size is not None else getattr(image_processor, "tile_size", 448)
        max_tiles_per_side = max_tiles_per_side if max_tiles_per_side is not None else getattr(image_processor, "max_tiles_per_side", 2)
        if not isinstance(image_processor, HybridImageProcessor):
            values = image_processor.to_dict()
            values.pop("image_processor_type", None)
            values.update(tile_size=tile_size, max_tiles_per_side=max_tiles_per_side)
            image_processor = HybridImageProcessor(**values)
        elif image_processor.tile_size != tile_size or image_processor.max_tiles_per_side != max_tiles_per_side:
            raise ValueError("Processor tiling options disagree with image_processor")
        super().__init__(image_processor=image_processor, tokenizer=tokenizer, chat_template=chat_template)
        self.tile_size = tile_size
        self.max_tiles_per_side = int(max_tiles_per_side)

    def save_pretrained(self, save_directory, **kwargs):
        # HF tokenizers do not persist assignments to padding_side automatically.
        self.tokenizer.init_kwargs["padding_side"] = self.tokenizer.padding_side
        return super().save_pretrained(save_directory, **kwargs)

    @classmethod
    def from_legacy(cls, processor):
        return cls(image_processor=processor.image_processor, tokenizer=processor.tokenizer,
                   tile_size=processor.tile_size, max_tiles_per_side=processor.max_tiles_per_side)

    @property
    def model_input_names(self):
        return ["input_ids", "attention_mask", "pixel_values", "image_grid_thw",
                "pixel_values_videos", "video_grid_thw", "second_per_grid_ts"]

    def __call__(self, text=None, images=None, *, videos_per_sample=None,
                 fps_per_sample=None, padding=True, return_tensors="pt", **kwargs):
        """Text is rendered ChatML; images is nested per sample (or flat for one sample).

        Use user_prompt() for the EXACT legacy single-turn prompt. No implicit chat
        template replacement is performed by this phase-1 compatibility interface.
        """
        if kwargs:
            raise TypeError(f"Unsupported processor arguments: {sorted(kwargs)}")
        if return_tensors != "pt":
            raise ValueError("Only return_tensors='pt' is supported")
        prompts = [text] if isinstance(text, str) else text
        if not prompts:
            raise ValueError("Supply text (rendered prompt) or a list of prompts")
        if images is not None:
            if isinstance(images, Image.Image):
                images = [[images]]
            elif len(prompts) == 1 and (not images or isinstance(images[0], Image.Image)):
                images = [list(images)]
        batch = self.encode_batch(prompts, images_per_sample=images,
                                  videos_per_sample=videos_per_sample,
                                  fps_per_sample=fps_per_sample, padding=padding)
        return BatchFeature({k: v for k, v in batch.items() if v is not None})

    def user_prompt(self, text, num_images=0, num_videos=0):
        # Video and image placeholders can be arranged manually in any order
        # by passing a custom prompt instead of this helper.
        content = (
            "<|vision_start|><|image_pad|><|vision_end|>" * num_images
            + "<|vision_start|><|video_pad|><|vision_end|>" * num_videos
            + "\n" + text
        )
        return "<|im_start|>user\n" + content + "<|im_end|>\n<|im_start|>assistant\n"

    def tile_image(self, image):
        return self.image_processor.tile_image(image)

    def encode_batch(
        self, prompts, images_per_sample=None,
        videos_per_sample=None, fps_per_sample=None, padding=True,
    ):
        n = len(prompts)
        if not n:
            raise ValueError("prompts must be nonempty")
        images_per_sample = images_per_sample if images_per_sample is not None else [[] for _ in range(n)]
        videos_per_sample = videos_per_sample if videos_per_sample is not None else [[] for _ in range(n)]
        fps_per_sample = fps_per_sample if fps_per_sample is not None else [[] for _ in range(n)]
        if not (len(images_per_sample) == len(videos_per_sample) == len(fps_per_sample) == n):
            raise ValueError("Batch field counts do not match prompts")

        texts, image_tiles, video_tiles = [], [], []
        image_grids, video_grids, seconds_per_grid = [], [], []
        for prompt, images, videos, fps_values in zip(
            prompts, images_per_sample, videos_per_sample, fps_per_sample,
        ):
            if len(fps_values) == 0 and len(videos) > 0:
                fps_values = [2.0] * len(videos)
            if len(fps_values) != len(videos):
                raise ValueError("One fps is required per video")
            parts = re.split(r"(<\|image_pad\|>|<\|video_pad\|>)", prompt)
            rebuilt = []
            img_i = vid_i = 0
            for part in parts:
                if part == self.IMAGE:
                    if img_i >= len(images):
                        raise ValueError("More image placeholders than images")
                    tiles, grid = self.tile_image(images[img_i])
                    image_tiles.extend(tiles)
                    image_grids.append(grid)
                    rebuilt.append(self.IMAGE * (grid[0] * grid[1] * grid[2] // 4))
                    img_i += 1
                elif part == self.VIDEO:
                    if vid_i >= len(videos):
                        raise ValueError("More video placeholders than videos")
                    frames = videos[vid_i]
                    fps = float(fps_values[vid_i])
                    if not frames or fps <= 0:
                        raise ValueError("Video frames cannot be empty, and fps must be positive")
                    first_spatial = None
                    for frame in frames:
                        tiles, grid = self.tile_image(frame)
                        if first_spatial is None:
                            first_spatial = grid[1:]
                        elif grid[1:] != first_spatial:
                            raise ValueError("All frames in the same video must have the same tile grid")
                        video_tiles.extend(tiles)
                    grid = [len(frames), *first_spatial]
                    video_grids.append(grid)
                    seconds_per_grid.append(1.0 / fps)
                    rebuilt.append(self.VIDEO * (grid[0] * grid[1] * grid[2] // 4))
                    vid_i += 1
                else:
                    rebuilt.append(part)
            if img_i != len(images) or vid_i != len(videos):
                raise ValueError("Visual inputs were not all consumed by prompt placeholders")
            texts.append("".join(rebuilt))

        text = self.tokenizer(texts, padding=padding, return_tensors="pt", add_special_tokens=False)
        result = {"input_ids": text["input_ids"], "attention_mask": text["attention_mask"]}
        result["image_grid_thw"] = torch.tensor(image_grids, dtype=torch.long) if image_grids else None
        result["video_grid_thw"] = torch.tensor(video_grids, dtype=torch.long) if video_grids else None
        result["second_per_grid_ts"] = torch.tensor(seconds_per_grid, dtype=torch.float32) if seconds_per_grid else None
        if image_tiles:
            result["pixel_values"] = self.image_processor(images=image_tiles, return_tensors="pt").pixel_values
            if tuple(result["pixel_values"].shape[1:]) != (3, 448, 448):
                raise ValueError("Image processor is not configured for 448x448 InternViT tiles")
        else:
            result["pixel_values"] = None
        if video_tiles:
            result["pixel_values_videos"] = self.image_processor(images=video_tiles, return_tensors="pt").pixel_values
            if tuple(result["pixel_values_videos"].shape[1:]) != (3, 448, 448):
                raise ValueError("Video frame processor must produce 448x448 InternViT tiles")
        else:
            result["pixel_values_videos"] = None

        img_id = self.tokenizer.convert_tokens_to_ids(self.IMAGE)
        vid_id = self.tokenizer.convert_tokens_to_ids(self.VIDEO)
        image_expected = sum(t * h * w // 4 for t, h, w in image_grids)
        video_expected = sum(t * h * w // 4 for t, h, w in video_grids)
        if int((result["input_ids"] == img_id).sum()) != image_expected:
            raise ValueError("Tokenized image placeholders differ from visual feature length")
        if int((result["input_ids"] == vid_id).sum()) != video_expected:
            raise ValueError("Tokenized video placeholders differ from visual feature length")
        return result

    def encode_images(self, prompts, images_per_sample, padding=True):
        return self.encode_batch(prompts, images_per_sample=images_per_sample, padding=padding)

    def encode_video(self, prompt, frames, fps=2.0):
        return self.encode_batch(
            [prompt], videos_per_sample=[[frames]], fps_per_sample=[[fps]], padding=False,
        )
