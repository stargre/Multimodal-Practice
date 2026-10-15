"""Exact legacy ImageOps.pad + row-major tiling, with saved CLIP preprocessing."""
import math
from PIL import Image, ImageOps
from transformers import CLIPImageProcessor


class HybridImageProcessor(CLIPImageProcessor):
    def __init__(self, tile_size=448, max_tiles_per_side=2, **kwargs):
        super().__init__(**kwargs)
        if tile_size != 448 or int(max_tiles_per_side) < 1:
            raise ValueError("Requires tile_size=448 and max_tiles_per_side >= 1")
        self.tile_size = int(tile_size)
        self.max_tiles_per_side = int(max_tiles_per_side)

    def tile_image(self, image):
        image = image.convert("RGB")
        rows = min(self.max_tiles_per_side, max(1, math.ceil(image.height / 448)))
        cols = min(self.max_tiles_per_side, max(1, math.ceil(image.width / 448)))
        canvas = ImageOps.pad(image, (cols * 448, rows * 448),
                              method=Image.Resampling.BICUBIC, color=(127, 127, 127))
        tiles = [canvas.crop((col * 448, row * 448, (col + 1) * 448, (row + 1) * 448))
                 for row in range(rows) for col in range(cols)]
        return tiles, [1, rows * 32, cols * 32]
