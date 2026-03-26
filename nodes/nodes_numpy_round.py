"""
Demonstrates numpy version-dependent rounding behavior.

Takes an image and two float scale factors. Computes output dimensions
via np.round() on (original_dim * factor). numpy 1.x rounds 0.5 away
from zero; numpy 2.x rounds 0.5 to even. Same code, different results.
The info string is burned onto the resized image.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io


def _burn_text(image_tensor: torch.Tensor, text: str) -> torch.Tensor:
    """Burn white text with black outline onto bottom of each image in batch."""
    B, H, W, C = image_tensor.shape
    out = []
    for i in range(B):
        # tensor (H,W,C) float 0-1 -> PIL RGB
        arr = (image_tensor[i].clamp(0, 1) * 255).byte().cpu().numpy()
        pil = Image.fromarray(arr, "RGB")
        draw = ImageDraw.Draw(pil)

        font_size = max(12, min(H // 40, 24))
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (W - tw) // 2
        y = H - th - max(8, H // 40)

        # black outline
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                draw.text((x + dx, y + dy), text, fill=(0, 0, 0), font=font)
        draw.text((x, y), text, fill=(255, 255, 255), font=font)

        t = torch.from_numpy(np.array(pil)).float() / 255.0
        out.append(t)
    return torch.stack(out)


class NumpyRoundUpscale_ZIMG(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NumpyRoundUpscale_ZIMG",
            display_name="Upscale by np.round_Isolated",
            category="image/upscaling",
            inputs=[
                io.Image.Input("image"),
                io.String.Input("width_factor", default="1.5"),
                io.String.Input("height_factor", default="2.5"),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, image, width_factor, height_factor) -> io.NodeOutput:
        wf_raw = float(width_factor)
        hf_raw = float(height_factor)
        wf_rounded = float(np.round(wf_raw))
        hf_rounded = float(np.round(hf_raw))
        # image shape: (B, H, W, C)
        _, h, w, _ = image.shape

        new_w = max(int(w * wf_rounded), 1)
        new_h = max(int(h * hf_rounded), 1)

        samples = image.movedim(-1, 1)
        resized = torch.nn.functional.interpolate(samples, size=(new_h, new_w), mode="bilinear", antialias=True)
        result = resized.movedim(1, -1)

        info = (
            f"numpy {np.__version__} | "
            f"np.round({wf_raw}) = {wf_rounded}, np.round({hf_raw}) = {hf_rounded} | "
            f"{w}x{h} * ({wf_rounded}, {hf_rounded}) = {new_w}x{new_h}"
        )

        result = _burn_text(result, info)
        return io.NodeOutput(result)


class NumpyRoundExtension_ISO(ComfyExtension):

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [NumpyRoundUpscale_ZIMG]


async def comfy_entrypoint() -> NumpyRoundExtension_ISO:
    return NumpyRoundExtension_ISO()
