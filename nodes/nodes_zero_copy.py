from __future__ import annotations

import logging
import torch
from comfy import model_management
from comfy_api.latest import ComfyExtension, io

logger = logging.getLogger(__name__)


class ZeroCopyArange(io.ComfyNode):
    """Create a tiny CUDA latent and report its device / data_ptr."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ZeroCopyArange",
            display_name="ZeroCopyArange",
            category="PyIsolated/Debug",
            description="Create small CUDA tensor and report device/data_ptr",
            inputs=[],
            outputs=[
                io.Latent.Output("latent", display_name="latent"),
                io.String.Output("device", display_name="device"),
                io.Int.Output("data_ptr", display_name="data_ptr"),
            ],
        )

    @classmethod
    def execute(cls) -> io.NodeOutput:
        device = torch.device(model_management.get_torch_device())
        t = torch.arange(8, device=device, dtype=torch.float32).reshape(1, 4, 1, 2)
        ptr = int(t.data_ptr())
        logger.warning("[ZeroCopyArange] device=%s data_ptr=%d shape=%s", t.device, ptr, tuple(t.shape))
        return io.NodeOutput({"samples": t}, str(t.device), ptr)


class ZeroCopyExtension_ISO(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [ZeroCopyArange]
