from __future__ import annotations

import json
import logging
import torch
from comfy import model_management
from comfy_api.latest import ComfyExtension, io

logger = logging.getLogger(__name__)

class ZeroCopyIPCTest_ISO(io.ComfyNode):
    """Report device/is_cuda/data_ptr for incoming latent without touching storage."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ZeroCopyIPCTest_ISO",
            display_name="ZeroCopyIPCTest_ISO",
            category="isolation/tests",
            description="Inspect latent tensor device/is_cuda/data_ptr",
            inputs=[io.Latent.Input("latent")],
            outputs=[
                io.Latent.Output("latent_out", display_name="latent"),
                io.String.Output("info"),
            ],
        )

    @classmethod
    def execute(cls, latent):
        samples = latent.get("samples") if isinstance(latent, dict) else None
        device_str = None
        is_cuda = None
        data_ptr = None
        shape = None
        if isinstance(samples, torch.Tensor):
            target = torch.device(model_management.get_torch_device())
            if samples.device != target:
                samples = samples.to(target, non_blocking=True)
                latent = {**latent, "samples": samples}
            device_str = str(samples.device)
            is_cuda = samples.is_cuda
            data_ptr = samples.data_ptr()
            shape = tuple(samples.shape)
        info = {
            "device": device_str,
            "is_cuda": is_cuda,
            "data_ptr": data_ptr,
            "shape": shape,
        }
        logger.warning("[ZeroCopyIPCTest_ISO] %s", info)
        return io.NodeOutput(latent, json.dumps(info))



class ZeroCopyTestExtension_ISO(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [ZeroCopyIPCTest_ISO]




async def comfy_entrypoint() -> ZeroCopyTestExtension_ISO:
    return ZeroCopyTestExtension_ISO()

