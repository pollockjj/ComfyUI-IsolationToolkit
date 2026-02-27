"""Isolated passthrough for ModelPatcher.

This node is a minimal exercise to prove ModelPatcher transport
host → isolated process → host without touching the object. It accepts a
model input and returns it unchanged.
"""

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override


class ModelPassthrough_ISO(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ModelPassthrough_ISO",
            category="isolation/testing",
            description="Passthrough ModelPatcher through isolation",
            inputs=[io.Model.Input("model")],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model) -> io.NodeOutput:
        # Stateless: return model as-is
        return io.NodeOutput(model)


class ModelPassthroughExtension_ISO(ComfyExtension):

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [ModelPassthrough_ISO]


async def comfy_entrypoint() -> ModelPassthroughExtension_ISO:
    return ModelPassthroughExtension_ISO()
