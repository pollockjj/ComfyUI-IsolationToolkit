from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

class wanBlockSwap_ISO(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='wanBlockSwap_ISO', category='', description='NOP', inputs=[io.Model.Input('model')], outputs=[io.Model.Output()], is_deprecated=True)

    @classmethod
    def execute(cls, model) -> io.NodeOutput:
        return io.NodeOutput(model)

class NopExtension_ISO(ComfyExtension):

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [wanBlockSwap_ISO]

async def comfy_entrypoint() -> NopExtension_ISO:
    return NopExtension_ISO()
