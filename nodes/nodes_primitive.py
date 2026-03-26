import sys
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

class String_ISO(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='PrimitiveString_ISO', display_name='String_ISO', category='utils/primitive', inputs=[io.String.Input('value')], outputs=[io.String.Output()])

    @classmethod
    def execute(cls, value: str) -> io.NodeOutput:
        return io.NodeOutput(value)

class StringMultiline_ISO(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='PrimitiveStringMultiline_ISO', display_name='String (Multiline)_ISO', category='utils/primitive', inputs=[io.String.Input('value', multiline=True)], outputs=[io.String.Output()])

    @classmethod
    def execute(cls, value: str) -> io.NodeOutput:
        return io.NodeOutput(value)

class Int_ISO(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='PrimitiveInt_ISO', display_name='Int_ISO', category='utils/primitive', inputs=[io.Int.Input('value', min=-sys.maxsize, max=sys.maxsize, control_after_generate=True)], outputs=[io.Int.Output()])

    @classmethod
    def execute(cls, value: int) -> io.NodeOutput:
        return io.NodeOutput(value)

class Float_ISO(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='PrimitiveFloat_ISO', display_name='Float_ISO', category='utils/primitive', inputs=[io.Float.Input('value', min=-sys.maxsize, max=sys.maxsize)], outputs=[io.Float.Output()])

    @classmethod
    def execute(cls, value: float) -> io.NodeOutput:
        return io.NodeOutput(value)

class Boolean_ISO(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='PrimitiveBoolean_ISO', display_name='Boolean_ISO', category='utils/primitive', inputs=[io.Boolean.Input('value')], outputs=[io.Boolean.Output()])

    @classmethod
    def execute(cls, value: bool) -> io.NodeOutput:
        return io.NodeOutput(value)

class PrimitivesExtension_ISO(ComfyExtension):

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [String_ISO, StringMultiline_ISO, Int_ISO, Float_ISO, Boolean_ISO]

async def comfy_entrypoint() -> PrimitivesExtension_ISO:
    return PrimitivesExtension_ISO()
