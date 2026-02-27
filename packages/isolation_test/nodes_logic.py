from typing import TypedDict
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from comfy_api.latest import _io

class SwitchNode_ISO(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        template = io.MatchType.Template('switch')
        return io.Schema(node_id='ComfySwitchNode_ISO', display_name='Switch_ISO', category='logic', is_experimental=True, inputs=[io.Boolean.Input('switch'), io.MatchType.Input('on_false', template=template, lazy=True, optional=True), io.MatchType.Input('on_true', template=template, lazy=True, optional=True)], outputs=[io.MatchType.Output(template=template, display_name='output')])

    @classmethod
    def check_lazy_status(cls, switch, on_false=..., on_true=...):
        if on_false is ...:
            return ['on_true']
        if on_true is ...:
            return ['on_false']
        if switch and on_true is None:
            return ['on_true']
        if not switch and on_false is None:
            return ['on_false']

    @classmethod
    def validate_inputs(cls, switch, on_false=..., on_true=...):
        if on_false is ... and on_true is ...:
            return 'At least one of on_false or on_true must be connected to Switch node'
        return True

    @classmethod
    def execute(cls, switch, on_true=..., on_false=...) -> io.NodeOutput:
        if on_true is ...:
            return io.NodeOutput(on_false)
        if on_false is ...:
            return io.NodeOutput(on_true)
        return io.NodeOutput(on_true if switch else on_false)

class DCTestNode_ISO(io.ComfyNode):

    class DCValues(TypedDict):
        combo: str
        string: str
        integer: int
        image: io.Image.Type
        subcombo: dict[str]

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='DCTestNode_ISO', display_name='DCTest_ISO', category='logic', is_output_node=True, inputs=[_io.DynamicCombo.Input('combo', options=[_io.DynamicCombo.Option('option1', [io.String.Input('string')]), _io.DynamicCombo.Option('option2', [io.Int.Input('integer')]), _io.DynamicCombo.Option('option3', [io.Image.Input('image')]), _io.DynamicCombo.Option('option4', [_io.DynamicCombo.Input('subcombo', options=[_io.DynamicCombo.Option('opt1', [io.Float.Input('float_x'), io.Float.Input('float_y')]), _io.DynamicCombo.Option('opt2', [io.Mask.Input('mask1', optional=True)])])])])], outputs=[io.AnyType.Output()])

    @classmethod
    def execute(cls, combo: DCValues) -> io.NodeOutput:
        combo_val = combo['combo']
        if combo_val == 'option1':
            return io.NodeOutput(combo['string'])
        elif combo_val == 'option2':
            return io.NodeOutput(combo['integer'])
        elif combo_val == 'option3':
            return io.NodeOutput(combo['image'])
        elif combo_val == 'option4':
            return io.NodeOutput(f'{combo['subcombo']}')
        else:
            raise ValueError(f'Invalid combo: {combo_val}')

class AutogrowNamesTestNode_ISO(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        template = _io.Autogrow.TemplateNames(input=io.Float.Input('float'), names=['a', 'b', 'c'])
        return io.Schema(node_id='AutogrowNamesTestNode_ISO', display_name='AutogrowNamesTest_ISO', category='logic', inputs=[_io.Autogrow.Input('autogrow', template=template)], outputs=[io.String.Output()])

    @classmethod
    def execute(cls, autogrow: _io.Autogrow.Type) -> io.NodeOutput:
        vals = list(autogrow.values())
        combined = ','.join([str(x) for x in vals])
        return io.NodeOutput(combined)

class AutogrowPrefixTestNode_ISO(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        template = _io.Autogrow.TemplatePrefix(input=io.Float.Input('float'), prefix='float', min=1, max=10)
        return io.Schema(node_id='AutogrowPrefixTestNode_ISO', display_name='AutogrowPrefixTest_ISO', category='logic', inputs=[_io.Autogrow.Input('autogrow', template=template)], outputs=[io.String.Output()])

    @classmethod
    def execute(cls, autogrow: _io.Autogrow.Type) -> io.NodeOutput:
        vals = list(autogrow.values())
        combined = ','.join([str(x) for x in vals])
        return io.NodeOutput(combined)

class LogicExtension_ISO(ComfyExtension):

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return []

async def comfy_entrypoint() -> LogicExtension_ISO:
    return LogicExtension_ISO()
