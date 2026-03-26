from comfy_api.latest import IO, ComfyExtension
from typing_extensions import override


class VAEDecode_ISO(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id='VAEDecode_ISO',
            category='latent',
            inputs=[
                IO.Latent.Input('samples'),
                IO.Vae.Input('vae')
            ],
            outputs=[IO.Image.Output()]
        )

    @classmethod
    def execute(cls, vae, samples) -> IO.NodeOutput:
        images = vae.decode(samples['samples'])
        if len(images.shape) == 5:  # Combine batches
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return IO.NodeOutput(images)

    decode = execute


class CLIPTextEncode_ISO(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id='CLIPTextEncode_ISO',
            category='conditioning',
            inputs=[
                IO.String.Input('text', multiline=True),
                IO.Clip.Input('clip')
            ],
            outputs=[IO.Conditioning.Output()]
        )

    @classmethod
    def execute(cls, clip, text) -> IO.NodeOutput:
        if clip is None:
            raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")
        tokens = clip.tokenize(text)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return IO.NodeOutput(conditioning)

    encode = execute


class VaeClipBasicExtension_ISO(ComfyExtension):

    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [VAEDecode_ISO, CLIPTextEncode_ISO]


async def comfy_entrypoint() -> VaeClipBasicExtension_ISO:
    return VaeClipBasicExtension_ISO()


