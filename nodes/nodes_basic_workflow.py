from __future__ import annotations

import os
from typing_extensions import override

import folder_paths
import nodes
from comfy_api.latest import ComfyExtension, io, ui


class CheckpointLoaderSimple_ISO(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CheckpointLoaderSimple_ISO",
            category="loaders",
            inputs=[
                io.Combo.Input(
                    "ckpt_name",
                    options=folder_paths.get_filename_list("checkpoints"),
                    tooltip="The name of the checkpoint (model) to load.",
                )
            ],
            outputs=[io.Model.Output(), io.Clip.Output(), io.Vae.Output()],
        )

    @classmethod
    def execute(cls, ckpt_name) -> io.NodeOutput:
        model, clip, vae = nodes.CheckpointLoaderSimple().load_checkpoint(ckpt_name)
        return io.NodeOutput(model, clip, vae)


class EmptyLatentImage_ISO(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmptyLatentImage_ISO",
            category="latent",
            inputs=[
                io.Int.Input("width", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=8),
                io.Int.Input("height", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=8),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, width, height, batch_size=1) -> io.NodeOutput:
        latent = nodes.EmptyLatentImage().generate(width, height, batch_size)[0]
        return io.NodeOutput(latent)


class LoadImage_ISO(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return io.Schema(
            node_id="LoadImage_ISO",
            category="image",
            inputs=[io.Combo.Input("image", options=sorted(files), upload=io.UploadType.image)],
            outputs=[io.Image.Output(), io.Mask.Output()],
        )

    @classmethod
    def execute(cls, image) -> io.NodeOutput:
        output_image, output_mask = nodes.LoadImage().load_image(image)
        return io.NodeOutput(output_image, output_mask)

    @classmethod
    def fingerprint_inputs(cls, image):
        return nodes.LoadImage.IS_CHANGED(image)

    @classmethod
    def validate_inputs(cls, image):
        return nodes.LoadImage.VALIDATE_INPUTS(image)


class SaveImage_ISO(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveImage_ISO",
            category="image",
            is_output_node=True,
            inputs=[
                io.Image.Input("images", tooltip="The images to save."),
                io.String.Input(
                    "filename_prefix",
                    default="ComfyUI",
                    tooltip=(
                        "The prefix for the file to save. Supports formatting "
                        "tokens like %date:yyyy-MM-dd%."
                    ),
                ),
            ],
            outputs=[],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
        )

    @classmethod
    def execute(
        cls,
        images,
        filename_prefix="ComfyUI",
        prompt=None,
        extra_pnginfo=None,
        PROMPT=None,
        EXTRA_PNGINFO=None,
        **kwargs,
    ) -> io.NodeOutput:
        # Hidden prompt metadata may arrive in upper-case legacy aliases.
        if prompt is None and PROMPT is not None:
            prompt = PROMPT
        if extra_pnginfo is None and EXTRA_PNGINFO is not None:
            extra_pnginfo = EXTRA_PNGINFO

        if hasattr(cls, "hidden") and cls.hidden is not None:
            cls.hidden.prompt = prompt
            cls.hidden.extra_pnginfo = extra_pnginfo

        saved = ui.ImageSaveHelper.get_save_images_ui(images, filename_prefix, cls=cls)
        return io.NodeOutput(ui=saved)


class BasicWorkflowExtension_ISO(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            CheckpointLoaderSimple_ISO,
            EmptyLatentImage_ISO,
            LoadImage_ISO,
            SaveImage_ISO,
        ]


async def comfy_entrypoint() -> BasicWorkflowExtension_ISO:
    return BasicWorkflowExtension_ISO()
