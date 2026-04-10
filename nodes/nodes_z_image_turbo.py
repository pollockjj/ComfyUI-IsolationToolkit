"""
Isolated node copies for the z_image_turbo workflow.

Every node used in image_z_image_turbo.json, recreated as an isolated V3 API
node with unique node_ids (*_ZIMG) to avoid collision with existing _ISO nodes.

Display names match the originals with an _Isolated suffix.
"""
from __future__ import annotations

import torch
import comfy.sd
import comfy.sample
import comfy.samplers
import comfy.model_management
import comfy.model_sampling
import comfy.utils
import folder_paths
import latent_preview
import nodes
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io, ui


# --------------------------------------------------------------------------- #
# 1. UNETLoader
# --------------------------------------------------------------------------- #
class UNETLoader_ZIMG(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UNETLoader_ZIMG",
            display_name="UNETLoader_Isolated",
            category="advanced/loaders",
            inputs=[
                io.Combo.Input("unet_name", options=folder_paths.get_filename_list("diffusion_models")),
                io.Combo.Input("weight_dtype", options=["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], advanced=True),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, unet_name, weight_dtype) -> io.NodeOutput:
        model_options: dict = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        return io.NodeOutput(model)

    load_unet = execute


# --------------------------------------------------------------------------- #
# 2. CLIPLoader
# --------------------------------------------------------------------------- #
class CLIPLoader_ZIMG(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CLIPLoader_ZIMG",
            display_name="CLIPLoader_Isolated",
            category="advanced/loaders",
            inputs=[
                io.Combo.Input("clip_name", options=folder_paths.get_filename_list("text_encoders")),
                io.Combo.Input(
                    "type",
                    options=[
                        "stable_diffusion", "stable_cascade", "sd3", "stable_audio",
                        "mochi", "ltxv", "pixart", "cosmos", "lumina2", "wan",
                        "hidream", "chroma", "ace", "omnigen2", "qwen_image",
                        "hunyuan_image", "flux2", "ovis", "longcat_image",
                    ],
                ),
                io.Combo.Input("device", options=["default", "cpu"], advanced=True),
            ],
            outputs=[io.Clip.Output()],
        )

    @classmethod
    def execute(cls, clip_name, type="stable_diffusion", device="default") -> io.NodeOutput:
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)

        model_options: dict = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")

        clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
            model_options=model_options,
        )
        return io.NodeOutput(clip)

    load_clip = execute


# --------------------------------------------------------------------------- #
# 3. VAELoader
# --------------------------------------------------------------------------- #
class VAELoader_ZIMG(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VAELoader_ZIMG",
            display_name="VAELoader_Isolated",
            category="loaders",
            inputs=[
                io.Combo.Input("vae_name", options=folder_paths.get_filename_list("vae")),
            ],
            outputs=[io.Vae.Output()],
        )

    @classmethod
    def execute(cls, vae_name) -> io.NodeOutput:
        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
        sd, metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)
        vae = comfy.sd.VAE(sd=sd, metadata=metadata)
        vae.throw_exception_if_invalid()
        return io.NodeOutput(vae)

    load_vae = execute


# --------------------------------------------------------------------------- #
# 4. CLIPTextEncode
# --------------------------------------------------------------------------- #
class CLIPTextEncode_ZIMG(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CLIPTextEncode_ZIMG",
            display_name="CLIPTextEncode_Isolated",
            category="conditioning",
            inputs=[
                io.String.Input("text", multiline=True),
                io.Clip.Input("clip"),
            ],
            outputs=[io.Conditioning.Output()],
        )

    @classmethod
    def execute(cls, clip, text) -> io.NodeOutput:
        if clip is None:
            raise RuntimeError(
                "ERROR: clip input is invalid: None\n\n"
                "If the clip is from a checkpoint loader node your checkpoint "
                "does not contain a valid clip or text encoder model."
            )
        tokens = clip.tokenize(text)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return io.NodeOutput(conditioning)

    encode = execute


# --------------------------------------------------------------------------- #
# 5. ConditioningZeroOut
# --------------------------------------------------------------------------- #
class ConditioningZeroOut_ZIMG(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ConditioningZeroOut_ZIMG",
            display_name="ConditioningZeroOut_Isolated",
            category="advanced/conditioning",
            inputs=[
                io.Conditioning.Input("conditioning"),
            ],
            outputs=[io.Conditioning.Output()],
        )

    @classmethod
    def execute(cls, conditioning) -> io.NodeOutput:
        c = []
        for t in conditioning:
            d = t[1].copy()
            pooled_output = d.get("pooled_output", None)
            if pooled_output is not None:
                d["pooled_output"] = torch.zeros_like(pooled_output)
            conditioning_lyrics = d.get("conditioning_lyrics", None)
            if conditioning_lyrics is not None:
                d["conditioning_lyrics"] = torch.zeros_like(conditioning_lyrics)
            n = [torch.zeros_like(t[0]), d]
            c.append(n)
        return io.NodeOutput(c)

    zero_out = execute


# --------------------------------------------------------------------------- #
# 6. ModelSamplingAuraFlow
# --------------------------------------------------------------------------- #
class ModelSamplingAuraFlow_ZIMG(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ModelSamplingAuraFlow_ZIMG",
            display_name="ModelSamplingAuraFlow_Isolated",
            category="advanced/model",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("shift", default=1.73, min=0.0, max=100.0, step=0.01),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, shift) -> io.NodeOutput:
        m = model.clone()

        sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
        sampling_type = comfy.model_sampling.CONST

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=shift, multiplier=1.0)
        m.add_object_patch("model_sampling", model_sampling)
        return io.NodeOutput(m)

    patch_aura = execute


# --------------------------------------------------------------------------- #
# 7. EmptySD3LatentImage
# --------------------------------------------------------------------------- #
class EmptySD3LatentImage_ZIMG(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmptySD3LatentImage_ZIMG",
            display_name="EmptySD3LatentImage_Isolated",
            category="latent/sd3",
            inputs=[
                io.Int.Input("width", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=1024, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, width, height, batch_size=1) -> io.NodeOutput:
        latent = torch.zeros(
            [batch_size, 16, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
        )
        return io.NodeOutput({"samples": latent, "downscale_ratio_spacial": 8})

    generate = execute


# --------------------------------------------------------------------------- #
# 8. KSampler
# --------------------------------------------------------------------------- #
class KSampler_ZIMG(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KSampler_ZIMG",
            display_name="KSampler_Isolated",
            category="sampling",
            inputs=[
                io.Model.Input("model"),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True),
                io.Int.Input("steps", default=20, min=1, max=10000),
                io.Float.Input("cfg", default=8.0, min=0.0, max=100.0, step=0.1, round=0.01),
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("latent_image"),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise) -> io.NodeOutput:
        latent = latent_image
        latent_samples = latent["samples"]
        latent_samples = comfy.sample.fix_empty_latent_channels(
            model, latent_samples, latent.get("downscale_ratio_spacial", None)
        )

        if denoise <= 0.0:
            out = latent.copy()
            return io.NodeOutput(out)

        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_samples, seed, batch_inds)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        samples = comfy.sample.sample(
            model, noise, steps, cfg, sampler_name, scheduler,
            positive, negative, latent_samples,
            denoise=denoise, disable_noise=False,
            start_step=None, last_step=None,
            force_full_denoise=False, noise_mask=noise_mask,
            callback=callback, disable_pbar=disable_pbar, seed=seed,
        )

        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out["samples"] = samples
        return io.NodeOutput(out)

    sample = execute


# --------------------------------------------------------------------------- #
# 9. VAEDecode
# --------------------------------------------------------------------------- #
class VAEDecode_ZIMG(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VAEDecode_ZIMG",
            display_name="VAEDecode_Isolated",
            category="latent",
            inputs=[
                io.Latent.Input("samples"),
                io.Vae.Input("vae"),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, vae, samples) -> io.NodeOutput:
        images = vae.decode(samples["samples"])
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return io.NodeOutput(images)

    decode = execute


# --------------------------------------------------------------------------- #
# 10. SaveImage
# --------------------------------------------------------------------------- #
class SaveImage_ZIMG(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveImage_ZIMG",
            display_name="SaveImage_Isolated",
            category="image",
            is_output_node=True,
            inputs=[
                io.Image.Input("images"),
                io.String.Input("filename_prefix", default="ComfyUI"),
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
        if prompt is None and PROMPT is not None:
            prompt = PROMPT
        if extra_pnginfo is None and EXTRA_PNGINFO is not None:
            extra_pnginfo = EXTRA_PNGINFO

        if hasattr(cls, "hidden") and cls.hidden is not None:
            cls.hidden.prompt = prompt
            cls.hidden.extra_pnginfo = extra_pnginfo

        import os
        if os.environ.get("PYISOLATE_SANDBOX_MODE") == "required":
            folder = io.FolderType.temp
        else:
            folder = io.FolderType.output

        saved = ui.SavedImages(
            ui.ImageSaveHelper.save_images(
                images,
                filename_prefix=filename_prefix,
                folder_type=folder,
                cls=cls,
            )
        )
        return io.NodeOutput(ui=saved)


# --------------------------------------------------------------------------- #
# Extension registration
# --------------------------------------------------------------------------- #
class ZImageTurboExtension_ISO(ComfyExtension):

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            UNETLoader_ZIMG,
            CLIPLoader_ZIMG,
            VAELoader_ZIMG,
            CLIPTextEncode_ZIMG,
            ConditioningZeroOut_ZIMG,
            ModelSamplingAuraFlow_ZIMG,
            EmptySD3LatentImage_ZIMG,
            KSampler_ZIMG,
            VAEDecode_ZIMG,
            SaveImage_ZIMG,
        ]


async def comfy_entrypoint() -> ZImageTurboExtension_ISO:
    return ZImageTurboExtension_ISO()
