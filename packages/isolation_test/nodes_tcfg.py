from typing_extensions import override
import torch
from comfy_api.latest import ComfyExtension, io

def score_tangential_damping(cond_score: torch.Tensor, uncond_score: torch.Tensor) -> torch.Tensor:
    """Drop tangential components from uncond score to align with cond score."""
    batch_num = cond_score.shape[0]
    cond_score_flat = cond_score.reshape(batch_num, 1, -1).float()
    uncond_score_flat = uncond_score.reshape(batch_num, 1, -1).float()
    score_matrix = torch.cat((uncond_score_flat, cond_score_flat), dim=1)
    try:
        _, _, Vh = torch.linalg.svd(score_matrix, full_matrices=False)
    except RuntimeError:
        _, _, Vh = torch.linalg.svd(score_matrix.cpu(), full_matrices=False)
    v1 = Vh[:, 0:1, :].to(uncond_score_flat.device)
    uncond_score_td = uncond_score_flat @ v1.transpose(-2, -1) * v1
    return uncond_score_td.reshape_as(uncond_score).to(uncond_score.dtype)

class TCFG_ISO(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='TCFG_ISO', display_name='Tangential Damping CFG_ISO', category='advanced/guidance', description='TCFG – Tangential Damping CFG (2503.18137)\n\nRefine the uncond (negative) to align with the cond (positive) for improving quality.', inputs=[io.Model.Input('model')], outputs=[io.Model.Output(display_name='patched_model')])

    @classmethod
    def execute(cls, model):
        m = model.clone()

        def tangential_damping_cfg(args):
            x = args['input']
            conds_out = args['conds_out']
            if len(conds_out) <= 1 or None in args['conds'][:2]:
                return conds_out
            cond_pred = conds_out[0]
            uncond_pred = conds_out[1]
            uncond_td = score_tangential_damping(x - cond_pred, x - uncond_pred)
            uncond_pred_td = x - uncond_td
            return [cond_pred, uncond_pred_td] + conds_out[2:]
        m.set_model_sampler_pre_cfg_function(tangential_damping_cfg)
        return io.NodeOutput(m)

class TcfgExtension_ISO(ComfyExtension):

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [TCFG_ISO]

async def comfy_entrypoint() -> TcfgExtension_ISO:
    return TcfgExtension_ISO()
