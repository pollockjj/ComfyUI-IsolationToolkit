# ComfyUI-IsolationToolkit

Example custom node pack demonstrating all standard process isolation methods for ComfyUI via `--use-process-isolation` and pyisolate.

Eight workflows progress from simple to complex, showing how to integrate isolation into existing custom nodes:

| # | Workflow | What It Shows |
|:--|:--|:--|
| 0 | No Isolation | Baseline — same workflow without isolation, proving zero behavioral change |
| 1 | String Concatenation | Simplest isolated node — string in, string out, no models |
| 2 | Conditioning + Latent | Passing conditioning and latent tensors across the process boundary |
| 3 | CLIP + VAE Objects | CLIP encoding and VAE decode via RPC proxies |
| 4 | ModelSampling | ModelSampling proxy — sigma conversion across processes |
| 5 | ModelPatcher | Full ModelPatcher proxy — model load, patch, apply, unpatch via RPC |
| 6 | End to End | Complete image generation pipeline running entirely in an isolated process |
| 7 | Shared Model + Conditioning | Two isolated KSampler passes sharing the same model and conditioning — zero-copy tensor transport |

## Requirements

- ComfyUI on the `pyisolate-support` branch (or master once merged)
- `pyisolate>=0.10.1` (included in `requirements.txt`)
- Launch with `--use-process-isolation --disable-cuda-malloc`

## Quick Start

1. Clone into `ComfyUI/custom_nodes/`
2. Start ComfyUI with `--use-process-isolation --disable-cuda-malloc`
3. Open `Workflow > Browse Workflow Templates`
4. Open the `ComfyUI-IsolationToolkit` section
5. Pick a workflow — they progress in complexity from 0 to 7

## For Custom Node Authors

Each workflow corresponds to a node implementation in `packages/`. The isolation manifest is declared in `pyproject.toml` under `[tool.comfy.isolation]`. No code changes are needed beyond the manifest — pyisolate handles environment provisioning, sandbox setup, and RPC proxy wiring automatically.
