# ComfyUI-IsolationToolkit

Example custom node pack demonstrating all standard process isolation methods for ComfyUI via `--use-process-isolation` and [pyisolate](https://github.com/Comfy-Org/pyisolate).

Eight workflows progress from simple to complex, showing how to integrate isolation into existing custom nodes.

## Requirements

- ComfyUI on the [`pyisolate-support`](https://github.com/Comfy-Org/ComfyUI/tree/pyisolate-support) branch (or master once merged)
- `pyisolate>=0.10.1` (included in `requirements.txt`)
- Launch with `--use-process-isolation --disable-cuda-malloc`

## Quick Start

1. Clone into `ComfyUI/custom_nodes/`
2. Start ComfyUI with `--use-process-isolation --disable-cuda-malloc`
3. Open `Workflow > Browse Workflow Templates`
4. Open the `ComfyUI-IsolationToolkit` section
5. Pick a workflow — they progress in complexity from 0 to 7

## Workflows

### 0 — No Isolation (Baseline)

No isolation — the same workflow running natively, proving zero behavioral change when `--use-process-isolation` is passed but the node is not isolated.

<p align="center">
  <a href="example_workflows/isolation_0_no_isolation.json">
    <img src="https://raw.githubusercontent.com/pollockjj/ComfyUI-IsolationToolkit/main/example_workflows/isolation_0_no_isolation.jpg" width="800">
  </a>
</p>

### 1 — String Concatenation

Simplest isolated node — string in, string out, no models involved. Proves the basic RPC round-trip works.

<p align="center">
  <a href="example_workflows/isolation_1_string_concatenation.json">
    <img src="https://raw.githubusercontent.com/pollockjj/ComfyUI-IsolationToolkit/main/example_workflows/isolation_1_string_concatenation.jpg" width="800">
  </a>
</p>

### 2 — Conditioning + Latent

Passing conditioning and latent tensors across the process boundary via zero-copy shared memory.

<p align="center">
  <a href="example_workflows/isolation_2_conditioning_latent.json">
    <img src="https://raw.githubusercontent.com/pollockjj/ComfyUI-IsolationToolkit/main/example_workflows/isolation_2_conditioning_latent.jpg" width="800">
  </a>
</p>

### 3 — CLIP + VAE Objects

CLIP encoding and VAE decode via RPC proxies — heavy objects stay on the host, child gets lightweight proxy handles.

<p align="center">
  <a href="example_workflows/isolation_3_clip_vae_objects.json">
    <img src="https://raw.githubusercontent.com/pollockjj/ComfyUI-IsolationToolkit/main/example_workflows/isolation_3_clip_vae_objects.jpg" width="800">
  </a>
</p>

### 4 — ModelSampling

ModelSampling proxy — sigma conversion and sampling configuration crossing the process boundary.

<p align="center">
  <a href="example_workflows/isolation_4_modelsampler.json">
    <img src="https://raw.githubusercontent.com/pollockjj/ComfyUI-IsolationToolkit/main/example_workflows/isolation_4_modelsampler.jpg" width="800">
  </a>
</p>

### 5 — ModelPatcher

Full ModelPatcher proxy — model load, patch, apply, unpatch via RPC. The most complex proxy in the system, with VRAM headroom pre-allocation before CUDA transfers.

<p align="center">
  <a href="example_workflows/isolation_5_modelpatcher.json">
    <img src="https://raw.githubusercontent.com/pollockjj/ComfyUI-IsolationToolkit/main/example_workflows/isolation_5_modelpatcher.jpg" width="800">
  </a>
</p>

### 6 — End to End

Complete image generation pipeline running entirely in an isolated process — checkpoint load through VAE decode.

<p align="center">
  <a href="example_workflows/isolation_6_end_to_end.json">
    <img src="https://raw.githubusercontent.com/pollockjj/ComfyUI-IsolationToolkit/main/example_workflows/isolation_6_end_to_end.jpg" width="800">
  </a>
</p>

### 7 — Shared Model + Conditioning (Zero-Copy)

Two isolated KSampler passes sharing the same model and conditioning — zero-copy tensor transport via CUDA IPC and `/dev/shm`.

<p align="center">
  <a href="example_workflows/isolation_7_zero_copy_share.json">
    <img src="https://raw.githubusercontent.com/pollockjj/ComfyUI-IsolationToolkit/main/example_workflows/isolation_7_zero_copy_share.jpg" width="800">
  </a>
</p>

## For Custom Node Authors

Each workflow corresponds to a node implementation in `packages/`. The isolation manifest is declared in `pyproject.toml` under `[tool.comfy.isolation]`. No code changes are needed beyond the manifest — pyisolate handles environment provisioning, sandbox setup, and RPC proxy wiring automatically.
