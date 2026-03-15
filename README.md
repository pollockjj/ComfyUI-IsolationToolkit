# ComfyUI-IsolationToolkit

Unified outward-facing isolation toolkit for ComfyOrg evaluation.

This package recreates and combines the node sets from:
- `ComfyUI-IsolationTest`
- `ComfyUI-PyIsolatedV3`
- `ComfyUI-APIsolated`

The original packages remain untouched for internal multi-extension testing.

## Quick Start
1. Install this custom node into `ComfyUI/custom_nodes/`.
2. Start ComfyUI.
3. Open `Workflow > Browse Workflow Templates`.
4. Open the `ComfyUI-IsolationToolkit` section.
5. Pick a template from `example_workflows/`.

## Template Format (Comfy-aligned)
- Official directory: `example_workflows/` at the custom-node root.
- Workflow template files: `*.json`
- Optional preview thumbnails: same filename with `.jpg`
  - Example: `quick_1_isolated_Ksampler.json` + `quick_1_isolated_Ksampler.jpg`
- Reference: https://docs.comfy.org/custom-nodes/walkthrough#workflow-templates

This repository now follows that format for outward-facing examples.

## Included Examples
- `example_workflows/`: Comfy template-browser-ready examples (json + thumbnails).
- `examples/`: copied stability-battery source set for direct/internal usage.

## Sealed Worker Coverage
- `quick_6_uv_sealed_worker.json` and `isolation_7_uv_sealed_worker.json` prove the hermetic `uv + sealed_worker` path under Linux `bwrap`.
- `quick_8_conda_sealed_worker.json` and `isolation_9_conda_sealed_worker.json` prove the hermetic `conda + sealed_worker` path under Linux `bwrap`.
- Both provisioners use the same sealed-worker rules: `share_torch = false`, no host `PYTHONPATH` leakage, no host runtime import leakage, and JSON transport for tensors/latents.
- The hermetic sandbox now runs with an explicit env allowlist instead of inheriting the host environment sweep.
- Writable paths are intentionally narrow: isolated children only get the configured sandbox writable set plus their provisioned environment, while the host Comfy tree and host site-packages stay read-only or absent.

## Notes
- Node type names are preserved.
- Internal layout uses `packages/` to avoid breaking existing source modules while presenting one custom node externally.
