# ComfyUI-IsolationTest

Large compatibility surface for isolating Comfy-style node implementations.

## What Changed
- Extension registration is now auto-discovered from `nodes_*.py`.
- Manual import/registration boilerplate has been removed.

## Why
- Lower maintenance cost as node modules are added/removed.
- Reduces drift and registration bugs in long extension lists.

## Intended Use
- Core isolation parity testing (`*_ISO` nodes).
- Workflow compatibility validation against standard Comfy graph patterns.
