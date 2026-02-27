# ComfyUI-APIsolated

Isolated API-provider node pack (network-enabled, non-sandbox).

## What Changed
- Extension registration is now auto-discovered from `nodes_*.py`.
- Manual composite registration list has been removed.

## Why
- Easier provider maintenance and cleaner onboarding for new API node modules.
- Reduces hand-edited registry churn.

## Intended Use
- Demonstrate that isolation supports API-node workflows cleanly.
- Validate auth/hidden-input flow through isolated execution.
