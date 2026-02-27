# ComfyUI-PyIsolatedV3

Isolation toolkit nodes for:
- V3 isolated node examples
- proxy/regression tests for host<->child boundaries
- practical control helpers used by stability workflows

## Node Groups
- Core: `PyIsolated*`, `ZeroCopyArange`, `GateAny`, `FreeMemoryImagePassthrough`
- Proxy tests: `ProxyTest*`
- Experimental (opt-in): `AdversarialSummary`, `SecurityAudit`

## Experimental Toggle
Experimental nodes are disabled by default for cleaner user experience.

Enable with:
```bash
export COMFY_ISOLATION_TOOLKIT_EXPERIMENTAL=1
```

## Purpose
This pack is intended for validating isolation transport, proxy behavior, and
workflow gating patterns without requiring users to understand isolation internals.
