from __future__ import annotations

import logging
import os
from typing import Awaitable, Callable

from comfy_api.latest import ComfyExtension, IO

from .packages.api_isolated import comfy_entrypoint as api_entrypoint
from .packages.isolation_test import comfy_entrypoint as isolation_test_entrypoint
from .packages.pyisolated_v3 import comfy_entrypoint as pyisolated_v3_entrypoint

logger = logging.getLogger(__name__)
STRICT_ENV = "COMFY_ISOLATION_TOOLKIT_STRICT"
ENABLE_API_ENV = "COMFY_ISOLATION_TOOLKIT_ENABLE_API"


async def _collect_nodes(
    label: str,
    entrypoint: Callable[[], Awaitable[ComfyExtension]],
    *,
    required: bool,
) -> list[type[IO.ComfyNode]]:
    try:
        extension = await entrypoint()
        return await extension.get_node_list()
    except Exception:
        if required or os.environ.get(STRICT_ENV) == "1":
            raise
        logger.warning(
            "][ ComfyUI-IsolationToolkit skipped optional bundle '%s' due to import/runtime error",
            label,
            exc_info=True,
        )
        return []


class UnifiedIsolationToolkitExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        node_list: list[type[IO.ComfyNode]] = []
        node_list.extend(
            await _collect_nodes("isolation_test", isolation_test_entrypoint, required=True)
        )
        node_list.extend(
            await _collect_nodes("pyisolated_v3", pyisolated_v3_entrypoint, required=True)
        )

        if os.environ.get(ENABLE_API_ENV, "1") == "1":
            node_list.extend(
                await _collect_nodes("api_isolated", api_entrypoint, required=False)
            )

        return node_list


async def comfy_entrypoint() -> UnifiedIsolationToolkitExtension:
    return UnifiedIsolationToolkitExtension()
