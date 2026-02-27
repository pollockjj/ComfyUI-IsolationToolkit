from __future__ import annotations

import importlib
import inspect
from pathlib import Path

from comfy_api.latest import ComfyExtension, IO

__all__ = ["comfy_entrypoint"]


def _discover_extensions() -> list[ComfyExtension]:
    package_dir = Path(__file__).parent
    module_names = sorted(
        p.stem for p in package_dir.glob("nodes_*.py") if p.is_file() and p.stem != "__init__"
    )
    discovered: list[ComfyExtension] = []

    for module_name in module_names:
        module = importlib.import_module(f".{module_name}", __name__)
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if not issubclass(obj, ComfyExtension) or obj is ComfyExtension:
                continue
            if obj.__module__ != module.__name__:
                continue
            if not obj.__name__.endswith("Extension_ISO"):
                continue
            discovered.append(obj())

    return discovered


class CompositeIsolationTestExtension(ComfyExtension):
    def __init__(self) -> None:
        self.extensions = _discover_extensions()

    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        nodes: list[type[IO.ComfyNode]] = []
        for ext in self.extensions:
            nodes.extend(await ext.get_node_list())
        return nodes


async def comfy_entrypoint():
    return CompositeIsolationTestExtension()
