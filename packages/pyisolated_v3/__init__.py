from __future__ import annotations

import os

from comfy_api.latest import ComfyExtension

from .nodes import (
    PyIsolatedTestNodeV3,
    PyIsolatedExecuteV3,
    PyIsolatedExecuteAdvancedV3,
    TestCLIPProxy_APISO,
    ZeroCopyArange,
)
from .nodes_adversarial import AdversarialSummary
from .nodes_free_memory import FreeMemoryImagePassthrough
from .nodes_gate import GateAny
from .nodes_proxy_test_cli_args import ProxyTestCliArgs
from .nodes_proxy_test_clip import ProxyTestCLIP
from .nodes_proxy_test_folder_paths import ProxyTestFolderPaths
from .nodes_proxy_test_latent_formats import ProxyTestLatentFormats
from .nodes_proxy_test_model_management import ProxyTestModelManagement
from .nodes_proxy_test_model_patcher import ProxyTestModelPatcher
from .nodes_proxy_test_model_sampler import ProxyTestModelSampler
from .nodes_proxy_test_preview_pipeline import ProxyTestPreviewPipeline
from .nodes_proxy_test_utils import ProxyTestUtils
from .nodes_proxy_test_vae import ProxyTestVAE
from .nodes_security_audit import SecurityAudit

EXPERIMENTAL_ENV = "COMFY_ISOLATION_TOOLKIT_EXPERIMENTAL"

CORE_NODES = [
    PyIsolatedTestNodeV3,
    PyIsolatedExecuteV3,
    PyIsolatedExecuteAdvancedV3,
    ZeroCopyArange,
    GateAny,
    FreeMemoryImagePassthrough,
    TestCLIPProxy_APISO,
]

PROXY_TEST_NODES = [
    ProxyTestModelManagement,
    ProxyTestFolderPaths,
    ProxyTestUtils,
    ProxyTestLatentFormats,
    ProxyTestModelPatcher,
    ProxyTestCLIP,
    ProxyTestVAE,
    ProxyTestModelSampler,
    ProxyTestCliArgs,
    ProxyTestPreviewPipeline,
]

EXPERIMENTAL_NODES = [
    AdversarialSummary,
    SecurityAudit,
]


class PyIsolatedExtension(ComfyExtension):
    async def get_node_list(self):
        node_list = list(CORE_NODES) + list(PROXY_TEST_NODES)
        if os.environ.get(EXPERIMENTAL_ENV) == "1":
            node_list.extend(EXPERIMENTAL_NODES)
        return node_list


async def comfy_entrypoint() -> PyIsolatedExtension:
    return PyIsolatedExtension()
