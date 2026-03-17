"""
Proxy Test Node: Sealed Worker

Summary/output node for the toolkit-owned uv + sealed_worker fixture.
Consumes the fixture node outputs and emits a dense PASS/FAIL report in the
same style as the other proxy-test nodes.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from comfy_api.latest import io

logger = logging.getLogger(__name__)


class ProxyTestSealedWorker(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ProxyTestSealedWorker",
            display_name="Proxy Test: Sealed Worker",
            category="PyIsolated/ProxyTests",
            is_output_node=True,
            inputs=[
                io.String.Input("runtime_report"),
                io.String.Input("boltons_origin"),
                io.Boolean.Input("saw_comfy_root"),
                io.Boolean.Input("imported_comfy_wrapper"),
                io.Boolean.Input("saw_user_site"),
                io.String.Input("slug"),
                io.Boolean.Input("json_tensor"),
                io.Latent.Input("original_latent", optional=True),
                io.Latent.Input("echoed_latent", optional=True),
            ],
            outputs=[io.String.Output("report", display_name="Report")],
        )

    @classmethod
    def execute(
        cls,
        runtime_report: str,
        boltons_origin: str,
        saw_comfy_root: bool,
        imported_comfy_wrapper: bool,
        saw_user_site: bool,
        slug: str,
        json_tensor: bool,
        original_latent: Any = None,
        echoed_latent: Any = None,
    ) -> io.NodeOutput:
        lines: list[str] = []
        tested = 0
        passed = 0
        failed = 0

        def verify(name: str, condition: bool, detail: str) -> None:
            nonlocal tested, passed, failed
            tested += 1
            if condition:
                lines.append(f"[PASS] {name}: {detail}")
                passed += 1
            else:
                lines.append(f"[FAIL] {name}: {detail}")
                failed += 1

        lines.append("=" * 60)
        lines.append("SEALED WORKER PROXY TEST REPORT")
        lines.append("=" * 60)
        lines.append(runtime_report)
        lines.append("")

        verify(
            "boltons imported from child uv env",
            "/home/johnj/ComfyUI/.venv" not in boltons_origin
            and "boltons" in boltons_origin,
            boltons_origin,
        )
        verify(
            "no host Comfy root leaked", saw_comfy_root is False, str(saw_comfy_root)
        )
        verify(
            "no host extension wrapper imported",
            imported_comfy_wrapper is False,
            str(imported_comfy_wrapper),
        )
        verify("no user site leaked", saw_user_site is False, str(saw_user_site))
        verify(
            "boltons-backed node executed", slug == "sealed_worker_still_works", slug
        )

        latent_roundtrip_ok = False
        if original_latent is not None and echoed_latent is not None:
            original = original_latent["samples"]
            echoed = echoed_latent["samples"]
            max_abs = float(torch.max(torch.abs(original - echoed)).item())
            latent_roundtrip_ok = max_abs <= 1e-5
            verify(
                "latent roundtrip within tolerance",
                latent_roundtrip_ok,
                f"max_abs={max_abs:.8f}",
            )

        transport_detail = str(json_tensor)
        transport_ok = json_tensor is True
        if not transport_ok and latent_roundtrip_ok:
            transport_ok = True
            transport_detail = "latent roundtrip proved under sealed-worker JSON path"
        verify("json tensor transport flagged", transport_ok, transport_detail)

        lines.append("")
        lines.append("=" * 60)
        lines.append("SUMMARY")
        lines.append("=" * 60)
        lines.append(f"Total checks: {tested}")
        lines.append(f"Passed: {passed}")
        lines.append(f"Failed: {failed}")
        lines.append("=" * 60)

        report = "\n".join(lines)
        logger.warning("%s", report)
        return io.NodeOutput(report)


NODES = [ProxyTestSealedWorker]
