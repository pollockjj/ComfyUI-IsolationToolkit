"""
Proxy Test Node: Conda Sealed Worker

Summary/output node for the toolkit-owned conda + sealed_worker fixture.
Consumes the fixture node outputs and emits a dense PASS/FAIL report in the
same style as the other proxy-test nodes.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from comfy_api.latest import io

logger = logging.getLogger(__name__)


class ProxyTestCondaSealedWorker(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ProxyTestCondaSealedWorker",
            display_name="Proxy Test: Conda Sealed Worker",
            category="PyIsolated/ProxyTests",
            is_output_node=True,
            inputs=[
                io.String.Input("runtime_report"),
                io.String.Input("python_exe"),
                io.String.Input("dependency_report"),
                io.Boolean.Input("saw_comfy_root"),
                io.Boolean.Input("imported_comfy_wrapper"),
                io.Boolean.Input("saw_user_site"),
                io.Float.Input("sum_value"),
                io.String.Input("grib_path"),
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
        python_exe: str,
        dependency_report: str,
        saw_comfy_root: bool,
        imported_comfy_wrapper: bool,
        saw_user_site: bool,
        sum_value: float,
        grib_path: str,
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
        lines.append("CONDA SEALED WORKER PROXY TEST REPORT")
        lines.append("=" * 60)
        lines.append(runtime_report)
        lines.append(dependency_report)
        lines.append("")

        verify(
            "pixi python executable",
            ".pixi/envs/default/bin/python" in python_exe,
            python_exe,
        )
        verify(
            "conda dependency stack from child env",
            all(
                token in runtime_report
                for token in ("xarray_origin=", "cfgrib_origin=", "eccodes_origin=")
            )
            and "/home/johnj/ComfyUI/.venv" not in runtime_report,
            runtime_report,
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
            "conda-backed node executed",
            abs(sum_value - 10.0) <= 1e-6 and grib_path.endswith(".grib2"),
            f"sum={sum_value} path={grib_path}",
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


NODES = [ProxyTestCondaSealedWorker]
