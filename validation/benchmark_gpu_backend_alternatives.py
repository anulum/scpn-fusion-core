#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Fail-closed measurement of WGPU and CUDA/JAX GPU backend alternatives.

The two lanes are intentionally independent:

* ``wgpu_physical`` measures the Rust ``fusion-gpu`` WGPU SOR benchmark only
  when the backend exposes a physical GPU adapter. CPU software Vulkan devices
  such as llvmpipe are blocked and never counted as GPU throughput.
* ``cuda_jax`` measures a deterministic JAX workload only when JAX reports a
  CUDA device.

The report is suitable for cloud runs because backend availability, command
lines, elapsed time, and output checksums are recorded in one JSON/Markdown
pair without merging the two evidence classes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "validation" / "reports"
DEFAULT_JSON = REPORT_DIR / "gpu_backend_alternatives.json"
DEFAULT_MD = REPORT_DIR / "gpu_backend_alternatives.md"
REPORT_METADATA = {
    "spdx_license": "AGPL-3.0-or-later",
    "commercial_license": "Commercial license available",
    "concepts_rights": "Concepts 1996-2026 Miroslav Sotek. All rights reserved.",
    "code_rights": "Code 2020-2026 Miroslav Sotek. All rights reserved.",
    "orcid": "0009-0009-3560-0851",
    "contact": "www.anulum.li | protoscience@anulum.li",
}

LANE_PASSED = "passed"
LANE_FAILED = "failed"
LANE_BLOCKED_CPU_ADAPTER = "blocked_cpu_adapter"
LANE_BLOCKED_MISSING_BACKEND = "blocked_missing_backend"


def _run_command(command: list[str], *, cwd: Path = ROOT, timeout_s: int = 180) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            check=False,
        )
        elapsed = time.perf_counter() - start
        return {
            "command": command,
            "elapsed_s": round(elapsed, 6),
            "exit_code": int(completed.returncode),
            "output": completed.stdout,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        return {
            "command": command,
            "elapsed_s": round(elapsed, 6),
            "exit_code": 124,
            "output": str(exc.stdout or "") + str(exc.stderr or ""),
            "timed_out": True,
        }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _array_sha256(array: NDArray[np.float64]) -> str:
    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    return _sha256_bytes(contiguous.tobytes())


def _parse_vulkan_devices(output: str) -> list[dict[str, str]]:
    devices: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if re.match(r"GPU\d+:", line):
            if current:
                devices.append(current)
            current = {"slot": line.rstrip(":")}
            continue
        if current is None or "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        if key in {"deviceName", "deviceType", "driverName", "driverInfo", "vendorID"}:
            current[key] = value
    if current:
        devices.append(current)
    return devices


def _physical_vulkan_device_present(devices: list[dict[str, str]]) -> bool:
    return any(
        device.get("deviceType")
        in {"PHYSICAL_DEVICE_TYPE_DISCRETE_GPU", "PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU"}
        for device in devices
    )


def _summarise_criterion_times(output: str) -> dict[str, str]:
    timings: dict[str, str] = {}
    current_name = ""
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line.startswith("Benchmarking "):
            current_name = line.removeprefix("Benchmarking ").split(":", 1)[0]
        elif current_name and line.startswith("time:"):
            timings[current_name] = line
            current_name = ""
    return timings


def _parse_fusion_gpu_metadata(output: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line.startswith("fusion_gpu_available="):
            metadata["fusion_gpu_available"] = line.split("=", 1)[1].strip() == "true"
        elif line.startswith("fusion_gpu_adapter="):
            adapter = line.split("=", 1)[1].strip()
            metadata["fusion_gpu_adapter"] = adapter
            metadata["fusion_gpu_adapter_physical"] = (
                "DiscreteGpu" in adapter or "IntegratedGpu" in adapter
            )
    return metadata


def measure_wgpu_physical(*, run_bench: bool, timeout_s: int) -> dict[str, Any]:
    """Measure the Rust WGPU lane, blocking CPU software adapters."""
    nvidia = _run_command(["nvidia-smi"], timeout_s=30)
    vulkan = _run_command(["vulkaninfo", "--summary"], timeout_s=30)
    vulkan_devices = _parse_vulkan_devices(str(vulkan["output"])) if vulkan["exit_code"] == 0 else []
    physical_vulkan = _physical_vulkan_device_present(vulkan_devices)

    lane: dict[str, Any] = {
        "lane": "wgpu_physical",
        "status": LANE_BLOCKED_MISSING_BACKEND,
        "nvidia_smi_exit_code": nvidia["exit_code"],
        "vulkaninfo_exit_code": vulkan["exit_code"],
        "vulkan_devices": vulkan_devices,
        "physical_vulkan_device_present": physical_vulkan,
        "command": None,
        "elapsed_s": None,
        "criterion_timings": {},
    }

    if not physical_vulkan:
        software_devices = [
            device for device in vulkan_devices if device.get("deviceType") == "PHYSICAL_DEVICE_TYPE_CPU"
        ]
        lane["status"] = LANE_BLOCKED_CPU_ADAPTER if software_devices else LANE_BLOCKED_MISSING_BACKEND
        lane["blocker"] = (
            "vulkan_wgpu_exposes_only_cpu_software_adapter"
            if software_devices
            else "no_physical_vulkan_wgpu_adapter_detected"
        )
        return lane

    if not run_bench:
        lane["status"] = LANE_BLOCKED_MISSING_BACKEND
        lane["blocker"] = "wgpu_benchmark_not_requested"
        return lane

    command = [
        "cargo",
        "bench",
        "--manifest-path",
        "scpn-fusion-rs/crates/fusion-gpu/Cargo.toml",
        "--bench",
        "gpu_sor_bench",
    ]
    bench = _run_command(command, timeout_s=timeout_s)
    output = str(bench["output"])
    fusion_gpu_metadata = _parse_fusion_gpu_metadata(output)
    lane.update(
        {
            "command": command,
            "elapsed_s": bench["elapsed_s"],
            "exit_code": bench["exit_code"],
            "timed_out": bench["timed_out"],
            "output_sha256": _sha256_bytes(output.encode("utf-8")),
            "criterion_timings": _summarise_criterion_times(output),
            **fusion_gpu_metadata,
        }
    )
    if bench["exit_code"] == 0:
        if fusion_gpu_metadata.get("fusion_gpu_adapter_physical", False):
            lane["status"] = LANE_PASSED
        else:
            lane["status"] = LANE_BLOCKED_CPU_ADAPTER
            lane["blocker"] = "wgpu_benchmark_did_not_report_physical_adapter"
    elif "No suitable physical GPU adapter" in output or "llvmpipe" in output:
        lane["status"] = LANE_BLOCKED_CPU_ADAPTER
        lane["blocker"] = "wgpu_selected_cpu_adapter"
    else:
        lane["status"] = LANE_FAILED
        lane["blocker"] = "wgpu_benchmark_command_failed"
    return lane


def _jax_devices_summary() -> tuple[bool, list[str], str]:
    try:
        import jax  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on optional backend
        return False, [], f"jax_unavailable:{type(exc).__name__}:{exc}"
    devices = [str(device) for device in jax.devices()]
    has_cuda = any("cuda" in str(device).lower() or "gpu" in str(device).lower() for device in devices)
    return has_cuda, devices, str(getattr(jax, "__version__", "unknown"))


def measure_cuda_jax(*, run_bench: bool, size: int, repeats: int) -> dict[str, Any]:
    """Measure a deterministic CUDA/JAX workload when a CUDA device exists."""
    has_cuda, devices, version_or_error = _jax_devices_summary()
    lane: dict[str, Any] = {
        "lane": "cuda_jax",
        "status": LANE_BLOCKED_MISSING_BACKEND,
        "jax_version_or_error": version_or_error,
        "jax_devices": devices,
        "cuda_device_present": has_cuda,
        "size": size,
        "repeats": repeats,
        "elapsed_s": None,
        "median_s": None,
        "result_sha256": None,
    }
    if not has_cuda:
        lane["blocker"] = "jax_cuda_device_not_detected"
        return lane
    if not run_bench:
        lane["blocker"] = "cuda_jax_benchmark_not_requested"
        return lane

    try:
        import jax  # type: ignore[import-not-found]
        import jax.numpy as jnp  # type: ignore[import-not-found]

        base = np.arange(size * size, dtype=np.float32).reshape(size, size) / float(size * size)
        a = jnp.asarray(base)
        b = jnp.asarray(base.T + np.float32(0.125))

        def workload(x: Any, y: Any) -> Any:
            z = x
            for _ in range(4):
                z = jnp.tanh((z @ y) * np.float32(0.01) + z)
            return z

        compiled = jax.jit(workload)
        compiled(a, b).block_until_ready()
        samples: list[float] = []
        result = None
        start_total = time.perf_counter()
        for _ in range(repeats):
            start = time.perf_counter()
            result = compiled(a, b).block_until_ready()
            samples.append(time.perf_counter() - start)
        elapsed = time.perf_counter() - start_total
        result_np = np.asarray(result, dtype=np.float64)
        lane.update(
            {
                "status": LANE_PASSED,
                "elapsed_s": round(float(elapsed), 6),
                "median_s": round(float(np.median(np.asarray(samples, dtype=np.float64))), 6),
                "min_s": round(float(np.min(np.asarray(samples, dtype=np.float64))), 6),
                "max_s": round(float(np.max(np.asarray(samples, dtype=np.float64))), 6),
                "result_sha256": _array_sha256(result_np),
                "result_l2_norm": round(float(np.linalg.norm(result_np)), 12),
            }
        )
    except Exception as exc:  # pragma: no cover - backend dependent
        lane.update(
            {
                "status": LANE_FAILED,
                "blocker": "cuda_jax_benchmark_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    return lane


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    wgpu = measure_wgpu_physical(run_bench=not args.probe_only, timeout_s=args.wgpu_timeout_s)
    cuda = measure_cuda_jax(run_bench=not args.probe_only, size=args.jax_size, repeats=args.jax_repeats)
    lanes = [wgpu, cuda]
    return {
        "_metadata": REPORT_METADATA,
        "schema": "gpu-backend-alternatives.v1",
        "benchmark_id": "gpu_backend_alternatives",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "probe_only": bool(args.probe_only),
        "lanes": lanes,
        "summary": {
            lane["lane"]: lane["status"] for lane in lanes
        },
        "publishable_gpu_throughput": any(lane["status"] == LANE_PASSED for lane in lanes),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
        "<!-- Commercial license available -->",
        "<!-- Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->",
        "<!-- Code 2020-2026 Miroslav Sotek. All rights reserved. -->",
        "<!-- ORCID: 0009-0009-3560-0851 -->",
        "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
        "<!-- SCPN Fusion Core - GPU Backend Alternatives Benchmark -->",
        "",
        "# GPU Backend Alternatives",
        "",
        f"- Schema: `{report['schema']}`",
        f"- Generated UTC: `{report['generated_at_utc']}`",
        f"- Publishable GPU throughput present: `{report['publishable_gpu_throughput']}`",
        "",
        "| Lane | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for lane in report["lanes"]:
        evidence = ""
        if lane["lane"] == "wgpu_physical":
            selected = str(lane.get("fusion_gpu_adapter", "")).strip()
            devices = lane.get("vulkan_devices", [])
            device_names = ", ".join(str(device.get("deviceName", "unknown")) for device in devices)
            if selected:
                evidence = f"selected: {selected}; inventory: {device_names}"
            else:
                evidence = device_names or str(lane.get("blocker", "no device evidence"))
        elif lane["lane"] == "cuda_jax":
            devices = lane.get("jax_devices", [])
            evidence = ", ".join(str(device) for device in devices) or str(
                lane.get("blocker", "no JAX device evidence")
            )
        lines.append(f"| `{lane['lane']}` | `{lane['status']}` | {evidence} |")
    lines.extend(["", "## Timing Details", ""])
    for lane in report["lanes"]:
        lines.append(f"### `{lane['lane']}`")
        lines.append("")
        if lane["lane"] == "wgpu_physical":
            timings = lane.get("criterion_timings", {})
            if timings:
                lines.extend(["| Benchmark | Criterion time line |", "| --- | --- |"])
                for name, timing in timings.items():
                    lines.append(f"| `{name}` | `{timing}` |")
            else:
                lines.append(f"- Blocker: `{lane.get('blocker', 'none')}`")
        elif lane["lane"] == "cuda_jax":
            if lane.get("status") == LANE_PASSED:
                lines.append(f"- Median: `{lane['median_s']}` s")
                lines.append(f"- Result SHA-256: `{lane['result_sha256']}`")
            else:
                lines.append(f"- Blocker: `{lane.get('blocker', 'none')}`")
        lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON, help="JSON report path.")
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD, help="Markdown report path.")
    parser.add_argument("--probe-only", action="store_true", help="Probe backends without running kernels.")
    parser.add_argument("--jax-size", type=int, default=256, help="Square JAX matrix size.")
    parser.add_argument("--jax-repeats", type=int, default=5, help="JAX timed repetitions.")
    parser.add_argument("--wgpu-timeout-s", type=int, default=240, help="WGPU bench timeout.")
    parser.add_argument("--strict", action="store_true", help="Return non-zero unless at least one lane passes.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    if args.strict and not bool(report["publishable_gpu_throughput"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
