#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Fail-closed GPU Phase 1 readiness gate.

This gate verifies that the Rust/wgpu SOR prototype surfaces required by the
GPU roadmap are present and separates static implementation readiness from
missing hardware benchmark artifacts. It never promotes the lane to production
scaling readiness without an explicit tracked GPU benchmark report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "validation" / "reports"
DEFAULT_JSON_REPORT = REPORT_DIR / "gpu_phase1_readiness.json"
DEFAULT_MD_REPORT = REPORT_DIR / "gpu_phase1_readiness.md"
DEFAULT_BENCHMARK_ARTIFACTS = [
    REPORT_DIR / "gpu_backend_alternatives.json",
    REPORT_DIR / "gpu_kernel_benchmark_results.json",
    REPORT_DIR / "gpu_sor_benchmark_results.json",
    ROOT / "artifacts" / "gpu_kernel_benchmark_results.json",
]


def _rel(path: Path, *, root: Path) -> str:
    resolved = path if path.is_absolute() else root / path
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def _sha256_present(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _physical_gpu_adapter_present(payload: dict[str, Any]) -> bool:
    """Return true only for hardware GPU adapter evidence.

    WGPU can run through software Vulkan adapters such as llvmpipe. Those rows
    may be useful for shader smoke tests, but they are not GPU benchmark
    evidence and must not satisfy the readiness gate.
    """
    if payload.get("physical_gpu_adapter") is True:
        return True
    device_type = str(payload.get("adapter_device_type", ""))
    return device_type in {"DiscreteGpu", "IntegratedGpu"}


def _load_benchmark_artifact(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _dual_backend_report_ready(payload: dict[str, Any]) -> bool:
    if payload.get("schema") != "gpu-backend-alternatives.v1":
        return False
    lanes = payload.get("lanes", [])
    if not isinstance(lanes, list):
        return False
    for lane in lanes:
        if not isinstance(lane, dict) or lane.get("status") != "passed":
            continue
        if lane.get("lane") == "wgpu_physical":
            if (
                lane.get("fusion_gpu_adapter_physical") is True
                or lane.get("physical_vulkan_device_present") is True
            ):
                return True
        if lane.get("lane") == "cuda_jax":
            if lane.get("cuda_device_present") is True and _sha256_present(
                lane.get("result_sha256")
            ):
                return True
    return False


def evaluate_gpu_phase1_readiness(
    *,
    root: Path = ROOT,
    benchmark_report_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Evaluate GPU Phase 1 readiness from tracked implementation surfaces."""
    paths = {
        "gpu_crate": root / "scpn-fusion-rs" / "crates" / "fusion-gpu" / "Cargo.toml",
        "gpu_solver": root / "scpn-fusion-rs" / "crates" / "fusion-gpu" / "src" / "lib.rs",
        "gpu_shader": root / "scpn-fusion-rs" / "crates" / "fusion-gpu" / "src" / "gs_solver.wgsl",
        "gpu_bench": root
        / "scpn-fusion-rs"
        / "crates"
        / "fusion-gpu"
        / "benches"
        / "gpu_sor_bench.rs",
        "cpu_sor": root / "scpn-fusion-rs" / "crates" / "fusion-math" / "src" / "sor.rs",
    }
    surface_checks = {key: path.exists() for key, path in paths.items()}
    benchmark_paths = benchmark_report_paths or DEFAULT_BENCHMARK_ARTIFACTS
    artifacts: list[dict[str, Any]] = []
    for path in benchmark_paths:
        artifact = _load_benchmark_artifact(path)
        if artifact is not None:
            artifacts.append({"path": _rel(path, root=root), "payload": artifact})

    benchmark_artifact_ready = any(
        bool(
            _dual_backend_report_ready(artifact["payload"])
            or (
                artifact["payload"].get("gpu_available") is True
                and artifact["payload"].get("solver") in {"wgpu_sor", "gpu_sor"}
                and _physical_gpu_adapter_present(artifact["payload"])
                and (
                    _sha256_present(artifact["payload"].get("output_sha256"))
                    or _sha256_present(artifact["payload"].get("result_sha256"))
                )
            )
        )
        for artifact in artifacts
    )
    phase1_static_ready = all(surface_checks.values())
    accepted = bool(phase1_static_ready and benchmark_artifact_ready)
    blockers: list[str] = []
    if not phase1_static_ready:
        blockers.extend(key for key, ready in surface_checks.items() if not ready)
    if not benchmark_artifact_ready:
        blockers.append("tracked_gpu_physical_wgpu_sor_benchmark_artifact_missing")
    return {
        "schema": "gpu-phase1-readiness.v1",
        "benchmark_id": "gpu_phase1_backend_readiness",
        "status": "accepted_gpu_phase1_backend_readiness"
        if accepted
        else "blocked_gpu_phase1_backend_readiness",
        "accepted_phase1_readiness": accepted,
        "phase1_static_implementation_ready": phase1_static_ready,
        "production_scaling_ready": False,
        "checks": {
            **surface_checks,
            "tracked_gpu_benchmark_artifact_ready": benchmark_artifact_ready,
        },
        "inputs": {key: _rel(path, root=root) for key, path in paths.items()},
        "candidate_benchmark_artifacts": [_rel(path, root=root) for path in benchmark_paths],
        "observed_benchmark_artifact_count": len(artifacts),
        "blockers": blockers,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render GPU Phase 1 readiness as Markdown."""
    lines = [
        "# GPU Phase 1 Readiness",
        "",
        "This gate is fail-closed. Static Rust/wgpu implementation surfaces are",
        "reported separately from hardware benchmark evidence.",
        "",
        f"- Schema: `{report['schema']}`",
        f"- Status: `{report['status']}`",
        f"- Accepted Phase 1 readiness: `{report['accepted_phase1_readiness']}`",
        f"- Static implementation ready: `{report['phase1_static_implementation_ready']}`",
        f"- Production scaling ready: `{report['production_scaling_ready']}`",
        "",
        "## Checks",
        "",
        "| Check | Ready |",
        "| --- | ---: |",
    ]
    for key, ready in report["checks"].items():
        lines.append(f"| `{key}` | `{ready}` |")
    lines.extend(["", "## Blockers", ""])
    if report["blockers"]:
        for blocker in report["blockers"]:
            lines.append(f"- `{blocker}`")
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def run_benchmark(
    *,
    json_report_path: Path = DEFAULT_JSON_REPORT,
    md_report_path: Path = DEFAULT_MD_REPORT,
    write: bool = True,
) -> dict[str, Any]:
    """Run the GPU Phase 1 readiness gate."""
    report = evaluate_gpu_phase1_readiness()
    if write:
        json_report_path.parent.mkdir(parents=True, exist_ok=True)
        json_report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        md_report_path.write_text(render_markdown(report), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    """Run GPU Phase 1 readiness, write outputs, and apply strict checks if needed.

    Args:
        argv: Optional CLI arguments. If ``None``, parse from process arguments.

    Returns:
        ``0`` when readiness is accepted or non-strict mode is used,
        ``1`` when strict mode is enabled and readiness is blocked.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Run without writing reports.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero unless Phase 1 readiness is accepted.",
    )
    args = parser.parse_args(argv)
    report = run_benchmark(write=not args.check)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if (not args.strict or report["accepted_phase1_readiness"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
