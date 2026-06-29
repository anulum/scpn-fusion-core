#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Dispatcher Kernel-Tier Benchmark
"""Benchmark the canonical fastest-first dispatcher kernel tiers."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REPORT = ROOT / "validation" / "reports" / "dispatcher_kernel_tiers_benchmark.json"
SCHEMA = "scpn-fusion-core.dispatcher-kernel-tiers-benchmark.v1"
KERNEL_NAMES = (
    "shafranov_bv",
    "solve_coil_currents",
    "measure_magnetics",
    "multigrid_solve",
    "simulate_tearing_mode",
)
DEFAULT_REPEATS = 3

sys.path.insert(0, str(SRC))

from scpn_fusion.core import _multi_compat as multi  # noqa: E402
from scpn_fusion.fallback_telemetry import (  # noqa: E402
    reset_fallback_telemetry,
    snapshot_fallback_telemetry,
)

FloatArray: TypeAlias = NDArray[np.float64]
KernelImpl: TypeAlias = Callable[..., Any]
MetricRow: TypeAlias = dict[str, Any]


@dataclass(frozen=True)
class KernelCase:
    """Deterministic benchmark input for one dispatcher kernel."""

    name: str
    call: Callable[[KernelImpl], Any]


def _host_load_average() -> list[float]:
    """Return load-average metadata when the host exposes it."""
    try:
        return [float(value) for value in os.getloadavg()]
    except OSError:
        return []


def _multigrid_problem() -> tuple[FloatArray, FloatArray, float, float, float, float, int, int]:
    """Return a compact fixed-boundary Grad-Shafranov multigrid benchmark case."""
    nr = nz = 33
    r_min, r_max, z_min, z_max = 1.2, 2.2, -0.5, 0.5
    rr, zz = np.meshgrid(np.linspace(r_min, r_max, nr), np.linspace(z_min, z_max, nz))
    source = np.asarray(-rr * np.exp(-((rr - 1.7) ** 2 + zz**2) / 0.05), dtype=np.float64)
    psi_bc = np.zeros((nz, nr), dtype=np.float64)
    return source, psi_bc, r_min, r_max, z_min, z_max, nr, nz


def _checksum(value: Any) -> float:
    """Return a deterministic numeric checksum for nested kernel outputs."""
    if isinstance(value, np.ndarray):
        return float(np.sum(np.asarray(value, dtype=np.float64), dtype=np.float64))
    if isinstance(value, np.generic):
        return float(value)
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, tuple | list):
        return float(sum(_checksum(item) for item in value))
    raise TypeError(f"Unsupported benchmark output type: {type(value).__name__}")


def _output_shape(value: Any) -> str:
    """Return a compact structural descriptor for benchmark output."""
    if isinstance(value, np.ndarray):
        return f"ndarray{tuple(int(part) for part in value.shape)}"
    if isinstance(value, np.generic | bool | int | float):
        return type(value).__name__
    if isinstance(value, tuple):
        return "tuple[" + ",".join(_output_shape(item) for item in value) + "]"
    if isinstance(value, list):
        return "list[" + ",".join(_output_shape(item) for item in value) + "]"
    return type(value).__name__


def _registered_tiers(kernel_name: str) -> list[str]:
    """Return registered tier names without availability markers."""
    tiers = multi.registered_kernels()[kernel_name]
    return [tier.rstrip("*") for tier in tiers]


def _clear_dispatch_cache(kernel_name: str) -> None:
    """Clear the dispatcher cache entry for one benchmark kernel."""
    with multi._registry_lock:
        multi._dispatch_cache.pop(kernel_name, None)


def _kernel_cases() -> list[KernelCase]:
    """Build the deterministic benchmark input deck."""
    psi = np.full((33, 33), 0.7, dtype=np.float64)
    source, psi_bc, r_min, r_max, z_min, z_max, nr, nz = _multigrid_problem()

    return [
        KernelCase(
            "shafranov_bv",
            lambda impl: impl(6.2, 2.0, 15.0, beta_p=0.5, li=0.8),
        ),
        KernelCase(
            "solve_coil_currents",
            lambda impl: impl([0.01, 0.02, 0.015], -0.05, ridge_lambda=1e-3),
        ),
        KernelCase(
            "measure_magnetics",
            lambda impl: impl(psi, 33, 33, 3.0, 9.0, -3.5, 3.5),
        ),
        KernelCase(
            "multigrid_solve",
            lambda impl: impl(
                source,
                psi_bc,
                r_min,
                r_max,
                z_min,
                z_max,
                nr,
                nz,
                tol=1e-6,
                max_cycles=120,
            ),
        ),
        KernelCase(
            "simulate_tearing_mode",
            lambda impl: impl(256, seed=2026, beta_p=0.8, w_crit=0.05),
        ),
    ]


def _run_case(case: KernelCase, *, repeats: int) -> MetricRow:
    """Run one benchmark case and return tier, timing, and output metadata."""
    _clear_dispatch_cache(case.name)
    impl = multi.dispatch(case.name)
    selected_tier = multi.dispatch_tier(case.name)
    best_s: float | None = None
    result: Any = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = case.call(impl)
        elapsed = time.perf_counter() - start
        best_s = elapsed if best_s is None else min(best_s, elapsed)
    assert best_s is not None
    return {
        "kernel": case.name,
        "registered_tiers": _registered_tiers(case.name),
        "selected_tier": selected_tier,
        "selected_provider": getattr(impl, "__name__", type(impl).__name__),
        "repeats": repeats,
        "wall_time_s_min": float(best_s),
        "output_shape": _output_shape(result),
        "output_checksum": _checksum(result),
        "passes": True,
    }


def _validate_numpy_fallback_telemetry(cases: Sequence[KernelCase]) -> MetricRow:
    """Force Rust unavailable and verify NumPy fallback telemetry for every kernel."""
    multi._ensure_probed()
    original_availability = dict(multi._availability)
    reset_fallback_telemetry()
    rows: list[MetricRow] = []
    try:
        multi._availability[multi.BackendTier.RUST] = False
        for case in cases:
            _clear_dispatch_cache(case.name)
            impl = multi.dispatch(case.name)
            selected_tier = multi.dispatch_tier(case.name)
            result = case.call(impl)
            rows.append(
                {
                    "kernel": case.name,
                    "registered_tiers": _registered_tiers(case.name),
                    "selected_tier": selected_tier,
                    "selected_provider": getattr(impl, "__name__", type(impl).__name__),
                    "output_checksum": _checksum(result),
                    "passes": selected_tier == "numpy",
                }
            )
        snapshot = snapshot_fallback_telemetry()
    finally:
        multi._availability.clear()
        multi._availability.update(original_availability)
        for case in cases:
            _clear_dispatch_cache(case.name)
        reset_fallback_telemetry()

    event_count = int(snapshot["domain_counts"].get("multi_backend", 0))
    return {
        "passes": all(bool(row["passes"]) for row in rows) and event_count == len(cases),
        "forced_unavailable_tier": "rust",
        "expected_selected_tier": "numpy",
        "fallback_event_count": event_count,
        "rows": rows,
        "snapshot": snapshot,
    }


def build_report(*, repeats: int = DEFAULT_REPEATS) -> MetricRow:
    """Build the dispatcher kernel-tier benchmark report."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    cases = _kernel_cases()
    rows = [_run_case(case, repeats=repeats) for case in cases]
    telemetry = _validate_numpy_fallback_telemetry(cases)
    all_numpy_floor_passed = all(
        "numpy" in row["registered_tiers"] and bool(row["passes"]) for row in telemetry["rows"]
    )
    gate_passes = (
        len(rows) == len(KERNEL_NAMES)
        and {str(row["kernel"]) for row in rows} == set(KERNEL_NAMES)
        and all(bool(row["passes"]) for row in rows)
        and all_numpy_floor_passed
        and bool(telemetry["passes"])
    )
    return {
        "schema": SCHEMA,
        "benchmark": "dispatcher_kernel_tiers",
        "benchmark_evidence": {
            "classification": "local_non_isolated_regression",
            "command": "PYTHONPATH=src python benchmarks/bench_dispatcher_kernel_tiers.py",
            "platform": platform.platform(),
            "python": platform.python_version(),
            "host_load_average": _host_load_average(),
            "timing_note": (
                "Wall-clock timings are local workstation regression evidence only; "
                "tier selection and fallback telemetry are the accepted contract."
            ),
        },
        "kernels": rows,
        "fallback_telemetry_validation": telemetry,
        "gate": {
            "passes": gate_passes,
            "kernel_count": len(rows),
            "all_numpy_floor_passed": all_numpy_floor_passed,
            "fallback_telemetry_validation_passed": bool(telemetry["passes"]),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark CLI and write the JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPORT, help="JSON report output path.")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help="Timing repeats.")
    args = parser.parse_args(argv)

    report = build_report(repeats=int(args.repeats))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return 0 if bool(report["gate"]["passes"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
