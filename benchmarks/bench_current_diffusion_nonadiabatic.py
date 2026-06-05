#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Non-Adiabatic Current Diffusion Benchmark
"""Benchmark the non-adiabatic MIF/FRC flux-evolution carrier."""

from __future__ import annotations

import json
from importlib import import_module
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REPORT = ROOT / "validation" / "reports" / "current_diffusion_nonadiabatic_benchmark.json"
CRITERION_ROOT = ROOT / "scpn-fusion-rs" / "target" / "criterion" / "current_diffusion_nonadiabatic"

sys.path.insert(0, str(SRC))

solve_flux_evolution_nonadiabatic = import_module(
    "scpn_fusion.core.current_diffusion"
).solve_flux_evolution_nonadiabatic

FloatArray: TypeAlias = NDArray[np.float64]

_REPEATS = 5
_CASES = [(64, 128), (256, 128), (1024, 128)]


def _benchmark_case(n_rho: int, n_steps: int) -> dict[str, Any]:
    rho: FloatArray = np.linspace(0.0, 1.0, n_rho, dtype=np.float64)
    psi0: FloatArray = 0.08 - 0.02 * rho**2
    dt = 1.0e-8

    def e_theta(time_s: float, grid: FloatArray) -> FloatArray:
        return 25.0 + time_s / (n_steps * dt) + grid

    def j_theta(_: float, grid: FloatArray) -> FloatArray:
        return 1.0e5 * (1.0 - 0.3 * grid)

    elapsed_s: list[float] = []
    checksum = 0.0
    max_abs_update_residual = 0.0
    max_abs_source_increment = 0.0
    max_abs_damping_decrement = 0.0
    for _ in range(_REPEATS):
        start = time.perf_counter()
        trajectory = solve_flux_evolution_nonadiabatic(
            rho,
            psi0,
            tau_psi_fn=lambda _: 2.0e-6,
            R_null_t=lambda _: 0.18,
            E_theta_t=e_theta,
            eta_spitzer_fn=lambda grid: np.full_like(grid, 2.5e-6),
            J_theta_t=j_theta,
            dt=dt,
            n_steps=n_steps,
        )
        elapsed_s.append(time.perf_counter() - start)
        checksum = float(np.sum(trajectory.psi[-1], dtype=np.float64))
        max_abs_update_residual = float(np.max(np.abs(trajectory.update_residual)))
        max_abs_source_increment = float(np.max(np.abs(trajectory.source_increment)))
        max_abs_damping_decrement = float(np.max(np.abs(trajectory.damping_decrement)))

    return {
        "language": "python",
        "grid_points": n_rho,
        "steps": n_steps,
        "repeats": _REPEATS,
        "mean_wall_time_s": float(np.mean(elapsed_s)),
        "min_wall_time_s": float(np.min(elapsed_s)),
        "max_wall_time_s": float(np.max(elapsed_s)),
        "final_psi_checksum": checksum,
        "max_abs_update_residual": max_abs_update_residual,
        "max_abs_source_increment": max_abs_source_increment,
        "max_abs_damping_decrement": max_abs_damping_decrement,
    }


def _criterion_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for estimate_path in sorted(CRITERION_ROOT.glob("rust_*_rho_*_steps/new/estimates.json")):
        bench_name = estimate_path.parent.parent.name
        parts = bench_name.split("_")
        if len(parts) != 5 or parts[0] != "rust" or parts[2] != "rho" or parts[4] != "steps":
            continue
        estimates = json.loads(estimate_path.read_text(encoding="utf-8"))
        slope = estimates["slope"]
        confidence = slope["confidence_interval"]
        rows.append(
            {
                "language": "rust",
                "grid_points": int(parts[1]),
                "steps": int(parts[3]),
                "source": "criterion_estimates_json",
                "mean_wall_time_s": float(slope["point_estimate"]) / 1.0e9,
                "confidence_level": float(confidence["confidence_level"]),
                "ci_lower_s": float(confidence["lower_bound"]) / 1.0e9,
                "ci_upper_s": float(confidence["upper_bound"]) / 1.0e9,
                "estimate_path": str(estimate_path.relative_to(ROOT)),
            }
        )
    return rows


def main() -> None:
    load_before = os.getloadavg()
    rows = [_benchmark_case(n_rho, n_steps) for n_rho, n_steps in _CASES]
    rust_rows = _criterion_rows()
    rows.extend(rust_rows)
    load_after = os.getloadavg()
    report = {
        "schema_version": 2,
        "benchmark": "current_diffusion_nonadiabatic",
        "benchmark_evidence_class": "local_regression_non_isolated",
        "production_claim_allowed": False,
        "isolation_method": "none",
        "command": "PYTHONPATH=src python benchmarks/bench_current_diffusion_nonadiabatic.py",
        "rust_criterion_command": (
            "cargo bench -p fusion-core --bench current_diffusion_nonadiabatic_bench "
            "-- --sample-size 10"
        ),
        "rust_rows_status": "present" if rust_rows else "not_found_run_cargo_bench_first",
        "diagnostics_contract": {
            "balance": "psi[n+1] = psi[n] - damping_decrement[n] + source_increment[n]",
            "update_residual_gate": "local regression report records max_abs_update_residual",
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "load_average_before": list(load_before),
            "load_average_after": list(load_after),
        },
        "rows": rows,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
