#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FRC Rigid-Rotor Benchmark
"""Benchmark and compare FRC no-rotation analytical solver surfaces."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess  # nosec B404
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from scpn_fusion.core.frc_rigid_rotor import RigidRotorFRCInputs, solve_frc_equilibrium

FloatArray: TypeAlias = NDArray[np.float64]
_RUST_PROJECT = _REPO / "scpn-fusion-rs"
_REPORT_JSON = _REPO / "validation" / "reports" / "frc_rigid_rotor_benchmark.json"
_GRIDS = (64, 256, 1024)
_REPEATS = 5


def _inputs() -> RigidRotorFRCInputs:
    return RigidRotorFRCInputs(
        n0=2.0e20,
        T_i_eV=10_000.0,
        T_e_eV=5_000.0,
        theta_dot=0.0,
        R_s=0.20,
        B_ext=5.0,
        delta=0.02,
    )


def _rho(n: int) -> FloatArray:
    return cast(FloatArray, np.linspace(0.0, 0.4, n))


def _checksum(values: FloatArray) -> float:
    weights: FloatArray = cast(FloatArray, np.arange(1, values.size + 1, dtype=np.float64))
    return float(np.sum(weights * np.asarray(values, dtype=np.float64)))


def _python_metrics() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    inputs = _inputs()
    for n in _GRIDS:
        rho = _rho(n)
        timings: list[float] = []
        state = solve_frc_equilibrium(inputs, rho)
        for _ in range(_REPEATS):
            start = time.perf_counter()
            state = solve_frc_equilibrium(inputs, rho)
            timings.append(time.perf_counter() - start)
        rows.append(
            {
                "grid_points": n,
                "wall_time_s_min": min(timings),
                "wall_time_s_median": float(np.median(np.asarray(timings))),
                "wall_time_s_repeats": timings,
                "r_null_m": state.R_null,
                "s_parameter": state.s_parameter,
                "energy_j_per_m": state.energy_J,
                "pressure_balance_ratio": state.pressure_balance_ratio,
                "force_balance_residual_linf": state.force_balance_residual_linf,
                "b_z_checksum": _checksum(state.B_z),
                "psi_checksum": _checksum(state.psi),
                "p_checksum": _checksum(state.p),
            }
        )
    return rows


def _rust_metrics() -> tuple[str, list[dict[str, Any]] | None]:
    cargo = shutil.which("cargo")
    if cargo is None:
        return "blocked_cargo_not_found", None
    # Fixed Cargo command, no shell, benchmark-only path.
    completed = subprocess.run(  # nosec B603
        [
            cargo,
            "run",
            "--release",
            "-q",
            "-p",
            "fusion-physics",
            "--example",
            "frc_rigid_rotor_metrics",
            "--",
            *[str(n) for n in _GRIDS],
        ],
        cwd=_RUST_PROJECT,
        check=False,
        text=True,
        capture_output=True,
        timeout=240,
    )
    if completed.returncode != 0:
        return f"blocked_rust_example_failed: {completed.stderr.strip()[-500:]}", None
    payload = json.loads(completed.stdout)
    return "available", cast(list[dict[str, Any]], payload["grids"])


def _pyo3_metrics() -> tuple[str, list[dict[str, Any]] | None]:
    try:
        scpn_fusion_rs = cast(Any, __import__("scpn_fusion_rs"))
    except ImportError:
        return "blocked_extension_not_importable", None
    if not hasattr(scpn_fusion_rs, "py_solve_frc_equilibrium"):
        return "blocked_extension_without_frc_surface", None
    rows: list[dict[str, Any]] = []
    inputs = _inputs()
    for n in _GRIDS:
        rho = _rho(n)
        timings: list[float] = []
        state = scpn_fusion_rs.py_solve_frc_equilibrium(
            rho,
            inputs.n0,
            inputs.T_i_eV,
            inputs.T_e_eV,
            inputs.theta_dot,
            inputs.R_s,
            inputs.B_ext,
            inputs.delta,
            1.0e-10,
        )
        for _ in range(_REPEATS):
            start = time.perf_counter()
            state = scpn_fusion_rs.py_solve_frc_equilibrium(
                rho,
                inputs.n0,
                inputs.T_i_eV,
                inputs.T_e_eV,
                inputs.theta_dot,
                inputs.R_s,
                inputs.B_ext,
                inputs.delta,
                1.0e-10,
            )
            timings.append(time.perf_counter() - start)
        rows.append(
            {
                "grid_points": n,
                "wall_time_s_min": min(timings),
                "wall_time_s_median": float(np.median(np.asarray(timings))),
                "wall_time_s_repeats": timings,
                "r_null_m": float(state["R_null"]),
                "s_parameter": float(state["s_parameter"]),
                "energy_j_per_m": float(state["energy_J"]),
                "pressure_balance_ratio": float(state["pressure_balance_ratio"]),
                "force_balance_residual_linf": float(state["force_balance_residual_linf"]),
                "b_z_checksum": _checksum(cast(FloatArray, state["B_z"])),
                "psi_checksum": _checksum(cast(FloatArray, state["psi"])),
                "p_checksum": _checksum(cast(FloatArray, state["p"])),
            }
        )
    return "available", rows


def _relative_error(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1.0)


def _compare_surface(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]] | None,
    status: str,
) -> dict[str, Any]:
    if candidate is None:
        return {"status": status, "parity_passed": False, "comparisons": []}
    by_grid = {int(row["grid_points"]): row for row in candidate}
    comparisons: list[dict[str, Any]] = []
    parity_passed = True
    for ref in reference:
        grid = int(ref["grid_points"])
        cand = by_grid.get(grid)
        if cand is None:
            comparisons.append({"grid_points": grid, "status": "blocked_missing_grid"})
            parity_passed = False
            continue
        checks = {
            "r_null_abs_error": abs(float(cand["r_null_m"]) - float(ref["r_null_m"])),
            "s_parameter_rel_error": _relative_error(
                float(cand["s_parameter"]),
                float(ref["s_parameter"]),
            ),
            "energy_rel_error": _relative_error(
                float(cand["energy_j_per_m"]),
                float(ref["energy_j_per_m"]),
            ),
            "pressure_balance_rel_error": _relative_error(
                float(cand["pressure_balance_ratio"]),
                float(ref["pressure_balance_ratio"]),
            ),
            "force_balance_linf_rel_error": _relative_error(
                float(cand["force_balance_residual_linf"]),
                float(ref["force_balance_residual_linf"]),
            ),
            "b_z_checksum_abs_error": abs(
                float(cand["b_z_checksum"]) - float(ref["b_z_checksum"])
            ),
            "psi_checksum_abs_error": abs(
                float(cand["psi_checksum"]) - float(ref["psi_checksum"])
            ),
            "p_checksum_rel_error": _relative_error(
                float(cand["p_checksum"]),
                float(ref["p_checksum"]),
            ),
        }
        passed = (
            checks["r_null_abs_error"] <= 1.0e-12
            and checks["s_parameter_rel_error"] <= 1.0e-12
            and checks["energy_rel_error"] <= 1.0e-12
            and checks["pressure_balance_rel_error"] <= 1.0e-12
            and checks["force_balance_linf_rel_error"] <= 1.0e-9
            and checks["b_z_checksum_abs_error"] <= 1.0e-9
            and checks["psi_checksum_abs_error"] <= 1.0e-10
            and checks["p_checksum_rel_error"] <= 1.0e-12
        )
        parity_passed = parity_passed and passed
        comparisons.append({"grid_points": grid, "status": "passed" if passed else "failed", **checks})
    return {"status": status, "parity_passed": parity_passed, "comparisons": comparisons}


def main() -> None:
    _REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    python_rows = _python_metrics()
    rust_status, rust_rows = _rust_metrics()
    pyo3_status, pyo3_rows = _pyo3_metrics()
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "benchmark": "frc_rigid_rotor_no_rotation_analytical",
        "contract": "Steinhauer 2011 no-rotation analytical axial-field profile",
        "claim_boundary": "No rotating rigid-rotor BVP, no full FRC transport, no reduced-order substitution.",
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
        },
        "grids": list(_GRIDS),
        "repeats": _REPEATS,
        "surfaces": {
            "python_numpy": {"status": "available", "metrics": python_rows},
            "rust_fusion_physics": {
                "status": rust_status,
                "metrics": rust_rows,
                "parity": _compare_surface(python_rows, rust_rows, rust_status),
            },
            "python_pyo3": {
                "status": pyo3_status,
                "metrics": pyo3_rows,
                "parity": _compare_surface(python_rows, pyo3_rows, pyo3_status),
            },
            "go": {
                "status": "not_applicable_no_frc_surface",
                "reason": "No equivalent FRC rigid-rotor solver is exposed in the Go surface.",
            },
            "julia": {
                "status": "not_applicable_no_frc_surface",
                "reason": "No equivalent FRC rigid-rotor solver is exposed in the Julia surface.",
            },
            "lean": {
                "status": "not_applicable_no_frc_surface",
                "reason": "No equivalent executable FRC rigid-rotor solver is exposed in the Lean surface.",
            },
        },
    }
    _REPORT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("| Surface | Status | Parity | Grid 1024 median (s) |")
    print("|---|---|---:|---:|")
    surfaces = cast(dict[str, dict[str, Any]], report["surfaces"])
    for name, surface in surfaces.items():
        metrics = surface.get("metrics")
        median = "n/a"
        if isinstance(metrics, list) and metrics:
            median = f"{float(metrics[-1].get('wall_time_s_median', metrics[-1].get('wall_time_s', 0.0))):.6e}"
        parity_payload = surface.get("parity", {})
        parity = parity_payload.get("parity_passed", "n/a") if isinstance(parity_payload, dict) else "n/a"
        print(f"| {name} | {surface.get('status', 'unknown')} | {parity} | {median} |")
    print(f"\nWrote {_REPORT_JSON.relative_to(_REPO)}")


if __name__ == "__main__":
    main()
