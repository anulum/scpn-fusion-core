#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FRC Rigid Rotor Benchmark
"""Benchmark the accepted no-rotation FRC analytical equilibrium contract."""

from __future__ import annotations

import json
import os
import platform
import subprocess  # nosec B404
import sys
import time
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REPORT = ROOT / "validation" / "reports" / "frc_rigid_rotor_benchmark.json"
RUST_MANIFEST = ROOT / "scpn-fusion-rs" / "Cargo.toml"

sys.path.insert(0, str(SRC))

from scpn_fusion.core.frc_rigid_rotor import (  # noqa: E402
    ELEMENTARY_CHARGE_C,
    MU_0,
    RigidRotorFRCInputs,
    solve_frc_equilibrium,
)

try:
    scpn_fusion_rs = cast(Any, __import__("scpn_fusion_rs"))
    HAS_PYO3 = hasattr(scpn_fusion_rs, "py_solve_frc_equilibrium")
except ImportError:
    scpn_fusion_rs = cast(Any, None)
    HAS_PYO3 = False

FloatArray: TypeAlias = NDArray[np.float64]
MetricRow: TypeAlias = dict[str, Any]

_GRIDS = [65, 129, 257, 513]
_REPEATS = 5
_PARITY_CASES: list[tuple[float | None, int, float, float]] = [
    (0.012, 33, 2.75, 0.16),
    (0.014, 65, 3.25, 0.18),
    (0.016, 129, 3.75, 0.19),
    (0.018, 257, 4.25, 0.20),
    (0.020, 33, 4.75, 0.21),
    (0.022, 65, 5.25, 0.22),
    (0.024, 129, 5.75, 0.23),
    (0.026, 257, 6.25, 0.24),
    (0.028, 33, 6.75, 0.25),
    (0.030, 65, 7.25, 0.26),
    (0.032, 129, 7.75, 0.27),
    (0.034, 257, 8.25, 0.28),
    (None, 33, 3.50, 0.17),
    (None, 65, 5.00, 0.20),
    (None, 129, 6.50, 0.24),
    (None, 257, 8.00, 0.28),
]


def _pressure_matched_density_m3(b_ext: float) -> float:
    return float(b_ext**2 / (2.0 * MU_0) / ((10_000.0 + 5_000.0) * ELEMENTARY_CHARGE_C))


def _inputs() -> RigidRotorFRCInputs:
    return _inputs_for_case(delta=0.02, b_ext=5.0, r_s=0.20)


def _inputs_for_case(delta: float | None, b_ext: float, r_s: float) -> RigidRotorFRCInputs:
    return RigidRotorFRCInputs(
        n0=_pressure_matched_density_m3(b_ext),
        T_i_eV=10_000.0,
        T_e_eV=5_000.0,
        theta_dot=0.0,
        R_s=r_s,
        B_ext=b_ext,
        delta=delta,
    )


def _rho(grid_points: int, r_s: float = 0.20) -> FloatArray:
    return np.linspace(0.0, 2.0 * r_s, grid_points, dtype=np.float64)


def _checksum(values: FloatArray) -> float:
    return float(np.sum(values, dtype=np.float64))


def _state_metric_row(
    *,
    label: str,
    grid_points: int,
    elapsed_s: float,
    state: Any,
    delta: float | None,
    b_ext: float,
    r_s: float,
) -> MetricRow:
    return {
        "case_label": label,
        "delta_m": float(state.delta),
        "input_delta_m": delta,
        "input_b_ext_t": b_ext,
        "input_separatrix_radius_m": r_s,
        "grid_points": grid_points,
        "wall_time_s": elapsed_s,
        "r_null_m": float(state.R_null),
        "target_separatrix_radius_m": float(state.target_separatrix_radius_m),
        "separatrix_radius_error_m": float(state.separatrix_radius_error_m),
        "separatrix_index": int(state.separatrix_index),
        "field_reversal_passed": bool(state.field_reversal_passed),
        "s_parameter": float(state.s_parameter),
        "energy_j_per_m": float(state.energy_J),
        "converged": bool(state.converged),
        "residual": float(state.residual),
        "psi_axis_wb": float(state.psi_axis_Wb),
        "psi_separatrix_wb": float(state.psi_separatrix_Wb),
        "psi_normalized_axis_error": float(state.psi_normalized_axis_error),
        "psi_normalized_separatrix": float(state.psi_normalized_separatrix),
        "psi_normalized_separatrix_error": float(state.psi_normalized_separatrix_error),
        "psi_normalized_residual_linf": float(state.psi_normalized_residual_linf),
        "psi_normalized_monotonic_passed": bool(state.psi_normalized_monotonic_passed),
        "psi_normalized_bounds_passed": bool(state.psi_normalized_bounds_passed),
        "pressure_balance_ratio": float(state.pressure_balance_ratio),
        "pressure_balance_residual_linf": float(state.pressure_balance_residual_linf),
        "pressure_balance_residual_l2": float(state.pressure_balance_residual_l2),
        "pressure_gradient_residual_linf": float(state.pressure_gradient_residual_linf),
        "pressure_gradient_residual_l2": float(state.pressure_gradient_residual_l2),
        "peak_pressure_pa": float(state.peak_pressure_pa),
        "density_peak_m3": float(state.density_peak_m3),
        "input_density_m3": float(state.input_density_m3),
        "central_density_residual_m3": float(state.central_density_residual_m3),
        "central_density_relative_error": float(state.central_density_relative_error),
        "beta_peak": float(state.beta_peak),
        "beta_separatrix_average": float(state.beta_separatrix_average),
        "particle_line_density_m1": float(state.particle_line_density_m1),
        "separatrix_pressure_energy_j_m": float(state.separatrix_pressure_energy_J_m),
        "separatrix_magnetic_deficit_energy_j_m": float(
            state.separatrix_magnetic_deficit_energy_J_m
        ),
        "separatrix_energy_closure_relative_error": float(
            state.separatrix_energy_closure_relative_error
        ),
        "input_thermal_pressure_pa": float(state.input_thermal_pressure_pa),
        "thermal_pressure_ratio": float(state.thermal_pressure_ratio),
        "flux_derivative_residual_linf": float(state.flux_derivative_residual_linf),
        "flux_derivative_residual_l2": float(state.flux_derivative_residual_l2),
        "ampere_residual_linf": float(state.ampere_residual_linf),
        "ampere_residual_l2": float(state.ampere_residual_l2),
        "peak_j_theta_a_m2": float(state.peak_j_theta_A_m2),
        "separatrix_bz_gradient_t_m": float(state.separatrix_bz_gradient_T_m),
        "separatrix_expected_bz_gradient_t_m": float(state.separatrix_expected_bz_gradient_T_m),
        "separatrix_gradient_relative_error": float(state.separatrix_gradient_relative_error),
        "separatrix_current_density_a_m2": float(state.separatrix_current_density_A_m2),
        "separatrix_expected_current_density_a_m2": float(
            state.separatrix_expected_current_density_A_m2
        ),
        "separatrix_current_density_relative_error": float(
            state.separatrix_current_density_relative_error
        ),
        "sheet_current_integral_a_m": float(state.sheet_current_integral_A_m),
        "expected_sheet_current_integral_a_m": float(state.expected_sheet_current_integral_A_m),
        "sheet_current_integral_relative_error": float(state.sheet_current_integral_relative_error),
        "force_balance_residual_linf": float(state.force_balance_residual_linf),
        "force_balance_residual_l2": float(state.force_balance_residual_l2),
        "psi_checksum": _checksum(state.psi),
        "psi_normalized_checksum": _checksum(state.psi_normalized),
        "bz_checksum": _checksum(state.B_z),
        "jtheta_checksum": _checksum(state.J_theta),
        "pressure_checksum": _checksum(state.p),
        "density_checksum": _checksum(state.density_m3),
    }


def _pyo3_metric_row(
    *,
    label: str,
    grid_points: int,
    elapsed_s: float,
    state: MetricRow,
    delta: float | None,
    b_ext: float,
    r_s: float,
) -> MetricRow:
    return {
        "case_label": label,
        "delta_m": float(state["delta"]),
        "input_delta_m": delta,
        "input_b_ext_t": b_ext,
        "input_separatrix_radius_m": r_s,
        "grid_points": grid_points,
        "wall_time_s": elapsed_s,
        "r_null_m": float(state["R_null"]),
        "target_separatrix_radius_m": float(state["target_separatrix_radius_m"]),
        "separatrix_radius_error_m": float(state["separatrix_radius_error_m"]),
        "separatrix_index": int(state["separatrix_index"]),
        "field_reversal_passed": bool(state["field_reversal_passed"]),
        "s_parameter": float(state["s_parameter"]),
        "energy_j_per_m": float(state["energy_J"]),
        "converged": bool(state["converged"]),
        "residual": float(state["residual"]),
        "psi_axis_wb": float(state["psi_axis_Wb"]),
        "psi_separatrix_wb": float(state["psi_separatrix_Wb"]),
        "psi_normalized_axis_error": float(state["psi_normalized_axis_error"]),
        "psi_normalized_separatrix": float(state["psi_normalized_separatrix"]),
        "psi_normalized_separatrix_error": float(state["psi_normalized_separatrix_error"]),
        "psi_normalized_residual_linf": float(state["psi_normalized_residual_linf"]),
        "psi_normalized_monotonic_passed": bool(state["psi_normalized_monotonic_passed"]),
        "psi_normalized_bounds_passed": bool(state["psi_normalized_bounds_passed"]),
        "pressure_balance_ratio": float(state["pressure_balance_ratio"]),
        "pressure_balance_residual_linf": float(state["pressure_balance_residual_linf"]),
        "pressure_balance_residual_l2": float(state["pressure_balance_residual_l2"]),
        "pressure_gradient_residual_linf": float(state["pressure_gradient_residual_linf"]),
        "pressure_gradient_residual_l2": float(state["pressure_gradient_residual_l2"]),
        "peak_pressure_pa": float(state["peak_pressure_pa"]),
        "density_peak_m3": float(state["density_peak_m3"]),
        "input_density_m3": float(state["input_density_m3"]),
        "central_density_residual_m3": float(state["central_density_residual_m3"]),
        "central_density_relative_error": float(state["central_density_relative_error"]),
        "beta_peak": float(state["beta_peak"]),
        "beta_separatrix_average": float(state["beta_separatrix_average"]),
        "particle_line_density_m1": float(state["particle_line_density_m1"]),
        "separatrix_pressure_energy_j_m": float(state["separatrix_pressure_energy_J_m"]),
        "separatrix_magnetic_deficit_energy_j_m": float(
            state["separatrix_magnetic_deficit_energy_J_m"]
        ),
        "separatrix_energy_closure_relative_error": float(
            state["separatrix_energy_closure_relative_error"]
        ),
        "input_thermal_pressure_pa": float(state["input_thermal_pressure_pa"]),
        "thermal_pressure_ratio": float(state["thermal_pressure_ratio"]),
        "flux_derivative_residual_linf": float(state["flux_derivative_residual_linf"]),
        "flux_derivative_residual_l2": float(state["flux_derivative_residual_l2"]),
        "ampere_residual_linf": float(state["ampere_residual_linf"]),
        "ampere_residual_l2": float(state["ampere_residual_l2"]),
        "peak_j_theta_a_m2": float(state["peak_j_theta_A_m2"]),
        "separatrix_bz_gradient_t_m": float(state["separatrix_bz_gradient_T_m"]),
        "separatrix_expected_bz_gradient_t_m": float(state["separatrix_expected_bz_gradient_T_m"]),
        "separatrix_gradient_relative_error": float(state["separatrix_gradient_relative_error"]),
        "separatrix_current_density_a_m2": float(state["separatrix_current_density_A_m2"]),
        "separatrix_expected_current_density_a_m2": float(
            state["separatrix_expected_current_density_A_m2"]
        ),
        "separatrix_current_density_relative_error": float(
            state["separatrix_current_density_relative_error"]
        ),
        "sheet_current_integral_a_m": float(state["sheet_current_integral_A_m"]),
        "expected_sheet_current_integral_a_m": float(state["expected_sheet_current_integral_A_m"]),
        "sheet_current_integral_relative_error": float(
            state["sheet_current_integral_relative_error"]
        ),
        "force_balance_residual_linf": float(state["force_balance_residual_linf"]),
        "force_balance_residual_l2": float(state["force_balance_residual_l2"]),
        "psi_checksum": _checksum(cast(FloatArray, np.asarray(state["psi"], dtype=np.float64))),
        "psi_normalized_checksum": _checksum(
            cast(FloatArray, np.asarray(state["psi_normalized"], dtype=np.float64))
        ),
        "bz_checksum": _checksum(cast(FloatArray, np.asarray(state["B_z"], dtype=np.float64))),
        "jtheta_checksum": _checksum(
            cast(FloatArray, np.asarray(state["J_theta"], dtype=np.float64))
        ),
        "pressure_checksum": _checksum(cast(FloatArray, np.asarray(state["p"], dtype=np.float64))),
        "density_checksum": _checksum(
            cast(FloatArray, np.asarray(state["density_m3"], dtype=np.float64))
        ),
    }


def _python_grid_metrics() -> list[MetricRow]:
    rows: list[MetricRow] = []
    inputs = _inputs()
    for grid_points in _GRIDS:
        rho = _rho(grid_points)
        elapsed: list[float] = []
        state: Any = None
        for _ in range(_REPEATS):
            start = time.perf_counter()
            state = solve_frc_equilibrium(inputs, rho)
            elapsed.append(time.perf_counter() - start)
        rows.append(
            _state_metric_row(
                label="grid_convergence_pressure_matched_no_rotation",
                grid_points=grid_points,
                elapsed_s=float(np.median(elapsed)),
                state=state,
                delta=inputs.delta,
                b_ext=inputs.B_ext,
                r_s=inputs.R_s,
            )
        )
    return rows


def _python_parameter_case_metrics() -> list[MetricRow]:
    rows: list[MetricRow] = []
    for index, (delta, grid_points, b_ext, r_s) in enumerate(_PARITY_CASES):
        inputs = _inputs_for_case(delta=delta, b_ext=b_ext, r_s=r_s)
        rho = _rho(grid_points, r_s)
        start = time.perf_counter()
        state = solve_frc_equilibrium(inputs, rho)
        rows.append(
            _state_metric_row(
                label=f"mif_frc_no_rotation_parity_{index:02}",
                grid_points=grid_points,
                elapsed_s=time.perf_counter() - start,
                state=state,
                delta=delta,
                b_ext=b_ext,
                r_s=r_s,
            )
        )
    return rows


def _rust_metrics() -> tuple[str, list[MetricRow] | None, list[MetricRow] | None]:
    command = [
        "cargo",
        "run",
        "--manifest-path",
        str(RUST_MANIFEST),
        "--package",
        "fusion-physics",
        "--example",
        "frc_rigid_rotor_metrics",
        "--quiet",
    ]
    try:
        # Fixed local benchmark argv, no shell.
        completed = subprocess.run(  # nosec B603
            command,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return "unavailable", None, None
    payload = json.loads(completed.stdout)
    return (
        "available",
        cast(list[MetricRow], payload["grids"]),
        cast(list[MetricRow], payload["parameter_cases"]),
    )


def _pyo3_grid_metrics() -> tuple[str, list[MetricRow] | None]:
    if not HAS_PYO3:
        return "unavailable", None
    rows: list[MetricRow] = []
    inputs = _inputs()
    for grid_points in _GRIDS:
        rho = _rho(grid_points)
        elapsed: list[float] = []
        state: MetricRow = {}
        for _ in range(_REPEATS):
            start = time.perf_counter()
            state = cast(
                MetricRow,
                scpn_fusion_rs.py_solve_frc_equilibrium(
                    rho,
                    inputs.n0,
                    inputs.T_i_eV,
                    inputs.T_e_eV,
                    inputs.theta_dot,
                    inputs.R_s,
                    inputs.B_ext,
                    inputs.delta,
                    1.0e-10,
                ),
            )
            elapsed.append(time.perf_counter() - start)
        rows.append(
            _pyo3_metric_row(
                label="grid_convergence_pressure_matched_no_rotation",
                grid_points=grid_points,
                elapsed_s=float(np.median(elapsed)),
                state=state,
                delta=inputs.delta,
                b_ext=inputs.B_ext,
                r_s=inputs.R_s,
            )
        )
    return "available", rows


def _pyo3_parameter_case_metrics() -> tuple[str, list[MetricRow] | None]:
    if not HAS_PYO3:
        return "unavailable", None
    rows: list[MetricRow] = []
    for index, (delta, grid_points, b_ext, r_s) in enumerate(_PARITY_CASES):
        inputs = _inputs_for_case(delta=delta, b_ext=b_ext, r_s=r_s)
        rho = _rho(grid_points, r_s)
        start = time.perf_counter()
        state = cast(
            MetricRow,
            scpn_fusion_rs.py_solve_frc_equilibrium(
                rho,
                inputs.n0,
                inputs.T_i_eV,
                inputs.T_e_eV,
                inputs.theta_dot,
                inputs.R_s,
                inputs.B_ext,
                inputs.delta,
                1.0e-10,
            ),
        )
        rows.append(
            _pyo3_metric_row(
                label=f"mif_frc_no_rotation_parity_{index:02}",
                grid_points=grid_points,
                elapsed_s=time.perf_counter() - start,
                state=state,
                delta=delta,
                b_ext=b_ext,
                r_s=r_s,
            )
        )
    return "available", rows


def _max_abs_delta(left: list[MetricRow], right: list[MetricRow], key: str) -> float:
    return max(
        abs(float(item[key]) - float(other[key])) for item, other in zip(left, right, strict=True)
    )


def _parity(
    reference: list[MetricRow], candidate: list[MetricRow] | None, status: str
) -> MetricRow:
    if candidate is None or status != "available":
        return {"status": status}
    return {
        "status": status,
        "same_row_count": len(reference) == len(candidate),
        "max_r_null_delta_m": _max_abs_delta(reference, candidate, "r_null_m"),
        "max_psi_axis_delta_wb": _max_abs_delta(reference, candidate, "psi_axis_wb"),
        "max_psi_normalized_checksum_delta": _max_abs_delta(
            reference, candidate, "psi_normalized_checksum"
        ),
        "max_ampere_linf_delta": _max_abs_delta(reference, candidate, "ampere_residual_linf"),
        "max_force_linf_delta": _max_abs_delta(reference, candidate, "force_balance_residual_linf"),
        "all_field_reversal_passed": all(bool(row["field_reversal_passed"]) for row in candidate),
        "all_psi_normalized_monotonic_passed": all(
            bool(row["psi_normalized_monotonic_passed"]) for row in candidate
        ),
        "all_psi_normalized_bounds_passed": all(
            bool(row["psi_normalized_bounds_passed"]) for row in candidate
        ),
        "metrics": candidate,
    }


def _grid_convergence(rows: list[MetricRow]) -> MetricRow:
    ampere_values = [float(row["ampere_residual_linf"]) for row in rows]
    force_values = [float(row["force_balance_residual_linf"]) for row in rows]
    return {
        "ampere_residual_linf": ampere_values,
        "force_balance_residual_linf": force_values,
        "ampere_refines": ampere_values[-1] < ampere_values[0],
        "force_finite": all(np.isfinite(value) for value in force_values),
        "psi_normalized_bounds_all_passed": all(
            bool(row["psi_normalized_bounds_passed"]) for row in rows
        ),
        "psi_normalized_monotonic_all_passed": all(
            bool(row["psi_normalized_monotonic_passed"]) for row in rows
        ),
    }


def _benchmark_evidence() -> MetricRow:
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    load_average = os.getloadavg() if hasattr(os, "getloadavg") else None
    return {
        "classification": "local_non_isolated_regression",
        "command": "PYTHONPATH=src python benchmarks/bench_frc_rigid_rotor.py",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_affinity": affinity,
        "host_load_average": load_average,
        "timing_note": (
            "Wall-clock timings are local workstation regression evidence only unless "
            "paired with isolated-core benchmark metadata."
        ),
    }


def main() -> None:
    python_rows = _python_grid_metrics()
    python_case_rows = _python_parameter_case_metrics()
    rust_status, rust_rows, rust_case_rows = _rust_metrics()
    pyo3_status, pyo3_rows = _pyo3_grid_metrics()
    pyo3_case_status, pyo3_case_rows = _pyo3_parameter_case_metrics()
    if pyo3_status != pyo3_case_status:
        pyo3_status = "partial"

    report: MetricRow = {
        "schema_version": 2,
        "benchmark_id": "frc_rigid_rotor_no_rotation_analytical",
        "benchmark": "frc_rigid_rotor",
        "contract": "Steinhauer no-rotation FRC analytical equilibrium",
        "benchmark_evidence": _benchmark_evidence(),
        "scope": (
            "Accepted no-rotation MIF/FRC analytical contract only; rotating rigid-rotor "
            "BVP parity remains fail-closed until the finite-temperature rotation model lands."
        ),
        "local_timing_notice": (
            "Wall-clock timings are local workstation regression evidence only unless paired "
            "with isolated-core benchmark metadata."
        ),
        "python_numpy": {
            "status": "available",
            "repeats": _REPEATS,
            "metrics": python_rows,
            "grid_convergence": _grid_convergence(python_rows),
        },
        "rust_fusion_physics": _parity(python_rows, rust_rows, rust_status),
        "python_pyo3": _parity(python_rows, pyo3_rows, pyo3_status),
        "parameter_case_parity": {
            "case_count": len(_PARITY_CASES),
            "reference_surface": "python_numpy",
            "python_numpy": {"status": "available", "metrics": python_case_rows},
            "rust_fusion_physics": _parity(python_case_rows, rust_case_rows, rust_status),
            "python_pyo3": _parity(python_case_rows, pyo3_case_rows, pyo3_status),
            "go": {"status": "not_applicable_no_frc_surface"},
            "julia": {"status": "not_applicable_no_frc_surface"},
            "lean": {"status": "not_applicable_no_frc_surface"},
        },
        "polyglot_surface_contract": {
            "python_numpy": "reference analytical implementation",
            "rust_fusion_physics": "native solver implementation",
            "python_pyo3": "Rust implementation exposed through PyO3",
            "go": "not applicable; no FRC equilibrium surface is exposed",
            "julia": "not applicable; no FRC equilibrium surface is exposed",
            "lean": "not applicable; no FRC equilibrium surface is exposed",
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
