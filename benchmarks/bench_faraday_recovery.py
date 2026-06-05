#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Faraday Recovery Benchmark
"""Benchmark the classical MIF/FRC Faraday recovery contract."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from scpn_fusion.core.faraday_recovery import (
    FaradayRecoveryTrajectoryPoint,
    coil_source_work_from_voltage_driven_compression,
    compression_flux_budget_from_pulsed_compression,
    compression_flux_budget_from_voltage_driven_compression,
    compression_trajectory_diagnostics_from_pulsed_compression,
    compression_trajectory_diagnostics_from_voltage_driven_compression,
    compression_work_from_pulsed_compression,
    compression_work_from_voltage_driven_compression,
    faraday_trajectory_from_pulsed_compression,
    faraday_trajectory_from_voltage_driven_compression,
    integrated_recovery_energy,
)
from scpn_fusion.core import solve_frc_equilibrium
from scpn_fusion.core.frc_rigid_rotor import ELEMENTARY_CHARGE_C, MU_0, RigidRotorFRCInputs
from scpn_fusion.core.pulsed_compression import (
    CoilGeometry,
    PulsedCompressionConfig,
    initial_pulsed_compression_state,
    run_pulsed_compression,
    run_voltage_driven_pulsed_compression,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "validation" / "reports" / "faraday_recovery_benchmark.json"
SOURCE_PATHS = [
    REPO_ROOT / "src" / "scpn_fusion" / "core" / "faraday_recovery.py",
    REPO_ROOT / "src" / "scpn_fusion" / "core" / "pulsed_compression.py",
    REPO_ROOT / "scpn-fusion-rs" / "crates" / "fusion-physics" / "src" / "faraday_recovery.rs",
    REPO_ROOT
    / "scpn-fusion-rs"
    / "crates"
    / "fusion-physics"
    / "src"
    / "compression"
    / "pulsed.rs",
    Path(__file__).resolve(),
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trajectory(samples: int) -> list[FaradayRecoveryTrajectoryPoint]:
    duration = 1.0e-6
    times = np.linspace(0.0, duration, samples, dtype=np.float64)
    return [
        FaradayRecoveryTrajectoryPoint(
            t_s=float(t_s),
            separatrix_radius_m=float(0.15 + 4.0e3 * t_s),
            b_ext_t=20.0,
        )
        for t_s in times
    ]


def _python_case(samples: int) -> dict[str, Any]:
    timings = []
    state = None
    trace = _trajectory(samples)
    for _ in range(7):
        start_ns = time.perf_counter_ns()
        state = integrated_recovery_energy(trace, 6, 0.08)
        timings.append(float(time.perf_counter_ns() - start_ns))
    if state is None:
        raise RuntimeError("benchmark did not run")
    return {
        "language": "python",
        "case": f"python_{samples}_samples",
        "samples": samples,
        "mean_seconds": float(np.mean(np.asarray(timings, dtype=np.float64)) * 1.0e-9),
        "samples_seconds": [value * 1.0e-9 for value in timings],
        "recovered_energy_j": state.recovered_energy_j,
        "max_abs_back_emf_v": state.max_abs_back_emf_v,
        "max_abs_load_current_a": state.max_abs_load_current_a,
        "max_abs_flux_rate_field_term_wb_s": state.max_abs_flux_rate_field_term_wb_s,
        "max_abs_flux_rate_radial_term_wb_s": state.max_abs_flux_rate_radial_term_wb_s,
        "max_abs_flux_rate_total_wb_s": state.max_abs_flux_rate_total_wb_s,
        "flux_derivative_residual_linf": state.flux_derivative_residual_linf,
        "flux_derivative_residual_l2": state.flux_derivative_residual_l2,
        "flux_derivative_closure_passed": state.flux_derivative_closure_passed,
        "budget_claim_status": state.budget_claim_status,
        "source_budget_claim_status": state.source_budget_claim_status,
        "compression_flux_budget_claim_status": state.compression_flux_budget_claim_status,
        "compression_trajectory_diagnostics_claim_status": (
            state.compression_trajectory_diagnostics_claim_status
        ),
    }


def _compression_config() -> PulsedCompressionConfig:
    b_ext = 5.0
    t_i = 10_000.0
    t_e = 5_000.0
    n0 = b_ext**2 / (2.0 * MU_0) / ((t_i + t_e) * ELEMENTARY_CHARGE_C)
    equilibrium = solve_frc_equilibrium(
        RigidRotorFRCInputs(
            n0=n0,
            T_i_eV=t_i,
            T_e_eV=t_e,
            theta_dot=0.0,
            R_s=0.20,
            B_ext=b_ext,
            delta=0.02,
        ),
        np.linspace(0.0, 0.4, 65, dtype=np.float64),
    )
    return PulsedCompressionConfig(
        equilibrium=equilibrium,
        coil=CoilGeometry(
            N_turns=80,
            L_coil_m=1.0,
            R_coil_m=0.35,
            L_inductance_H=2.0e-6,
            R_resistance_ohm=0.02,
            bank_voltage_max_V=20_000.0,
        ),
        coil_current_t=lambda _t: 5.0e5,
        plasma_mass_kg=2.0e-5,
        ion_temperature_eV=t_i,
        electron_temperature_eV=t_e,
    )


def _voltage_driven_compression_config() -> PulsedCompressionConfig:
    config = _compression_config()
    return replace(config, coil_current_t=lambda _t: 1.0)


def _python_compression_coupled_case(steps: int) -> dict[str, Any]:
    timings = []
    report = None
    compression_work = 0.0
    for _ in range(5):
        config = _compression_config()
        start_ns = time.perf_counter_ns()
        states = run_pulsed_compression(
            initial_pulsed_compression_state(config),
            config,
            1.0e-9,
            steps,
        )
        trajectory = faraday_trajectory_from_pulsed_compression(states)
        compression_work = compression_work_from_pulsed_compression(states)
        compression_flux_budget = compression_flux_budget_from_pulsed_compression(states)
        compression_trajectory_diagnostics = (
            compression_trajectory_diagnostics_from_pulsed_compression(
                states,
                radius_floor_m=config.min_radius_m,
            )
        )
        report = integrated_recovery_energy(
            trajectory,
            config.coil.N_turns,
            config.coil.R_resistance_ohm,
            compression_work_j=compression_work,
            compression_flux_budget=compression_flux_budget,
            compression_trajectory_diagnostics=compression_trajectory_diagnostics,
        )
        timings.append(float(time.perf_counter_ns() - start_ns))
    if report is None:
        raise RuntimeError("compression-coupled benchmark did not run")
    return {
        "language": "python",
        "case": f"python_fus_c6_coupled_{steps}_steps",
        "steps": steps,
        "samples": len(report.samples),
        "mean_seconds": float(np.mean(np.asarray(timings, dtype=np.float64)) * 1.0e-9),
        "samples_seconds": [value * 1.0e-9 for value in timings],
        "recovered_energy_j": report.recovered_energy_j,
        "compression_work_j": compression_work,
        "energy_budget_relative_error": report.energy_budget_relative_error,
        "energy_budget_passed": report.energy_budget_passed,
        "flux_derivative_residual_linf": report.flux_derivative_residual_linf,
        "flux_derivative_residual_l2": report.flux_derivative_residual_l2,
        "flux_derivative_closure_passed": report.flux_derivative_closure_passed,
        "budget_claim_status": report.budget_claim_status,
        "compression_flux_budget_claim_status": report.compression_flux_budget_claim_status,
        "compression_trajectory_diagnostics_claim_status": (
            report.compression_trajectory_diagnostics_claim_status
        ),
        "compression_trajectory_diagnostics_passed": (
            report.compression_trajectory_diagnostics_passed
        ),
        "compression_trajectory_min_radius_m": (
            report.compression_trajectory_diagnostics.min_radius_m
            if report.compression_trajectory_diagnostics is not None
            else None
        ),
        "compression_trajectory_compression_ratio": (
            report.compression_trajectory_diagnostics.compression_ratio
            if report.compression_trajectory_diagnostics is not None
            else None
        ),
        "compression_trajectory_max_abs_radial_acceleration_m_s2": (
            report.compression_trajectory_diagnostics.max_abs_radial_acceleration_m_s2
            if report.compression_trajectory_diagnostics is not None
            else None
        ),
        "compression_flux_update_residual_abs_max": (
            report.compression_flux_budget.update_residual_abs_max
            if report.compression_flux_budget is not None
            else None
        ),
        "compression_flux_source_increment_checksum": (
            report.compression_flux_budget.source_increment_checksum
            if report.compression_flux_budget is not None
            else None
        ),
        "compression_flux_damping_decrement_checksum": (
            report.compression_flux_budget.damping_decrement_checksum
            if report.compression_flux_budget is not None
            else None
        ),
        "max_abs_back_emf_v": report.max_abs_back_emf_v,
        "max_abs_load_current_a": report.max_abs_load_current_a,
        "max_abs_flux_rate_field_term_wb_s": report.max_abs_flux_rate_field_term_wb_s,
        "max_abs_flux_rate_radial_term_wb_s": report.max_abs_flux_rate_radial_term_wb_s,
        "max_abs_flux_rate_total_wb_s": report.max_abs_flux_rate_total_wb_s,
    }


def _python_voltage_driven_coupled_case(steps: int) -> dict[str, Any]:
    timings = []
    report = None
    compression_work = 0.0
    source_work = 0.0
    for _ in range(5):
        config = _voltage_driven_compression_config()
        start_ns = time.perf_counter_ns()
        result = run_voltage_driven_pulsed_compression(
            config,
            lambda _t: 20_000.0,
            1.0e-9,
            steps,
            initial_current_A=5.0e5,
        )
        trajectory = faraday_trajectory_from_voltage_driven_compression(result)
        compression_work = compression_work_from_voltage_driven_compression(result)
        compression_flux_budget = compression_flux_budget_from_voltage_driven_compression(result)
        compression_trajectory_diagnostics = (
            compression_trajectory_diagnostics_from_voltage_driven_compression(
                result,
                radius_floor_m=config.min_radius_m,
            )
        )
        source_work = coil_source_work_from_voltage_driven_compression(result)
        report = integrated_recovery_energy(
            trajectory,
            config.coil.N_turns,
            config.coil.R_resistance_ohm,
            compression_work_j=compression_work,
            coil_source_work_j=source_work,
            compression_flux_budget=compression_flux_budget,
            compression_trajectory_diagnostics=compression_trajectory_diagnostics,
        )
        timings.append(float(time.perf_counter_ns() - start_ns))
    if report is None:
        raise RuntimeError("voltage-driven compression-coupled benchmark did not run")
    return {
        "language": "python",
        "case": f"python_fus_c6_voltage_driven_{steps}_steps",
        "steps": steps,
        "samples": len(report.samples),
        "mean_seconds": float(np.mean(np.asarray(timings, dtype=np.float64)) * 1.0e-9),
        "samples_seconds": [value * 1.0e-9 for value in timings],
        "recovered_energy_j": report.recovered_energy_j,
        "compression_work_j": compression_work,
        "coil_source_work_j": source_work,
        "energy_budget_relative_error": report.energy_budget_relative_error,
        "energy_budget_passed": report.energy_budget_passed,
        "budget_claim_status": report.budget_claim_status,
        "source_energy_budget_relative_error": report.source_energy_budget_relative_error,
        "source_energy_budget_passed": report.source_energy_budget_passed,
        "source_budget_claim_status": report.source_budget_claim_status,
        "flux_derivative_residual_linf": report.flux_derivative_residual_linf,
        "flux_derivative_residual_l2": report.flux_derivative_residual_l2,
        "flux_derivative_closure_passed": report.flux_derivative_closure_passed,
        "compression_flux_budget_claim_status": report.compression_flux_budget_claim_status,
        "compression_trajectory_diagnostics_claim_status": (
            report.compression_trajectory_diagnostics_claim_status
        ),
        "compression_trajectory_diagnostics_passed": (
            report.compression_trajectory_diagnostics_passed
        ),
        "compression_trajectory_min_radius_m": (
            report.compression_trajectory_diagnostics.min_radius_m
            if report.compression_trajectory_diagnostics is not None
            else None
        ),
        "compression_trajectory_compression_ratio": (
            report.compression_trajectory_diagnostics.compression_ratio
            if report.compression_trajectory_diagnostics is not None
            else None
        ),
        "compression_trajectory_max_abs_radial_acceleration_m_s2": (
            report.compression_trajectory_diagnostics.max_abs_radial_acceleration_m_s2
            if report.compression_trajectory_diagnostics is not None
            else None
        ),
        "compression_flux_update_residual_abs_max": (
            report.compression_flux_budget.update_residual_abs_max
            if report.compression_flux_budget is not None
            else None
        ),
        "compression_flux_source_increment_checksum": (
            report.compression_flux_budget.source_increment_checksum
            if report.compression_flux_budget is not None
            else None
        ),
        "compression_flux_damping_decrement_checksum": (
            report.compression_flux_budget.damping_decrement_checksum
            if report.compression_flux_budget is not None
            else None
        ),
        "max_abs_back_emf_v": report.max_abs_back_emf_v,
        "max_abs_load_current_a": report.max_abs_load_current_a,
        "max_abs_flux_rate_field_term_wb_s": report.max_abs_flux_rate_field_term_wb_s,
        "max_abs_flux_rate_radial_term_wb_s": report.max_abs_flux_rate_radial_term_wb_s,
        "max_abs_flux_rate_total_wb_s": report.max_abs_flux_rate_total_wb_s,
    }


def _criterion_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    criterion_root = REPO_ROOT / "scpn-fusion-rs" / "target" / "criterion" / "faraday_recovery"
    for estimates_path in sorted(criterion_root.glob("rust_*_samples/new/estimates.json")):
        with estimates_path.open("r", encoding="utf-8") as handle:
            estimates = json.load(handle)
        benchmark_name = estimates_path.parents[1].name
        parts = benchmark_name.split("_")
        try:
            samples = int(parts[1])
        except (IndexError, ValueError):
            samples = None
        rows.append(
            {
                "language": "rust",
                "case": benchmark_name,
                "samples": samples,
                "mean_seconds": float(estimates["mean"]["point_estimate"]) * 1.0e-9,
                "stddev_seconds": float(estimates["std_dev"]["point_estimate"]) * 1.0e-9,
                "criterion_estimates": str(estimates_path.relative_to(REPO_ROOT)),
                "compression_flux_budget_claim_status": ("blocked_missing_compression_flux_budget"),
                "compression_trajectory_diagnostics_claim_status": (
                    "blocked_missing_compression_trajectory_diagnostics"
                ),
                "flux_derivative_closure_status": "asserted_in_rust_criterion_harness",
                "flux_rate_term_status": "asserted_in_rust_criterion_harness",
            }
        )
    for estimates_path in sorted(
        criterion_root.glob("rust_fus_c6_coupled_*_steps/new/estimates.json")
    ):
        with estimates_path.open("r", encoding="utf-8") as handle:
            estimates = json.load(handle)
        benchmark_name = estimates_path.parents[1].name
        parts = benchmark_name.split("_")
        steps = _steps_from_benchmark_parts(parts)
        rows.append(
            {
                "language": "rust",
                "case": benchmark_name,
                "steps": steps,
                "mean_seconds": float(estimates["mean"]["point_estimate"]) * 1.0e-9,
                "stddev_seconds": float(estimates["std_dev"]["point_estimate"]) * 1.0e-9,
                "criterion_estimates": str(estimates_path.relative_to(REPO_ROOT)),
                "compression_flux_budget_claim_status": "asserted_in_rust_criterion_harness",
                "compression_trajectory_diagnostics_claim_status": (
                    "asserted_in_rust_criterion_harness"
                ),
                "flux_derivative_closure_status": "computed_in_rust_criterion_harness",
                "flux_rate_term_status": "computed_in_rust_criterion_harness",
            }
        )
    for estimates_path in sorted(
        criterion_root.glob("rust_fus_c6_voltage_driven_*_steps/new/estimates.json")
    ):
        with estimates_path.open("r", encoding="utf-8") as handle:
            estimates = json.load(handle)
        benchmark_name = estimates_path.parents[1].name
        parts = benchmark_name.split("_")
        steps = _steps_from_benchmark_parts(parts)
        rows.append(
            {
                "language": "rust",
                "case": benchmark_name,
                "steps": steps,
                "mean_seconds": float(estimates["mean"]["point_estimate"]) * 1.0e-9,
                "stddev_seconds": float(estimates["std_dev"]["point_estimate"]) * 1.0e-9,
                "criterion_estimates": str(estimates_path.relative_to(REPO_ROOT)),
                "compression_flux_budget_claim_status": "asserted_in_rust_criterion_harness",
                "compression_trajectory_diagnostics_claim_status": (
                    "asserted_in_rust_criterion_harness"
                ),
                "flux_derivative_closure_status": "computed_in_rust_criterion_harness",
                "flux_rate_term_status": "computed_in_rust_criterion_harness",
            }
        )
    return rows


def _steps_from_benchmark_parts(parts: list[str]) -> int | None:
    for index, part in enumerate(parts):
        if part == "steps" and index > 0:
            try:
                return int(parts[index - 1])
            except ValueError:
                return None
    return None


def build_report(cases: Iterable[int] = (64, 256, 1024)) -> dict[str, Any]:
    rows = [_python_case(samples) for samples in cases]
    rows.extend(_python_compression_coupled_case(steps) for steps in (64, 256))
    rows.extend(_python_voltage_driven_coupled_case(steps) for steps in (64, 256))
    rows.extend(_criterion_rows())
    return {
        "schema": "scpn-fusion-core.faraday_recovery_benchmark.v6",
        "claim_boundary": (
            "Local non-isolated regression evidence for the exact classical Faraday recovery "
            "contract over supplied trajectories, including internal FUS-C.6 supplied-current "
            "compression-work sidecars, FUS-C.6 flux-budget and trajectory-quality sidecars, "
            "and voltage-driven coil-source sidecars. This is not Slough compression-work "
            "acceptance evidence."
        ),
        "physics_contract": {
            "flux": "Phi = B_ext*pi*R_s^2",
            "emf": "EMF = -N*pi*(R_s^2*dB_ext/dt + 2*B_ext*R_s*dR_s/dt)",
            "faraday_law_closure": "finite_difference(Phi) + EMF/N_turns = 0",
            "flux_rate_terms": "dPhi/dt = pi*R_s^2*dB_ext/dt + 2*pi*B_ext*R_s*dR_s/dt",
            "load_power": "P = EMF^2/R_load",
            "internal_fus_c6_budget": (
                "evaluated when supplied-current or voltage-driven compression states provide "
                "compression_work_j"
            ),
            "internal_fus_c6_source_budget": (
                "evaluated when voltage-driven coil-circuit states provide coil_source_work_j"
            ),
            "internal_fus_c6_flux_budget": (
                "evaluated when compression states provide source/damping/update-residual "
                "flux-budget sidecars"
            ),
            "internal_fus_c6_trajectory_diagnostics": (
                "evaluated when compression states provide validated min-radius, acceleration, "
                "radius-floor, turning-point, compression-ratio, and all-flux-budget diagnostics"
            ),
            "external_slough_acceptance": "blocked_missing_public_digitised_reference",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "isolated": False,
        },
        "source_checksums": {
            str(path.relative_to(REPO_ROOT)): _sha256(path) for path in SOURCE_PATHS
        },
        "results": rows,
    }


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = build_report()
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(REPORT_PATH), "rows": len(report["results"])}, sort_keys=True))


if __name__ == "__main__":
    main()
