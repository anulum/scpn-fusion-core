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
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from scpn_fusion.core.faraday_recovery import (
    FaradayRecoveryTrajectoryPoint,
    compression_work_from_pulsed_compression,
    faraday_trajectory_from_pulsed_compression,
    integrated_recovery_energy,
)
from scpn_fusion.core import solve_frc_equilibrium
from scpn_fusion.core.frc_rigid_rotor import ELEMENTARY_CHARGE_C, MU_0, RigidRotorFRCInputs
from scpn_fusion.core.pulsed_compression import (
    CoilGeometry,
    PulsedCompressionConfig,
    initial_pulsed_compression_state,
    run_pulsed_compression,
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
        "budget_claim_status": state.budget_claim_status,
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
        report = integrated_recovery_energy(
            trajectory,
            config.coil.N_turns,
            config.coil.R_resistance_ohm,
            compression_work_j=compression_work,
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
        "budget_claim_status": report.budget_claim_status,
        "max_abs_back_emf_v": report.max_abs_back_emf_v,
        "max_abs_load_current_a": report.max_abs_load_current_a,
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
            }
        )
    for estimates_path in sorted(
        criterion_root.glob("rust_fus_c6_coupled_*_steps/new/estimates.json")
    ):
        with estimates_path.open("r", encoding="utf-8") as handle:
            estimates = json.load(handle)
        benchmark_name = estimates_path.parents[1].name
        parts = benchmark_name.split("_")
        try:
            steps = int(parts[4])
        except (IndexError, ValueError):
            steps = None
        rows.append(
            {
                "language": "rust",
                "case": benchmark_name,
                "steps": steps,
                "mean_seconds": float(estimates["mean"]["point_estimate"]) * 1.0e-9,
                "stddev_seconds": float(estimates["std_dev"]["point_estimate"]) * 1.0e-9,
                "criterion_estimates": str(estimates_path.relative_to(REPO_ROOT)),
            }
        )
    return rows


def build_report(cases: Iterable[int] = (64, 256, 1024)) -> dict[str, Any]:
    rows = [_python_case(samples) for samples in cases]
    rows.extend(_python_compression_coupled_case(steps) for steps in (64, 256))
    rows.extend(_criterion_rows())
    return {
        "schema": "scpn-fusion-core.faraday_recovery_benchmark.v1",
        "claim_boundary": (
            "Local non-isolated regression evidence for the exact classical Faraday recovery "
            "contract over supplied trajectories, including internal FUS-C.6 supplied-current "
            "compression-work sidecars. This is not Slough compression-work acceptance evidence."
        ),
        "physics_contract": {
            "flux": "Phi = B_ext*pi*R_s^2",
            "emf": "EMF = -N*pi*(R_s^2*dB_ext/dt + 2*B_ext*R_s*dR_s/dt)",
            "load_power": "P = EMF^2/R_load",
            "internal_fus_c6_budget": (
                "evaluated when supplied-current compression states provide compression_work_j"
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
