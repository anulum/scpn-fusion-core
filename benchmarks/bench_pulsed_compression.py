#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Pulsed Compression Benchmark
"""Benchmark the MIF/FRC pulsed-compression contract."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from scpn_fusion.core import RigidRotorFRCInputs, solve_frc_equilibrium
from scpn_fusion.core.frc_rigid_rotor import ELEMENTARY_CHARGE_C, MU_0
from scpn_fusion.core.pulsed_compression import (
    CoilGeometry,
    PulsedCompressionConfig,
    initial_pulsed_compression_state,
    run_pulsed_compression,
    run_voltage_driven_pulsed_compression,
    slough_fig5_acceptance_status,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "validation" / "reports" / "pulsed_compression_benchmark.json"
SOURCE_PATHS = [
    REPO_ROOT / "src" / "scpn_fusion" / "core" / "pulsed_compression.py",
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


def _config() -> PulsedCompressionConfig:
    b_ext = 5.0
    t_i = 10_000.0
    t_e = 5_000.0
    density = b_ext**2 / (2.0 * MU_0) / ((t_i + t_e) * ELEMENTARY_CHARGE_C)
    rho = np.linspace(0.0, 0.4, 129, dtype=np.float64)
    equilibrium = solve_frc_equilibrium(
        RigidRotorFRCInputs(
            n0=density,
            T_i_eV=t_i,
            T_e_eV=t_e,
            theta_dot=0.0,
            R_s=0.2,
            B_ext=b_ext,
            delta=0.02,
        ),
        rho,
    )
    coil = CoilGeometry(
        N_turns=80,
        L_coil_m=1.0,
        R_coil_m=0.35,
        L_inductance_H=2.0e-6,
        R_resistance_ohm=0.02,
        bank_voltage_max_V=20_000.0,
    )
    return PulsedCompressionConfig(
        equilibrium=equilibrium,
        coil=coil,
        coil_current_t=lambda _t: 5.0e5,
        plasma_mass_kg=2.0e-5,
        ion_temperature_eV=t_i,
        electron_temperature_eV=t_e,
        tau_psi_s=np.inf,
    )


def _python_case(steps: int) -> dict[str, Any]:
    cfg = _config()
    timings = []
    final_state = None
    for _ in range(5):
        initial = initial_pulsed_compression_state(cfg)
        start_ns = time.perf_counter_ns()
        states = run_pulsed_compression(initial, cfg, 1.0e-9, steps)
        timings.append(float(time.perf_counter_ns() - start_ns))
        final_state = states[-1]
    if final_state is None:
        raise RuntimeError("benchmark did not run")
    return {
        "language": "python",
        "case": f"python_{steps}_steps",
        "steps": steps,
        "mean_seconds": float(np.mean(np.asarray(timings, dtype=np.float64)) * 1.0e-9),
        "samples_seconds": [value * 1.0e-9 for value in timings],
        "final_radius_m": final_state.R_s_m,
        "final_ion_temperature_eV": final_state.T_i_eV,
        "final_beta": final_state.beta,
        "energy_balance_residual": final_state.energy_balance_residual,
        "flux_coupling_status": final_state.flux_coupling_status,
    }


def _python_voltage_driven_case(steps: int) -> dict[str, Any]:
    cfg = _config()
    timings = []
    final_state = None
    final_circuit = None
    for _ in range(5):
        start_ns = time.perf_counter_ns()
        result = run_voltage_driven_pulsed_compression(
            cfg,
            lambda _t: 20_000.0,
            1.0e-9,
            steps,
            initial_current_A=5.0e5,
        )
        timings.append(float(time.perf_counter_ns() - start_ns))
        final_state = result.compression[-1]
        final_circuit = result.coil_circuit[-1]
    if final_state is None or final_circuit is None:
        raise RuntimeError("voltage-driven benchmark did not run")
    return {
        "language": "python",
        "case": f"python_voltage_driven_{steps}_steps",
        "steps": steps,
        "mean_seconds": float(np.mean(np.asarray(timings, dtype=np.float64)) * 1.0e-9),
        "samples_seconds": [value * 1.0e-9 for value in timings],
        "final_radius_m": final_state.R_s_m,
        "final_ion_temperature_eV": final_state.T_i_eV,
        "final_beta": final_state.beta,
        "final_coil_current_A": final_circuit.current_A,
        "coil_circuit_energy_balance_residual": final_circuit.energy_balance_residual,
        "compression_energy_balance_residual": final_state.energy_balance_residual,
        "drive_status": "exact_lumped_rl_bank_limited",
        "flux_coupling_status": final_state.flux_coupling_status,
    }


def _criterion_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    criterion_root = REPO_ROOT / "scpn-fusion-rs" / "target" / "criterion" / "pulsed_compression"
    for estimates_path in sorted(criterion_root.glob("rust_*_steps/new/estimates.json")):
        with estimates_path.open("r", encoding="utf-8") as handle:
            estimates = json.load(handle)
        benchmark_name = estimates_path.parents[1].name
        parts = benchmark_name.split("_")
        steps = None
        for index, part in enumerate(parts):
            if part == "steps" and index > 0:
                try:
                    steps = int(parts[index - 1])
                except ValueError:
                    steps = None
        row: dict[str, Any] = {
            "language": "rust",
            "case": benchmark_name,
            "steps": steps,
            "mean_seconds": float(estimates["mean"]["point_estimate"]) * 1.0e-9,
            "stddev_seconds": float(estimates["std_dev"]["point_estimate"]) * 1.0e-9,
            "criterion_estimates": str(estimates_path.relative_to(REPO_ROOT)),
        }
        if "voltage_driven" in benchmark_name:
            row["drive_status"] = "exact_lumped_rl_bank_limited"
        rows.append(row)
    return rows


def build_report() -> dict[str, Any]:
    rows = [_python_case(steps) for steps in (64, 256, 1024)]
    rows.extend(_python_voltage_driven_case(steps) for steps in (64, 256))
    rows.extend(_criterion_rows())
    rows.append(
        {
            "language": "external_reference",
            **slough_fig5_acceptance_status(),
        }
    )
    return {
        "schema": "scpn-fusion-core.pulsed_compression_benchmark.v1",
        "claim_boundary": (
            "Local non-isolated regression evidence for supplied-current and exact lumped "
            "R-L voltage-driven pulsed-compression dynamics wired to the Ono non-adiabatic "
            "flux carrier. Slough Fig. 5 parity remains blocked until a public digitised "
            "reference trajectory exists."
        ),
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
