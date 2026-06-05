#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — MRTI Benchmark
"""Benchmark the analytical MRTI growth/spectrum tracker."""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from scpn_fusion.core import solve_frc_equilibrium
from scpn_fusion.core.frc_rigid_rotor import ELEMENTARY_CHARGE_C, MU_0, RigidRotorFRCInputs
from scpn_fusion.core.mrti import MRTISpectrumTracker, track_mrti_from_pulsed_compression
from scpn_fusion.core.mrti import effective_acceleration_from_pulsed_compression
from scpn_fusion.core.pulsed_compression import (
    CoilGeometry,
    PulsedCompressionConfig,
    initial_pulsed_compression_state,
    run_pulsed_compression,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "validation" / "reports" / "mrti_benchmark.json"
SOURCE_PATHS = [
    REPO_ROOT / "src" / "scpn_fusion" / "core" / "mrti.py",
    REPO_ROOT / "src" / "scpn_fusion" / "core" / "pulsed_compression.py",
    REPO_ROOT / "scpn-fusion-rs" / "crates" / "fusion-physics" / "src" / "mrti.rs",
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


def _python_case(n_modes: int, steps: int) -> dict[str, Any]:
    samples = []
    for _ in range(5):
        tracker = MRTISpectrumTracker(
            k_max_m_inv=1.0e4,
            n_modes=n_modes,
            initial_perturbation_m=1.0e-9,
            rho_kg_m3=1.0e-3,
            saturation_threshold_m=1.0e-3,
        )
        start_ns = time.perf_counter_ns()
        state = tracker.state()
        for _step in range(steps):
            state = tracker.step(2.0e-7, 6.5e6, B_perp_t=8.0e-4)
        elapsed_ns = time.perf_counter_ns() - start_ns
        samples.append(float(elapsed_ns))
    mean_ns = float(np.mean(np.asarray(samples, dtype=np.float64)))
    return {
        "language": "python",
        "case": f"python_{n_modes}_modes_{steps}_steps",
        "n_modes": n_modes,
        "steps": steps,
        "mean_seconds": mean_ns * 1.0e-9,
        "samples_seconds": [value * 1.0e-9 for value in samples],
        "final_time_s": state.t_s,
        "final_max_amplitude_m": state.max_amplitude_m,
        "max_log_amplitude": state.max_log_amplitude,
        "amplitude_overflow_limited": state.amplitude_overflow_limited,
        "saturation_warning": state.saturation_warning,
        "fastest_growing_k_m_inv": state.fastest_growing_k_m_inv,
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
        tau_psi_s=np.inf,
    )


def _python_compression_coupled_case(n_modes: int, steps: int) -> dict[str, Any]:
    samples = []
    final_state = None
    final_compression = None
    final_acceleration = None
    for _ in range(5):
        config = _compression_config()
        tracker = MRTISpectrumTracker(
            k_max_m_inv=1.0e4,
            n_modes=n_modes,
            initial_perturbation_m=1.0e-9,
            rho_kg_m3=1.0e-3,
            saturation_threshold_m=1.0e-3,
        )
        start_ns = time.perf_counter_ns()
        compression_states = run_pulsed_compression(
            initial_pulsed_compression_state(config),
            config,
            1.0e-9,
            steps,
        )
        snapshots = track_mrti_from_pulsed_compression(compression_states, tracker)
        samples.append(float(time.perf_counter_ns() - start_ns))
        final_state = snapshots[-1]
        final_compression = compression_states[-1]
        final_acceleration = effective_acceleration_from_pulsed_compression(compression_states)
    if final_state is None or final_compression is None:
        raise RuntimeError("compression-coupled MRTI benchmark did not run")
    if final_acceleration is None:
        raise RuntimeError("compression-coupled MRTI acceleration did not run")
    mean_ns = float(np.mean(np.asarray(samples, dtype=np.float64)))
    return {
        "language": "python",
        "case": f"python_fus_c6_coupled_{n_modes}_modes_{steps}_steps",
        "n_modes": n_modes,
        "steps": steps,
        "mean_seconds": mean_ns * 1.0e-9,
        "samples_seconds": [value * 1.0e-9 for value in samples],
        "final_time_s": final_state.t_s,
        "final_radius_m": final_compression.R_s_m,
        "compression_work_j": final_compression.compression_work_J,
        "acceleration_source": "fus_c6_state_radial_acceleration_m_s2",
        "final_effective_acceleration_m_s2": float(final_acceleration[-1]),
        "max_effective_acceleration_m_s2": float(np.max(final_acceleration)),
        "final_max_amplitude_m": final_state.max_amplitude_m,
        "max_log_amplitude": final_state.max_log_amplitude,
        "amplitude_overflow_limited": final_state.amplitude_overflow_limited,
        "saturation_warning": final_state.saturation_warning,
        "fastest_growing_k_m_inv": final_state.fastest_growing_k_m_inv,
        "coupling_status": "internal_fus_c6_pulsed_compression_adapter",
    }


def _criterion_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    criterion_root = REPO_ROOT / "scpn-fusion-rs" / "target" / "criterion" / "mrti"
    for estimates_path in sorted(criterion_root.glob("rust_*_modes_*_steps/new/estimates.json")):
        with estimates_path.open("r", encoding="utf-8") as handle:
            estimates = json.load(handle)
        benchmark_name = estimates_path.parents[1].name
        parts = benchmark_name.split("_")
        n_modes = None
        steps = None
        for index, part in enumerate(parts):
            if part == "modes" and index > 0:
                try:
                    n_modes = int(parts[index - 1])
                except ValueError:
                    n_modes = None
            if part == "steps" and index > 0:
                try:
                    steps = int(parts[index - 1])
                except ValueError:
                    steps = None
        row: dict[str, Any] = {
            "language": "rust",
            "case": benchmark_name,
            "n_modes": n_modes,
            "steps": steps,
            "mean_seconds": float(estimates["mean"]["point_estimate"]) * 1.0e-9,
            "stddev_seconds": float(estimates["std_dev"]["point_estimate"]) * 1.0e-9,
            "criterion_estimates": str(estimates_path.relative_to(REPO_ROOT)),
        }
        if "fus_c6_coupled" in benchmark_name:
            row["coupling_status"] = "internal_fus_c6_pulsed_compression_adapter"
            row["acceleration_source"] = "fus_c6_state_radial_acceleration_m_s2"
        row["log_amplitude_status"] = "asserted_finite_in_rust_criterion_harness"
        rows.append(row)
    return rows


def build_report(
    cases: Iterable[tuple[int, int]] = ((64, 512), (256, 512), (1024, 512)),
) -> dict[str, Any]:
    rows = [_python_case(n_modes, steps) for n_modes, steps in cases]
    rows.extend(_python_compression_coupled_case(n_modes, 64) for n_modes in (64, 256))
    rows.extend(_criterion_rows())
    return {
        "schema": "scpn-fusion-core.mrti_benchmark.v3",
        "claim_boundary": (
            "Local non-isolated MRTI analytical growth/spectrum regression evidence. "
            "Internal FUS-C.6 coupled rows consume the supplied-current "
            "pulsed-compression trajectory and its force-balance acceleration. "
            "External nonlinear MRTI parity remains blocked."
        ),
        "physics_contract": {
            "name": "MIF/FRC linear MRTI growth spectrum",
            "equation": "gamma^2 = k*a_eff - k^2*B_perp^2/(mu0*rho)",
            "amplitude_evolution": "log(A_i) <- log(A_i) + max(gamma_i*dt, 0)",
            "stabilization": "negative radicands clipped to zero for magnetic-tension-stabilized modes",
            "not_claimed": "external Slough same-case MRTI parity or nonlinear saturation physics",
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
