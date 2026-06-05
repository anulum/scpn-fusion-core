# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FRC Pulsed Compression Quickstart
"""Run the accepted FUS-C.6 FRC pulsed-compression trajectory contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_src_on_path(root: Path) -> None:
    src_path = root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def _pressure_matched_density_m3(t_i_ev: float, t_e_ev: float, b_ext_t: float) -> float:
    from scpn_fusion.core.frc_rigid_rotor import ELEMENTARY_CHARGE_C, MU_0

    external_pressure_pa = b_ext_t * b_ext_t / (2.0 * MU_0)
    return float(external_pressure_pa / ((t_i_ev + t_e_ev) * ELEMENTARY_CHARGE_C))


def run_case(
    *,
    steps: int = 256,
    dt_s: float = 2.0e-8,
    b_ext_t: float = 5.0,
    r_s_m: float = 0.20,
    delta_m: float = 0.020,
    t_i_ev: float = 10_000.0,
    t_e_ev: float = 5_000.0,
    coil_turns: int = 32,
    coil_length_m: float = 0.40,
    coil_current_a: float = 160_000.0,
) -> dict[str, Any]:
    """Run a deterministic pulsed-compression trajectory.

    The returned payload separates scalar diagnostics from sampled time series
    so it can be written as JSON without losing physical units.
    """
    if steps < 1:
        raise ValueError("steps must be positive")

    root = _repo_root()
    _ensure_src_on_path(root)

    from scpn_fusion.core import (
        CoilGeometry,
        PulsedCompressionConfig,
        RigidRotorFRCInputs,
        initial_pulsed_compression_state,
        run_pulsed_compression,
        solve_frc_equilibrium,
    )

    density_m3 = _pressure_matched_density_m3(t_i_ev, t_e_ev, b_ext_t)
    frc_inputs = RigidRotorFRCInputs(
        n0=density_m3,
        T_i_eV=t_i_ev,
        T_e_eV=t_e_ev,
        theta_dot=0.0,
        R_s=r_s_m,
        B_ext=b_ext_t,
        delta=delta_m,
    )
    equilibrium = solve_frc_equilibrium(frc_inputs, np.linspace(0.0, 2.0 * r_s_m, 401))
    config = PulsedCompressionConfig(
        equilibrium=equilibrium,
        coil=CoilGeometry(
            N_turns=coil_turns,
            L_coil_m=coil_length_m,
            R_coil_m=0.35,
            L_inductance_H=2.0e-6,
            R_resistance_ohm=0.02,
            bank_voltage_max_V=20_000.0,
        ),
        coil_current_t=lambda _t: coil_current_a,
        plasma_mass_kg=2.0e-5,
        ion_temperature_eV=t_i_ev,
        electron_temperature_eV=t_e_ev,
    )
    initial = initial_pulsed_compression_state(config)
    trajectory = run_pulsed_compression(initial, config, dt_s, steps)
    first = trajectory[0]
    last = trajectory[-1]

    diagnostics: dict[str, float | int] = {
        "steps": int(steps),
        "dt_s": float(dt_s),
        "coil_turns": int(coil_turns),
        "coil_length_m": float(coil_length_m),
        "coil_current_A": float(coil_current_a),
        "initial_radius_m": float(first.R_s_m),
        "final_radius_m": float(last.R_s_m),
        "initial_T_i_eV": float(first.T_i_eV),
        "final_T_i_eV": float(last.T_i_eV),
        "initial_beta": float(first.beta),
        "final_beta": float(last.beta),
        "initial_flux_checksum": float(first.flux_psi_checksum),
        "final_flux_checksum": float(last.flux_psi_checksum),
    }

    return {
        "diagnostics": diagnostics,
        "time_s": [float(state.t_s) for state in trajectory],
        "radius_m": [float(state.R_s_m) for state in trajectory],
        "radial_velocity_m_s": [float(state.dR_s_dt_m_s) for state in trajectory],
        "T_i_eV": [float(state.T_i_eV) for state in trajectory],
        "T_e_eV": [float(state.T_e_eV) for state in trajectory],
        "B_ext_T": [float(state.B_ext_T) for state in trajectory],
        "beta": [float(state.beta) for state in trajectory],
        "flux_checksum": [float(state.flux_psi_checksum) for state in trajectory],
    }


def _write_json(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the accepted FUS-C.6 FRC pulsed-compression quickstart.",
    )
    parser.add_argument("--steps", type=int, default=256, help="Number of compression steps.")
    parser.add_argument("--dt-s", type=float, default=2.0e-8, help="Time step [s].")
    parser.add_argument("--b-ext-t", type=float, default=5.0, help="Initial external field [T].")
    parser.add_argument("--r-s-m", type=float, default=0.20, help="Initial separatrix radius [m].")
    parser.add_argument("--delta-m", type=float, default=0.020, help="FRC layer thickness [m].")
    parser.add_argument(
        "--t-i-ev", type=float, default=10_000.0, help="Initial ion temperature [eV]."
    )
    parser.add_argument(
        "--t-e-ev", type=float, default=5_000.0, help="Initial electron temperature [eV]."
    )
    parser.add_argument("--coil-turns", type=int, default=32, help="Solenoid turns.")
    parser.add_argument("--coil-length-m", type=float, default=0.40, help="Solenoid length [m].")
    parser.add_argument("--coil-current-a", type=float, default=160_000.0, help="Coil current [A].")
    parser.add_argument("--output-json", type=Path, help="Optional JSON output path.")
    args = parser.parse_args(argv)

    result = run_case(
        steps=args.steps,
        dt_s=args.dt_s,
        b_ext_t=args.b_ext_t,
        r_s_m=args.r_s_m,
        delta_m=args.delta_m,
        t_i_ev=args.t_i_ev,
        t_e_ev=args.t_e_ev,
        coil_turns=args.coil_turns,
        coil_length_m=args.coil_length_m,
        coil_current_a=args.coil_current_a,
    )

    if args.output_json is not None:
        _write_json(result, args.output_json)

    diagnostics = result["diagnostics"]
    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
