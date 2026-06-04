# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FRC Rigid-Rotor Quickstart
"""Reproduce the accepted Steinhauer no-rotation FRC analytical contract."""

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
    grid_points: int = 401,
    b_ext_t: float = 5.0,
    r_s_m: float = 0.20,
    delta_m: float = 0.020,
    t_i_ev: float = 10_000.0,
    t_e_ev: float = 5_000.0,
) -> dict[str, Any]:
    """Run the accepted no-rotation FRC analytical equilibrium.

    The returned arrays expose the solved radial profiles. The scalar
    diagnostics are suitable for JSON summaries and validation gates.
    """
    if grid_points < 9:
        raise ValueError("grid_points must be at least 9")

    root = _repo_root()
    _ensure_src_on_path(root)

    from scpn_fusion.core import RigidRotorFRCInputs, solve_frc_equilibrium
    from scpn_fusion.core.frc_rigid_rotor import validate_equilibrium

    inputs = RigidRotorFRCInputs(
        n0=_pressure_matched_density_m3(t_i_ev, t_e_ev, b_ext_t),
        T_i_eV=t_i_ev,
        T_e_eV=t_e_ev,
        theta_dot=0.0,
        R_s=r_s_m,
        B_ext=b_ext_t,
        delta=delta_m,
    )
    rho = np.linspace(0.0, 2.0 * r_s_m, grid_points)
    state = solve_frc_equilibrium(inputs, rho)
    report = validate_equilibrium(state)

    input_summary: dict[str, float | int] = {
        "grid_points": int(grid_points),
        "B_ext_T": float(b_ext_t),
        "R_s_m": float(r_s_m),
        "delta_m": float(delta_m),
        "T_i_eV": float(t_i_ev),
        "T_e_eV": float(t_e_ev),
        "n0_m3": float(inputs.n0),
    }
    diagnostics: dict[str, float | bool | str] = {
        "model": state.model,
        "validation_passed": bool(report.passed),
        "field_reversal_passed": bool(report.field_reversal_passed),
        "R_null_m": float(state.R_null),
        "separatrix_radius_error_m": float(state.separatrix_radius_error_m),
        "s_parameter": float(state.s_parameter),
        "beta_peak": float(state.beta_peak),
        "pressure_balance_residual_linf": float(state.pressure_balance_residual_linf),
        "ampere_residual_linf": float(state.ampere_residual_linf),
        "flux_derivative_residual_linf": float(state.flux_derivative_residual_linf),
        "psi_normalized_residual_linf": float(state.psi_normalized_residual_linf),
        "separatrix_energy_closure_relative_error": float(
            state.separatrix_energy_closure_relative_error
        ),
        "force_balance_residual_linf": float(state.force_balance_residual_linf),
    }

    return {
        "inputs": input_summary,
        "diagnostics": diagnostics,
        "rho_m": state.rho,
        "B_z_T": state.B_z,
        "pressure_Pa": state.p,
        "psi_normalized": state.psi_normalized,
        "density_m3": state.density_m3,
        "beta": state.beta,
    }


def _json_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "inputs": result["inputs"],
        "diagnostics": result["diagnostics"],
        "samples": {
            "rho_m": np.asarray(result["rho_m"], dtype=float).tolist(),
            "B_z_T": np.asarray(result["B_z_T"], dtype=float).tolist(),
            "pressure_Pa": np.asarray(result["pressure_Pa"], dtype=float).tolist(),
            "psi_normalized": np.asarray(result["psi_normalized"], dtype=float).tolist(),
        },
    }


def _write_json(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_payload(result)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_figure(result: dict[str, Any], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required when --output-figure is used") from exc

    rho = np.asarray(result["rho_m"], dtype=float)
    b_z = np.asarray(result["B_z_T"], dtype=float)
    pressure = np.asarray(result["pressure_Pa"], dtype=float)
    pressure_mpa = pressure / 1.0e6

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7.0, 6.0))
    axes[0].plot(rho, b_z, color="tab:blue")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("B_z [T]")
    axes[0].set_title("Steinhauer no-rotation FRC analytical field")
    axes[1].plot(rho, pressure_mpa, color="tab:orange")
    axes[1].set_xlabel("r [m]")
    axes[1].set_ylabel("p [MPa]")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the accepted Steinhauer no-rotation FRC analytical quickstart.",
    )
    parser.add_argument("--grid-points", type=int, default=401, help="Number of radial samples.")
    parser.add_argument("--b-ext-t", type=float, default=5.0, help="External axial field [T].")
    parser.add_argument("--r-s-m", type=float, default=0.20, help="Separatrix radius [m].")
    parser.add_argument("--delta-m", type=float, default=0.020, help="Layer thickness [m].")
    parser.add_argument("--t-i-ev", type=float, default=10_000.0, help="Ion temperature [eV].")
    parser.add_argument("--t-e-ev", type=float, default=5_000.0, help="Electron temperature [eV].")
    parser.add_argument("--output-json", type=Path, help="Optional JSON output path.")
    parser.add_argument("--output-figure", type=Path, help="Optional PNG figure output path.")
    args = parser.parse_args(argv)

    result = run_case(
        grid_points=args.grid_points,
        b_ext_t=args.b_ext_t,
        r_s_m=args.r_s_m,
        delta_m=args.delta_m,
        t_i_ev=args.t_i_ev,
        t_e_ev=args.t_e_ev,
    )
    if args.output_json is not None:
        _write_json(result, args.output_json)
    if args.output_figure is not None:
        _write_figure(result, args.output_figure)

    diagnostics = result["diagnostics"]
    print(
        "FRC quickstart: "
        f"validation_passed={diagnostics['validation_passed']} "
        f"R_null_m={diagnostics['R_null_m']:.6g} "
        f"s={diagnostics['s_parameter']:.6g} "
        f"beta_peak={diagnostics['beta_peak']:.6g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
