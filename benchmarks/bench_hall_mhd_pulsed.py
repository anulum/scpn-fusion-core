#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Pulsed Hall-MHD Benchmark
"""Benchmark the accepted axisymmetric FUS-C.2 Hall-MHD flux carrier."""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from scpn_fusion.core import RigidRotorFRCInputs, solve_frc_equilibrium
from scpn_fusion.core.frc_rigid_rotor import ELEMENTARY_CHARGE_C, MU_0
from scpn_fusion.core.hall_mhd_pulsed import (
    HallMHDPulsedConfig,
    gkeyll_small_hall_acceptance_status,
    initial_hall_mhd_pulsed_state,
    ono_fig4_acceptance_status,
    run_hall_mhd_pulsed,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "validation" / "reports" / "hall_mhd_pulsed_benchmark.json"
SOURCE_PATHS = [
    REPO_ROOT / "src" / "scpn_fusion" / "core" / "hall_mhd_pulsed.py",
    REPO_ROOT / "scpn-fusion-rs" / "crates" / "fusion-physics" / "src" / "hall_mhd_pulsed.rs",
    Path(__file__).resolve(),
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _equilibrium(n_grid: int) -> Any:
    t_i = 10_000.0
    t_e = 5_000.0
    b_ext = 5.0
    n0 = b_ext**2 / (2.0 * MU_0) / ((t_i + t_e) * ELEMENTARY_CHARGE_C)
    rho = np.linspace(0.0, 0.4, n_grid, dtype=np.float64)
    return solve_frc_equilibrium(
        RigidRotorFRCInputs(
            n0=n0,
            T_i_eV=t_i,
            T_e_eV=t_e,
            theta_dot=0.0,
            R_s=0.2,
            B_ext=b_ext,
            delta=0.02,
        ),
        rho,
    )


def _python_case(n_grid: int, steps: int) -> dict[str, Any]:
    eq = _equilibrium(n_grid)
    cfg = HallMHDPulsedConfig(
        equilibrium=eq,
        B_ext_t=lambda _t: 5.0,
        tau_psi_s=5.0e-6,
        electron_temperature_eV=5_000.0,
        E_theta_t=lambda _t, rho: np.full_like(rho, 2.5),
        J_theta_t=lambda _t, rho: np.full_like(rho, 2.0e5),
    )
    samples = []
    final_state = None
    for _ in range(5):
        initial = initial_hall_mhd_pulsed_state(cfg)
        start_ns = time.perf_counter_ns()
        trajectory = run_hall_mhd_pulsed(initial, cfg, 1.0e-8, steps)
        samples.append(float(time.perf_counter_ns() - start_ns))
        final_state = trajectory[-1]
    if final_state is None:
        raise RuntimeError("benchmark did not run")
    return {
        "language": "python",
        "case": f"python_{n_grid}_grid_{steps}_steps",
        "n_grid": n_grid,
        "steps": steps,
        "mean_seconds": float(np.mean(np.asarray(samples, dtype=np.float64)) * 1.0e-9),
        "samples_seconds": [value * 1.0e-9 for value in samples],
        "final_time_s": final_state.t_s,
        "final_energy_proxy_J_m": final_state.energy_proxy_J_m,
        "source_residual_linf": final_state.source_residual_linf,
        "external_parity_status": final_state.external_parity_status,
    }


def _criterion_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    criterion_root = REPO_ROOT / "scpn-fusion-rs" / "target" / "criterion" / "hall_mhd_pulsed"
    for estimates_path in sorted(criterion_root.glob("rust_*_grid_256_steps/new/estimates.json")):
        with estimates_path.open("r", encoding="utf-8") as handle:
            estimates = json.load(handle)
        benchmark_name = estimates_path.parents[1].name
        parts = benchmark_name.split("_")
        try:
            n_grid = int(parts[1])
        except (IndexError, ValueError):
            n_grid = None
        rows.append(
            {
                "language": "rust",
                "case": benchmark_name,
                "n_grid": n_grid,
                "steps": 256,
                "mean_seconds": float(estimates["mean"]["point_estimate"]) * 1.0e-9,
                "stddev_seconds": float(estimates["std_dev"]["point_estimate"]) * 1.0e-9,
                "criterion_estimates": str(estimates_path.relative_to(REPO_ROOT)),
            }
        )
    return rows


def build_report(
    cases: Iterable[tuple[int, int]] = ((64, 256), (256, 256), (1024, 256)),
) -> dict[str, Any]:
    rows = [_python_case(n_grid, steps) for n_grid, steps in cases]
    rows.extend(_criterion_rows())
    rows.append({"language": "external_reference", **gkeyll_small_hall_acceptance_status()})
    rows.append({"language": "external_reference", **ono_fig4_acceptance_status()})
    return {
        "schema": "scpn-fusion-core.hall_mhd_pulsed_benchmark.v1",
        "claim_boundary": (
            "Local non-isolated axisymmetric Ono Eq. 8 Hall-MHD flux-carrier regression evidence. "
            "Full 2D two-fluid Hall-MHD, Gkeyll/BOUT++ parity, and Ono figure reproduction remain blocked."
        ),
        "physics_contract": {
            "name": "MIF/FRC axisymmetric pulsed Hall-MHD flux carrier",
            "equation": "dpsi/dt = -psi/tau_psi + R_null*E_theta - eta_spitzer*J_theta",
            "integrator": "implicit backward-Euler damping with explicit Ono source",
            "not_claimed": "full 2D two-fluid Hall-MHD or external-code same-case parity",
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
