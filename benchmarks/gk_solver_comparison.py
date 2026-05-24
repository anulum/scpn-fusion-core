# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Nonlinear GK Solver Benchmark
"""Benchmark NumPy and JAX nonlinear GK paths on current public APIs."""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from scpn_fusion.core.gk_nonlinear import NonlinearGKConfig, NonlinearGKSolver
from scpn_fusion.core.jax_gk_nonlinear import JaxNonlinearGKSolver, jax_available

ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = ROOT / "validation" / "reports" / "gk_nonlinear_solver_comparison.json"
REPORT_MD = ROOT / "validation" / "reports" / "gk_nonlinear_solver_comparison.md"


def _benchmark_config(collision_model: str) -> NonlinearGKConfig:
    return NonlinearGKConfig(
        n_kx=4,
        n_ky=4,
        n_theta=8,
        n_vpar=8,
        n_mu=6,
        n_species=2,
        dt=0.01,
        n_steps=8,
        save_interval=4,
        collisions=True,
        collision_model=collision_model,
        nonlinear=True,
        hyper_coeff=0.05,
        cfl_adapt=False,
    )


def _run_numpy(cfg: NonlinearGKConfig) -> dict[str, Any]:
    solver = NonlinearGKSolver(cfg)
    t0 = time.perf_counter()
    result = solver.run(solver.init_state(amplitude=1e-5, seed=23))
    elapsed = time.perf_counter() - t0
    return {
        "backend": "numpy",
        "elapsed_s": elapsed,
        "converged": result.converged,
        "chi_i": result.chi_i,
        "chi_e": result.chi_e,
        "phi_rms_final": float(result.phi_rms_t[-1]) if len(result.phi_rms_t) else 0.0,
    }


def _run_jax(cfg: NonlinearGKConfig) -> dict[str, Any]:
    solver = JaxNonlinearGKSolver(cfg)
    t0 = time.perf_counter()
    result = solver.run(solver._np_solver.init_state(amplitude=1e-5, seed=23))
    elapsed = time.perf_counter() - t0
    return {
        "backend": "jax" if jax_available() else "numpy_fallback",
        "elapsed_s": elapsed,
        "converged": result.converged,
        "chi_i": result.chi_i,
        "chi_e": result.chi_e,
        "phi_rms_final": float(result.phi_rms_t[-1]) if len(result.phi_rms_t) else 0.0,
    }


def _sugama_moment_residuals(cfg: NonlinearGKConfig) -> dict[str, float]:
    solver = NonlinearGKSolver(cfg)
    state = solver.init_state(amplitude=1e-4, seed=29)
    collision = solver.collide(state.f[0])
    vpar = solver.vpar[None, None, None, :, None]
    mu = solver.mu[None, None, None, None, :]
    energy = 0.5 * vpar**2 + mu
    dv = solver.dvpar * solver.dmu
    density = np.sum(collision * dv, axis=(-2, -1))
    momentum = np.sum(collision * vpar * dv, axis=(-2, -1))
    energy_moment = np.sum(collision * energy * dv, axis=(-2, -1))
    return {
        "density_max_abs": float(np.max(np.abs(density))),
        "parallel_momentum_max_abs": float(np.max(np.abs(momentum))),
        "energy_max_abs": float(np.max(np.abs(energy_moment))),
    }


def run_benchmark() -> dict[str, Any]:
    cases: dict[str, Any] = {}
    for collision_model in ("krook", "sugama"):
        cfg = _benchmark_config(collision_model)
        cases[collision_model] = {
            "config": asdict(cfg),
            "numpy": _run_numpy(cfg),
            "jax": _run_jax(cfg),
        }
        if collision_model == "sugama":
            cases[collision_model]["moment_residuals"] = _sugama_moment_residuals(cfg)

    return {
        "benchmark": "gk_nonlinear_solver_comparison",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "jax_available": jax_available(),
        "cases": cases,
    }


def _write_reports(report: dict[str, Any]) -> None:
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Nonlinear GK Solver Comparison",
        "",
        f"- Benchmark: `{report['benchmark']}`",
        f"- Python: `{report['python']}`",
        f"- Platform: `{report['platform']}`",
        f"- JAX available: `{report['jax_available']}`",
        "",
        "| Collision model | Backend | Elapsed s | Converged | chi_i | chi_e | phi_rms_final |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, case in report["cases"].items():
        for backend_name in ("numpy", "jax"):
            row = case[backend_name]
            lines.append(
                "| {name} | {backend} | {elapsed:.6f} | {converged} | {chi_i:.6e} | "
                "{chi_e:.6e} | {phi:.6e} |".format(
                    name=name,
                    backend=row["backend"],
                    elapsed=row["elapsed_s"],
                    converged=row["converged"],
                    chi_i=row["chi_i"],
                    chi_e=row["chi_e"],
                    phi=row["phi_rms_final"],
                )
            )

    residuals = report["cases"]["sugama"]["moment_residuals"]
    lines.extend(
        [
            "",
            "## Sugama Moment Residuals",
            "",
            "| Moment | Max abs residual |",
            "|---|---:|",
            f"| density | {residuals['density_max_abs']:.6e} |",
            f"| parallel_momentum | {residuals['parallel_momentum_max_abs']:.6e} |",
            f"| energy | {residuals['energy_max_abs']:.6e} |",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    report = run_benchmark()
    _write_reports(report)
    print(f"Wrote {REPORT_JSON}")
    print(f"Wrote {REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
