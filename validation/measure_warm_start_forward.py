# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — warm-start evidence for the compiled predictive forward (IDA pattern)
"""Measure the warm-start behaviour of the compiled predictive forward (evidence generator).

The IDA MAP/MCMC loop and real-time control both evaluate NEIGHBOURING parameter points:
the previous equilibrium is an in-basin initial guess. Two things make the warm path fast
and both are measured here, honestly separated:

1. **Correctness**: a warm-started solve (``psi_init`` = base solution) retains the standard
   current ramp. The 65² regression proves that jumping directly to the full nonlinear
   operator with ``ip_ramp=1`` can create a persistent Anderson limit cycle even from an
   in-basin seed. The warm solve must land on the SAME fixed point as a cold solve of the
   perturbed problem at the unchanged tolerance.
2. **Speed (indicative)**: wall-clock of cold vs warm solves at 33²/65²/129² on this host,
   all repeats recorded. Host-load caveat applies; the claimable number is a dedicated-host
   run of this same generator.

Run: ``python validation/measure_warm_start_forward.py``
Output: ``artifacts/rung2_mg_preconditioner/warm_start_forward.json``
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from scpn_fusion.core.jax_free_boundary_predictive import DEFAULT_N_ITER, build_response_matrix
from scpn_fusion.core.jax_o_point import smooth_axis_flux
from scpn_fusion.core.jax_predictive_forward_compiled import (
    solve_predictive_equilibrium_compiled,
)
from scpn_fusion.core.jax_x_point import smooth_xpoint_flux

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "rung2_mg_preconditioner" / "warm_start_forward.json"

COIL_R = jnp.array([1.2, 2.3, 1.2, 2.3, 1.6, 1.5])
COIL_Z = jnp.array([0.9, 0.9, -0.9, -0.9, 1.3, -1.35])
COIL_I = jnp.array([-3.0e5, -3.0e5, -3.0e5, -3.0e5, -1.0e5, -6.0e5])
PSIN = jnp.linspace(0.0, 1.0, 6)
PPRIME = jnp.array([-8.0e4, -6.0e4, -4.0e4, -2.0e4, -0.7e4, 0.0])
FFPRIME = jnp.array([-1.2, -0.9, -0.6, -0.3, -0.1, 0.0])
IP = 1.0e6
PERTURB_REL = 0.005  # ±0.5 % coil perturbation — the documented in-basin scale
REPEATS = 3
GRIDS = (33, 65, 129)

_LOGIC_SOURCES = (
    "src/scpn_fusion/core/jax_predictive_forward_compiled.py",
    "src/scpn_fusion/core/jax_free_boundary_predictive.py",
    "src/scpn_fusion/core/jax_free_boundary_gs.py",
    "src/scpn_fusion/core/jax_plasma_support.py",
    "src/scpn_fusion/core/jax_continuation_history.py",
    "src/scpn_fusion/core/jax_multigrid_precond.py",
    "src/scpn_fusion/core/jax_predictive_checkpoint_trace.py",
    "src/scpn_fusion/core/jax_equilibrium_solver.py",
    "src/scpn_fusion/core/jax_o_point.py",
    "src/scpn_fusion/core/jax_x_point.py",
)


def _digest_paths(rels: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for rel in sorted(rels):
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update((REPO / rel).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _time_one(fn) -> float:
    t0 = time.perf_counter()
    jax.block_until_ready(fn())
    return (time.perf_counter() - t0) * 1e3


def main() -> None:
    device = jax.devices()[0]
    rows = []
    equivalences: dict[str, float | None] = {}
    convergence: dict[str, dict[str, dict[str, int | bool]]] = {}
    for n in GRIDS:
        r = jnp.linspace(1.0, 2.5, n)
        z = jnp.linspace(-1.4, 1.4, n)
        m, b, s = build_response_matrix(r, z)
        ci_pert = COIL_I * (1.0 + PERTURB_REL)

        def cold(ci):
            return solve_predictive_equilibrium_compiled(
                ci, PPRIME, FFPRIME, r, z, COIL_R, COIL_Z, PSIN, IP, m, b, s
            )

        def warm(ci, psi0):
            return solve_predictive_equilibrium_compiled(
                ci,
                PPRIME,
                FFPRIME,
                r,
                z,
                COIL_R,
                COIL_Z,
                PSIN,
                IP,
                m,
                b,
                s,
                psi_init=psi0,
            )

        psi_base, base_iterations = solve_predictive_equilibrium_compiled(
            COIL_I,
            PPRIME,
            FFPRIME,
            r,
            z,
            COIL_R,
            COIL_Z,
            PSIN,
            IP,
            m,
            b,
            s,
            return_iterations=True,
        )
        psi_base = jax.block_until_ready(psi_base)
        psi_cold_pert, cold_iterations = solve_predictive_equilibrium_compiled(
            ci_pert,
            PPRIME,
            FFPRIME,
            r,
            z,
            COIL_R,
            COIL_Z,
            PSIN,
            IP,
            m,
            b,
            s,
            return_iterations=True,
        )
        psi_cold_pert = jax.block_until_ready(psi_cold_pert)
        psi_warm, warm_iterations = solve_predictive_equilibrium_compiled(
            ci_pert,
            PPRIME,
            FFPRIME,
            r,
            z,
            COIL_R,
            COIL_Z,
            PSIN,
            IP,
            m,
            b,
            s,
            psi_init=psi_base,
            return_iterations=True,
        )
        psi_warm = jax.block_until_ready(psi_warm)  # also warms the jit cache
        span = abs(
            float(smooth_axis_flux(psi_cold_pert)) - float(smooth_xpoint_flux(psi_cold_pert, r, z))
        )
        equiv = float(jnp.max(jnp.abs(psi_warm - psi_cold_pert))) / span
        warm_key = f"{n}x{n}"
        warm_converged = all(
            iterations < DEFAULT_N_ITER
            for iterations in (base_iterations, cold_iterations, warm_iterations)
        )
        equivalences[warm_key] = equiv if warm_converged else None
        convergence[warm_key] = {
            "base": {
                "iterations": base_iterations,
                "converged": base_iterations < DEFAULT_N_ITER,
            },
            "cold_perturbed": {
                "iterations": cold_iterations,
                "converged": cold_iterations < DEFAULT_N_ITER,
            },
            "warm_perturbed": {
                "iterations": warm_iterations,
                "converged": warm_iterations < DEFAULT_N_ITER,
            },
        }
        print(
            f"{n}^2 warm-vs-cold fixed-point agreement (span-rel): "
            f"{equiv:.3e} ({'claimable' if warm_converged else 'NON-CONVERGED'})",
            flush=True,
        )

        def warm_mgr(ci, psi0):
            return solve_predictive_equilibrium_compiled(
                ci,
                PPRIME,
                FFPRIME,
                r,
                z,
                COIL_R,
                COIL_Z,
                PSIN,
                IP,
                m,
                b,
                s,
                psi_init=psi0,
                inner_solver="mg_richardson",
                inner_cycles=2,
            )

        psi_combo, combo_iterations = solve_predictive_equilibrium_compiled(
            ci_pert,
            PPRIME,
            FFPRIME,
            r,
            z,
            COIL_R,
            COIL_Z,
            PSIN,
            IP,
            m,
            b,
            s,
            psi_init=psi_base,
            inner_solver="mg_richardson",
            inner_cycles=2,
            return_iterations=True,
        )
        psi_combo = jax.block_until_ready(psi_combo)
        combo_key = f"{n}x{n}_warm_mg_richardson2"
        combo_equiv = float(jnp.max(jnp.abs(psi_combo - psi_cold_pert))) / span
        combo_converged = all(
            iterations < DEFAULT_N_ITER
            for iterations in (base_iterations, cold_iterations, combo_iterations)
        )
        equivalences[combo_key] = combo_equiv if combo_converged else None
        convergence[combo_key] = {
            "base": {
                "iterations": base_iterations,
                "converged": base_iterations < DEFAULT_N_ITER,
            },
            "cold_perturbed": {
                "iterations": cold_iterations,
                "converged": cold_iterations < DEFAULT_N_ITER,
            },
            "warm_mg_richardson2_perturbed": {
                "iterations": combo_iterations,
                "converged": combo_iterations < DEFAULT_N_ITER,
            },
        }
        cold_ms = [_time_one(lambda: cold(ci_pert)) for _ in range(REPEATS)]
        warm_ms = [_time_one(lambda: warm(ci_pert, psi_base)) for _ in range(REPEATS)]
        warm_mgr_ms = [_time_one(lambda: warm_mgr(ci_pert, psi_base)) for _ in range(REPEATS)]
        rows.append(
            {
                "grid": f"{n}x{n}",
                "cold_ms": cold_ms,
                "warm_ms": warm_ms,
                "warm_mg_richardson2_ms": warm_mgr_ms,
                "cold_median_ms": sorted(cold_ms)[REPEATS // 2],
                "warm_median_ms": sorted(warm_ms)[REPEATS // 2],
                "warm_mg_richardson2_median_ms": sorted(warm_mgr_ms)[REPEATS // 2],
            }
        )
        print(
            f"{n}^2 cold {rows[-1]['cold_median_ms']:.0f} ms  "
            f"warm {rows[-1]['warm_median_ms']:.0f} ms  "
            f"warm+mgR2 {rows[-1]['warm_mg_richardson2_median_ms']:.0f} ms",
            flush=True,
        )

    record = {
        "task": "warm-start behaviour of the compiled predictive forward (IDA pattern)",
        "method": (
            "base solve at nominal coil currents; perturb all coil currents by +0.5% (the "
            "documented in-basin FD-validation scale); warm solve = psi_init=base with the "
            "standard current ramp retained for cross-grid nonlinear stability; correctness "
            "= warm solve agrees "
            "with the COLD solve of the perturbed problem at span-relative tolerance"
        ),
        "correctness_load_independent": {
            "warm_vs_cold_fixed_point_span_rel": equivalences,
            "convergence": convergence,
            "policy": (
                "an equivalence value is claimable only when the base, cold-perturbed, and "
                "candidate solves all terminate before n_iter; otherwise it is null"
            ),
        },
        "timings_indicative": {
            "host": f"{platform.node()} ({platform.machine()}) - host-load caveat applies; "
            "the claimable number is this generator on a dedicated host",
            "device": f"{device.device_kind} ({device.platform})",
            "rows": rows,
        },
        "settings": {
            "perturb_rel": PERTURB_REL,
            "repeats": REPEATS,
            "n_iter": DEFAULT_N_ITER,
        },
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "jax_version": jax.__version__,
        "provenance": {
            "generator": "validation/measure_warm_start_forward.py",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "logic_sources": list(_LOGIC_SOURCES),
            "logic_sources_sha256": _digest_paths(_LOGIC_SOURCES),
            "pinned_environment": "requirements/full.txt (hash-pinned) for exact reproduction",
            "pinned_requirements_sha256": hashlib.sha256(
                (REPO / "requirements" / "full.txt").read_bytes()
            ).hexdigest(),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
