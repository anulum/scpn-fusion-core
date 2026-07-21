# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — reproducible adjoint-vs-FD coil-gradient sweep (evidence generator)
"""Reproducible adjoint-vs-warm-FD coil-gradient sweep for the predictive solver.

Regenerates ``artifacts/coilgrad_adjoint_fd_evidence.json`` with FULL-precision values (no
rounding) so every published agreement figure has a committed, re-runnable generator — the
distinct-eye finding G1 on the earlier hand-assembled evidence file.

Method (the validated warm-FD protocol): the adjoint is ``jax.grad`` through the implicit-diff
custom VJP of ``solve_predictive_equilibrium_diff``; the finite difference is CENTRAL and
**warm-started from the base solution** (``psi_init = psi_base``) so both perturbed solves stay
in the converged basin. The sweep covers a strong coil and a weak coil at steps 100 A–3 kA;
the truncation signature (error GROWING with the step) is part of the record — a 3 kA step is
a ~0.5 % coil perturbation where the axis-flux response is visibly nonlinear, which is where
the historical "~3 %" artefact figure came from.

Case: the synthetic diverted coilset of the test suite (33², no external data needed).

Run: ``python validation/measure_coilgrad_adjoint_fd.py``
Output: ``artifacts/coilgrad_adjoint_fd_evidence.json``
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from scpn_fusion.core.jax_free_boundary_predictive import (
    build_response_matrix,
    solve_predictive_equilibrium,
    solve_predictive_equilibrium_diff,
)
from scpn_fusion.core.jax_o_point import smooth_axis_flux

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "coilgrad_adjoint_fd_evidence.json"

# The synthetic diverted case of tests/test_jax_free_boundary_predictive.py, verbatim.
_R = jnp.linspace(1.0, 2.5, 33)
_Z = jnp.linspace(-1.4, 1.4, 33)
_COIL_R = jnp.array([1.2, 2.3, 1.2, 2.3, 1.6, 1.5])
_COIL_Z = jnp.array([0.9, 0.9, -0.9, -0.9, 1.3, -1.35])
_COIL_I = jnp.array([-3.0e5, -3.0e5, -3.0e5, -3.0e5, -1.0e5, -6.0e5])
_PSIN = jnp.linspace(0.0, 1.0, 6)
_PPRIME = jnp.array([-8.0e4, -6.0e4, -4.0e4, -2.0e4, -0.7e4, 0.0])
_FFPRIME = jnp.array([-1.2, -0.9, -0.6, -0.3, -0.1, 0.0])
_IP = 1.0e6
_N_ITER = 150

COILS = (5, 4)  # strongest (divertor) and weakest coil of the set
EPS_SWEEP_A = (100.0, 300.0, 1000.0, 3000.0)


def _solve(coil_i: jnp.ndarray, response, psi_init: jnp.ndarray | None) -> jnp.ndarray:
    m, b, s = response
    return solve_predictive_equilibrium(
        _COIL_I * 0 + coil_i,
        _PPRIME,
        _FFPRIME,
        _R,
        _Z,
        _COIL_R,
        _COIL_Z,
        _PSIN,
        _IP,
        m,
        b,
        s,
        psi_init=psi_init,
        n_iter=_N_ITER,
    )


def main() -> None:
    response = build_response_matrix(_R, _Z)
    m, b, s = response

    def loss(ci: jnp.ndarray) -> jnp.ndarray:
        psi = solve_predictive_equilibrium_diff(
            ci, _PPRIME, _FFPRIME, _R, _Z, _COIL_R, _COIL_Z, _PSIN, _IP, m, b, s, n_iter=_N_ITER
        )
        return smooth_axis_flux(psi)

    psi_base = _solve(_COIL_I, response, None)
    adjoint_g = jax.grad(loss)(_COIL_I)
    print("adjoint g_ci:", [float(v) for v in adjoint_g], flush=True)

    rows = []
    for idx in COILS:
        for eps in EPS_SWEEP_A:
            e = jnp.zeros_like(_COIL_I).at[idx].set(eps)
            psi_p = _solve(_COIL_I + e, response, psi_base)
            psi_m = _solve(_COIL_I - e, response, psi_base)
            fd = float((smooth_axis_flux(psi_p) - smooth_axis_flux(psi_m)) / (2.0 * eps))
            adj = float(adjoint_g[idx])
            rel = abs(adj - fd) / (abs(fd) + 1e-300)
            rows.append({"coil": idx, "eps_A": eps, "warm_fd": fd, "adjoint": adj, "rel_diff": rel})
            print(rows[-1], flush=True)

    small = [r for r in rows if r["eps_A"] <= 300.0]
    worst_small = max(r["rel_diff"] for r in small)
    record = {
        "task": "adjoint-vs-warm-FD coil-gradient sweep (reproducible generator, finding G1)",
        "case": "synthetic diverted coilset, 33x33, tests/test_jax_free_boundary_predictive.py",
        "loss": "smooth_axis_flux(solve_predictive_equilibrium_diff(...)), n_iter=150",
        "method": {
            "adjoint": "jax.grad through the custom_vjp implicit-diff adjoint "
            "(Jacobi-preconditioned BiCGSTAB)",
            "fd": "central FD, WARM-STARTED from the base solution (psi_init=psi_base) to "
            "stay in-basin; sweep the step and require convergence toward the adjoint "
            "(truncation error grows with the step; a real missing term does not)",
        },
        "adjoint_g_ci": [float(v) for v in adjoint_g],
        "measurements": rows,
        "guarded_claim": (
            f"agreement within {worst_small:.3e} relative at 100-300 A steps on the tested "
            "strong and weak coils (this run, full precision); larger steps show the "
            "GROWING-with-eps truncation signature — the origin of the historical '~3 %' "
            "artefact figure"
        ),
        "regression_guard": "tests/test_jax_free_boundary_predictive.py::"
        "test_coil_gradient_matches_finite_difference (eps=300 A, rel<1e-3)",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host_note": f"{platform.node()} — gradients/counts are load-independent; "
        "no timings recorded",
        "jax_version": jax.__version__,
        "provenance": {
            "generator": "validation/measure_coilgrad_adjoint_fd.py",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "pinned_environment": "requirements/full.txt (hash-pinned) for exact reproduction",
            "pinned_requirements_sha256": hashlib.sha256(
                (REPO / "requirements" / "full.txt").read_bytes()
            ).hexdigest(),
        },
    }
    OUT.write_text(json.dumps(record, indent=2) + "\n")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
