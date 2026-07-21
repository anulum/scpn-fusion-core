# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — measured Krylov iteration counts for the multigrid GS preconditioner
"""Measure BiCGSTAB iteration counts with/without the geometric-multigrid preconditioner.

Fidelity-curve Rung 2, local (host-load-independent) half of the evidence: the number of
Krylov iterations needed to solve the predictive solver's linear GS system to a given true
relative residual is a property of the operator, the right-hand side and the preconditioner —
not of the host load. Wall-clock timings are deliberately NOT recorded here; the honest FP64
wall-clock benchmark against the ~20 ms/129² contract bar is the dedicated-hardware (A100)
half of Rung 2.

Method: for each grid (33², 65², 129²) build the identity-wall GS operator and a GS-like
right-hand side (smooth interior source + non-trivial Dirichlet ring), then bisect the
smallest ``maxiter`` whose returned iterate satisfies ``‖A x − b‖/‖b‖ ≤ tol`` — BiCGSTAB's
internal stopping test is disabled so the count is exact.

Run: ``python validation/measure_mg_preconditioner_iterations.py``
Output: ``artifacts/rung2_mg_preconditioner/iteration_counts.json``
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from scpn_fusion.core.jax_free_boundary_predictive import _gs_operator_flat
from scpn_fusion.core.jax_multigrid_precond import build_gs_mg_preconditioner

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "artifacts" / "rung2_mg_preconditioner"

GRIDS = (33, 65, 129)
TOLS = (1.0e-8, 1.0e-11)  # the second is the predictive solver's own _BICGSTAB_TOL
MAX_CAP = 20000


def _case(n: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float, float]:
    r = jnp.linspace(1.0, 2.5, n)
    z = jnp.linspace(-1.4, 1.4, n)
    rr, zz = jnp.meshgrid(r, z)
    interior = -3.0e-1 * rr * jnp.exp(-((rr - 1.7) ** 2 + zz**2) / 0.15)
    ring = 0.05 * rr + 0.02 * zz
    rhs = interior.at[0, :].set(ring[0, :]).at[-1, :].set(ring[-1, :])
    rhs = rhs.at[:, 0].set(ring[:, 0]).at[:, -1].set(ring[:, -1])
    return r, z, rhs.reshape(-1), float(r[1] - r[0]), float(z[1] - z[0])


def _min_iters(operator, rhs: jnp.ndarray, m, tol: float) -> int:
    b_norm = float(jnp.linalg.norm(rhs))

    def achieved(maxiter: int) -> bool:
        x, _ = jax.scipy.sparse.linalg.bicgstab(
            operator, rhs, tol=0.0, atol=1e-300, maxiter=maxiter, M=m
        )
        res = float(jnp.linalg.norm(operator(x) - rhs)) / b_norm
        return bool(np.isfinite(res)) and res <= tol

    lo, hi = 1, 1
    while not achieved(hi):
        lo, hi = hi, hi * 2
        if hi > MAX_CAP:
            return -1  # honest failure marker: not reached within the cap
    while lo < hi:
        mid = (lo + hi) // 2
        if achieved(mid):
            hi = mid
        else:
            lo = mid + 1
    return hi


def main() -> None:
    rows = []
    for n in GRIDS:
        r, _z, rhs, d_r, d_z = _case(n)
        shape = (n, n)

        def operator(pf: jnp.ndarray, _shape=shape, _r=r, _dr=d_r, _dz=d_z) -> jnp.ndarray:
            return _gs_operator_flat(pf, _shape, _r, jnp.asarray(_dr), jnp.asarray(_dz))

        m = build_gs_mg_preconditioner(shape, r, d_r, d_z)
        for tol in TOLS:
            plain = _min_iters(operator, rhs, None, tol)
            mg = _min_iters(operator, rhs, m, tol)
            speedup = (plain / mg) if (plain > 0 and mg > 0) else None
            rows.append(
                {
                    "grid": f"{n}x{n}",
                    "rel_tol": tol,
                    "bicgstab_iters_plain": plain,
                    "bicgstab_iters_mg_preconditioned": mg,
                    "iteration_ratio_plain_over_mg": speedup,
                }
            )
            print(rows[-1], flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "task": "Rung 2 (forward speed) — local, host-load-independent half of the evidence",
        "metric": (
            "minimal BiCGSTAB maxiter whose iterate satisfies the TRUE relative residual "
            "(internal stopping disabled); iteration counts are independent of host load"
        ),
        "note": (
            "one MG-preconditioned iteration costs more than a plain one (a V-cycle per "
            "matvec pair); the wall-clock verdict against the ~20 ms/129^2 contract bar is "
            "the dedicated-hardware FP64 benchmark, deliberately not measured here"
        ),
        "case": "synthetic GS-like RHS (smooth interior source + non-trivial Dirichlet ring)",
        "preconditioner": "build_gs_mg_preconditioner defaults (1 V-cycle, 2+2 RB-GS sweeps)",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host_note": f"{platform.node()} ({platform.machine()}) — counts only, no timings",
        "measurements": rows,
        "provenance": {
            "generator": "validation/measure_mg_preconditioner_iterations.py",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "pinned_environment": "requirements/full.txt (hash-pinned) for exact reproduction",
            "pinned_requirements_sha256": hashlib.sha256(
                (REPO / "requirements" / "full.txt").read_bytes()
            ).hexdigest(),
        },
    }
    out = OUT_DIR / "iteration_counts.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
