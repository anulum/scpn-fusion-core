# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — same-host wall-clock comparison vs FreeGS (free-boundary solver class)
"""Same-host wall-clock comparison: our predictive free-boundary solve vs FreeGS.

Honest scope, stated up front:

- The two solvers run **their own canonical diverted test cases** (FreeGS: ``TestTokamak``
  example with its X-point constraints; ours: the synthetic diverted coilset of the test
  suite). This is a **solver-class comparison at matched grid resolution, FP64, same host**
  — NOT the same physical problem solved twice. The accuracy link between the two codes is
  established separately: our predictive solver reproduces the SHA-verified FreeGS DIII-D
  reference to ≈ 0.9 % of the ψ span from a cold start (Rung-1 record).
- FreeGS is NumPy/SciPy CPU code by design; ours is JAX (CPU or GPU). Both timings are
  reported with the device stated. Warm-up (JIT compile) is excluded for JAX; FreeGS gets an
  untimed warm-up solve as well so both measure a steady-state re-solve.
- All repeats are recorded; medians reported; no cherry-picking.

Run: ``python validation/benchmark_freegs_comparison.py``
Output: ``artifacts/rung2_mg_preconditioner/freegs_comparison_<hostname>.json``
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

import freegs

from scpn_fusion.core.jax_free_boundary_predictive import (
    build_response_matrix,
    solve_predictive_equilibrium,
)

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "artifacts" / "rung2_mg_preconditioner"

GRIDS = (65, 129)
REPEATS = 5

COIL_R = jnp.array([1.2, 2.3, 1.2, 2.3, 1.6, 1.5])
COIL_Z = jnp.array([0.9, 0.9, -0.9, -0.9, 1.3, -1.35])
COIL_I = jnp.array([-3.0e5, -3.0e5, -3.0e5, -3.0e5, -1.0e5, -6.0e5])
PSIN = jnp.linspace(0.0, 1.0, 6)
PPRIME = jnp.array([-8.0e4, -6.0e4, -4.0e4, -2.0e4, -0.7e4, 0.0])
FFPRIME = jnp.array([-1.2, -0.9, -0.6, -0.3, -0.1, 0.0])
IP = 1.0e6


def _freegs_solve(n: int) -> None:
    """One full FreeGS free-boundary solve of its canonical diverted example at n×n."""
    tokamak = freegs.machine.TestTokamak()
    eq = freegs.Equilibrium(tokamak=tokamak, Rmin=0.1, Rmax=2.0, Zmin=-1.0, Zmax=1.0, nx=n, ny=n)
    profiles = freegs.jtor.ConstrainPaxisIp(eq, 1e3, 2e5, 2.0)
    xpoints = [(1.1, -0.6), (1.1, 0.8)]
    isoflux = [(1.1, -0.6, 1.1, 0.6)]
    constrain = freegs.control.constrain(xpoints=xpoints, isoflux=isoflux)
    freegs.solve(eq, profiles, constrain, show=False)


def _ours_solve(n: int, response, use_mg: bool) -> jnp.ndarray:
    r = jnp.linspace(1.0, 2.5, n)
    z = jnp.linspace(-1.4, 1.4, n)
    m, b, s = response
    return solve_predictive_equilibrium(
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
        use_mg_preconditioner=use_mg,
    )


def _time_repeats(fn) -> list[float]:
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return times


def main() -> None:
    device = jax.devices()[0]
    results = []
    for n in GRIDS:
        # FreeGS (CPU by design) — untimed warm-up, then timed repeats.
        _freegs_solve(n)
        t_freegs = _time_repeats(lambda n=n: _freegs_solve(n))
        results.append(
            {
                "solver": "freegs",
                "case": "TestTokamak diverted example (its own canonical case)",
                "grid": f"{n}x{n}",
                "device": "cpu (NumPy/SciPy by design)",
                "repeats_ms": t_freegs,
                "median_ms": sorted(t_freegs)[len(t_freegs) // 2],
            }
        )
        print(results[-1]["solver"], n, f"{results[-1]['median_ms']:.0f} ms", flush=True)

        r = jnp.linspace(1.0, 2.5, n)
        z = jnp.linspace(-1.4, 1.4, n)
        response = build_response_matrix(r, z)
        jax.block_until_ready(response)
        for use_mg in (False, True):
            psi = jax.block_until_ready(_ours_solve(n, response, use_mg))  # warm-up
            assert psi.dtype == jnp.float64
            t_ours = _time_repeats(
                lambda n=n, response=response, use_mg=use_mg: jax.block_until_ready(
                    _ours_solve(n, response, use_mg)
                )
            )
            results.append(
                {
                    "solver": "scpn_fusion predictive" + (" + mg preconditioner" if use_mg else ""),
                    "case": "synthetic diverted coilset (our canonical case)",
                    "grid": f"{n}x{n}",
                    "device": f"{device.device_kind} ({device.platform})",
                    "repeats_ms": t_ours,
                    "median_ms": sorted(t_ours)[len(t_ours) // 2],
                }
            )
            print(results[-1]["solver"], n, f"{results[-1]['median_ms']:.0f} ms", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "task": "same-host wall-clock comparison of free-boundary solver classes",
        "honest_scope": (
            "each solver runs ITS OWN canonical diverted case at matched grid resolution "
            "and FP64 — a solver-class comparison, not the same physical problem twice; "
            "the accuracy link is the separate Rung-1 record (our predictive solve "
            "reproduces the SHA-verified FreeGS DIII-D reference to ~0.9% of the psi span "
            "from a cold start); FreeGS is CPU NumPy/SciPy by design and its timing is a "
            "CPU timing, stated as such"
        ),
        "host": f"{platform.node()} ({platform.machine()})",
        "jax_device": f"{device.device_kind} ({device.platform})",
        "freegs_version": freegs.__version__,
        "jax_version": jax.__version__,
        "repeats": REPEATS,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results": results,
        "provenance": {
            "generator": "validation/benchmark_freegs_comparison.py",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    out = OUT_DIR / f"freegs_comparison_{platform.node()}.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
