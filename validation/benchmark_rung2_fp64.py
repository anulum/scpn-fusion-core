# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — honest FP64 wall-clock benchmark of the predictive free-boundary solve
"""Honest FP64 wall-clock benchmark of the predictive free-boundary solve (Rung 2, GPU half).

Intended for a DEDICATED, otherwise-idle host (the contract bar of ~20 ms per 129² solve with
~10 iterations is only meaningful against dedicated hardware; a loaded development host
produces load-contaminated numbers, which this project treats as indicative-only and never
publishes as performance claims).

Methodology (stated in the output record):
- ``jax_enable_x64`` is asserted ON and the solved ψ dtype is verified float64 — no silent
  FP32 substitution;
- one full warm-up solve per configuration is excluded (JIT compilation + autotuning);
- each configuration is repeated ``REPEATS`` times with ``block_until_ready`` fencing; the
  record keeps EVERY repeat plus median/min — no cherry-picking;
- both the plain and the multigrid-preconditioned forward solve are measured on 33², 65² and
  129², cold start (vacuum ψ init) with the validated Anderson defaults;
- the device, platform, JAX version and per-run iteration setting are recorded verbatim.

Run: ``python validation/benchmark_rung2_fp64.py``
Output: ``artifacts/rung2_mg_preconditioner/fp64_benchmark_<hostname>.json``
"""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from scpn_fusion.core.jax_free_boundary_predictive import (
    build_response_matrix,
    solve_predictive_equilibrium,
)

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "artifacts" / "rung2_mg_preconditioner"

GRIDS = (33, 65, 129)
REPEATS = 5
N_ITER = 120  # validated Anderson default (early-stops at DEFAULT_TOL)

# The synthetic diverted coilset of the test suite, grid-resolution independent.
COIL_R = jnp.array([1.2, 2.3, 1.2, 2.3, 1.6, 1.5])
COIL_Z = jnp.array([0.9, 0.9, -0.9, -0.9, 1.3, -1.35])
COIL_I = jnp.array([-3.0e5, -3.0e5, -3.0e5, -3.0e5, -1.0e5, -6.0e5])
PSIN = jnp.linspace(0.0, 1.0, 6)
PPRIME = jnp.array([-8.0e4, -6.0e4, -4.0e4, -2.0e4, -0.7e4, 0.0])
FFPRIME = jnp.array([-1.2, -0.9, -0.6, -0.3, -0.1, 0.0])
IP = 1.0e6


def _solve(n: int, response, use_mg: bool) -> jnp.ndarray:
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
        n_iter=N_ITER,
        use_mg_preconditioner=use_mg,
    )


def main() -> None:
    assert jax.config.jax_enable_x64, "FP64 must be enabled for this benchmark"
    device = jax.devices()[0]
    results = []
    for n in GRIDS:
        r = jnp.linspace(1.0, 2.5, n)
        z = jnp.linspace(-1.4, 1.4, n)
        response = build_response_matrix(r, z)
        jax.block_until_ready(response)
        for use_mg in (False, True):
            label = f"{n}x{n} {'mg' if use_mg else 'plain'}"
            psi = jax.block_until_ready(_solve(n, response, use_mg))  # warm-up, excluded
            assert psi.dtype == jnp.float64, f"{label}: got {psi.dtype}, not float64"
            times = []
            for _ in range(REPEATS):
                t0 = time.perf_counter()
                jax.block_until_ready(_solve(n, response, use_mg))
                times.append(time.perf_counter() - t0)
            times_ms = [t * 1e3 for t in times]
            row = {
                "grid": f"{n}x{n}",
                "preconditioner": "mg_vcycle" if use_mg else "none",
                "repeats_ms": times_ms,
                "median_ms": sorted(times_ms)[len(times_ms) // 2],
                "min_ms": min(times_ms),
                "psi_dtype": str(psi.dtype),
            }
            results.append(row)
            print(
                f"{label}: median {row['median_ms']:.1f} ms  min {row['min_ms']:.1f} ms", flush=True
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "task": "Rung 2 (forward speed) — dedicated-hardware FP64 wall-clock half",
        "contract_bar_note": (
            "reference figure: ~20 ms per 129^2 solve with ~10 iterations (partner solver); "
            "this measures the full cold-start Anderson solve (n_iter cap 120, early-stop at "
            "the validated tolerance) — iteration regimes differ and are stated, not hidden"
        ),
        "methodology": (
            "warm-up excluded; block_until_ready fencing; all repeats recorded; FP64 asserted "
            "on the solved field; dedicated idle host required for the numbers to be a claim"
        ),
        "device": f"{device.device_kind} ({device.platform})",
        "jax_version": jax.__version__,
        "host": f"{platform.node()} ({platform.machine()})",
        "n_iter_cap": N_ITER,
        "repeats": REPEATS,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results": results,
    }
    out = OUT_DIR / f"fp64_benchmark_{platform.node()}.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
