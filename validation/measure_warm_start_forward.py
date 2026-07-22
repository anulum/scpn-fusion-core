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

1. **Correctness**: a warm-started solve (``psi_init`` = base solution, ``ip_ramp=1`` so the
   early stop is armed immediately — the ramp exists for cold-start robustness and would
   otherwise force 30 iterations) must land on the SAME fixed point as a cold solve of the
   perturbed problem. Asserted at span-relative tolerance for a ±0.5 % coil perturbation
   (the FD-validation scale, documented in-basin).
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

from scpn_fusion.core.jax_free_boundary_predictive import build_response_matrix
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
    "src/scpn_fusion/core/jax_multigrid_precond.py",
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
    equivalences = {}
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
                ip_ramp=1,
            )

        psi_base = jax.block_until_ready(cold(COIL_I))
        psi_cold_pert = jax.block_until_ready(cold(ci_pert))
        psi_warm = jax.block_until_ready(warm(ci_pert, psi_base))  # also warms the jit cache
        span = abs(
            float(smooth_axis_flux(psi_cold_pert)) - float(smooth_xpoint_flux(psi_cold_pert, r, z))
        )
        equiv = float(jnp.max(jnp.abs(psi_warm - psi_cold_pert))) / span
        equivalences[f"{n}x{n}"] = equiv
        print(f"{n}^2 warm-vs-cold fixed-point agreement (span-rel): {equiv:.3e}", flush=True)

        cold_ms = [_time_one(lambda: cold(ci_pert)) for _ in range(REPEATS)]
        warm_ms = [_time_one(lambda: warm(ci_pert, psi_base)) for _ in range(REPEATS)]
        rows.append(
            {
                "grid": f"{n}x{n}",
                "cold_ms": cold_ms,
                "warm_ms": warm_ms,
                "cold_median_ms": sorted(cold_ms)[REPEATS // 2],
                "warm_median_ms": sorted(warm_ms)[REPEATS // 2],
            }
        )
        print(
            f"{n}^2 cold {rows[-1]['cold_median_ms']:.0f} ms  "
            f"warm {rows[-1]['warm_median_ms']:.0f} ms",
            flush=True,
        )

    record = {
        "task": "warm-start behaviour of the compiled predictive forward (IDA pattern)",
        "method": (
            "base solve at nominal coil currents; perturb all coil currents by +0.5% (the "
            "documented in-basin FD-validation scale); warm solve = psi_init=base + "
            "ip_ramp=1 (the ramp exists for cold-start robustness and would otherwise force "
            "30 iterations before the early stop can fire); correctness = warm solve agrees "
            "with the COLD solve of the perturbed problem at span-relative tolerance"
        ),
        "correctness_load_independent": {"warm_vs_cold_fixed_point_span_rel": equivalences},
        "timings_indicative": {
            "host": f"{platform.node()} ({platform.machine()}) - host-load caveat applies; "
            "the claimable number is this generator on a dedicated host",
            "device": f"{device.device_kind} ({device.platform})",
            "rows": rows,
        },
        "settings": {"perturb_rel": PERTURB_REL, "repeats": REPEATS},
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
