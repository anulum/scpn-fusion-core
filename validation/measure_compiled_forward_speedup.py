# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — measured evidence for the compiled predictive forward (eager vs jit loop)
"""Measure the compiled predictive forward against the eager reference (evidence generator).

Records, with full provenance (generator + logic-source digests + pinned environment — the
schema the Lane-2 distinct-eye review converged on):

- the **fixed-point equivalence** of the compiled and eager solvers (span-relative), which is
  the load-independent correctness fact;
- indicative wall-clock: eager vs compiled-warm at 33² (the speedup anchor) and compiled-warm
  at 33²/65²/129², with one-time compile cost stated separately. Timings on a development
  host are load-contaminated and are labelled INDICATIVE — the claimable number is the
  dedicated-hardware re-benchmark (separate lane).

Run: ``python validation/measure_compiled_forward_speedup.py``
Output: ``artifacts/rung2_mg_preconditioner/compiled_forward_speedup.json``
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

from scpn_fusion.core.jax_free_boundary_predictive import (
    build_response_matrix,
    solve_predictive_equilibrium,
)
from scpn_fusion.core.jax_o_point import smooth_axis_flux
from scpn_fusion.core.jax_predictive_forward_compiled import (
    solve_predictive_equilibrium_compiled,
)
from scpn_fusion.core.jax_x_point import smooth_xpoint_flux

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "rung2_mg_preconditioner" / "compiled_forward_speedup.json"

COIL_R = jnp.array([1.2, 2.3, 1.2, 2.3, 1.6, 1.5])
COIL_Z = jnp.array([0.9, 0.9, -0.9, -0.9, 1.3, -1.35])
COIL_I = jnp.array([-3.0e5, -3.0e5, -3.0e5, -3.0e5, -1.0e5, -6.0e5])
PSIN = jnp.linspace(0.0, 1.0, 6)
PPRIME = jnp.array([-8.0e4, -6.0e4, -4.0e4, -2.0e4, -0.7e4, 0.0])
FFPRIME = jnp.array([-1.2, -0.9, -0.6, -0.3, -0.1, 0.0])
IP = 1.0e6
REPEATS = 3
COMPILED_GRIDS = (33, 65, 129)

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


def _case(n: int):
    r = jnp.linspace(1.0, 2.5, n)
    z = jnp.linspace(-1.4, 1.4, n)
    return r, z, build_response_matrix(r, z)


def _time_one(fn) -> float:
    t0 = time.perf_counter()
    jax.block_until_ready(fn())
    return (time.perf_counter() - t0) * 1e3


def main() -> None:
    device = jax.devices()[0]
    r33, z33, resp33 = _case(33)
    m, b, s = resp33

    def eager():
        return solve_predictive_equilibrium(
            COIL_I, PPRIME, FFPRIME, r33, z33, COIL_R, COIL_Z, PSIN, IP, m, b, s
        )

    def compiled(r, z, resp):
        mm, bb, ss = resp
        return solve_predictive_equilibrium_compiled(
            COIL_I, PPRIME, FFPRIME, r, z, COIL_R, COIL_Z, PSIN, IP, mm, bb, ss
        )

    # Correctness fact first (load-independent): same fixed point.
    psi_e = jax.block_until_ready(eager())
    compile_ms_33 = _time_one(lambda: compiled(r33, z33, resp33))  # includes trace+compile
    psi_c = jax.block_until_ready(compiled(r33, z33, resp33))
    span = abs(float(smooth_axis_flux(psi_e)) - float(smooth_xpoint_flux(psi_e, r33, z33)))
    equivalence = float(jnp.max(jnp.abs(psi_c - psi_e))) / span
    print(f"fixed-point equivalence (33^2, span-rel): {equivalence:.3e}", flush=True)

    eager_ms = [_time_one(eager) for _ in range(REPEATS)]
    print(f"eager 33^2: {[f'{t:.0f}' for t in eager_ms]} ms", flush=True)

    compiled_rows = []
    for n in COMPILED_GRIDS:
        r, z, resp = _case(n)
        compile_ms = _time_one(lambda: compiled(r, z, resp)) if n != 33 else compile_ms_33
        warm_ms = [_time_one(lambda: compiled(r, z, resp)) for _ in range(REPEATS)]
        row = {
            "grid": f"{n}x{n}",
            "compile_plus_first_solve_ms": compile_ms,
            "warm_ms": warm_ms,
            "warm_median_ms": sorted(warm_ms)[len(warm_ms) // 2],
        }
        compiled_rows.append(row)
        print(f"compiled {n}^2 warm: {[f'{t:.0f}' for t in warm_ms]} ms", flush=True)

    record = {
        "task": "compiled predictive forward (whole Anderson loop under jit) vs eager",
        "correctness_load_independent": {
            "fixed_point_equivalence_span_rel_33sq": equivalence,
            "note": "compiled and eager reach the same equilibrium; warm repeats are "
            "bit-deterministic (asserted in tests/test_jax_predictive_forward_compiled.py)",
        },
        "timings_indicative": {
            "host": f"{platform.node()} ({platform.machine()}) - LOADED development host; "
            "indicative only, NOT a performance claim; the claimable number is the "
            "dedicated-hardware re-benchmark",
            "device": f"{device.device_kind} ({device.platform})",
            "eager_33sq_ms": eager_ms,
            "compiled": compiled_rows,
            "speedup_anchor_33sq": (sorted(eager_ms)[len(eager_ms) // 2])
            / (sorted(compiled_rows[0]["warm_ms"])[len(compiled_rows[0]["warm_ms"]) // 2]),
        },
        "settings": {"repeats": REPEATS, "defaults": "validated Anderson defaults, MG on"},
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "jax_version": jax.__version__,
        "provenance": {
            "generator": "validation/measure_compiled_forward_speedup.py",
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
