#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Deterministic Replay Certificate
"""Generate and verify the deterministic replay certificate (M-2).

The certificate proves that a seeded control episode — an equilibrium
multigrid solve, a batched multi-layer UPDE phase-control segment, and a
seeded tearing-mode indicator trace — replays BIT-IDENTICALLY: every
trajectory is hashed (SHA-256 over canonical little-endian float64 bytes)
and the certificate carries the component hashes, a combined hash, and an
environment manifest.

Claim boundary (revised after the first real two-machine comparison,
CI run 28841804121 on 2026-07-07):

- Same-machine, cross-run bit-identity is the ASSERTED contract for both
  tiers (double-run verified at generation and by the test lane).
- Cross-machine bit-identity is ENVIRONMENT-CONDITIONAL, not universal:
  the first CI comparison against this rig's certificate showed that
  components exercising vectorised transcendentals (np.exp in the
  equilibrium source, np.sin in the phase step) diverge across CPU
  microarchitectures — numpy dispatches SIMD kernels at runtime, so an
  identical wheel can round differently on different hardware. The
  ``disruption_indicator`` component (scalar-RNG arithmetic, no vectorised
  transcendentals) DID reproduce bit-identically across machines and is
  the asserted cross-machine invariant.
- ``fastest_tier`` hashes are RECORDED evidence for the generating
  machine only (platform libm differences apply on top).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CERTIFICATE = ROOT / "validation" / "reference_data" / "replay" / "replay_certificate.json"
SCHEMA = "scpn-fusion-core.replay-certificate.v1"
EPISODE_SEED = 2026

sys.path.insert(0, str(SRC))

FloatArray = npt.NDArray[np.float64]


def _hash_array(values: FloatArray) -> str:
    """SHA-256 over canonical little-endian float64 bytes."""
    canonical = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _episode_equilibrium(tier: str) -> FloatArray:
    """Seeded multigrid Grad-Shafranov solve segment."""
    from scpn_fusion.core import _multi_compat as multi
    from scpn_fusion.core import _multi_compat_providers as providers

    n = 65
    rng = np.random.default_rng(EPISODE_SEED)
    r_axis = np.linspace(4.0, 8.0, n)
    z_axis = np.linspace(-4.0, 4.0, n)
    r_grid, z_grid = np.meshgrid(r_axis, z_axis)
    source = -np.exp(-((r_grid - 6.0) ** 2 + z_grid**2) / 0.4)
    psi_bc = np.zeros((n, n))
    psi_bc[0, :] = psi_bc[-1, :] = psi_bc[:, 0] = psi_bc[:, -1] = (
        1e-3 * rng.standard_normal(n)[0]
    )  # deterministic scalar boundary offset

    impl = (
        providers._numpy_multigrid_solve if tier == "numpy" else multi.dispatch("multigrid_solve")
    )
    psi, residual, n_cycles, converged = impl(
        source, psi_bc, 4.0, 8.0, -4.0, 4.0, n, n, tol=1e-8, max_cycles=200
    )
    if not converged:
        raise RuntimeError("replay episode equilibrium solve did not converge")
    trace = np.concatenate([np.asarray(psi, dtype=np.float64).ravel(), [float(residual)]])
    return trace


def _episode_phase_control(tier: str) -> FloatArray:
    """Seeded batched UPDE phase-control segment."""
    from scpn_fusion.core import _multi_compat as multi
    from scpn_fusion.core import _multi_compat_providers as providers

    rng = np.random.default_rng(EPISODE_SEED + 1)
    layers, n_per = 4, 96
    total = layers * n_per
    theta0 = rng.uniform(-np.pi, np.pi, size=total)
    omega = rng.normal(0.0, 0.15, size=total)
    offsets = np.arange(0, total + 1, n_per, dtype=np.intp)
    K = 0.6 * np.eye(layers) + 0.08 * rng.uniform(size=(layers, layers))
    alpha = np.zeros((layers, layers))
    zeta = np.full(layers, 0.4)

    impl = providers._numpy_upde_run if tier == "numpy" else multi.dispatch("upde_run")
    out = impl(
        theta0,
        omega,
        offsets,
        K,
        alpha,
        zeta,
        n_steps=250,
        dt=2e-3,
        psi_global=0.3,
        actuation_gain=1.0,
        pac_gamma=0.25,
        wrap=True,
    )
    return np.concatenate(
        [
            np.asarray(out["theta_final"], dtype=np.float64).ravel(),
            np.asarray(out["R_global_hist"], dtype=np.float64).ravel(),
            np.asarray(out["V_global_hist"], dtype=np.float64).ravel(),
        ]
    )


def _episode_disruption_indicator(tier: str) -> FloatArray:
    """Seeded tearing-mode indicator trace segment."""
    from scpn_fusion.core import _multi_compat as multi
    from scpn_fusion.core import _multi_compat_providers as providers

    impl = (
        providers._numpy_simulate_tearing_mode
        if tier == "numpy"
        else multi.dispatch("simulate_tearing_mode")
    )
    signal, label, ttd = impl(800, seed=EPISODE_SEED + 2)
    return np.concatenate(
        [np.asarray(signal, dtype=np.float64).ravel(), [float(label), float(ttd)]]
    )


_COMPONENTS = {
    "equilibrium_multigrid": _episode_equilibrium,
    "phase_control_upde": _episode_phase_control,
    "disruption_indicator": _episode_disruption_indicator,
}


def run_episode(tier: str) -> dict[str, str]:
    """Run every episode component on *tier* and return its hash map."""
    if tier not in ("numpy", "fastest"):
        raise ValueError("tier must be 'numpy' or 'fastest'")
    return {name: _hash_array(fn(tier)) for name, fn in _COMPONENTS.items()}


def _combined_hash(component_hashes: dict[str, str]) -> str:
    """Order-independent combined hash over the component hash map."""
    canonical = json.dumps(component_hashes, sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _environment_manifest() -> dict[str, Any]:
    """Record the environment the certificate was generated in."""
    from scpn_fusion.core import _multi_compat as multi

    tiers = {
        name: multi.dispatch_tier(name)
        for name in ("multigrid_solve", "upde_run", "simulate_tearing_mode")
    }
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "fastest_tier_selection": tiers,
    }


def build_certificate() -> dict[str, Any]:
    """Generate the replay certificate payload (double-run verified)."""
    numpy_first = run_episode("numpy")
    numpy_second = run_episode("numpy")
    if numpy_first != numpy_second:
        raise RuntimeError("NumPy floor failed same-process replay - episode is not deterministic")
    fastest_first = run_episode("fastest")
    fastest_second = run_episode("fastest")
    if fastest_first != fastest_second:
        raise RuntimeError("fastest tier failed same-process replay - tier is not deterministic")

    return {
        "schema": SCHEMA,
        "episode_seed": EPISODE_SEED,
        "numpy_floor": {
            "component_hashes": numpy_first,
            "combined_hash": _combined_hash(numpy_first),
            "claim": (
                "asserted same-machine cross-run; cross-machine only for "
                "components without vectorised transcendentals "
                "(disruption_indicator) - numpy runtime SIMD dispatch rounds "
                "exp/sin differently across CPU microarchitectures"
            ),
        },
        "fastest_tier": {
            "component_hashes": fastest_first,
            "combined_hash": _combined_hash(fastest_first),
            "claim": (
                "recorded for the generating machine (same-machine cross-run "
                "asserted by tests); cross-machine NOT asserted (platform libm)"
            ),
        },
        "environment": _environment_manifest(),
    }


def verify_certificate(path: Path) -> dict[str, Any]:
    """Re-run the episode and compare against a committed certificate."""
    committed = cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
    if committed.get("schema") != SCHEMA:
        raise ValueError(f"unexpected certificate schema: {committed.get('schema')!r}")

    numpy_now = run_episode("numpy")
    committed_numpy = committed["numpy_floor"]["component_hashes"]
    numpy_match = numpy_now == committed_numpy

    fastest_now = run_episode("fastest")
    fastest_match = fastest_now == committed["fastest_tier"]["component_hashes"]

    verifier_env = _environment_manifest()
    certificate_env = committed["environment"]
    environment_matches = all(
        verifier_env.get(key) == certificate_env.get(key)
        for key in ("python", "numpy", "platform", "machine")
    )

    return {
        "numpy_floor_bit_identical": numpy_match,
        "numpy_component_matches": {
            name: numpy_now.get(name) == committed_numpy.get(name) for name in committed_numpy
        },
        "fastest_tier_bit_identical": fastest_match,
        "numpy_component_hashes": numpy_now,
        "fastest_component_hashes": fastest_now,
        "environment_matches": environment_matches,
        "verifier_environment": verifier_env,
        "certificate_environment": certificate_env,
    }


def main(argv: list[str] | None = None) -> int:
    """Generate or verify the replay certificate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="verify against the committed file")
    parser.add_argument("--output", type=Path, default=CERTIFICATE)
    args = parser.parse_args(argv)

    if args.verify:
        result = verify_certificate(args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
        if result["environment_matches"]:
            return 0 if result["numpy_floor_bit_identical"] else 1
        # Different machine class: only the transcendental-free component is
        # the asserted cross-machine invariant.
        return 0 if result["numpy_component_matches"].get("disruption_indicator") else 1

    certificate = build_certificate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
