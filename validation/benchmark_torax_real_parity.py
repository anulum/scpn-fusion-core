#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Real-TORAX Profile Comparison Gate
"""Compare our 1.5D transport against a really-executed TORAX reference.

The reference artifact (``validation/reference_data/torax/``) was produced by
actually running the open-source TORAX code (Apache-2.0, DeepMind) through
``validation/torax_reference_runner.py`` in its dedicated virtual environment
(``.venv-torax``; TORAX requires numpy>=2/jax>=0.10, incompatible with the
project pins). The artifact carries full provenance (TORAX version, config
SHA-256, timestamp), and this lane runs everywhere against the committed
artifact, so its checks are deterministic.

Claim boundary: this gate asserts reference integrity, finite solver output,
and that the divergence metrics are RECORDED — it does not claim physics
equivalence. The first real-reference comparison exposed a genuine finding in
our solver, documented in ``solver_stability_findings``: the 1.5D steady state
is timestep-dependent (Ti0 ~0.8 keV at dt=0.1 s vs a ~22.5 keV transient at
dt=0.5 s on iter_config with 50 MW), and the dt=0.5 s trajectory enters a
numerical crash-rebuild limit cycle after ~12 s with no sawtooth model in the
lane to explain it. Equivalence thresholds may only be introduced after that
correctness row is closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from importlib import import_module
from typing import Any, cast

import numpy as np
import numpy.typing as npt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REFERENCE = ROOT / "validation" / "reference_data" / "torax" / "torax_basic_config_profiles.json"
REPORT = ROOT / "validation" / "reports" / "torax_real_parity.json"
SCHEMA = "scpn-fusion-core.torax-real-parity.v1"

COARSE_DT_S = 0.5
COARSE_STEPS = 44
FINE_DT_S = 0.1
FINE_STEPS = 80
P_AUX_MW = 50.0

sys.path.insert(0, str(SRC))


def _load_reference() -> dict[str, Any]:
    """Load and integrity-check the committed TORAX reference artifact."""
    payload = cast(dict[str, Any], json.loads(REFERENCE.read_text(encoding="utf-8")))
    provenance = payload["provenance"]
    required = ("code", "torax_version", "config_name", "config_sha256", "licence")
    missing = [key for key in required if not provenance.get(key)]
    if missing:
        raise ValueError(f"TORAX reference provenance incomplete: missing {missing}")
    if provenance["code"] != "TORAX":
        raise ValueError("reference artifact is not a TORAX export")
    return payload


def _profile_checksum(profiles: dict[str, Any]) -> str:
    """Return a stable checksum of the reference profile payload."""
    canonical = json.dumps(profiles, sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _evolve_trajectory(dt: float, steps: int) -> dict[str, Any]:
    """Evolve our 1.5D transport and record the core-Ti trajectory."""
    transport_module = import_module("scpn_fusion.core.integrated_transport_solver")
    solver = transport_module.TransportSolver(str(ROOT / "validation" / "iter_config.json"))
    core_ti: list[float] = []
    for _ in range(steps):
        solver.evolve_profiles(dt, P_AUX_MW)
        core_ti.append(float(solver.Ti[0]))
    te = np.asarray(solver.Te, dtype=np.float64)
    rho = np.asarray(solver.rho, dtype=np.float64)
    trajectory = np.asarray(core_ti, dtype=np.float64)
    tail = trajectory[-8:]
    # Period-2 alternation detector: large swing between consecutive samples
    # while the two-step difference stays small.
    swings = np.abs(np.diff(tail))
    period2 = np.abs(tail[2:] - tail[:-2])
    limit_cycle = bool(np.max(swings) > 2.0 and np.median(period2) < 0.5)
    return {
        "dt_s": dt,
        "steps": steps,
        "final_core_ti_kev": float(trajectory[-1]),
        "peak_core_ti_kev": float(np.max(trajectory)),
        "limit_cycle_detected": limit_cycle,
        "finite": bool(np.all(np.isfinite(trajectory)) and np.all(np.isfinite(te))),
        "te_kev": te,
        "rho_norm": rho,
    }


def _normalised_shape(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Normalise a profile to its core value for shape comparison."""
    core = float(values[0])
    return values / max(abs(core), 1e-30)


def build_report() -> dict[str, Any]:
    """Build the real-TORAX comparison report payload."""
    reference = _load_reference()
    ref_profiles = reference["profiles"]
    ref_rho = np.asarray(ref_profiles["rho_norm"], dtype=np.float64)
    ref_te = np.asarray(ref_profiles["T_e_keV"], dtype=np.float64)

    coarse = _evolve_trajectory(COARSE_DT_S, COARSE_STEPS)
    fine = _evolve_trajectory(FINE_DT_S, FINE_STEPS)

    fine_te_on_ref = np.interp(ref_rho, fine["rho_norm"], fine["te_kev"])
    shape_delta = _normalised_shape(fine_te_on_ref) - _normalised_shape(ref_te)
    reference_shape_norm = float(np.linalg.norm(_normalised_shape(ref_te)))
    shape_rel_l2 = float(np.linalg.norm(shape_delta)) / max(reference_shape_norm, 1e-30)

    integrity_ok = True  # _load_reference raised otherwise
    finite_ok = bool(coarse["finite"] and fine["finite"])
    metrics_recorded = True
    passed = integrity_ok and finite_ok and metrics_recorded

    for run in (coarse, fine):
        run.pop("te_kev", None)
        run.pop("rho_norm", None)

    return {
        "schema": SCHEMA,
        "status": (
            "real_torax_reference_acquired_divergence_documented"
            if passed
            else "failed_reference_or_solver_integrity"
        ),
        "passes_thresholds": passed,
        "physics_equivalence_claimed": False,
        "claim_boundary": (
            "Comparison against a really-executed TORAX basic_config run; the "
            "reduced 1.5D solver and TORAX differ by design and by the open "
            "solver-stability finding below. Metrics are recorded divergence, "
            "not equivalence."
        ),
        "reference": {
            "artifact": str(REFERENCE.relative_to(ROOT)),
            "provenance": reference["provenance"],
            "profile_checksum_sha256": _profile_checksum(ref_profiles),
            "final_time_s": reference["final_time_s"],
            "torax_core_te_kev": float(ref_te[0]),
        },
        "our_solver": {
            "config": "validation/iter_config.json",
            "p_aux_mw": P_AUX_MW,
            "coarse_run": coarse,
            "fine_run": fine,
        },
        "divergence_metrics": {
            "core_te_ratio_fine_over_torax": float(fine_te_on_ref[0] / max(ref_te[0], 1e-30)),
            "normalised_te_shape_rel_l2_fine": shape_rel_l2,
        },
        "solver_stability_findings": {
            "steady_state_dt_dependence": (
                f"final core Ti {fine['final_core_ti_kev']:.2f} keV at dt={FINE_DT_S} s vs "
                f"peak {coarse['peak_core_ti_kev']:.2f} keV at dt={COARSE_DT_S} s "
                "on the same config and heating"
            ),
            "limit_cycle_at_coarse_dt": coarse["limit_cycle_detected"],
            "sawtooth_model_present_in_lane": False,
            "disposition": (
                "open correctness row: root-cause the source/diffusion splitting "
                "dt-dependence before any TORAX equivalence threshold is set"
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Run the real-TORAX comparison gate and write the tracked report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPORT)
    args = parser.parse_args(argv)

    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["divergence_metrics"], indent=2, sort_keys=True))
    print(json.dumps(report["solver_stability_findings"], indent=2, sort_keys=True))
    print(f"status: {report['status']}")
    return 0 if report["passes_thresholds"] else 1


if __name__ == "__main__":
    sys.exit(main())
