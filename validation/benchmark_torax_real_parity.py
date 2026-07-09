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
our solver (a timestep-dependent steady state plus a period-2 crash-rebuild
limit cycle at dt=0.5 s); its four numerical root causes were fixed on
2026-07-07 and ``solver_stability_findings`` now records the dt-consistency
of the steady state (coarse/fine core ratio inside
``STEADY_STATE_CORE_RATIO_BAND``) together with the limit-cycle detector.
Equivalence thresholds versus TORAX remain unset because the transport
models still differ by design (fixed chi versus TORAX's transport model).
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
COARSE_STEPS = 400
FINE_DT_S = 0.1
FINE_STEPS = 2000
P_AUX_MW = 50.0
# Both runs integrate to t = 200 s so the steady states (not transients,
# which differ at first order in dt) are compared for dt-consistency.
STEADY_STATE_CORE_RATIO_BAND = (0.97, 1.03)

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


def _environment_record(*, include_runtime_environment: bool) -> dict[str, Any]:
    """Return deterministic or runtime environment metadata for the report."""
    if not include_runtime_environment:
        return {
            "runtime_recorded": False,
            "python": None,
            "platform": None,
        }
    return {
        "runtime_recorded": True,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def build_report(*, include_runtime_environment: bool = False) -> dict[str, Any]:
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
            "steady_state_core_ratio_coarse_over_fine": (
                float(coarse["final_core_ti_kev"] / max(fine["final_core_ti_kev"], 1e-30))
            ),
            "steady_state_dt_consistent": bool(
                STEADY_STATE_CORE_RATIO_BAND[0]
                <= coarse["final_core_ti_kev"] / max(fine["final_core_ti_kev"], 1e-30)
                <= STEADY_STATE_CORE_RATIO_BAND[1]
            ),
            "steady_state_core_ratio_band": list(STEADY_STATE_CORE_RATIO_BAND),
            "limit_cycle_at_coarse_dt": coarse["limit_cycle_detected"],
            "sawtooth_model_present_in_lane": False,
            "disposition": (
                "resolved 2026-07-07: the dt-dependent steady state and the "
                "period-2 crash-rebuild cycle were numerical, not modelled "
                "physics — (a) explicit-Euler impurity diffusion violated its "
                "CFL limit by ~2400x at transport steps and used a "
                "non-conservative axis-amplified divergence, blowing the "
                "impurity profile into the sanitiser ceiling; (b) the model "
                "had no impurity sink, so every trajectory ended in radiative "
                "collapse; (c) the stiff radiation sink was explicit, giving "
                "a dt-dependent period-2 map; (d) identity boundary rows in "
                "the CN tridiagonal left dt-scaled source terms in the "
                "boundary rhs entries that leaked into the interior through "
                "the off-diagonal coupling. Fixed by implicit CN impurity "
                "diffusion with a tau_imp residence-time loss, Patankar "
                "implicit radiation sinks, and folding the physical boundary "
                "conditions into the solve. Equivalence thresholds versus "
                "TORAX remain unset: the transport models still differ by "
                "design (fixed chi versus TORAX's transport model)."
            ),
        },
        "environment": _environment_record(
            include_runtime_environment=include_runtime_environment,
        ),
    }


def _report_digest(report: dict[str, Any]) -> str:
    """Return a deterministic digest for a real-TORAX report payload."""
    encoded = json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def check_report(
    path: Path = REPORT,
    *,
    include_runtime_environment: bool = False,
) -> list[str]:
    """Return drift errors for the tracked real-TORAX parity report."""
    expected = build_report(include_runtime_environment=include_runtime_environment)
    errors: list[str] = []
    if not path.exists():
        errors.append(f"missing TORAX real-parity report: {path}")
        return errors
    observed = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    if _report_digest(observed) != _report_digest(expected):
        errors.append("tracked TORAX real-parity report is stale")
    return errors


def main(argv: list[str] | None = None) -> int:
    """Run the real-TORAX comparison gate and write the tracked report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPORT)
    parser.add_argument("--check", action="store_true", help="Check the tracked report for drift.")
    parser.add_argument(
        "--include-runtime-environment",
        action="store_true",
        help="Embed the current Python and platform strings in the report.",
    )
    args = parser.parse_args(argv)

    if args.check:
        errors = check_report(
            args.output,
            include_runtime_environment=args.include_runtime_environment,
        )
        for error in errors:
            print(f"TORAX REAL PARITY ERROR: {error}", file=sys.stderr)
        if errors:
            return 1
        print(f"TORAX real-parity report is up to date: {args.output}")
        return 0

    report = build_report(include_runtime_environment=args.include_runtime_environment)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["divergence_metrics"], indent=2, sort_keys=True))
    print(json.dumps(report["solver_stability_findings"], indent=2, sort_keys=True))
    print(f"status: {report['status']}")
    return 0 if report["passes_thresholds"] else 1


if __name__ == "__main__":
    sys.exit(main())
