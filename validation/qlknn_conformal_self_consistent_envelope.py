#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — self-consistent profile envelope from the QLKNN conformal intervals
"""Carry the QLKNN-10D conformal intervals self-consistently into a steady transport profile.

:mod:`validation.qlknn_conformal_profile_envelope` froze the turbulent diffusivity at each
end of a conformal interval, so the diffusivity could not respond to the profiles it
produced. Here it does. On the same declared L-mode case, with the same certified weights
and the same bound (:func:`~validation.qlknn_conformal_profile_envelope.bounded_coefficients`),
every transport step:

1. evaluates the neural closure on the current profiles, through the solver's public
   ``update_transport_model`` with the neural backend;
2. recovers the turbulent part as ``chi - chi_base`` (``chi_base`` from the public
   Chang-Hinton profile) and ``D_n``, moves it by the interval half-width and clips it at
   zero;
3. moves the applied coefficients a fraction ``RELAXATION`` of the way to that target; and
4. advances one step with the ``fixed_coefficients`` backend.

A run stops when the relative difference between the target and the applied coefficients
falls below ``RESIDUAL_TOLERANCE``. At that point the profiles are steady and the applied
diffusivity equals the bounded closure of the profiles it produced: a self-consistent steady
state. The relaxation sets only the path to that state. It is needed because the closure is
stiff: applied without relaxation, the lagged closure oscillates from step to step.

The solver is not modified and no private member is used. The report gates that the nominal
run (zero half-width) reproduces the neural steady state, that the nominal fixed point does
not depend on the relaxation, that every run converged, and that the interval widths equal
those of the frozen study's tracked report.

Usage: python validation/qlknn_conformal_self_consistent_envelope.py [--output report.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scpn_fusion.core.integrated_transport_solver import TransportSolver
from tools import generate_surrogate_conformal_certificates as certificates
from validation import qlknn_conformal_profile_envelope as frozen
from validation import qlknn_conformal_regime_coverage as regimes
from validation import qlknn_conformal_variants as variants

FloatArray = NDArray[np.float64]
REPORT_PATH = ROOT / "validation/reports/qlknn_conformal_self_consistent_envelope.json"
MARKDOWN_PATH = ROOT / "validation/reports/qlknn_conformal_self_consistent_envelope.md"
SCHEMA = "scpn-fusion-core.qlknn-conformal-self-consistent-envelope.v1"
RELAXATION = 0.005
CHECK_RELAXATION = 0.02
MAX_STEPS = 40_000
RESIDUAL_TOLERANCE = 1.0e-10
RELAXATION_INDEPENDENCE_TOLERANCE = 1.0e-8
WIDTH_TOLERANCE = 1.0e-12
VARIANTS = ("certificate", "normalised_prediction")
SOURCE_PATHS = (
    "validation/qlknn_conformal_self_consistent_envelope.py",
    "validation/qlknn_conformal_profile_envelope.py",
    "src/scpn_fusion/core/integrated_transport_solver_model.py",
    "src/scpn_fusion/core/_integrated_transport_solver_model_backend.py",
    "src/scpn_fusion/core/integrated_transport_solver_coupling.py",
    "src/scpn_fusion/core/integrated_transport_solver_runtime.py",
    "src/scpn_fusion/core/neural_transport.py",
)
MARKDOWN_HEADER = (
    "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
    "<!-- Commercial license available -->",
    "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
    "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
    "<!-- ORCID: 0009-0009-3560-0851 -->",
    "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
    "<!-- SCPN Fusion Core — self-consistent profile envelope from the QLKNN conformal "
    "intervals -->",
)


@dataclass(frozen=True)
class Intervals:
    """The certified weights and both interval variants' widths at the certificate's level."""

    alpha: float
    weights: Path
    weights_sha256: str
    radius: dict[str, float]
    normalised_quantile: dict[str, float]


def load_intervals(certificate_path: Path) -> Intervals:
    """Return the certificate's radii and the normalised score's quantiles.

    Raises
    ------
    ValueError
        If the weights file differs from the certificate's digest.
    """
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    entry = certificate["certificates"][0]
    alpha = float(certificate["method"]["alpha"])
    weights = ROOT / entry["model"]["weights"]
    digest = entry["model"]["weights_file_sha256"]
    if hashlib.sha256(weights.read_bytes()).hexdigest() != digest:
        raise ValueError("weights differ from the certificate's digest")
    data = certificates.residual_scores(
        weights_path=weights,
        calibration_path=ROOT / entry["calibration"]["split"],
        holdout_path=ROOT / entry["holdout"]["split"],
    )
    radius = {channel["name"]: float(channel["radius"]) for channel in entry["channels"]}
    quantile = {
        name: variants.rank_quantile(
            data.calibration_scores[:, index]
            / (np.abs(data.calibration_predictions[:, index]) + radius[name]),
            alpha,
        )
        for index, name in enumerate(frozen.CHANNELS)
    }
    return Intervals(alpha, weights, digest, radius, quantile)


def half_widths(
    variant: str, turbulent: dict[str, FloatArray], intervals: Intervals
) -> dict[str, FloatArray]:
    """Return each channel's half-width for ``variant`` at the given turbulent diffusivity."""
    if variant == "certificate":
        return {
            name: np.full_like(turbulent[name], intervals.radius[name]) for name in frozen.CHANNELS
        }
    if variant == "normalised_prediction":
        return {
            name: intervals.normalised_quantile[name]
            * (np.abs(turbulent[name]) + intervals.radius[name])
            for name in frozen.CHANNELS
        }
    raise ValueError(f"unknown interval variant {variant!r}")


def make_solver(config_path: Path, weights: Path) -> TransportSolver:
    """Return the declared case's solver with the certified neural weights."""
    solver = TransportSolver(str(config_path), multi_ion=False)
    solver.set_neoclassical(**frozen.NEOCLASSICAL)
    solver.neural_transport_weights_path = str(weights)
    return solver


def closure_target(
    solver: TransportSolver, p_aux_mw: float, variant: str, sign: int, intervals: Intervals
) -> tuple[FloatArray, dict[str, FloatArray]]:
    """Evaluate the bounded neural closure on the solver's current profiles.

    Returns the stacked ``(chi_e, chi_i, D_n)`` target and the half-widths used.
    """
    solver.transport_backend = "neural_transport"
    solver.update_transport_model(p_aux_mw)
    chi_base = np.asarray(solver.chang_hinton_chi_profile(), dtype=np.float64)
    turbulent = {
        "chi_e": np.asarray(solver.chi_e, dtype=np.float64) - chi_base,
        "chi_i": np.asarray(solver.chi_i, dtype=np.float64) - chi_base,
        "D_e": np.asarray(solver.D_n, dtype=np.float64),
    }
    widths = half_widths(variant, turbulent, intervals)
    target = np.stack(frozen.bounded_coefficients(chi_base, turbulent, widths, sign))
    return target, widths


def relative_residual(target: FloatArray, applied: FloatArray) -> float:
    """Return ``|target - applied| / |applied|`` in the Euclidean norm."""
    return float(np.linalg.norm(target - applied) / np.linalg.norm(applied))


def relax(applied: FloatArray, target: FloatArray, factor: float) -> FloatArray:
    """Move ``applied`` a fraction ``factor`` of the way to ``target``."""
    return (1.0 - factor) * applied + factor * target


def summary(result: dict[str, Any]) -> dict[str, float]:
    """Return the core and mean ion temperature and the confinement time of a run result."""
    return {
        "T_core_keV": float(result["T_core"]),
        "T_avg_keV": float(result["T_avg"]),
        "tau_e_s": float(result["tau_e"]),
    }


def self_consistent_run(
    config_path: Path,
    p_aux_mw: float,
    intervals: Intervals,
    variant: str,
    sign: int,
    relaxation: float = RELAXATION,
) -> dict[str, Any]:
    """Run the bounded closure to its self-consistent steady state, or to ``MAX_STEPS``."""
    solver = make_solver(config_path, intervals.weights)
    applied: FloatArray | None = None
    residual = math.inf
    result: dict[str, Any] = {}
    widths: dict[str, FloatArray] = {}
    steps = 0
    for steps in range(1, MAX_STEPS + 1):
        target, widths = closure_target(solver, p_aux_mw, variant, sign, intervals)
        if applied is None:
            applied = target
        else:
            residual = relative_residual(target, applied)
            applied = relax(applied, target, relaxation)
        solver.transport_backend = "fixed_coefficients"
        solver.chi_e, solver.chi_i, solver.D_n = (row.copy() for row in applied)
        result = solver.run_to_steady_state(P_aux=p_aux_mw, n_steps=1, dt=frozen.DT_S)
        if residual < RESIDUAL_TOLERANCE:
            break
    return {
        **summary(result),
        "steps": steps,
        "final_relative_residual": residual,
        "converged": residual < RESIDUAL_TOLERANCE,
        "median_half_width_m2_s": {
            name: float(np.median(widths[name])) for name in frozen.CHANNELS
        },
    }


def neural_reference(
    config_path: Path, weights: Path, p_aux_mw: float
) -> tuple[dict[str, float], float]:
    """Run the neural closure as the frozen study does; return its state and last change."""
    solver = make_solver(config_path, weights)
    solver.transport_backend = "neural_transport"
    solver.run_to_steady_state(P_aux=p_aux_mw, n_steps=frozen.STEPS, dt=frozen.DT_S)
    before = np.array(solver.Ti, dtype=np.float64)
    result = solver.run_to_steady_state(P_aux=p_aux_mw, n_steps=frozen.CHECK_STEPS, dt=frozen.DT_S)
    change = float(np.max(np.abs(solver.Ti - before)) / np.max(solver.Ti))
    return summary(result), change


def widths_match(intervals: Intervals, frozen_report: dict[str, Any]) -> bool:
    """Return whether the widths equal those recorded by the frozen study's report."""
    pairs = [
        (intervals.radius, frozen_report["certificate_radius_m2_s"]),
        (intervals.normalised_quantile, frozen_report["normalised_prediction_quantile"]),
    ]
    return intervals.weights_sha256 == frozen_report["weights_sha256"] and all(
        math.isclose(ours[name], theirs[name], rel_tol=WIDTH_TOLERANCE)
        for ours, theirs in pairs
        for name in frozen.CHANNELS
    )


def _relative_difference(first: dict[str, float], second: dict[str, float]) -> float:
    return max(abs(first[key] / second[key] - 1.0) for key in first)


def build_report(
    certificate_path: Path = regimes.CERTIFICATE_PATH,
    config_path: Path = frozen.CONFIG_PATH,
    p_aux_mw: float = frozen.P_AUX_MW,
    frozen_report_path: Path = frozen.REPORT_PATH,
) -> dict[str, Any]:
    """Run the neural reference, the nominal run at two relaxations, and every interval end."""
    if not 0.0 < p_aux_mw <= frozen.PEDESTAL_THRESHOLD_MW:
        raise ValueError(
            f"P_aux must be in (0, {frozen.PEDESTAL_THRESHOLD_MW}] MW, "
            "where no pedestal modifier acts"
        )
    intervals = load_intervals(certificate_path)
    frozen_bytes = frozen_report_path.read_bytes()
    frozen_report = json.loads(frozen_bytes)
    neural, neural_change = neural_reference(config_path, intervals.weights, p_aux_mw)
    nominal = self_consistent_run(config_path, p_aux_mw, intervals, VARIANTS[0], 0)
    check = self_consistent_run(
        config_path, p_aux_mw, intervals, VARIANTS[0], 0, relaxation=CHECK_RELAXATION
    )
    envelopes = {
        variant: {
            "lower_turbulence": self_consistent_run(config_path, p_aux_mw, intervals, variant, -1),
            "upper_turbulence": self_consistent_run(config_path, p_aux_mw, intervals, variant, 1),
        }
        for variant in VARIANTS
    }
    keys = ("T_core_keV", "T_avg_keV", "tau_e_s")
    reproduction = abs(nominal["T_core_keV"] / neural["T_core_keV"] - 1.0)
    independence = _relative_difference(
        {key: nominal[key] for key in keys}, {key: check[key] for key in keys}
    )
    runs = [nominal, check, *(run for ends in envelopes.values() for run in ends.values())]
    checks = {
        "widths_match_frozen_report": widths_match(intervals, frozen_report),
        "neural_run_reached_steady_state": neural_change < frozen.STEADY_TOLERANCE,
        "nominal_reproduces_neural_steady_state": reproduction < frozen.STEADY_TOLERANCE,
        "nominal_fixed_point_independent_of_relaxation": (
            independence < RELAXATION_INDEPENDENCE_TOLERANCE
        ),
        "every_run_converged": all(run["converged"] for run in runs),
    }
    return {
        "schema": SCHEMA,
        "alpha": intervals.alpha,
        "case": {
            "config": config_path.relative_to(ROOT).as_posix()
            if config_path.is_relative_to(ROOT)
            else str(config_path),
            "p_aux_mw": p_aux_mw,
            "neoclassical": frozen.NEOCLASSICAL,
            "dt_s": frozen.DT_S,
        },
        "iteration": {
            "relaxation": RELAXATION,
            "check_relaxation": CHECK_RELAXATION,
            "max_steps": MAX_STEPS,
            "residual_tolerance": RESIDUAL_TOLERANCE,
        },
        "weights_sha256": intervals.weights_sha256,
        "certificate_radius_m2_s": intervals.radius,
        "normalised_prediction_quantile": intervals.normalised_quantile,
        "neural_steady_state": neural,
        "neural_steady_state_relative_change": neural_change,
        "nominal": nominal,
        "nominal_at_check_relaxation": check,
        "nominal_reproduction_relative_error": reproduction,
        "relaxation_independence_relative_difference": independence,
        "envelopes": envelopes,
        "frozen_counterpart": {
            "report": frozen_report_path.relative_to(ROOT).as_posix()
            if frozen_report_path.is_relative_to(ROOT)
            else str(frozen_report_path),
            "report_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
            "envelopes": frozen_report["envelopes"],
        },
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
        "source_revision": variants.git_revision(),
        "source_sha256": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in SOURCE_PATHS
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render the nominal state and each interval end beside its frozen counterpart."""
    neural, nominal = report["neural_steady_state"], report["nominal"]
    lines = [
        *MARKDOWN_HEADER,
        "# Self-consistent profile envelope from the QLKNN-10D conformal intervals",
        "",
        f"Status: **{report['status'].upper()}**. Level {1 - report['alpha']:.2f}; "
        f"P_aux {report['case']['p_aux_mw']} MW on `{report['case']['config']}`; relaxation "
        f"{report['iteration']['relaxation']}, residual tolerance "
        f"{report['iteration']['residual_tolerance']:.0e}.",
        "",
        f"Neural steady state: T_core {neural['T_core_keV']:.4f} keV. Nominal fixed point: "
        f"T_core {nominal['T_core_keV']:.4f} keV after {nominal['steps']} steps.",
        "",
        "| Interval | Turbulence | Steps | T_core (keV) | T_avg (keV) | tau_E (s) "
        "| Frozen T_core (keV) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for variant, envelope in report["envelopes"].items():
        for end in ("lower_turbulence", "upper_turbulence"):
            row = envelope[end]
            counterpart = report["frozen_counterpart"]["envelopes"][variant][end]
            lines.append(
                f"| {variant} | {end.split('_')[0]} | {row['steps']} "
                f"| {row['T_core_keV']:.4f} | {row['T_avg_keV']:.4f} | {row['tau_e_s']:.4f} "
                f"| {counterpart['T_core_keV']:.4f} |"
            )
    failed = [name for name, ok in report["checks"].items() if not ok]
    lines += [
        "",
        "Failed checks: " + (", ".join(failed) if failed else "none") + ".",
        "",
        f"Source revision: `{report['source_revision']}`",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Write the JSON and Markdown reports; exit 1 when a gate fails."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--certificate", type=Path, default=regimes.CERTIFICATE_PATH)
    parser.add_argument("--config", type=Path, default=frozen.CONFIG_PATH)
    parser.add_argument("--p-aux", type=float, default=frozen.P_AUX_MW)
    parser.add_argument("--frozen-report", type=Path, default=frozen.REPORT_PATH)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    parser.add_argument("--markdown", type=Path, default=MARKDOWN_PATH)
    args = parser.parse_args(argv)
    report = build_report(
        args.certificate.resolve(), args.config.resolve(), args.p_aux, args.frozen_report.resolve()
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
