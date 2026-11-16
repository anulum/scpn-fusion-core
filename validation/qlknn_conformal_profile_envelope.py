#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — frozen-coefficient profile envelope from the QLKNN conformal intervals
"""Carry the QLKNN-10D conformal intervals into a steady transport profile.

The integrated transport solver is run to steady state on a declared L-mode case with the
neural transport closure, using the certified weights. Its total diffusivities are
``max(chi_base + chi_turb, 1e-6)`` and ``D_n = max(d_turb, 0.1 chi_base)``, with ``chi_base``
the Chang-Hinton neoclassical value; below 30 MW of auxiliary power no pedestal modifier
acts, so the turbulent part is recovered as ``chi - chi_base``. That turbulent part is then
moved to the lower and upper end of a conformal interval and frozen, and the solver is run
again to steady state with those fixed coefficients, from the neural steady state.

Two interval variants are carried, both at the certificate's level and both calibrated on
its calibration split: the certificate's constant radius ``r``, and the ``|y_hat| + r``
normalised score, whose half-width is ``q (|y_hat| + r)``. Lower bounds are clipped at
zero turbulent diffusivity, as the certificate's interval is.

This is a frozen-coefficient sensitivity, not a self-consistent propagation: the
diffusivities do not respond to the profiles they produce. The report gates that the
neural run used the certified weights and reached steady state, and that a frozen run at
the nominal coefficients reproduces the neural steady state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scpn_fusion.core.integrated_transport_solver import TransportSolver
from tools import generate_surrogate_conformal_certificates as certificates
from validation import qlknn_conformal_regime_coverage as regimes
from validation import qlknn_conformal_variants as variants

FloatArray = NDArray[np.float64]
REPORT_PATH = ROOT / "validation/reports/qlknn_conformal_profile_envelope.json"
MARKDOWN_PATH = ROOT / "validation/reports/qlknn_conformal_profile_envelope.md"
CONFIG_PATH = ROOT / "validation/iter_config.json"
SCHEMA = "scpn-fusion-core.qlknn-conformal-profile-envelope.v1"
NEOCLASSICAL = {
    "R0": 6.2,
    "a": 2.0,
    "B0": 5.3,
    "A_ion": 2.5,
    "Z_eff": 1.7,
    "q0": 1.0,
    "q_edge": 3.0,
}
P_AUX_MW = 20.0
PEDESTAL_THRESHOLD_MW = 30.0
STEPS = 1200
CHECK_STEPS = 200
DT_S = 0.01
STEADY_TOLERANCE = 1.0e-3
CHANNELS = ("chi_e", "chi_i", "D_e")
MARKDOWN_HEADER = (
    "<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->",
    "<!-- Commercial license available -->",
    "<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->",
    "<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->",
    "<!-- ORCID: 0009-0009-3560-0851 -->",
    "<!-- Contact: www.anulum.li | protoscience@anulum.li -->",
    "<!-- SCPN Fusion Core — profile envelope from the QLKNN conformal intervals -->",
)


def bounded_coefficients(
    chi_base: FloatArray,
    turbulent: dict[str, FloatArray],
    half_width: dict[str, FloatArray],
    sign: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return ``(chi_e, chi_i, D_n)`` with each turbulent part moved by ``sign`` half-widths.

    The turbulent parts are clipped at zero and recombined as the neural closure does.
    """
    moved = {name: np.maximum(turbulent[name] + sign * half_width[name], 0.0) for name in CHANNELS}
    return (
        np.maximum(chi_base + moved["chi_e"], 1e-6),
        np.maximum(chi_base + moved["chi_i"], 1e-6),
        np.maximum(moved["D_e"], 0.1 * chi_base),
    )


def _solver(config: Path, weights: Path) -> TransportSolver:
    solver = TransportSolver(str(config), multi_ion=False)
    solver.set_neoclassical(**NEOCLASSICAL)
    solver.neural_transport_weights_path = str(weights)
    return solver


def _summary(result: dict[str, Any]) -> dict[str, float]:
    return {
        "T_core_keV": float(result["T_core"]),
        "T_avg_keV": float(result["T_avg"]),
        "tau_e_s": float(result["tau_e"]),
    }


def build_report(
    certificate_path: Path = regimes.CERTIFICATE_PATH,
    config_path: Path = CONFIG_PATH,
    p_aux_mw: float = P_AUX_MW,
) -> dict[str, Any]:
    """Run the neural steady state and the frozen runs at each interval end."""
    if not 0.0 < p_aux_mw <= PEDESTAL_THRESHOLD_MW:
        raise ValueError(
            f"P_aux must be in (0, {PEDESTAL_THRESHOLD_MW}] MW, where no pedestal modifier acts"
        )
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    entry = certificate["certificates"][0]
    alpha = float(certificate["method"]["alpha"])
    weights = ROOT / entry["model"]["weights"]
    calibration = ROOT / entry["calibration"]["split"]
    holdout = ROOT / entry["holdout"]["split"]
    if hashlib.sha256(weights.read_bytes()).hexdigest() != entry["model"]["weights_file_sha256"]:
        raise ValueError("weights differ from the certificate's digest")
    data = certificates.residual_scores(
        weights_path=weights, calibration_path=calibration, holdout_path=holdout
    )
    radius = {channel["name"]: float(channel["radius"]) for channel in entry["channels"]}
    normalised_q = {
        name: variants.rank_quantile(
            data.calibration_scores[:, index]
            / (np.abs(data.calibration_predictions[:, index]) + radius[name]),
            alpha,
        )
        for index, name in enumerate(CHANNELS)
    }

    neural = _solver(config_path, weights)
    neural.transport_backend = "neural_transport"
    neural.run_to_steady_state(P_aux=p_aux_mw, n_steps=STEPS, dt=DT_S)
    before = np.array(neural.Ti, dtype=np.float64)
    reference = neural.run_to_steady_state(P_aux=p_aux_mw, n_steps=CHECK_STEPS, dt=DT_S)
    steady_change = float(np.max(np.abs(neural.Ti - before)) / np.max(neural.Ti))
    state = {name: np.array(getattr(neural, name), dtype=np.float64) for name in ("Ti", "Te", "ne")}
    chi_base = np.asarray(neural.chang_hinton_chi_profile(), dtype=np.float64)
    turbulent = {
        "chi_e": np.asarray(neural.chi_e, dtype=np.float64) - chi_base,
        "chi_i": np.asarray(neural.chi_i, dtype=np.float64) - chi_base,
        "D_e": np.asarray(neural.D_n, dtype=np.float64),
    }
    widths = {
        "certificate": {name: np.full_like(chi_base, radius[name]) for name in CHANNELS},
        "normalised_prediction": {
            name: normalised_q[name] * (np.abs(turbulent[name]) + radius[name]) for name in CHANNELS
        },
    }

    def frozen(coefficients: tuple[FloatArray, FloatArray, FloatArray]) -> dict[str, float]:
        solver = _solver(config_path, weights)
        for name, values in state.items():
            setattr(solver, name, values.copy())
        solver.transport_backend = "fixed_coefficients"
        solver.chi_e, solver.chi_i, solver.D_n = coefficients
        return _summary(solver.run_to_steady_state(P_aux=p_aux_mw, n_steps=STEPS, dt=DT_S))

    nominal = frozen(bounded_coefficients(chi_base, turbulent, widths["certificate"], 0))
    envelopes = {
        variant: {
            "lower_turbulence": frozen(bounded_coefficients(chi_base, turbulent, width, -1)),
            "upper_turbulence": frozen(bounded_coefficients(chi_base, turbulent, width, 1)),
            "median_half_width_m2_s": {name: float(np.median(width[name])) for name in CHANNELS},
        }
        for variant, width in widths.items()
    }
    neural_summary = _summary(reference)
    reproduced = abs(nominal["T_core_keV"] / neural_summary["T_core_keV"] - 1.0)
    checks = {
        "neural_run_reached_steady_state": steady_change < STEADY_TOLERANCE,
        "nominal_frozen_run_reproduces_neural_steady_state": reproduced < STEADY_TOLERANCE,
    }
    return {
        "schema": SCHEMA,
        "alpha": alpha,
        "case": {
            "config": config_path.relative_to(ROOT).as_posix()
            if config_path.is_relative_to(ROOT)
            else str(config_path),
            "p_aux_mw": p_aux_mw,
            "neoclassical": NEOCLASSICAL,
            "steps": STEPS,
            "dt_s": DT_S,
        },
        "weights_sha256": entry["model"]["weights_file_sha256"],
        "certificate_radius_m2_s": radius,
        "normalised_prediction_quantile": normalised_q,
        "neural_steady_state": neural_summary,
        "steady_state_relative_change": steady_change,
        "nominal_frozen": nominal,
        "nominal_reproduction_relative_error": reproduced,
        "median_turbulent_diffusivity_m2_s": {
            name: float(np.median(turbulent[name])) for name in CHANNELS
        },
        "envelopes": envelopes,
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
        "source_revision": variants.git_revision(),
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render the nominal state and each variant's envelope."""
    nominal = report["neural_steady_state"]
    lines = [
        *MARKDOWN_HEADER,
        "# Profile envelope from the QLKNN-10D conformal intervals (frozen coefficients)",
        "",
        f"Status: **{report['status'].upper()}**. Level {1 - report['alpha']:.2f}; "
        f"P_aux {report['case']['p_aux_mw']} MW on `{report['case']['config']}`. A "
        "frozen-coefficient sensitivity, not a self-consistent propagation.",
        "",
        f"Neural steady state: T_core {nominal['T_core_keV']:.3f} keV, T_avg "
        f"{nominal['T_avg_keV']:.3f} keV, tau_E {nominal['tau_e_s']:.3f} s.",
        "",
        "| Interval | Turbulence | T_core (keV) | T_avg (keV) | tau_E (s) |",
        "|---|---|---:|---:|---:|",
    ]
    for variant, envelope in report["envelopes"].items():
        for end in ("lower_turbulence", "upper_turbulence"):
            row = envelope[end]
            lines.append(
                f"| {variant} | {end.split('_')[0]} | {row['T_core_keV']:.3f} "
                f"| {row['T_avg_keV']:.3f} | {row['tau_e_s']:.3f} |"
            )
    lines += ["", f"Source revision: `{report['source_revision']}`", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Write the JSON and Markdown reports; exit 1 when a gate fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certificate", type=Path, default=regimes.CERTIFICATE_PATH)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--p-aux", type=float, default=P_AUX_MW)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    parser.add_argument("--markdown", type=Path, default=MARKDOWN_PATH)
    args = parser.parse_args(argv)
    report = build_report(args.certificate.resolve(), args.config.resolve(), args.p_aux)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
