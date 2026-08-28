# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Stellarator SNN Stability-Control Validation
"""Validate reduced stellarator geometry and SNN stability control synthetically."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scpn_fusion.core.equilibrium_3d import FourierMode3D, VMECStyleEquilibrium3D
from scpn_fusion.core.geometry_3d import Reactor3DBuilder
from scpn_fusion.scpn.compiler import FusionCompiler
from scpn_fusion.scpn.contracts import ControlScales, ControlTargets
from scpn_fusion.scpn.controller import NeuroSymbolicController
from scpn_fusion.scpn.structure import StochasticPetriNet


ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA_VERSION = 2
REPORT_KIND = "stellarator_snn_stability_control_validation"
REPORT_PAYLOAD_KEY = "stellarator_snn_stability_control_validation"


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= 1.")
    checked = int(value)
    if checked < 1:
        raise ValueError(f"{name} must be an integer >= 1.")
    return checked


def _nonnegative_finite(value: float, *, name: str) -> float:
    checked = float(value)
    if not np.isfinite(checked):
        raise ValueError(f"{name} must be finite.")
    if checked < 0.0:
        raise ValueError(f"{name} must be >= 0.")
    return checked


def _percentage(value: float, *, name: str) -> float:
    checked = float(value)
    if not np.isfinite(checked):
        raise ValueError(f"{name} must be finite.")
    if checked < 0.0:
        raise ValueError(f"{name} must be in [0, 100].")
    if checked > 100.0:
        raise ValueError(f"{name} must be in [0, 100].")
    return checked


def build_stellarator_snn_controller() -> NeuroSymbolicController:
    """Build the deterministic public controller used by the validation loop."""
    net = StochasticPetriNet()
    net.add_place("x_R_pos", initial_tokens=0.0)
    net.add_place("x_R_neg", initial_tokens=0.0)
    net.add_place("a_R_pos", initial_tokens=0.0)
    net.add_place("a_R_neg", initial_tokens=0.0)
    net.add_transition("T_Rp", threshold=0.1)
    net.add_transition("T_Rn", threshold=0.1)
    net.add_arc("x_R_pos", "T_Rp", weight=1.0)
    net.add_arc("x_R_neg", "T_Rn", weight=1.0)
    net.add_arc("T_Rp", "a_R_pos", weight=1.0)
    net.add_arc("T_Rn", "a_R_neg", weight=1.0)
    net.compile()

    artifact = (
        FusionCompiler(bitstream_length=1024, seed=99)
        .compile(net, firing_mode="binary")
        .export_artifact(
            name="stellarator_snn_stability_controller",
            dt_control_s=0.001,
            readout_config={
                "actions": [{"name": "dI_PF3_A", "pos_place": 2, "neg_place": 3}],
                "gains": [2000.0],
                "abs_max": [4000.0],
                "slew_per_s": [1e6],
            },
            injection_config=[
                {
                    "place_id": 0,
                    "source": "x_R_pos",
                    "scale": 1.0,
                    "offset": 0.0,
                    "clamp_0_1": True,
                },
                {
                    "place_id": 1,
                    "source": "x_R_neg",
                    "scale": 1.0,
                    "offset": 0.0,
                    "clamp_0_1": True,
                },
            ],
        )
    )
    return NeuroSymbolicController(
        artifact=artifact,
        seed_base=271828182,
        targets=ControlTargets(R_target_m=0.06, Z_target_m=0.0),
        scales=ControlScales(R_scale_m=0.08, Z_scale_m=1.0),
    )


def calculate_stability_metric(builder: Reactor3DBuilder, coupling: float, nfp: int) -> float:
    """Return field-line angular-step variability for one reduced configuration."""
    tracer = builder.create_fieldline_tracer(
        rotational_transform=0.44,
        helical_coupling_scale=float(coupling),
        radial_coupling_scale=0.05,
        nfp=nfp,
    )
    trace = tracer.trace_line(
        rho0=0.93,
        theta0=0.05,
        phi0=0.0,
        toroidal_turns=8,
        steps_per_turn=160,
    )
    return float(np.std(np.diff(trace.theta)) * 100.0)


def calculate_synthetic_reference_parity(
    candidate: VMECStyleEquilibrium3D,
    reference: VMECStyleEquilibrium3D,
    *,
    samples: int,
) -> float:
    """Compare two in-repository synthetic equilibria on deterministic samples."""
    rng = np.random.default_rng(7)
    rho = rng.uniform(0.25, 1.0, samples)
    theta = rng.uniform(0.0, 2.0 * np.pi, samples)
    phi = rng.uniform(0.0, 2.0 * np.pi, samples)
    candidate_r, candidate_z, _ = candidate.flux_to_cylindrical(rho, theta, phi)
    reference_r, reference_z, _ = reference.flux_to_cylindrical(rho, theta, phi)
    rmse = float(
        np.sqrt(np.mean((candidate_r - reference_r) ** 2 + (candidate_z - reference_z) ** 2))
    )
    scale = float(
        np.mean(
            np.sqrt((reference_r - reference.r_axis) ** 2 + (reference_z - reference.z_axis) ** 2)
        )
    )
    return float(np.clip(100.0 * (1.0 - rmse / max(scale, 1.0e-9)), 0.0, 100.0))


def run_campaign(
    *,
    iterations: int = 6,
    parity_samples: int = 720,
    max_final_instability_metric: float = 0.025,
    min_improvement_pct: float = 30.0,
    min_synthetic_reference_parity_pct: float = 95.0,
) -> dict[str, Any]:
    """Run reduced stellarator geometry, SNN control and synthetic parity gates."""
    checked_iterations = _positive_int(iterations, name="iterations")
    if checked_iterations < 2:
        raise ValueError("iterations must be >= 2.")
    checked_parity_samples = _positive_int(parity_samples, name="parity_samples")
    checked_final_metric = _nonnegative_finite(
        max_final_instability_metric,
        name="max_final_instability_metric",
    )
    checked_improvement = _percentage(min_improvement_pct, name="min_improvement_pct")
    checked_parity = _percentage(
        min_synthetic_reference_parity_pct,
        name="min_synthetic_reference_parity_pct",
    )

    t0 = time.perf_counter()
    base_equilibrium = VMECStyleEquilibrium3D(
        r_axis=2.0,
        z_axis=0.0,
        a_minor=0.52,
        kappa=1.65,
        triangularity=0.22,
        nfp=1,
    )
    base_builder = Reactor3DBuilder(equilibrium_3d=base_equilibrium, solve_equilibrium=False)
    stellarator_equilibrium = base_builder.build_stellarator_w7x_like_equilibrium(
        nfp=5,
        edge_ripple=0.09,
        vertical_ripple=0.05,
    )
    builder = Reactor3DBuilder(
        equilibrium_3d=stellarator_equilibrium,
        solve_equilibrium=False,
    )
    controller = build_stellarator_snn_controller()
    coupling = 0.18
    history: list[float] = []

    for iteration in range(checked_iterations):
        metric = calculate_stability_metric(builder, coupling, stellarator_equilibrium.nfp)
        history.append(metric)
        action = controller.step({"R_axis_m": metric, "Z_axis_m": 0.0}, iteration)
        correction = float(np.clip(action["dI_PF3_A"] / 5000.0, -0.35, 0.35))
        coupling = float(np.clip(coupling - 0.05 * correction, 0.03, 0.18))

    baseline_metric = history[0]
    final_metric = history[-1]
    improvement_pct = float(100.0 * (baseline_metric - final_metric) / max(baseline_metric, 1.0e-9))
    reference_modes = [
        FourierMode3D(
            m=mode.m,
            n=mode.n,
            r_cos=mode.r_cos * 1.012,
            r_sin=mode.r_sin * 1.012,
            z_cos=mode.z_cos * 1.012,
            z_sin=mode.z_sin * 1.012,
        )
        for mode in stellarator_equilibrium.modes
    ]
    synthetic_reference = VMECStyleEquilibrium3D(
        r_axis=stellarator_equilibrium.r_axis,
        z_axis=stellarator_equilibrium.z_axis,
        a_minor=stellarator_equilibrium.a_minor,
        kappa=stellarator_equilibrium.kappa,
        triangularity=stellarator_equilibrium.triangularity,
        nfp=stellarator_equilibrium.nfp,
        modes=reference_modes,
    )
    parity_pct = calculate_synthetic_reference_parity(
        stellarator_equilibrium,
        synthetic_reference,
        samples=checked_parity_samples,
    )
    passes = all(
        (
            final_metric <= checked_final_metric,
            improvement_pct >= checked_improvement,
            parity_pct >= checked_parity,
        )
    )
    return {
        "iterations": checked_iterations,
        "parity_samples": checked_parity_samples,
        "field_periods": int(stellarator_equilibrium.nfp),
        "baseline_instability_metric": baseline_metric,
        "final_instability_metric": final_metric,
        "improvement_pct": improvement_pct,
        "synthetic_reference_parity_pct": parity_pct,
        "thresholds": {
            "max_final_instability_metric": checked_final_metric,
            "min_improvement_pct": checked_improvement,
            "min_synthetic_reference_parity_pct": checked_parity,
        },
        "passes_thresholds": bool(passes),
        "runtime_seconds": float(time.perf_counter() - t0),
    }


def generate_report(**kwargs: Any) -> dict[str, Any]:
    """Generate the versioned stellarator SNN stability-control report."""
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_kind": REPORT_KIND,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        REPORT_PAYLOAD_KEY: run_campaign(**kwargs),
    }


def validate_report(report: dict[str, Any]) -> dict[str, Any]:
    """Validate the current serialized report contract and return its payload."""
    expected = {"schema_version", "report_kind", "generated_at_utc", REPORT_PAYLOAD_KEY}
    if set(report) != expected:
        raise ValueError("report keys do not match the current descriptive contract")
    if report["schema_version"] != REPORT_SCHEMA_VERSION:
        raise ValueError(f"unsupported report schema_version: {report['schema_version']!r}")
    if report["report_kind"] != REPORT_KIND:
        raise ValueError(f"unsupported report_kind: {report['report_kind']!r}")
    generated_at = report["generated_at_utc"]
    if not isinstance(generated_at, str):
        raise ValueError("generated_at_utc must be a non-empty string")
    if not generated_at:
        raise ValueError("generated_at_utc must be a non-empty string")
    payload = report[REPORT_PAYLOAD_KEY]
    if not isinstance(payload, dict):
        raise ValueError(f"{REPORT_PAYLOAD_KEY} must be an object")
    return payload


def render_markdown(report: dict[str, Any]) -> str:
    """Render stellarator SNN stability-control metrics as Markdown."""
    campaign = validate_report(report)
    thresholds = campaign["thresholds"]
    return "\n".join(
        [
            "# Stellarator SNN Stability-Control Validation",
            "",
            f"- Generated: `{report['generated_at_utc']}`",
            f"- Runtime: `{campaign['runtime_seconds']:.3f} s`",
            f"- Field periods: `{campaign['field_periods']}`",
            "",
            "## Reduced Stability Control",
            "",
            f"- Baseline instability metric: `{campaign['baseline_instability_metric']:.5f}`",
            f"- Final instability metric: `{campaign['final_instability_metric']:.5f}`",
            f"- Maximum final metric: `{thresholds['max_final_instability_metric']:.5f}`",
            f"- Improvement: `{campaign['improvement_pct']:.2f}%`",
            f"- Minimum improvement: `{thresholds['min_improvement_pct']:.2f}%`",
            "",
            "## In-Repository Synthetic-Reference Parity",
            "",
            f"- Parity samples: `{campaign['parity_samples']}`",
            f"- Parity score: `{campaign['synthetic_reference_parity_pct']:.2f}%`",
            f"- Minimum parity: `{thresholds['min_synthetic_reference_parity_pct']:.2f}%`",
            f"- Overall pass: `{'YES' if campaign['passes_thresholds'] else 'NO'}`",
            "",
        ]
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for stellarator SNN validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--parity-samples", type=int, default=720)
    parser.add_argument("--max-final-instability-metric", type=float, default=0.025)
    parser.add_argument("--min-improvement-pct", type=float, default=30.0)
    parser.add_argument("--min-synthetic-reference-parity-pct", type=float, default=95.0)
    parser.add_argument(
        "--output-json",
        default=str(
            ROOT / "validation" / "reports" / "stellarator_snn_stability_control_validation.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        default=str(
            ROOT / "validation" / "reports" / "stellarator_snn_stability_control_validation.md"
        ),
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run validation and write versioned JSON and Markdown reports."""
    args = parse_args(argv)
    report = generate_report(
        iterations=args.iterations,
        parity_samples=args.parity_samples,
        max_final_instability_metric=args.max_final_instability_metric,
        min_improvement_pct=args.min_improvement_pct,
        min_synthetic_reference_parity_pct=args.min_synthetic_reference_parity_pct,
    )
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output_md.write_text(render_markdown(report), encoding="utf-8")
    campaign = validate_report(report)
    print("Stellarator SNN stability-control validation complete.")
    print(
        f"final_metric={campaign['final_instability_metric']:.5f}, "
        f"improvement_pct={campaign['improvement_pct']:.2f}, "
        f"synthetic_reference_parity_pct={campaign['synthetic_reference_parity_pct']:.2f}, "
        f"passes_thresholds={campaign['passes_thresholds']}"
    )
    if args.strict and not campaign["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
