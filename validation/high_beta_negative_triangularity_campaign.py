#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — High-Beta Negative-Triangularity Campaign
"""Integrated reduced-order high-beta negative-triangularity scenario campaign."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scpn_fusion.core.divertor_thermal_sim import DivertorLab  # noqa: E402
from scpn_fusion.core.elm_model import ELMCrashModel, PeelingBallooningBoundary  # noqa: E402
from validation import task13_free_boundary_disruption_policy_recovery as task13  # noqa: E402
from validation import vertical_control_replay_benchmark as vertical_replay  # noqa: E402

SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class NegativeTriangularityScenario:
    """Reduced-order compact spherical-tokamak-like scenario contract."""

    scenario_id: str = "compact_negative_triangularity_high_beta_reduced_order_v1"
    major_radius_m: float = 1.8
    minor_radius_m: float = 1.2
    elongation_kappa: float = 2.2
    triangularity_delta: float = -0.3
    target_beta_fraction: float = 0.40
    plasma_current_MA: float = 8.0
    toroidal_field_T: float = 6.0
    q95: float = 5.5
    edge_alpha: float = 0.8
    edge_current_A_m2: float = 1.5e5
    edge_shear: float = 2.3
    p_sol_MW: float = 50.0
    divertor_expansion_factor: float = 45.0
    liquid_metal_flow_m_s: float = 8.0
    vertical_growth_rate_s_inv: float = 28.0
    vertical_actuator_gain_m_s2: float = 150.0
    vertical_damping_s_inv: float = 9.5


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _require_finite(name: str, value: float, *, minimum: float | None = None) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite.")
    if minimum is not None and out < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return out


def _validate_scenario(scenario: NegativeTriangularityScenario) -> None:
    if not scenario.scenario_id.strip():
        raise ValueError("scenario_id is required.")
    _require_finite("major_radius_m", scenario.major_radius_m, minimum=1.0e-12)
    _require_finite("minor_radius_m", scenario.minor_radius_m, minimum=1.0e-12)
    _require_finite("elongation_kappa", scenario.elongation_kappa, minimum=1.0)
    _require_finite("target_beta_fraction", scenario.target_beta_fraction, minimum=1.0e-12)
    _require_finite("plasma_current_MA", scenario.plasma_current_MA, minimum=1.0e-12)
    _require_finite("toroidal_field_T", scenario.toroidal_field_T, minimum=1.0e-12)
    _require_finite("q95", scenario.q95, minimum=1.0e-12)
    _require_finite("edge_shear", scenario.edge_shear, minimum=1.0e-12)
    _require_finite("p_sol_MW", scenario.p_sol_MW, minimum=1.0e-12)
    _require_finite(
        "divertor_expansion_factor", scenario.divertor_expansion_factor, minimum=1.0e-12
    )
    _require_finite("liquid_metal_flow_m_s", scenario.liquid_metal_flow_m_s, minimum=1.0e-12)
    if scenario.triangularity_delta >= 0.0:
        raise ValueError("triangularity_delta must be negative for this campaign.")


def _geometry_summary(scenario: NegativeTriangularityScenario) -> dict[str, Any]:
    aspect_ratio = scenario.major_radius_m / scenario.minor_radius_m
    passes = bool(
        1.15 <= aspect_ratio <= 1.65
        and scenario.elongation_kappa > 2.0
        and scenario.triangularity_delta < 0.0
        and 0.30 <= scenario.target_beta_fraction <= 0.45
    )
    return {
        "aspect_ratio": float(aspect_ratio),
        "major_radius_m": float(scenario.major_radius_m),
        "minor_radius_m": float(scenario.minor_radius_m),
        "elongation_kappa": float(scenario.elongation_kappa),
        "triangularity_delta": float(scenario.triangularity_delta),
        "target_beta_fraction": float(scenario.target_beta_fraction),
        "plasma_current_MA": float(scenario.plasma_current_MA),
        "toroidal_field_T": float(scenario.toroidal_field_T),
        "passes_geometry_contract": passes,
        "claim_boundary": (
            "Reduced-order scenario target only; this is not hardware evidence of beta-40 "
            "operation or external same-case parity."
        ),
    }


def _edge_elm_summary(scenario: NegativeTriangularityScenario) -> dict[str, Any]:
    boundary = PeelingBallooningBoundary(
        q95=scenario.q95,
        kappa=scenario.elongation_kappa,
        delta=scenario.triangularity_delta,
        a=scenario.minor_radius_m,
        R0=scenario.major_radius_m,
    )
    margin = boundary.stability_margin(
        alpha_edge=scenario.edge_alpha,
        j_edge=scenario.edge_current_A_m2,
        s_edge=scenario.edge_shear,
    )
    unstable = boundary.is_unstable(
        alpha_edge=scenario.edge_alpha,
        j_edge=scenario.edge_current_A_m2,
        s_edge=scenario.edge_shear,
    )
    crash = ELMCrashModel(f_elm_fraction=0.04).crash(
        T_ped=4.5,
        n_ped=4.0,
        W_ped=12.0,
        A_wet=8.0,
    )
    return {
        "q95": float(scenario.q95),
        "edge_alpha": float(scenario.edge_alpha),
        "edge_current_A_m2": float(scenario.edge_current_A_m2),
        "edge_shear": float(scenario.edge_shear),
        "peeling_ballooning_margin": float(margin),
        "peeling_ballooning_unstable": bool(unstable),
        "bounded_crash_stress_case": {
            "pedestal_energy_loss_MJ": float(crash.delta_W_MJ),
            "duration_ms": float(crash.duration_ms),
            "peak_heat_flux_MW_m2": float(crash.peak_heat_flux_MW_m2),
        },
        "passes_edge_contract": bool(margin > 0.05 and not unstable),
        "claim_boundary": (
            "The edge lane is a reduced peeling-ballooning stability screen plus a bounded "
            "ELM crash stress case, not an experimental ELM-free demonstration."
        ),
    }


def _divertor_summary(scenario: NegativeTriangularityScenario) -> dict[str, Any]:
    with redirect_stdout(io.StringIO()):
        lab = DivertorLab(
            P_sol_MW=scenario.p_sol_MW,
            R_major=scenario.major_radius_m,
            B_pol=2.5,
        )
        liquid = lab.simulate_temhd_liquid_metal(
            flow_velocity_m_s=scenario.liquid_metal_flow_m_s,
            expansion_factor=scenario.divertor_expansion_factor,
        )
        tungsten_temp_c, tungsten_status = lab.simulate_tungsten()

    return {
        "p_sol_MW": float(scenario.p_sol_MW),
        "expansion_factor": float(scenario.divertor_expansion_factor),
        "liquid_metal": liquid,
        "tungsten_reference": {
            "surface_temperature_c": float(tungsten_temp_c),
            "status": str(tungsten_status),
        },
        "passes_divertor_contract": bool(
            liquid["is_stable"]
            and float(liquid["stability_index"]) <= 1.0
            and float(liquid["surface_heat_flux_w_m2"]) <= 45.0e6
        ),
        "claim_boundary": (
            "Reduced TEMHD divertor response only; no material lifetime, erosion, or "
            "hardware first-wall qualification is implied."
        ),
    }


def _vertical_summary(scenario: NegativeTriangularityScenario) -> dict[str, Any]:
    profile = vertical_replay.MachineProfile(
        profile_id="negative_triangularity_compact",
        provenance="internal reduced-order compact negative-triangularity scenario fixture",
        plasma_current_MA=scenario.plasma_current_MA,
        elongation=scenario.elongation_kappa,
        vertical_growth_rate_s_inv=scenario.vertical_growth_rate_s_inv,
        actuator_gain_m_per_s2=scenario.vertical_actuator_gain_m_s2,
        damping_s_inv=scenario.vertical_damping_s_inv,
        wall_time_s=0.015,
    )
    report = vertical_replay.run_benchmark(machine_profile=profile)[
        "vertical_control_replay_benchmark"
    ]
    return {
        "profile_id": profile.profile_id,
        "passes_thresholds": bool(report["passes_thresholds"]),
        "release_gate_status": str(report["release_gate"]["status"]),
        "max_p95_abs_z_m": float(report["uncertainty_report"]["max_p95_abs_z_m"]),
        "trace_integrity": report["trace_integrity"],
        "plant_contract": {
            "contract_id": report["plant_contract"]["contract_id"],
            "source_module": report["plant_contract"]["source_module"],
            "deterministic_state_trajectory_pass": bool(
                report["plant_contract"]["deterministic_state_trajectory_pass"]
            ),
        },
        "claim_boundary": (
            "Reduced-order vertical replay evidence only; release gate remains bounded by "
            "the underlying replay contract and is not a full PCS hardware result."
        ),
    }


def _free_boundary_disruption_summary() -> dict[str, Any]:
    report = task13.run_campaign(seed=42, shot_length=104, control_dt_s=0.05)[
        "task13_free_boundary_disruption_policy_recovery"
    ]
    closed_loop = report["policy_closed_loop"]
    return {
        "passes_thresholds": bool(report["passes_thresholds"]),
        "peak_alert_level": int(closed_loop["peak_alert_level"]),
        "final_alert_level": int(closed_loop["final_alert_level"]),
        "max_disruption_risk": float(closed_loop["max_disruption_risk"]),
        "late_mean_disruption_risk": float(report["recovery_window"]["late_mean_disruption_risk"]),
        "p95_axis_error_m": float(closed_loop["p95_axis_error_m"]),
        "p95_xpoint_error_m": float(closed_loop["p95_xpoint_error_m"]),
        "failure_reasons": list(report["failure_reasons"]),
        "claim_boundary": (
            "Free-boundary disruption-policy recovery uses the deterministic Task 13 "
            "acceptance plant, not external machine hardware."
        ),
    }


def run_campaign(
    scenario: NegativeTriangularityScenario | None = None,
) -> dict[str, Any]:
    """Run the integrated reduced-order negative-triangularity scenario campaign."""

    scenario = scenario or NegativeTriangularityScenario()
    _validate_scenario(scenario)

    geometry = _geometry_summary(scenario)
    edge = _edge_elm_summary(scenario)
    divertor = _divertor_summary(scenario)
    vertical = _vertical_summary(scenario)
    free_boundary = _free_boundary_disruption_summary()
    components = {
        "geometry": geometry,
        "edge_elm": edge,
        "divertor": divertor,
        "vertical_control": vertical,
        "free_boundary_disruption_policy": free_boundary,
    }
    failure_reasons = [
        name
        for name, passes in {
            "geometry": geometry["passes_geometry_contract"],
            "edge_elm": edge["passes_edge_contract"],
            "divertor": divertor["passes_divertor_contract"],
            "vertical_control": vertical["passes_thresholds"],
            "free_boundary_disruption_policy": free_boundary["passes_thresholds"],
        }.items()
        if not bool(passes)
    ]
    scenario_payload = asdict(scenario)
    return {
        "SPDX-License-Identifier": "AGPL-3.0-or-later",
        "campaign_id": "high_beta_negative_triangularity_campaign",
        "schema_version": SCHEMA_VERSION,
        "scenario": scenario_payload,
        "scenario_checksum_sha256": _sha256_json(scenario_payload),
        "components": components,
        "component_checksum_sha256": _sha256_json(components),
        "acceptance": {
            "passes_thresholds": bool(not failure_reasons),
            "failure_reasons": failure_reasons,
            "accepted_claim": (
                "Reduced-order integrated scenario contract across geometry, edge, divertor, "
                "vertical replay, and free-boundary disruption-policy recovery."
            ),
            "blocked_claims": [
                "hardware_beta_40_operation",
                "experimental_negative_triangularity_elm_free_operation",
                "external_same_case_free_boundary_parity_for_this_scenario",
                "plant_certified_fault_tolerance",
            ],
        },
    }


def generate_report(
    scenario: NegativeTriangularityScenario | None = None,
) -> dict[str, Any]:
    """Generate a timestamped high-beta negative-triangularity campaign report."""

    report = run_campaign(scenario)
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    return report


def render_markdown(report: dict[str, Any]) -> str:
    """Render the integrated campaign report as Markdown."""

    scenario = report["scenario"]
    components = report["components"]
    acceptance = report["acceptance"]
    lines = [
        "# High-Beta Negative-Triangularity Integrated Campaign",
        "",
        f"- Generated: `{report.get('generated_at_utc', 'not-recorded')}`",
        f"- Scenario ID: `{scenario['scenario_id']}`",
        f"- Scenario checksum: `{report['scenario_checksum_sha256']}`",
        f"- Component checksum: `{report['component_checksum_sha256']}`",
        f"- Overall pass: `{'YES' if acceptance['passes_thresholds'] else 'NO'}`",
        "",
        "## Claim Boundary",
        "",
        acceptance["accepted_claim"],
        "",
        "Blocked claims:",
    ]
    lines.extend(f"- `{claim}`" for claim in acceptance["blocked_claims"])
    lines.extend(
        [
            "",
            "## Geometry",
            "",
            f"- Aspect ratio: `{components['geometry']['aspect_ratio']:.3f}`",
            f"- Elongation kappa: `{components['geometry']['elongation_kappa']:.3f}`",
            f"- Triangularity delta: `{components['geometry']['triangularity_delta']:.3f}`",
            f"- Target beta fraction: `{components['geometry']['target_beta_fraction']:.3f}`",
            "",
            "## Edge And Divertor",
            "",
            f"- Peeling-ballooning margin: `{components['edge_elm']['peeling_ballooning_margin']:.4f}`",
            f"- Peeling-ballooning unstable: `{components['edge_elm']['peeling_ballooning_unstable']}`",
            f"- Divertor stability index: `{components['divertor']['liquid_metal']['stability_index']:.4f}`",
            f"- Divertor surface heat flux: `{components['divertor']['liquid_metal']['surface_heat_flux_w_m2'] / 1.0e6:.3f} MW/m2`",
            "",
            "## Control And Disruption",
            "",
            f"- Vertical replay pass: `{components['vertical_control']['passes_thresholds']}`",
            f"- Vertical max P95 |z|: `{components['vertical_control']['max_p95_abs_z_m']:.6f} m`",
            f"- Free-boundary policy pass: `{components['free_boundary_disruption_policy']['passes_thresholds']}`",
            f"- Peak alert level: `{components['free_boundary_disruption_policy']['peak_alert_level']}`",
            f"- Late mean disruption risk: `{components['free_boundary_disruption_policy']['late_mean_disruption_risk']:.4f}`",
            "",
        ]
    )
    if acceptance["failure_reasons"]:
        lines.append(f"- Failure reasons: `{', '.join(acceptance['failure_reasons'])}`")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the integrated campaign and write JSON/Markdown reports."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        default=str(
            ROOT / "validation" / "reports" / "high_beta_negative_triangularity_campaign.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        default=str(
            ROOT / "validation" / "reports" / "high_beta_negative_triangularity_campaign.md"
        ),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    report = generate_report()
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    acceptance = report["acceptance"]
    print("High-beta negative-triangularity campaign complete.")
    print(
        "Summary -> "
        f"passes_thresholds={acceptance['passes_thresholds']}, "
        f"component_checksum={report['component_checksum_sha256']}"
    )
    if args.strict and not acceptance["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
