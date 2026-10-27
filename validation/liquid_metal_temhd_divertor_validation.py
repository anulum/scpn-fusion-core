# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Liquid-Metal TEMHD Divertor Validation
"""Validate reduced liquid-metal TEMHD flow across a synthetic 3D divertor sweep."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np

from scpn_fusion.core.divertor_thermal_sim import DivertorLab
from scpn_fusion.core.equilibrium_3d import FourierMode3D, VMECStyleEquilibrium3D


ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA_VERSION = 2
REPORT_KIND = "liquid_metal_temhd_divertor_validation"
REPORT_PAYLOAD_KEY = "liquid_metal_temhd_divertor_validation"


def _positive_finite(value: float, *, name: str) -> float:
    checked = float(value)
    if not np.isfinite(checked):
        raise ValueError(f"{name} must be finite.")
    if checked <= 0.0:
        raise ValueError(f"{name} must be > 0.")
    return checked


def _nonnegative_finite(value: float, *, name: str) -> float:
    checked = float(value)
    if not np.isfinite(checked):
        raise ValueError(f"{name} must be finite.")
    if checked < 0.0:
        raise ValueError(f"{name} must be >= 0.")
    return checked


def _unit_interval(value: float, *, name: str) -> float:
    checked = float(value)
    if not np.isfinite(checked):
        raise ValueError(f"{name} must be finite.")
    if checked < 0.0:
        raise ValueError(f"{name} must be in [0, 1].")
    if checked > 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return checked


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= 1.")
    checked = int(value)
    if checked < 1:
        raise ValueError(f"{name} must be an integer >= 1.")
    return checked


def run_campaign(
    *,
    slow_flow_velocity_m_s: float = 0.001,
    fast_flow_velocity_m_s: float = 10.0,
    expansion_factor: float = 40.0,
    toroidal_samples: int = 36,
    min_pressure_ratio_fast_to_slow: float = 1000.0,
    max_evap_ratio_fast_to_slow: float = 1.0,
    max_toroidal_stability_index: float = 1.0,
    min_toroidal_stability_rate: float = 0.95,
) -> dict[str, Any]:
    """Run the reduced TEMHD flow and synthetic toroidal-stability campaign.

    Parameters
    ----------
    slow_flow_velocity_m_s, fast_flow_velocity_m_s : float
        Positive comparison velocities; the fast velocity must exceed the slow.
    expansion_factor : float
        Positive divertor flux-expansion proxy applied to both flow cases.
    toroidal_samples : int
        Number of synthetic non-axisymmetric toroidal positions.
    min_pressure_ratio_fast_to_slow : float
        Minimum accepted fast-to-slow MHD pressure-loss ratio.
    max_evap_ratio_fast_to_slow : float
        Maximum accepted fast-to-slow evaporation-rate ratio.
    max_toroidal_stability_index : float
        Maximum accepted combined reduced-model index at each toroidal point.
    min_toroidal_stability_rate : float
        Minimum fraction of toroidal points within the combined-index bound.

    Returns
    -------
    dict[str, Any]
        Flow states, ratios, toroidal sweep metrics, thresholds and gate status.

    Raises
    ------
    ValueError
        If a velocity, sampling value or acceptance threshold is invalid.
    """
    slow_velocity = _positive_finite(slow_flow_velocity_m_s, name="slow_flow_velocity_m_s")
    fast_velocity = _positive_finite(fast_flow_velocity_m_s, name="fast_flow_velocity_m_s")
    if fast_velocity <= slow_velocity:
        raise ValueError("fast_flow_velocity_m_s must exceed slow_flow_velocity_m_s.")
    checked_expansion = _positive_finite(expansion_factor, name="expansion_factor")
    checked_samples = _positive_int(toroidal_samples, name="toroidal_samples")
    checked_pressure_ratio = _positive_finite(
        min_pressure_ratio_fast_to_slow,
        name="min_pressure_ratio_fast_to_slow",
    )
    checked_evap_ratio = _nonnegative_finite(
        max_evap_ratio_fast_to_slow,
        name="max_evap_ratio_fast_to_slow",
    )
    checked_index = _positive_finite(
        max_toroidal_stability_index,
        name="max_toroidal_stability_index",
    )
    checked_stability_rate = _unit_interval(
        min_toroidal_stability_rate,
        name="min_toroidal_stability_rate",
    )

    t0 = time.perf_counter()
    lab = DivertorLab(P_sol_MW=35.0, R_major=1.4, B_pol=2.3)
    slow = lab.simulate_temhd_liquid_metal(
        flow_velocity_m_s=slow_velocity,
        expansion_factor=checked_expansion,
    )
    fast = lab.simulate_temhd_liquid_metal(
        flow_velocity_m_s=fast_velocity,
        expansion_factor=checked_expansion,
    )

    equilibrium = VMECStyleEquilibrium3D(
        r_axis=1.4,
        z_axis=0.0,
        a_minor=0.45,
        kappa=1.75,
        triangularity=0.32,
        nfp=3,
        modes=[FourierMode3D(m=1, n=1, r_cos=0.05, z_sin=0.04)],
    )
    phis = np.linspace(0.0, 2.0 * np.pi, checked_samples, endpoint=False)
    thetas = np.full_like(phis, 0.8 * np.pi)
    rho = np.ones_like(phis)
    radius_values, _, _ = equilibrium.flux_to_cylindrical(rho, thetas, phis)
    mean_radius = float(np.mean(radius_values))
    modulation = 1.0 + 0.08 * (radius_values - mean_radius) / max(mean_radius, 1.0e-9)

    stability_indices: list[float] = []
    for factor in modulation:
        modulated_flux = cast(float, fast["surface_heat_flux_w_m2"]) * float(factor)
        stability_indices.append(
            float(
                modulated_flux / 45.0e6
                + cast(float, fast["pressure_loss_pa"]) / 8.0e5
                + cast(float, fast["evaporation_rate_kg_m2_s"]) / 1.0e-3
            )
        )
    indices = np.asarray(stability_indices, dtype=float)
    toroidal_stability_rate = float(np.mean(indices <= checked_index))

    pressure_ratio = float(
        cast(float, fast["pressure_loss_pa"]) / max(cast(float, slow["pressure_loss_pa"]), 1.0e-12)
    )
    evaporation_ratio = float(
        cast(float, fast["evaporation_rate_kg_m2_s"])
        / max(cast(float, slow["evaporation_rate_kg_m2_s"]), 1.0e-12)
    )
    flow_states_stable = cast(bool, slow["is_stable"]) & cast(bool, fast["is_stable"])
    passes = all(
        (
            flow_states_stable,
            pressure_ratio >= checked_pressure_ratio,
            evaporation_ratio < checked_evap_ratio,
            toroidal_stability_rate >= checked_stability_rate,
        )
    )

    return {
        "slow_flow": slow,
        "fast_flow": fast,
        "pressure_ratio_fast_to_slow": pressure_ratio,
        "evaporation_ratio_fast_to_slow": evaporation_ratio,
        "toroidal_samples": checked_samples,
        "toroidal_stability_index_min": float(np.min(indices)),
        "toroidal_stability_index_max": float(np.max(indices)),
        "toroidal_stability_rate": toroidal_stability_rate,
        "thresholds": {
            "min_pressure_ratio_fast_to_slow": checked_pressure_ratio,
            "max_evap_ratio_fast_to_slow": checked_evap_ratio,
            "max_toroidal_stability_index": checked_index,
            "min_toroidal_stability_rate": checked_stability_rate,
        },
        "passes_thresholds": bool(passes),
        "runtime_seconds": float(time.perf_counter() - t0),
    }


def generate_report(**kwargs: Any) -> dict[str, Any]:
    """Generate the versioned liquid-metal TEMHD divertor report."""
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_kind": REPORT_KIND,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        REPORT_PAYLOAD_KEY: run_campaign(**kwargs),
    }


def validate_report(report: dict[str, Any]) -> dict[str, Any]:
    """Validate the current serialized report contract and return its payload."""
    expected_keys = {
        "schema_version",
        "report_kind",
        "generated_at_utc",
        REPORT_PAYLOAD_KEY,
    }
    if set(report) != expected_keys:
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
    """Render the liquid-metal TEMHD divertor report as Markdown."""
    campaign = validate_report(report)
    thresholds = campaign["thresholds"]
    lines = [
        "# Liquid-Metal TEMHD Divertor Validation",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Runtime: `{campaign['runtime_seconds']:.3f} s`",
        "",
        "## Slow and Fast Flow",
        "",
        f"- Slow-flow stability index: `{campaign['slow_flow']['stability_index']:.4f}`",
        f"- Fast-flow stability index: `{campaign['fast_flow']['stability_index']:.4f}`",
        f"- Pressure ratio (fast/slow): `{campaign['pressure_ratio_fast_to_slow']:.1f}`",
        f"- Required pressure ratio: `>= {thresholds['min_pressure_ratio_fast_to_slow']:.1f}`",
        f"- Evaporation ratio (fast/slow): `{campaign['evaporation_ratio_fast_to_slow']:.4f}`",
        f"- Maximum evaporation ratio: `< {thresholds['max_evap_ratio_fast_to_slow']:.4f}`",
        "",
        "## Synthetic 3D Toroidal Sweep",
        "",
        f"- Samples: `{campaign['toroidal_samples']}`",
        f"- Stability-index range: `{campaign['toroidal_stability_index_min']:.4f}..{campaign['toroidal_stability_index_max']:.4f}`",
        f"- Maximum accepted index: `{thresholds['max_toroidal_stability_index']:.4f}`",
        f"- Stability rate: `{campaign['toroidal_stability_rate']:.3f}`",
        f"- Minimum stability rate: `{thresholds['min_toroidal_stability_rate']:.3f}`",
        f"- Overall pass: `{'YES' if campaign['passes_thresholds'] else 'NO'}`",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for liquid-metal TEMHD divertor validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slow-flow-velocity-m-s", type=float, default=0.001)
    parser.add_argument("--fast-flow-velocity-m-s", type=float, default=10.0)
    parser.add_argument("--expansion-factor", type=float, default=40.0)
    parser.add_argument("--toroidal-samples", type=int, default=36)
    parser.add_argument("--min-pressure-ratio-fast-to-slow", type=float, default=1000.0)
    parser.add_argument("--max-evap-ratio-fast-to-slow", type=float, default=1.0)
    parser.add_argument("--max-toroidal-stability-index", type=float, default=1.0)
    parser.add_argument("--min-toroidal-stability-rate", type=float, default=0.95)
    parser.add_argument(
        "--output-json",
        default=str(
            ROOT / "validation" / "reports" / "liquid_metal_temhd_divertor_validation.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "liquid_metal_temhd_divertor_validation.md"),
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run validation and write versioned JSON and Markdown reports."""
    args = parse_args(argv)
    report = generate_report(
        slow_flow_velocity_m_s=args.slow_flow_velocity_m_s,
        fast_flow_velocity_m_s=args.fast_flow_velocity_m_s,
        expansion_factor=args.expansion_factor,
        toroidal_samples=args.toroidal_samples,
        min_pressure_ratio_fast_to_slow=args.min_pressure_ratio_fast_to_slow,
        max_evap_ratio_fast_to_slow=args.max_evap_ratio_fast_to_slow,
        max_toroidal_stability_index=args.max_toroidal_stability_index,
        min_toroidal_stability_rate=args.min_toroidal_stability_rate,
    )
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output_md.write_text(render_markdown(report), encoding="utf-8")

    campaign = validate_report(report)
    print("Liquid-metal TEMHD divertor validation complete.")
    print(
        f"pressure_ratio={campaign['pressure_ratio_fast_to_slow']:.1f}, "
        f"evaporation_ratio={campaign['evaporation_ratio_fast_to_slow']:.4f}, "
        f"toroidal_stability_rate={campaign['toroidal_stability_rate']:.3f}, "
        f"passes_thresholds={campaign['passes_thresholds']}"
    )
    if args.strict and not campaign["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
