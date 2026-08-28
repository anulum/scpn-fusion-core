# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Compact-Reactor Engineering-Constraint Validation
"""Validate a synthetic compact-reactor scan against declared engineering constraints."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scpn_fusion.core.global_design_scanner import GlobalDesignExplorer


ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA_VERSION = 2
REPORT_KIND = "compact_reactor_engineering_constraint_validation"
REPORT_PAYLOAD_KEY = "compact_reactor_engineering_constraint_validation"


def _positive_finite(value: float, *, name: str) -> float:
    checked = float(value)
    if not np.isfinite(checked):
        raise ValueError(f"{name} must be finite.")
    if checked <= 0.0:
        raise ValueError(f"{name} must be > 0.")
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
    seed: int = 42,
    scan_samples: int = 2600,
    radius_min_m: float = 1.2,
    radius_max_m: float = 1.5,
    min_fusion_gain: float = 5.0,
    min_feasible_designs: int = 1,
    divertor_flux_cap_mw_m2: float = 45.0,
    zeff_cap: float = 0.4,
    hts_peak_cap_t: float = 21.0,
) -> dict[str, Any]:
    """Run the synthetic compact-reactor scan and evaluate its acceptance gate.

    Parameters
    ----------
    seed : int
        Deterministic NumPy generator seed passed to the scanner.
    scan_samples : int
        Number of synthetic design points sampled in the compact envelope.
    radius_min_m, radius_max_m : float
        Accepted major-radius window in metres.
    min_fusion_gain : float
        Strict lower bound for the reduced engineering-gain proxy.
    min_feasible_designs : int
        Minimum number of designs required to pass the campaign.
    divertor_flux_cap_mw_m2 : float
        Maximum HEAT-ML-shadowed divertor flux proxy in megawatts per square metre.
    zeff_cap : float
        Maximum reduced impurity proxy accepted by the scanner.
    hts_peak_cap_t : float
        Maximum reduced HTS peak-field proxy in tesla.

    Returns
    -------
    dict[str, Any]
        Scan configuration, feasible count, best synthetic design and gate status.

    Raises
    ------
    ValueError
        If a sample count, radius bound, threshold or engineering cap is invalid.
    """
    if isinstance(seed, bool):
        raise ValueError("seed must be an integer.")
    checked_seed = int(seed)
    checked_samples = _positive_int(scan_samples, name="scan_samples")
    checked_radius_min = _positive_finite(radius_min_m, name="radius_min_m")
    checked_radius_max = _positive_finite(radius_max_m, name="radius_max_m")
    if checked_radius_max <= checked_radius_min:
        raise ValueError("radius_max_m must be greater than radius_min_m.")
    checked_min_gain = _positive_finite(min_fusion_gain, name="min_fusion_gain")
    checked_min_feasible = _positive_int(min_feasible_designs, name="min_feasible_designs")
    checked_divertor_cap = _positive_finite(
        divertor_flux_cap_mw_m2,
        name="divertor_flux_cap_mw_m2",
    )
    checked_zeff_cap = _positive_finite(zeff_cap, name="zeff_cap")
    if checked_zeff_cap > 1.0:
        raise ValueError("zeff_cap must be <= 1.0.")
    checked_hts_cap = _positive_finite(hts_peak_cap_t, name="hts_peak_cap_t")

    t0 = time.perf_counter()
    explorer = GlobalDesignExplorer(
        "synthetic-compact-reactor-validation",
        divertor_flux_cap_mw_m2=checked_divertor_cap,
        zeff_cap=checked_zeff_cap,
        hts_peak_cap_t=checked_hts_cap,
    )
    frame = explorer.run_compact_scan(n_samples=checked_samples, seed=checked_seed)
    feasible = frame[
        (frame["Constraint_OK"])
        & (frame["R"] >= checked_radius_min)
        & (frame["R"] <= checked_radius_max)
        & (frame["Q"] > checked_min_gain)
    ]

    feasible_count = int(len(feasible))
    best_design: dict[str, float] | None = None
    if feasible_count > 0:
        best = feasible.loc[feasible["Cost"].idxmin()]
        best_design = {
            "radius_m": float(best["R"]),
            "magnetic_field_t": float(best["B"]),
            "plasma_current_ma": float(best["Ip"]),
            "fusion_gain": float(best["Q"]),
            "divertor_flux_mw_m2": float(best["Div_Load_Optimized"]),
            "zeff_proxy": float(best["Zeff_Est"]),
            "hts_peak_field_t": float(best["B_peak_HTS_T"]),
            "cost_proxy": float(best["Cost"]),
        }

    return {
        "seed": checked_seed,
        "scan_samples": checked_samples,
        "evaluated_designs": int(len(frame)),
        "feasible_designs": feasible_count,
        "thresholds": {
            "radius_min_m": checked_radius_min,
            "radius_max_m": checked_radius_max,
            "min_fusion_gain": checked_min_gain,
            "min_feasible_designs": checked_min_feasible,
            "divertor_flux_cap_mw_m2": checked_divertor_cap,
            "zeff_cap": checked_zeff_cap,
            "hts_peak_cap_t": checked_hts_cap,
        },
        "best_design": best_design,
        "passes_thresholds": feasible_count >= checked_min_feasible,
        "runtime_seconds": float(time.perf_counter() - t0),
    }


def generate_report(**kwargs: Any) -> dict[str, Any]:
    """Generate the versioned compact-reactor engineering-constraint report."""
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
    """Render the compact-reactor engineering-constraint report as Markdown."""
    campaign = validate_report(report)
    thresholds = campaign["thresholds"]
    best = campaign["best_design"]
    lines = [
        "# Compact-Reactor Engineering-Constraint Validation",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Runtime: `{campaign['runtime_seconds']:.3f} s`",
        f"- Evaluated synthetic designs: `{campaign['evaluated_designs']}`",
        f"- Feasible designs: `{campaign['feasible_designs']}`",
        "",
        "## Acceptance Thresholds",
        "",
        f"- Radius window: `{thresholds['radius_min_m']:.2f}..{thresholds['radius_max_m']:.2f} m`",
        f"- Fusion-gain proxy: `> {thresholds['min_fusion_gain']:.2f}`",
        f"- Minimum feasible designs: `{thresholds['min_feasible_designs']}`",
        f"- Divertor-flux proxy cap: `<= {thresholds['divertor_flux_cap_mw_m2']:.2f} MW/m2`",
        f"- Zeff proxy cap: `<= {thresholds['zeff_cap']:.3f}`",
        f"- HTS peak-field proxy cap: `<= {thresholds['hts_peak_cap_t']:.2f} T`",
        "",
        "## Best Feasible Synthetic Design",
        "",
    ]
    if best is None:
        lines.append("- None found in the configured scan.")
    else:
        lines.extend(
            [
                f"- Major radius: `{best['radius_m']:.3f} m`",
                f"- Magnetic field: `{best['magnetic_field_t']:.3f} T`",
                f"- Plasma current: `{best['plasma_current_ma']:.3f} MA`",
                f"- Fusion-gain proxy: `{best['fusion_gain']:.3f}`",
                f"- Divertor-flux proxy: `{best['divertor_flux_mw_m2']:.3f} MW/m2`",
                f"- Zeff proxy: `{best['zeff_proxy']:.3f}`",
                f"- HTS peak-field proxy: `{best['hts_peak_field_t']:.3f} T`",
                f"- Cost proxy: `{best['cost_proxy']:.3f}`",
            ]
        )
    lines.extend(
        [
            "",
            f"- Overall pass: `{'YES' if campaign['passes_thresholds'] else 'NO'}`",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for compact-reactor constraint validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scan-samples", type=int, default=2600)
    parser.add_argument("--radius-min-m", type=float, default=1.2)
    parser.add_argument("--radius-max-m", type=float, default=1.5)
    parser.add_argument("--min-fusion-gain", type=float, default=5.0)
    parser.add_argument("--min-feasible-designs", type=int, default=1)
    parser.add_argument("--divertor-flux-cap-mw-m2", type=float, default=45.0)
    parser.add_argument("--zeff-cap", type=float, default=0.4)
    parser.add_argument("--hts-peak-cap-t", type=float, default=21.0)
    parser.add_argument(
        "--output-json",
        default=str(
            ROOT
            / "validation"
            / "reports"
            / "compact_reactor_engineering_constraint_validation.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        default=str(
            ROOT / "validation" / "reports" / "compact_reactor_engineering_constraint_validation.md"
        ),
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the validation and write its versioned JSON and Markdown reports."""
    args = parse_args(argv)
    report = generate_report(
        seed=args.seed,
        scan_samples=args.scan_samples,
        radius_min_m=args.radius_min_m,
        radius_max_m=args.radius_max_m,
        min_fusion_gain=args.min_fusion_gain,
        min_feasible_designs=args.min_feasible_designs,
        divertor_flux_cap_mw_m2=args.divertor_flux_cap_mw_m2,
        zeff_cap=args.zeff_cap,
        hts_peak_cap_t=args.hts_peak_cap_t,
    )
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output_md.write_text(render_markdown(report), encoding="utf-8")

    campaign = validate_report(report)
    print("Compact-reactor engineering-constraint validation complete.")
    print(
        f"feasible_designs={campaign['feasible_designs']}, "
        f"passes_thresholds={campaign['passes_thresholds']}"
    )
    if args.strict and not campaign["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
