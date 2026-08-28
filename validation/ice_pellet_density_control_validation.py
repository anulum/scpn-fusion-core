# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Ice-Pellet Density-Control Validation
"""Validate ice-pellet fueling on reduced ITER-like density dynamics."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from scpn_fusion.control.fueling_mode import simulate_iter_density_control

REPORT_SCHEMA_VERSION = 2
REPORT_KIND = "ice_pellet_density_control_validation"


def _finite_non_negative(name: str, value: float) -> float:
    checked = float(value)
    if not np.isfinite(checked):
        raise ValueError(f"{name} must be finite.")
    if checked < 0.0:
        raise ValueError(f"{name} must be >= 0.")
    return checked


def run_validation(
    *,
    target_density: float = 1.0,
    initial_density: float = 0.82,
    steps: int = 3000,
    dt_s: float = 1e-3,
    max_final_abs_error: float = 1e-3,
) -> dict[str, Any]:
    """Run reduced density control and evaluate its public error threshold."""
    threshold = _finite_non_negative("max_final_abs_error", max_final_abs_error)
    result = simulate_iter_density_control(
        target_density=target_density,
        initial_density=initial_density,
        steps=steps,
        dt_s=dt_s,
    )
    return {
        "target_density": float(target_density),
        "initial_density": float(initial_density),
        "steps": result.steps,
        "dt_s": result.dt_s,
        "final_density": result.final_density,
        "final_abs_error": result.final_abs_error,
        "rmse": result.rmse,
        "max_final_abs_error": threshold,
        "passes_thresholds": bool(result.final_abs_error <= threshold),
    }


def generate_report(**kwargs: Any) -> dict[str, Any]:
    """Generate a schema-versioned ice-pellet density-control report."""
    started = time.perf_counter()
    payload = run_validation(**kwargs)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_kind": REPORT_KIND,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        REPORT_KIND: payload,
    }


def validate_report(report: dict[str, Any]) -> dict[str, Any]:
    """Validate the current serialized contract and return its payload."""
    expected = {
        "schema_version",
        "report_kind",
        "generated_at_utc",
        "runtime_seconds",
        REPORT_KIND,
    }
    if set(report) != expected:
        raise ValueError("report keys do not match the current descriptive contract")
    if report["schema_version"] != REPORT_SCHEMA_VERSION:
        raise ValueError(f"unsupported report schema_version: {report['schema_version']!r}")
    if report["report_kind"] != REPORT_KIND:
        raise ValueError(f"unsupported report_kind: {report['report_kind']!r}")
    generated_at = report["generated_at_utc"]
    if not isinstance(generated_at, str) or not generated_at:
        raise ValueError("generated_at_utc must be a non-empty string")
    runtime = report["runtime_seconds"]
    if not isinstance(runtime, (int, float)):
        raise ValueError("runtime_seconds must be a finite non-negative number")
    if not np.isfinite(runtime) or runtime < 0.0:
        raise ValueError("runtime_seconds must be a finite non-negative number")
    payload = report[REPORT_KIND]
    if not isinstance(payload, dict):
        raise ValueError(f"{REPORT_KIND} payload must be an object")
    return cast(dict[str, Any], payload)


def render_markdown(report: dict[str, Any]) -> str:
    """Render validated ice-pellet density-control metrics as Markdown."""
    result = validate_report(report)
    return "\n".join(
        [
            "# Ice-Pellet Density-Control Validation",
            "",
            f"- Report schema: `{report['schema_version']}`",
            f"- Generated: `{report['generated_at_utc']}`",
            f"- Runtime: `{report['runtime_seconds']:.3f} s`",
            "",
            "## Results",
            "",
            f"- Final density: `{result['final_density']:.6f}`",
            f"- Final absolute error: `{result['final_abs_error']:.6e}`",
            f"- RMSE: `{result['rmse']:.6e}`",
            f"- Maximum final absolute error: `{result['max_final_abs_error']:.6e}`",
            f"- Overall pass: `{'YES' if result['passes_thresholds'] else 'NO'}`",
            "",
        ]
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for density-control validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-density", type=float, default=1.0)
    parser.add_argument("--initial-density", type=float, default=0.82)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--dt-s", type=float, default=1e-3)
    parser.add_argument("--max-final-abs-error", type=float, default=1e-3)
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / f"{REPORT_KIND}.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / f"{REPORT_KIND}.md"),
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run validation and write versioned JSON and Markdown reports."""
    args = parse_args(argv)
    report = generate_report(
        target_density=args.target_density,
        initial_density=args.initial_density,
        steps=args.steps,
        dt_s=args.dt_s,
        max_final_abs_error=args.max_final_abs_error,
    )
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(f"{json.dumps(report, indent=2)}\n", encoding="utf-8")
    output_md.write_text(render_markdown(report), encoding="utf-8")
    result = validate_report(report)
    print("Ice-pellet density-control validation complete.")
    print(
        f"final_abs_error={result['final_abs_error']:.6e}, "
        f"threshold={result['max_final_abs_error']:.6e}, "
        f"passes_thresholds={result['passes_thresholds']}"
    )
    if args.strict and not result["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
