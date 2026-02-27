#!/usr/bin/env python3
# ----------------------------------------------------------------------
# SCPN Fusion Core -- Transport Uncertainty Envelope Benchmark
# ----------------------------------------------------------------------
"""Contract benchmark for transport uncertainty-envelope metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "validation"))

from validate_real_shots import THRESHOLDS, validate_transport


CONTRACT_THRESHOLDS = {
    "min_within_2sigma_fraction": float(THRESHOLDS["tau_e_2sigma_fraction"]),
    "max_abs_relative_error_p95": 2.5,
    "max_zscore_p95": 4.0,
}


def _render_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def run_benchmark(*, itpa_csv: Path) -> dict[str, Any]:
    transport = validate_transport(itpa_csv)
    envelope = transport.get("uncertainty_envelope", {})

    required_fields = (
        "abs_relative_error_p95",
        "residual_s_p05",
        "residual_s_p95",
        "sigma_s_p50",
        "sigma_s_p95",
        "zscore_p95",
        "within_1sigma_fraction",
        "within_2sigma_fraction",
    )
    envelope_fields_pass = bool(
        isinstance(envelope, dict)
        and all(name in envelope for name in required_fields)
        and all(np.isfinite(float(envelope[name])) for name in required_fields)
    )

    coverage_pass = bool(
        float(transport.get("within_2sigma_fraction", 0.0))
        >= CONTRACT_THRESHOLDS["min_within_2sigma_fraction"]
    )
    abs_relative_p95_pass = bool(
        float(envelope.get("abs_relative_error_p95", np.inf))
        <= CONTRACT_THRESHOLDS["max_abs_relative_error_p95"]
    )
    zscore_p95_pass = bool(
        float(envelope.get("zscore_p95", np.inf))
        <= CONTRACT_THRESHOLDS["max_zscore_p95"]
    )
    passes = bool(
        int(transport.get("n_shots", 0)) > 0
        and bool(transport.get("passes", False))
        and envelope_fields_pass
        and coverage_pass
        and abs_relative_p95_pass
        and zscore_p95_pass
    )

    return {
        "transport_uncertainty_envelope_benchmark": {
            "itpa_csv": _render_path(itpa_csv),
            "n_shots": int(transport.get("n_shots", 0)),
            "transport_pass": bool(transport.get("passes", False)),
            "envelope_fields_pass": envelope_fields_pass,
            "coverage_pass": coverage_pass,
            "abs_relative_p95_pass": abs_relative_p95_pass,
            "zscore_p95_pass": zscore_p95_pass,
            "passes_thresholds": passes,
            "thresholds": dict(CONTRACT_THRESHOLDS),
            "transport_summary": {
                "rmse_s": transport.get("rmse_s"),
                "rmse_relative": transport.get("rmse_relative"),
                "within_2sigma_fraction": transport.get("within_2sigma_fraction"),
                "uncertainty_envelope": envelope,
            },
        }
    }


def render_markdown(report: dict[str, Any]) -> str:
    g = report["transport_uncertainty_envelope_benchmark"]
    tr = g["transport_summary"]
    env = tr["uncertainty_envelope"]
    lines = [
        "# Transport Uncertainty Envelope Benchmark",
        "",
        f"- ITPA CSV: `{g['itpa_csv']}`",
        f"- Shots: `{g['n_shots']}`",
        f"- Transport pass: `{'YES' if g['transport_pass'] else 'NO'}`",
        f"- Envelope fields pass: `{'YES' if g['envelope_fields_pass'] else 'NO'}`",
        f"- Coverage pass: `{'YES' if g['coverage_pass'] else 'NO'}`",
        f"- |rel error| p95 pass: `{'YES' if g['abs_relative_p95_pass'] else 'NO'}`",
        f"- z-score p95 pass: `{'YES' if g['zscore_p95_pass'] else 'NO'}`",
        f"- Overall pass: `{'YES' if g['passes_thresholds'] else 'NO'}`",
        "",
        "| Metric | Value | Threshold |",
        "|--------|-------|-----------|",
        (
            f"| within_2sigma_fraction | {float(tr['within_2sigma_fraction']):.2f} | "
            f">= {CONTRACT_THRESHOLDS['min_within_2sigma_fraction']:.2f} |"
        ),
        (
            f"| abs_relative_error_p95 | {float(env.get('abs_relative_error_p95', 0.0)):.4f} | "
            f"<= {CONTRACT_THRESHOLDS['max_abs_relative_error_p95']:.2f} |"
        ),
        (
            f"| zscore_p95 | {float(env.get('zscore_p95', 0.0)):.4f} | "
            f"<= {CONTRACT_THRESHOLDS['max_zscore_p95']:.2f} |"
        ),
        (
            f"| sigma_s_p95 | {float(env.get('sigma_s_p95', 0.0)):.4f} | > 0.00 |"
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--itpa-csv",
        default=str(ROOT / "validation" / "reference_data" / "itpa" / "hmode_confinement.csv"),
    )
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / "transport_uncertainty_envelope_benchmark.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "transport_uncertainty_envelope_benchmark.md"),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    report = run_benchmark(itpa_csv=Path(args.itpa_csv))
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    g = report["transport_uncertainty_envelope_benchmark"]
    print("Transport uncertainty envelope benchmark complete.")
    print(
        "shots={shots}, coverage_pass={cov}, abs_relative_p95_pass={abs_pass}, "
        "zscore_p95_pass={z_pass}, pass={p}".format(
            shots=g["n_shots"],
            cov=g["coverage_pass"],
            abs_pass=g["abs_relative_p95_pass"],
            z_pass=g["zscore_p95_pass"],
            p=g["passes_thresholds"],
        )
    )
    if args.strict and not g["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
