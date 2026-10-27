# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Disruption Anomaly-Alarm Validation
"""Validate synthetic disruption anomaly-alarm sensitivity and specificity."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from scpn_fusion.control.disruption_predictor import run_anomaly_alarm_campaign

REPORT_SCHEMA_VERSION = 2
REPORT_KIND = "disruption_anomaly_alarm_validation"
PAYLOAD_KEYS = {
    "seed",
    "episodes",
    "window",
    "threshold",
    "true_positive_rate",
    "false_positive_rate",
    "p95_alarm_latency_steps",
    "min_true_positive_rate",
    "max_false_positive_rate",
    "max_p95_alarm_latency_steps",
    "passes_thresholds",
}


def run_validation(
    *,
    seed: int = 0,
    episodes: int = 128,
    window: int = 64,
    threshold: float = 0.50,
    min_true_positive_rate: float = 0.90,
    max_false_positive_rate: float = 0.10,
    max_p95_alarm_latency_steps: int = 24,
) -> dict[str, float | int | bool]:
    """Run the public synthetic anomaly-alarm campaign without changing its gate."""
    return run_anomaly_alarm_campaign(
        seed=seed,
        episodes=episodes,
        window=window,
        threshold=threshold,
        min_true_positive_rate=min_true_positive_rate,
        max_false_positive_rate=max_false_positive_rate,
        max_p95_alarm_latency_steps=max_p95_alarm_latency_steps,
    )


def generate_report(**kwargs: Any) -> dict[str, Any]:
    """Generate a schema-versioned anomaly-alarm validation report."""
    started = time.perf_counter()
    payload = run_validation(**kwargs)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_kind": REPORT_KIND,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        REPORT_KIND: payload,
    }


def _finite_probability(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite probability")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be a finite probability")
    return parsed


def _integer(name: str, value: object, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def validate_report(report: dict[str, Any]) -> dict[str, Any]:
    """Validate the descriptive serialized contract and threshold result."""
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
    if isinstance(runtime, bool) or not isinstance(runtime, (int, float)):
        raise ValueError("runtime_seconds must be a finite non-negative number")
    if not np.isfinite(runtime) or runtime < 0.0:
        raise ValueError("runtime_seconds must be a finite non-negative number")

    payload = report[REPORT_KIND]
    if not isinstance(payload, dict):
        raise ValueError(f"{REPORT_KIND} payload must be an object")
    if set(payload) != PAYLOAD_KEYS:
        raise ValueError("payload keys do not match the current descriptive contract")

    _integer("seed", payload["seed"], 0)
    _integer("episodes", payload["episodes"], 1)
    _integer("window", payload["window"], 16)
    alarm_threshold = _finite_probability("threshold", payload["threshold"])
    if alarm_threshold <= 0.0 or alarm_threshold >= 1.0:
        raise ValueError("threshold must be strictly between zero and one")
    tpr = _finite_probability("true_positive_rate", payload["true_positive_rate"])
    fpr = _finite_probability("false_positive_rate", payload["false_positive_rate"])
    min_tpr = _finite_probability("min_true_positive_rate", payload["min_true_positive_rate"])
    max_fpr = _finite_probability("max_false_positive_rate", payload["max_false_positive_rate"])
    latency = _integer("p95_alarm_latency_steps", payload["p95_alarm_latency_steps"], -1)
    max_latency = _integer(
        "max_p95_alarm_latency_steps",
        payload["max_p95_alarm_latency_steps"],
        0,
    )
    passes = payload["passes_thresholds"]
    if not isinstance(passes, bool):
        raise ValueError("passes_thresholds must be a boolean")
    expected_pass = bool(tpr >= min_tpr and fpr <= max_fpr and 0 <= latency <= max_latency)
    if passes is not expected_pass:
        raise ValueError("passes_thresholds is inconsistent with public thresholds")
    return cast(dict[str, Any], payload)


def render_markdown(report: dict[str, Any]) -> str:
    """Render the validated synthetic campaign without overstating fidelity."""
    result = validate_report(report)
    latency = result["p95_alarm_latency_steps"]
    latency_text = "not observed" if latency < 0 else f"{latency} steps"
    return "\n".join(
        [
            "# Disruption Anomaly-Alarm Validation",
            "",
            f"- Report schema: `{report['schema_version']}`",
            f"- Generated: `{report['generated_at_utc']}`",
            f"- Runtime: `{report['runtime_seconds']:.3f} s`",
            "- Evidence boundary: deterministic synthetic episodes; not experimental validation.",
            "",
            "## Results",
            "",
            f"- True-positive rate: `{result['true_positive_rate']:.6f}`",
            f"- Minimum true-positive rate: `{result['min_true_positive_rate']:.6f}`",
            f"- False-positive rate: `{result['false_positive_rate']:.6f}`",
            f"- Maximum false-positive rate: `{result['max_false_positive_rate']:.6f}`",
            f"- P95 alarm latency: `{latency_text}`",
            f"- Maximum P95 alarm latency: `{result['max_p95_alarm_latency_steps']} steps`",
            f"- Overall pass: `{'YES' if result['passes_thresholds'] else 'NO'}`",
            "",
        ]
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse public campaign, acceptance and output arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--min-true-positive-rate", type=float, default=0.90)
    parser.add_argument("--max-false-positive-rate", type=float, default=0.10)
    parser.add_argument("--max-p95-alarm-latency-steps", type=int, default=24)
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
    """Run validation, write reports and fail strict mode on a missed gate."""
    args = parse_args(argv)
    report = generate_report(
        seed=args.seed,
        episodes=args.episodes,
        window=args.window,
        threshold=args.threshold,
        min_true_positive_rate=args.min_true_positive_rate,
        max_false_positive_rate=args.max_false_positive_rate,
        max_p95_alarm_latency_steps=args.max_p95_alarm_latency_steps,
    )
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(f"{json.dumps(report, indent=2)}\n", encoding="utf-8")
    output_md.write_text(render_markdown(report), encoding="utf-8")
    result = validate_report(report)
    print("Disruption anomaly-alarm validation complete.")
    print(
        f"true_positive_rate={result['true_positive_rate']:.6f}, "
        f"false_positive_rate={result['false_positive_rate']:.6f}, "
        f"p95_alarm_latency_steps={result['p95_alarm_latency_steps']}, "
        f"passes_thresholds={result['passes_thresholds']}"
    )
    if args.strict and not result["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
