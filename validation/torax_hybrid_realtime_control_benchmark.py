# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — TORAX-Hybrid Realtime Control Benchmark
"""Deterministic TORAX-hybrid realtime control benchmark."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA_VERSION = 2
REPORT_KIND = "torax_hybrid_realtime_control_benchmark"
REPORT_PAYLOAD_KEY = "torax_hybrid_realtime_control_benchmark"

from scpn_fusion.control.torax_hybrid_loop import run_nstxu_torax_hybrid_campaign


def generate_report(
    *,
    seed: int = 42,
    episodes: int = 16,
    steps_per_episode: int = 220,
    min_disruption_avoidance_rate: float = 0.90,
    min_torax_parity_pct: float = 95.0,
    max_p95_loop_latency_ms: float = 1.0,
) -> dict[str, Any]:
    """Run TORAX-hybrid campaign and return deterministic performance metrics."""
    t0 = time.perf_counter()
    campaign = run_nstxu_torax_hybrid_campaign(
        seed=seed,
        episodes=episodes,
        steps_per_episode=steps_per_episode,
    )
    elapsed = time.perf_counter() - t0
    payload = {
        "seed": int(seed),
        "episodes": campaign.episodes,
        "steps_per_episode": campaign.steps_per_episode,
        "control_artifact_name": campaign.control_artifact_name,
        "disruption_avoidance_rate": campaign.disruption_avoidance_rate,
        "torax_parity_pct": campaign.torax_parity_pct,
        "p95_loop_latency_ms": campaign.p95_loop_latency_ms,
        "mean_risk": campaign.mean_risk,
        "thresholds": {
            "min_disruption_avoidance_rate": min_disruption_avoidance_rate,
            "min_torax_parity_pct": min_torax_parity_pct,
            "max_p95_loop_latency_ms": max_p95_loop_latency_ms,
        },
        "passes_thresholds": bool(
            campaign.disruption_avoidance_rate >= min_disruption_avoidance_rate
            and campaign.torax_parity_pct >= min_torax_parity_pct
            and campaign.p95_loop_latency_ms <= max_p95_loop_latency_ms
        ),
    }
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_kind": REPORT_KIND,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": elapsed,
        REPORT_PAYLOAD_KEY: payload,
    }


def validate_report(report: dict[str, Any]) -> dict[str, Any]:
    """Validate the current serialized report contract and return its payload."""
    expected_keys = {
        "schema_version",
        "report_kind",
        "generated_at_utc",
        "runtime_seconds",
        REPORT_PAYLOAD_KEY,
    }
    if set(report) != expected_keys:
        raise ValueError("report keys do not match the current descriptive contract")
    if report["schema_version"] != REPORT_SCHEMA_VERSION:
        raise ValueError(f"unsupported report schema_version: {report['schema_version']!r}")
    if report["report_kind"] != REPORT_KIND:
        raise ValueError(f"unsupported report_kind: {report['report_kind']!r}")
    generated_at = report["generated_at_utc"]
    if not isinstance(generated_at, str) or not generated_at:
        raise ValueError("generated_at_utc must be a non-empty string")
    runtime_seconds = report["runtime_seconds"]
    if not isinstance(runtime_seconds, (int, float)) or runtime_seconds < 0.0:
        raise ValueError("runtime_seconds must be a non-negative number")
    payload = report[REPORT_PAYLOAD_KEY]
    if not isinstance(payload, dict):
        raise ValueError(f"{REPORT_PAYLOAD_KEY} must be an object")
    return payload


def render_markdown(report: dict[str, Any]) -> str:
    """Render the current TORAX-hybrid benchmark report as Markdown."""
    benchmark = validate_report(report)
    thresholds = benchmark["thresholds"]
    lines = [
        "# TORAX-Hybrid Realtime Control Benchmark",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Runtime: `{report['runtime_seconds']:.3f} s`",
        f"- Seed: `{benchmark['seed']}`",
        "",
        "## Metrics",
        "",
        f"- Disruption avoidance rate: `{benchmark['disruption_avoidance_rate']:.3f}` "
        f"(threshold `>={thresholds['min_disruption_avoidance_rate']:.2f}`)",
        f"- TORAX parity: `{benchmark['torax_parity_pct']:.2f}%` "
        f"(threshold `>={thresholds['min_torax_parity_pct']:.1f}%`)",
        f"- P95 loop latency: `{benchmark['p95_loop_latency_ms']:.4f} ms` "
        f"(threshold `<= {thresholds['max_p95_loop_latency_ms']:.1f} ms`)",
        f"- Mean disruption risk: `{benchmark['mean_risk']:.4f}`",
        f"- Threshold pass: `{'YES' if benchmark['passes_thresholds'] else 'NO'}`",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the TORAX-hybrid benchmark CLI and export current report artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--steps-per-episode", type=int, default=220)
    parser.add_argument("--min-disruption-avoidance-rate", type=float, default=0.90)
    parser.add_argument("--min-torax-parity-pct", type=float, default=95.0)
    parser.add_argument("--max-p95-loop-latency-ms", type=float, default=1.0)
    parser.add_argument(
        "--output-json",
        default=str(
            ROOT / "validation" / "reports" / "torax_hybrid_realtime_control_benchmark.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "torax_hybrid_realtime_control_benchmark.md"),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    report = generate_report(
        seed=args.seed,
        episodes=args.episodes,
        steps_per_episode=args.steps_per_episode,
        min_disruption_avoidance_rate=args.min_disruption_avoidance_rate,
        min_torax_parity_pct=args.min_torax_parity_pct,
        max_p95_loop_latency_ms=args.max_p95_loop_latency_ms,
    )

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    benchmark = validate_report(report)
    print("TORAX-hybrid realtime control benchmark complete.")
    print(
        f"avoidance_rate={benchmark['disruption_avoidance_rate']:.3f}, "
        f"torax_parity_pct={benchmark['torax_parity_pct']:.2f}, "
        f"p95_loop_latency_ms={benchmark['p95_loop_latency_ms']:.4f}, "
        f"passes_thresholds={benchmark['passes_thresholds']}"
    )

    if args.strict and not benchmark["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
