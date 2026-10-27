# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — NSTX-U/SPARC Digital-Twin Ingestion Validation
"""Validate realtime NSTX-U/SPARC digital-twin ingestion and scenario planning."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA_VERSION = 2
REPORT_KIND = "nstx_u_sparc_digital_twin_ingestion"
REPORT_PAYLOAD_KEY = "nstx_u_sparc_digital_twin_ingestion"

from scpn_fusion.control.digital_twin_ingest import run_realtime_twin_session


def _run_machine(
    machine: str,
    seed: int,
    samples: int,
    *,
    chaos_dropout_prob: float = 0.0,
    chaos_noise_std: float = 0.0,
) -> dict[str, Any]:
    return run_realtime_twin_session(
        machine,
        seed=int(seed),
        samples=int(samples),
        dt_ms=5,
        horizon=24,
        plan_every=8,
        max_buffer=512,
        chaos_dropout_prob=float(chaos_dropout_prob),
        chaos_noise_std=float(chaos_noise_std),
    )


def run_campaign(
    *,
    seed: int = 42,
    samples_per_machine: int = 320,
    chaos_dropout_prob: float = 0.0,
    chaos_noise_std: float = 0.0,
    min_planning_success_rate: float = 0.90,
    max_mean_risk: float = 0.75,
    max_p95_latency_ms: float = 6.0,
) -> dict[str, Any]:
    """Run the synthetic machine campaigns and evaluate their acceptance gates.

    Parameters
    ----------
    seed : int
        Base random seed; the SPARC replay uses the next integer.
    samples_per_machine : int
        Number of five-millisecond telemetry packets generated per machine.
    chaos_dropout_prob : float
        Independent probability of replacing each telemetry channel with zero.
    chaos_noise_std : float
        Standard deviation of Gaussian noise added to every telemetry channel.
    min_planning_success_rate : float
        Minimum accepted fraction of scenario plans that pass their runtime gate.
    max_mean_risk : float
        Maximum accepted mean disruption-risk score for each machine.
    max_p95_latency_ms : float
        Maximum accepted P95 deterministic planning-latency estimate in milliseconds.

    Returns
    -------
    dict[str, Any]
        Per-machine metrics, chaos accounting, thresholds, runtime, and aggregate
        acceptance status.

    Raises
    ------
    ValueError
        If a sample count, chaos parameter, or acceptance threshold is invalid.
    """
    samples_per_machine = int(samples_per_machine)
    if samples_per_machine < 32:
        raise ValueError("samples_per_machine must be >= 32.")
    dropout = float(chaos_dropout_prob)
    if not np.isfinite(dropout) or dropout < 0.0 or dropout > 1.0:
        raise ValueError("chaos_dropout_prob must be finite and in [0, 1].")
    noise_std = float(chaos_noise_std)
    if not np.isfinite(noise_std) or noise_std < 0.0:
        raise ValueError("chaos_noise_std must be finite and >= 0.")
    min_success = float(min_planning_success_rate)
    if not np.isfinite(min_success) or min_success < 0.0 or min_success > 1.0:
        raise ValueError("min_planning_success_rate must be finite and in [0, 1].")
    max_risk = float(max_mean_risk)
    if not np.isfinite(max_risk) or max_risk < 0.0 or max_risk > 1.0:
        raise ValueError("max_mean_risk must be finite and in [0, 1].")
    max_latency = float(max_p95_latency_ms)
    if not np.isfinite(max_latency) or max_latency <= 0.0:
        raise ValueError("max_p95_latency_ms must be finite and > 0.")

    t0 = time.perf_counter()
    machines = ["NSTX-U", "SPARC"]
    per_machine = [
        _run_machine(
            machines[0],
            seed=seed,
            samples=samples_per_machine,
            chaos_dropout_prob=dropout,
            chaos_noise_std=noise_std,
        ),
        _run_machine(
            machines[1],
            seed=seed + 1,
            samples=samples_per_machine,
            chaos_dropout_prob=dropout,
            chaos_noise_std=noise_std,
        ),
    ]
    passes = bool(
        all(
            float(machine["planning_success_rate"]) >= min_success
            and float(machine["mean_risk"]) <= max_risk
            and float(machine["p95_latency_ms"]) <= max_latency
            for machine in per_machine
        )
    )
    chaos_channels_total = int(sum(int(m["chaos_channels_total"]) for m in per_machine))
    chaos_dropouts_total = int(sum(int(m["chaos_dropouts_total"]) for m in per_machine))
    chaos_noise_injections_total = int(
        sum(int(m["chaos_noise_injections_total"]) for m in per_machine)
    )

    return {
        "seed": int(seed),
        "samples_per_machine": int(samples_per_machine),
        "chaos_dropout_prob": dropout,
        "chaos_noise_std": noise_std,
        "chaos_channels_total": chaos_channels_total,
        "chaos_dropouts_total": chaos_dropouts_total,
        "chaos_dropout_rate": float(chaos_dropouts_total / max(chaos_channels_total, 1)),
        "chaos_noise_injections_total": chaos_noise_injections_total,
        "chaos_noise_injection_rate": float(
            chaos_noise_injections_total / max(chaos_channels_total, 1)
        ),
        "thresholds": {
            "min_planning_success_rate": min_success,
            "max_mean_risk": max_risk,
            "max_p95_latency_ms": max_latency,
        },
        "machines": per_machine,
        "passes_thresholds": passes,
        "runtime_seconds": float(time.perf_counter() - t0),
    }


def generate_report(**kwargs: Any) -> dict[str, Any]:
    """Generate the versioned digital-twin ingestion report."""
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
    if not isinstance(generated_at, str) or not generated_at:
        raise ValueError("generated_at_utc must be a non-empty string")
    payload = report[REPORT_PAYLOAD_KEY]
    if not isinstance(payload, dict):
        raise ValueError(f"{REPORT_PAYLOAD_KEY} must be an object")
    return payload


def render_markdown(report: dict[str, Any]) -> str:
    """Render the current digital-twin ingestion report as Markdown."""
    g = validate_report(report)
    th = g["thresholds"]
    lines = [
        "# NSTX-U/SPARC Digital-Twin Ingestion Validation",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Runtime: `{g['runtime_seconds']:.3f} s`",
        "",
        "## Thresholds",
        "",
        f"- Planning success rate: `>= {th['min_planning_success_rate']:.2f}`",
        f"- Mean risk: `<= {th['max_mean_risk']:.2f}`",
        f"- P95 latency: `<= {th['max_p95_latency_ms']:.1f} ms`",
        "",
        "## Chaos Campaign",
        "",
        f"- Config dropout probability: `{100.0 * g['chaos_dropout_prob']:.2f}%`",
        f"- Config noise std: `{g['chaos_noise_std']:.6f}`",
        f"- Observed dropout rate: `{100.0 * g['chaos_dropout_rate']:.2f}%`",
        f"- Observed noise injection rate: `{100.0 * g['chaos_noise_injection_rate']:.2f}%`",
        "",
    ]

    for machine in g["machines"]:
        lines.extend(
            [
                f"## {machine['machine']}",
                "",
                f"- Plan count: `{machine['plan_count']}`",
                f"- Planning success rate: `{machine['planning_success_rate']:.3f}`",
                f"- Mean risk: `{machine['mean_risk']:.3f}`",
                f"- P95 latency: `{machine['p95_latency_ms']:.4f} ms`",
                f"- Chaos dropout rate: `{100.0 * machine['chaos_dropout_rate']:.2f}%`",
                f"- Chaos noise injection rate: `{100.0 * machine['chaos_noise_injection_rate']:.2f}%`",
                f"- Pass: `{'YES' if machine['passes_thresholds'] else 'NO'}`",
                "",
            ]
        )

    lines.append(f"- Overall pass: `{'YES' if g['passes_thresholds'] else 'NO'}`")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the digital-twin ingestion campaign and write report artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-machine", type=int, default=320)
    parser.add_argument("--chaos-dropout-prob", type=float, default=0.0)
    parser.add_argument("--chaos-noise-std", type=float, default=0.0)
    parser.add_argument("--min-planning-success-rate", type=float, default=0.90)
    parser.add_argument("--max-mean-risk", type=float, default=0.75)
    parser.add_argument("--max-p95-latency-ms", type=float, default=6.0)
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / "nstx_u_sparc_digital_twin_ingestion.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "nstx_u_sparc_digital_twin_ingestion.md"),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    report = generate_report(
        seed=args.seed,
        samples_per_machine=args.samples_per_machine,
        chaos_dropout_prob=args.chaos_dropout_prob,
        chaos_noise_std=args.chaos_noise_std,
        min_planning_success_rate=args.min_planning_success_rate,
        max_mean_risk=args.max_mean_risk,
        max_p95_latency_ms=args.max_p95_latency_ms,
    )
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    g = validate_report(report)
    print("NSTX-U/SPARC digital-twin ingestion validation complete.")
    print(f"passes_thresholds={g['passes_thresholds']}")

    if args.strict and not g["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
