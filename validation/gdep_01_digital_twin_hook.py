# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — GDEP-01 Digital Twin Hook Validation
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""GDEP-01: realtime digital-twin ingest + SNN scenario planning validation."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

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
) -> dict[str, Any]:
    samples_per_machine = int(samples_per_machine)
    if samples_per_machine < 32:
        raise ValueError("samples_per_machine must be >= 32.")
    dropout = float(chaos_dropout_prob)
    if not np.isfinite(dropout) or dropout < 0.0 or dropout > 1.0:
        raise ValueError("chaos_dropout_prob must be finite and in [0, 1].")
    noise_std = float(chaos_noise_std)
    if not np.isfinite(noise_std) or noise_std < 0.0:
        raise ValueError("chaos_noise_std must be finite and >= 0.")

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
    passes = bool(all(m["passes_thresholds"] for m in per_machine))
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
            "min_planning_success_rate": 0.90,
            "max_mean_risk": 0.75,
            "max_p95_latency_ms": 6.0,
        },
        "machines": per_machine,
        "passes_thresholds": passes,
        "runtime_seconds": float(time.perf_counter() - t0),
    }


def generate_report(**kwargs: Any) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gdep_01": run_campaign(**kwargs),
    }


def render_markdown(report: dict[str, Any]) -> str:
    g = report["gdep_01"]
    th = g["thresholds"]
    lines = [
        "# GDEP-01 Digital Twin Hook Validation",
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-machine", type=int, default=320)
    parser.add_argument("--chaos-dropout-prob", type=float, default=0.0)
    parser.add_argument("--chaos-noise-std", type=float, default=0.0)
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / "gdep_01_digital_twin_hook.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "gdep_01_digital_twin_hook.md"),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    report = generate_report(
        seed=args.seed,
        samples_per_machine=args.samples_per_machine,
        chaos_dropout_prob=args.chaos_dropout_prob,
        chaos_noise_std=args.chaos_noise_std,
    )
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    g = report["gdep_01"]
    print("GDEP-01 digital twin hook validation complete.")
    print(f"passes_thresholds={g['passes_thresholds']}")

    if args.strict and not g["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
