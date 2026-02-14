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

from scpn_fusion.control.digital_twin_ingest import RealtimeTwinHook, generate_emulated_stream


def _apply_chaos_monkey(
    packet,
    *,
    rng: np.random.Generator,
    dropout_prob: float,
    gaussian_noise_std: float,
):
    from scpn_fusion.control.digital_twin_ingest import TelemetryPacket

    drop = float(np.clip(dropout_prob, 0.0, 1.0))
    sigma = max(float(gaussian_noise_std), 0.0)

    def channel(value: float) -> float:
        if drop > 0.0 and float(rng.random()) < drop:
            return 0.0
        if sigma > 0.0:
            return float(value + rng.normal(0.0, sigma))
        return float(value)

    return TelemetryPacket(
        t_ms=int(packet.t_ms),
        machine=str(packet.machine),
        ip_ma=channel(packet.ip_ma),
        beta_n=channel(packet.beta_n),
        q95=channel(packet.q95),
        density_1e19=max(0.0, channel(packet.density_1e19)),
    )


def _run_machine(
    machine: str,
    seed: int,
    samples: int,
    *,
    chaos_dropout_prob: float = 0.0,
    chaos_noise_std: float = 0.0,
) -> dict[str, Any]:
    stream = generate_emulated_stream(machine, seed=seed, samples=samples, dt_ms=5)
    hook = RealtimeTwinHook(machine, seed=seed)
    rng = np.random.default_rng(int(seed) + 2026)
    plans = []
    for i, packet in enumerate(stream):
        noisy_packet = _apply_chaos_monkey(
            packet,
            rng=rng,
            dropout_prob=chaos_dropout_prob,
            gaussian_noise_std=chaos_noise_std,
        )
        hook.ingest(noisy_packet)
        if i % 8 == 0 and i > 0:
            plans.append(hook.scenario_plan(horizon=24))

    if not plans:
        return {
            "machine": machine,
            "planning_success_rate": 0.0,
            "mean_risk": 1.0,
            "p95_latency_ms": 999.0,
            "plan_count": 0,
            "passes_thresholds": False,
        }

    success_rate = float(np.mean([1.0 if p["passes"] else 0.0 for p in plans]))
    mean_risk = float(np.mean([float(p["mean_risk"]) for p in plans]))
    p95_latency = float(np.percentile([float(p["latency_ms"]) for p in plans], 95))
    return {
        "machine": machine,
        "planning_success_rate": success_rate,
        "mean_risk": mean_risk,
        "p95_latency_ms": p95_latency,
        "plan_count": int(len(plans)),
        "passes_thresholds": bool(success_rate >= 0.90 and p95_latency <= 6.0 and mean_risk <= 0.75),
    }


def run_campaign(
    *,
    seed: int = 42,
    samples_per_machine: int = 320,
    chaos_dropout_prob: float = 0.0,
    chaos_noise_std: float = 0.0,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    machines = ["NSTX-U", "SPARC"]
    per_machine = [
        _run_machine(
            machines[0],
            seed=seed,
            samples=samples_per_machine,
            chaos_dropout_prob=chaos_dropout_prob,
            chaos_noise_std=chaos_noise_std,
        ),
        _run_machine(
            machines[1],
            seed=seed + 1,
            samples=samples_per_machine,
            chaos_dropout_prob=chaos_dropout_prob,
            chaos_noise_std=chaos_noise_std,
        ),
    ]
    passes = bool(all(m["passes_thresholds"] for m in per_machine))

    return {
        "seed": int(seed),
        "samples_per_machine": int(samples_per_machine),
        "chaos_dropout_prob": float(np.clip(chaos_dropout_prob, 0.0, 1.0)),
        "chaos_noise_std": float(max(chaos_noise_std, 0.0)),
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
