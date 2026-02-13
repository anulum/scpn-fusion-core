# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Control Resilience Campaign
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Deterministic fault/noise campaign for disruption-control resilience."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scpn_fusion.control.disruption_predictor import run_fault_noise_campaign


def generate_campaign_report(
    *,
    seed: int = 42,
    episodes: int = 64,
    window: int = 128,
    noise_std: float = 0.03,
    bit_flip_interval: int = 11,
    recovery_window: int = 6,
    recovery_epsilon: float = 0.03,
) -> dict[str, Any]:
    start = time.perf_counter()
    metrics = run_fault_noise_campaign(
        seed=seed,
        episodes=episodes,
        window=window,
        noise_std=noise_std,
        bit_flip_interval=bit_flip_interval,
        recovery_window=recovery_window,
        recovery_epsilon=recovery_epsilon,
    )
    elapsed = time.perf_counter() - start
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": elapsed,
        "campaign": metrics,
    }


def render_markdown(report: dict[str, Any]) -> str:
    c = report["campaign"]
    lines = [
        "# Control Resilience Campaign",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Runtime: `{report['runtime_seconds']:.3f} s`",
        f"- Seed: `{c['seed']}`",
        f"- Episodes: `{c['episodes']}`",
        f"- Window: `{c['window']}`",
        f"- Noise std: `{c['noise_std']}`",
        f"- Bit-flip interval: `{c['bit_flip_interval']}`",
        f"- Fault count: `{c['fault_count']}`",
        "",
        "## Metrics",
        "",
        f"- Mean abs risk error: `{c['mean_abs_risk_error']:.6f}`",
        f"- P95 abs risk error: `{c['p95_abs_risk_error']:.6f}`",
        f"- P95 recovery steps: `{c['recovery_steps_p95']:.3f}`",
        f"- Recovery success rate: `{c['recovery_success_rate']:.3f}`",
        f"- Threshold pass: `{'YES' if c['passes_thresholds'] else 'NO'}`",
        "",
        "## Thresholds",
        "",
        f"- Max mean abs risk error: `{c['thresholds']['max_mean_abs_risk_error']}`",
        f"- Max P95 abs risk error: `{c['thresholds']['max_p95_abs_risk_error']}`",
        f"- Max P95 recovery steps: `{c['thresholds']['max_recovery_steps_p95']}`",
        f"- Min recovery success rate: `{c['thresholds']['min_recovery_success_rate']}`",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--noise-std", type=float, default=0.03)
    parser.add_argument("--bit-flip-interval", type=int, default=11)
    parser.add_argument("--recovery-window", type=int, default=6)
    parser.add_argument("--recovery-epsilon", type=float, default=0.03)
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / "control_resilience_campaign.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "control_resilience_campaign.md"),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when thresholds are not met.",
    )
    args = parser.parse_args(argv)

    report = generate_campaign_report(
        seed=args.seed,
        episodes=args.episodes,
        window=args.window,
        noise_std=args.noise_std,
        bit_flip_interval=args.bit_flip_interval,
        recovery_window=args.recovery_window,
        recovery_epsilon=args.recovery_epsilon,
    )

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    c = report["campaign"]
    print("Control resilience campaign complete.")
    print(
        "mean_abs_risk_error="
        f"{c['mean_abs_risk_error']:.6f}, "
        f"p95_abs_risk_error={c['p95_abs_risk_error']:.6f}, "
        f"recovery_steps_p95={c['recovery_steps_p95']:.3f}, "
        f"recovery_success_rate={c['recovery_success_rate']:.3f}, "
        f"passes_thresholds={c['passes_thresholds']}"
    )

    if args.strict and not c["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
