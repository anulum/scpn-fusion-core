# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — GAI-02 TORAX Hybrid Validation
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""GAI-02: deterministic TORAX-hybrid realtime loop validation (synthetic v1)."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

from scpn_fusion.control.torax_hybrid_loop import run_nstxu_torax_hybrid_campaign


def generate_report(
    *,
    seed: int = 42,
    episodes: int = 16,
    steps_per_episode: int = 220,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    campaign = run_nstxu_torax_hybrid_campaign(
        seed=seed,
        episodes=episodes,
        steps_per_episode=steps_per_episode,
    )
    elapsed = time.perf_counter() - t0
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": elapsed,
        "gai_02": {
            "seed": int(seed),
            "episodes": campaign.episodes,
            "steps_per_episode": campaign.steps_per_episode,
            "disruption_avoidance_rate": campaign.disruption_avoidance_rate,
            "torax_parity_pct": campaign.torax_parity_pct,
            "p95_loop_latency_ms": campaign.p95_loop_latency_ms,
            "mean_risk": campaign.mean_risk,
            "thresholds": {
                "min_disruption_avoidance_rate": 0.90,
                "min_torax_parity_pct": 95.0,
                "max_p95_loop_latency_ms": 1.0,
            },
            "passes_thresholds": campaign.passes_thresholds,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    g = report["gai_02"]
    th = g["thresholds"]
    lines = [
        "# GAI-02 TORAX Hybrid Validation",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Runtime: `{report['runtime_seconds']:.3f} s`",
        f"- Seed: `{g['seed']}`",
        "",
        "## Metrics",
        "",
        f"- Disruption avoidance rate: `{g['disruption_avoidance_rate']:.3f}` (threshold `>={th['min_disruption_avoidance_rate']:.2f}`)",
        f"- TORAX parity: `{g['torax_parity_pct']:.2f}%` (threshold `>={th['min_torax_parity_pct']:.1f}%`)",
        f"- P95 loop latency: `{g['p95_loop_latency_ms']:.4f} ms` (threshold `<= {th['max_p95_loop_latency_ms']:.1f} ms`)",
        f"- Mean disruption risk: `{g['mean_risk']:.4f}`",
        f"- Threshold pass: `{'YES' if g['passes_thresholds'] else 'NO'}`",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--steps-per-episode", type=int, default=220)
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / "gai_02_torax_hybrid.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "gai_02_torax_hybrid.md"),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    report = generate_report(
        seed=args.seed,
        episodes=args.episodes,
        steps_per_episode=args.steps_per_episode,
    )

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    g = report["gai_02"]
    print("GAI-02 TORAX-hybrid validation complete.")
    print(
        f"avoidance_rate={g['disruption_avoidance_rate']:.3f}, "
        f"torax_parity_pct={g['torax_parity_pct']:.2f}, "
        f"p95_loop_latency_ms={g['p95_loop_latency_ms']:.4f}, "
        f"passes_thresholds={g['passes_thresholds']}"
    )

    if args.strict and not g["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
