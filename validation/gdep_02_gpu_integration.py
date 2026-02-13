# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — GDEP-02 GPU Integration Validation
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""GDEP-02: GPU-sim bridge validation for multigrid and SNN inference lanes."""

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

from scpn_fusion.core.gpu_runtime import GPURuntimeBridge


def run_campaign(*, trials: int = 64, grid_size: int = 64) -> dict[str, Any]:
    t0 = time.perf_counter()
    bridge = GPURuntimeBridge(seed=42)
    bench = bridge.benchmark_pair(trials=trials, grid_size=grid_size)

    thresholds = {
        "max_gpu_multigrid_p95_ms_est": 2.0,
        "max_gpu_snn_p95_ms_est": 1.0,
        "min_multigrid_speedup_est": 4.0,
        "min_snn_speedup_est": 4.0,
    }
    passes = bool(
        bench["gpu_sim"]["multigrid_p95_ms_est"] <= thresholds["max_gpu_multigrid_p95_ms_est"]
        and bench["gpu_sim"]["snn_p95_ms_est"] <= thresholds["max_gpu_snn_p95_ms_est"]
        and bench["multigrid_speedup_est"] >= thresholds["min_multigrid_speedup_est"]
        and bench["snn_speedup_est"] >= thresholds["min_snn_speedup_est"]
    )

    return {
        "trials": int(trials),
        "grid_size": int(grid_size),
        "benchmarks": bench,
        "thresholds": thresholds,
        "passes_thresholds": passes,
        "runtime_seconds": float(time.perf_counter() - t0),
    }


def generate_report(**kwargs: Any) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gdep_02": run_campaign(**kwargs),
    }


def render_markdown(report: dict[str, Any]) -> str:
    g = report["gdep_02"]
    th = g["thresholds"]
    b = g["benchmarks"]
    lines = [
        "# GDEP-02 GPU Integration Validation",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Runtime: `{g['runtime_seconds']:.3f} s`",
        "",
        "## GPU-Sim P95 Estimate",
        "",
        f"- Multigrid: `{b['gpu_sim']['multigrid_p95_ms_est']:.4f} ms` (threshold `<= {th['max_gpu_multigrid_p95_ms_est']:.1f} ms`)",
        f"- SNN inference: `{b['gpu_sim']['snn_p95_ms_est']:.4f} ms` (threshold `<= {th['max_gpu_snn_p95_ms_est']:.1f} ms`)",
        "",
        "## Estimated Speedups",
        "",
        f"- Multigrid speedup: `{b['multigrid_speedup_est']:.2f}x` (threshold `>= {th['min_multigrid_speedup_est']:.1f}x`)",
        f"- SNN speedup: `{b['snn_speedup_est']:.2f}x` (threshold `>= {th['min_snn_speedup_est']:.1f}x`)",
        f"- Threshold pass: `{'YES' if g['passes_thresholds'] else 'NO'}`",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / "gdep_02_gpu_integration.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "gdep_02_gpu_integration.md"),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    report = generate_report(trials=args.trials, grid_size=args.grid_size)
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    g = report["gdep_02"]
    print("GDEP-02 GPU integration validation complete.")
    print(f"passes_thresholds={g['passes_thresholds']}")

    if args.strict and not g["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
