# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Deterministic CPU/GPU-Sim Runtime Benchmark
"""Benchmark deterministic CPU and GPU-sim multigrid and SNN runtime lanes."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA_VERSION = 2
REPORT_KIND = "deterministic_cpu_gpu_sim_runtime_benchmark"
REPORT_PAYLOAD_KEY = "deterministic_cpu_gpu_sim_runtime_benchmark"

from scpn_fusion.core.gpu_runtime import GPURuntimeBridge


def _positive_finite(value: float, *, name: str) -> float:
    checked = float(value)
    if not np.isfinite(checked) or checked <= 0.0:
        raise ValueError(f"{name} must be finite and > 0.")
    return checked


def run_campaign(
    *,
    trials: int = 64,
    grid_size: int = 64,
    max_gpu_sim_multigrid_p95_ms_est: float = 2.0,
    max_gpu_sim_snn_p95_ms_est: float = 1.0,
    min_multigrid_speedup_est: float = 4.0,
    min_snn_speedup_est: float = 4.0,
) -> dict[str, Any]:
    """Benchmark deterministic CPU/GPU-sim lanes and evaluate acceptance gates.

    Parameters
    ----------
    trials : int
        Number of repeated multigrid and SNN kernel executions per backend.
    grid_size : int
        Width and height of the square multigrid field.
    max_gpu_sim_multigrid_p95_ms_est : float
        Maximum accepted GPU-sim multigrid P95 operation-count estimate.
    max_gpu_sim_snn_p95_ms_est : float
        Maximum accepted GPU-sim SNN P95 operation-count estimate.
    min_multigrid_speedup_est : float
        Minimum accepted estimated multigrid speedup over the CPU lane.
    min_snn_speedup_est : float
        Minimum accepted estimated SNN speedup over the CPU lane.

    Returns
    -------
    dict[str, Any]
        Benchmarks, configured thresholds, runtime and aggregate gate status.

    Raises
    ------
    ValueError
        If a threshold is non-finite or not strictly positive, or if the
        runtime bridge rejects the trial or grid bounds.
    """
    max_multigrid = _positive_finite(
        max_gpu_sim_multigrid_p95_ms_est,
        name="max_gpu_sim_multigrid_p95_ms_est",
    )
    max_snn = _positive_finite(
        max_gpu_sim_snn_p95_ms_est,
        name="max_gpu_sim_snn_p95_ms_est",
    )
    min_multigrid = _positive_finite(
        min_multigrid_speedup_est,
        name="min_multigrid_speedup_est",
    )
    min_snn = _positive_finite(min_snn_speedup_est, name="min_snn_speedup_est")

    t0 = time.perf_counter()
    bridge = GPURuntimeBridge(seed=42)
    bench = bridge.benchmark_pair(trials=trials, grid_size=grid_size)
    gpu_sim = cast(dict[str, float], bench["gpu_sim"])
    multigrid_speedup = cast(float, bench["multigrid_speedup_est"])
    snn_speedup = cast(float, bench["snn_speedup_est"])

    thresholds = {
        "max_gpu_sim_multigrid_p95_ms_est": max_multigrid,
        "max_gpu_sim_snn_p95_ms_est": max_snn,
        "min_multigrid_speedup_est": min_multigrid,
        "min_snn_speedup_est": min_snn,
    }
    passes = bool(
        gpu_sim["multigrid_p95_ms_est"] <= max_multigrid
        and gpu_sim["snn_p95_ms_est"] <= max_snn
        and multigrid_speedup >= min_multigrid
        and snn_speedup >= min_snn
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
    """Generate the versioned deterministic runtime benchmark report."""
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
    """Render the deterministic CPU/GPU-sim benchmark metrics as Markdown."""
    g = validate_report(report)
    th = g["thresholds"]
    b = g["benchmarks"]
    lines = [
        "# Deterministic CPU/GPU-Sim Runtime Benchmark",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Runtime: `{g['runtime_seconds']:.3f} s`",
        "",
        "## GPU-Sim P95 Estimate",
        "",
        f"- Multigrid: `{b['gpu_sim']['multigrid_p95_ms_est']:.4f} ms` (threshold `<= {th['max_gpu_sim_multigrid_p95_ms_est']:.1f} ms`)",
        f"- SNN inference: `{b['gpu_sim']['snn_p95_ms_est']:.4f} ms` (threshold `<= {th['max_gpu_sim_snn_p95_ms_est']:.1f} ms`)",
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
    """Run the deterministic runtime benchmark and write report artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--max-gpu-sim-multigrid-p95-ms-est", type=float, default=2.0)
    parser.add_argument("--max-gpu-sim-snn-p95-ms-est", type=float, default=1.0)
    parser.add_argument("--min-multigrid-speedup-est", type=float, default=4.0)
    parser.add_argument("--min-snn-speedup-est", type=float, default=4.0)
    parser.add_argument(
        "--output-json",
        default=str(
            ROOT / "validation" / "reports" / "deterministic_cpu_gpu_sim_runtime_benchmark.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        default=str(
            ROOT / "validation" / "reports" / "deterministic_cpu_gpu_sim_runtime_benchmark.md"
        ),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    report = generate_report(
        trials=args.trials,
        grid_size=args.grid_size,
        max_gpu_sim_multigrid_p95_ms_est=args.max_gpu_sim_multigrid_p95_ms_est,
        max_gpu_sim_snn_p95_ms_est=args.max_gpu_sim_snn_p95_ms_est,
        min_multigrid_speedup_est=args.min_multigrid_speedup_est,
        min_snn_speedup_est=args.min_snn_speedup_est,
    )
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    g = validate_report(report)
    print("Deterministic CPU/GPU-sim runtime benchmark complete.")
    print(f"passes_thresholds={g['passes_thresholds']}")

    if args.strict and not g["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
