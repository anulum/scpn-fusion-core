# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — GyroSwin-Like Turbulence Surrogate Benchmark
"""Deterministic GyroSwin-like turbulence surrogate benchmark."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA_VERSION = 2
REPORT_KIND = "gyro_swin_turbulence_surrogate_benchmark"
REPORT_PAYLOAD_KEY = "turbulence_surrogate_benchmark"

from scpn_fusion.core.gyro_swin_surrogate import (
    GyroSwinLikeSurrogate,
    benchmark_speedup,
    generate_synthetic_gyrokinetic_dataset,
    rmse_percent,
)


def run_campaign(
    *,
    seed: int = 42,
    train_samples: int = 2048,
    eval_samples: int = 384,
    benchmark_samples: int = 96,
    rmse_threshold_pct: float = 10.0,
    speedup_threshold: float = 1000.0,
) -> dict[str, Any]:
    """Run the synthetic turbulence surrogate campaign and return RMSE/speedup metrics."""
    t0 = time.perf_counter()

    train = generate_synthetic_gyrokinetic_dataset(seed=seed, samples=train_samples)
    eval_set = generate_synthetic_gyrokinetic_dataset(seed=seed + 1, samples=eval_samples)

    surrogate = GyroSwinLikeSurrogate(hidden_dim=64, ridge=5e-4, seed=seed)
    surrogate.fit(train.features, train.chi_i)
    pred = surrogate.predict(eval_set.features)
    err_pct = rmse_percent(eval_set.chi_i, pred)

    bench_count = max(32, min(int(benchmark_samples), eval_set.features.shape[0]))
    speed = benchmark_speedup(eval_set.features[:bench_count], surrogate)

    elapsed = time.perf_counter() - t0
    return {
        "seed": int(seed),
        "train_samples": int(train_samples),
        "eval_samples": int(eval_samples),
        "benchmark_samples": int(bench_count),
        "rmse_pct": float(err_pct),
        "rmse_threshold_pct": rmse_threshold_pct,
        "speedup_vs_gene_proxy": float(speed.speedup),
        "speedup_threshold": speedup_threshold,
        "gene_proxy_s_per_sample": float(speed.gene_proxy_s_per_sample),
        "surrogate_s_per_sample": float(speed.surrogate_s_per_sample),
        "passes_thresholds": bool(
            err_pct <= rmse_threshold_pct and speed.speedup >= speedup_threshold
        ),
        "runtime_seconds": float(elapsed),
    }


def generate_report(**kwargs: Any) -> dict[str, Any]:
    """Generate the versioned benchmark report payload from a campaign run."""
    campaign = run_campaign(**kwargs)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_kind": REPORT_KIND,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        REPORT_PAYLOAD_KEY: campaign,
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
    """Render benchmark results from the current report contract as Markdown."""
    benchmark = validate_report(report)
    lines = [
        "# GyroSwin-Like Turbulence Surrogate Benchmark",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Runtime: `{benchmark['runtime_seconds']:.3f} s`",
        f"- Seed: `{benchmark['seed']}`",
        "",
        "## Metrics",
        "",
        f"- RMSE (% of mean target): `{benchmark['rmse_pct']:.3f}%` "
        f"(threshold `{benchmark['rmse_threshold_pct']:.1f}%`)",
        f"- Speedup vs GENE-like proxy: `{benchmark['speedup_vs_gene_proxy']:.1f}x` "
        f"(threshold `{benchmark['speedup_threshold']:.1f}x`)",
        f"- Baseline time/sample: `{benchmark['gene_proxy_s_per_sample']:.3e} s`",
        f"- Surrogate time/sample: `{benchmark['surrogate_s_per_sample']:.3e} s`",
        f"- Threshold pass: `{'YES' if benchmark['passes_thresholds'] else 'NO'}`",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark CLI and write versioned JSON and Markdown outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--eval-samples", type=int, default=384)
    parser.add_argument("--benchmark-samples", type=int, default=96)
    parser.add_argument("--rmse-threshold-pct", type=float, default=10.0)
    parser.add_argument("--speedup-threshold", type=float, default=1000.0)
    parser.add_argument(
        "--output-json",
        default=str(
            ROOT / "validation" / "reports" / "gyro_swin_turbulence_surrogate_benchmark.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        default=str(
            ROOT / "validation" / "reports" / "gyro_swin_turbulence_surrogate_benchmark.md"
        ),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    report = generate_report(
        seed=args.seed,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        benchmark_samples=args.benchmark_samples,
        rmse_threshold_pct=args.rmse_threshold_pct,
        speedup_threshold=args.speedup_threshold,
    )

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    benchmark = validate_report(report)
    print("GyroSwin-like turbulence surrogate benchmark complete.")
    print(
        f"rmse_pct={benchmark['rmse_pct']:.3f}, "
        f"speedup_vs_gene_proxy={benchmark['speedup_vs_gene_proxy']:.1f}x, "
        f"passes_thresholds={benchmark['passes_thresholds']}"
    )

    if args.strict and not benchmark["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
