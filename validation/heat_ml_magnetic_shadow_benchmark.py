# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — HEAT-ML Magnetic-Shadow Benchmark
"""HEAT-ML magnetic-shadow surrogate benchmark for the design scanner."""

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
REPORT_KIND = "heat_ml_magnetic_shadow_benchmark"
REPORT_PAYLOAD_KEY = "heat_ml_magnetic_shadow_benchmark"

from scpn_fusion.core.global_design_scanner import GlobalDesignExplorer
from scpn_fusion.core.heat_ml_shadow_surrogate import (
    HeatMLShadowSurrogate,
    benchmark_inference_seconds,
    generate_shadow_dataset,
    rmse_percent,
)


def run_campaign(
    *,
    seed: int = 42,
    train_samples: int = 2048,
    eval_samples: int = 512,
    scan_samples: int = 600,
    rmse_threshold_pct: float = 10.0,
    inference_threshold_seconds: float = 1.0,
    reduction_threshold_pct: float = 8.0,
) -> dict[str, Any]:
    """Run the HEAT-ML surrogate campaign and return inference/divertor metrics."""
    t0 = time.perf_counter()

    train = generate_shadow_dataset(seed=seed, samples=train_samples)
    eval_set = generate_shadow_dataset(seed=seed + 1, samples=eval_samples)

    model = HeatMLShadowSurrogate(ridge=1e-4)
    model.fit(train.features, train.shadow_fraction)
    pred = model.predict_shadow_fraction(eval_set.features)
    rmse_pct_val = rmse_percent(eval_set.shadow_fraction, pred)
    inference_seconds = benchmark_inference_seconds(model, samples=200_000)

    explorer = GlobalDesignExplorer("dummy")
    df = explorer.run_scan(n_samples=scan_samples, seed=seed)
    baseline = np.maximum(df["Div_Load_Baseline"].to_numpy(dtype=float), 1e-9)
    optimized = df["Div_Load_Optimized"].to_numpy(dtype=float)
    reduction_pct = 100.0 * (baseline - optimized) / baseline
    mean_reduction_pct = float(np.mean(reduction_pct))

    passes = bool(
        rmse_pct_val <= rmse_threshold_pct
        and inference_seconds <= inference_threshold_seconds
        and mean_reduction_pct >= reduction_threshold_pct
    )

    return {
        "seed": int(seed),
        "train_samples": int(train_samples),
        "eval_samples": int(eval_samples),
        "scan_samples": int(scan_samples),
        "rmse_pct": float(rmse_pct_val),
        "rmse_threshold_pct": rmse_threshold_pct,
        "inference_seconds_200k": float(inference_seconds),
        "inference_threshold_seconds": inference_threshold_seconds,
        "mean_divertor_reduction_pct": float(mean_reduction_pct),
        "reduction_threshold_pct": reduction_threshold_pct,
        "passes_thresholds": passes,
        "runtime_seconds": float(time.perf_counter() - t0),
    }


def generate_report(**kwargs: Any) -> dict[str, Any]:
    """Generate the versioned HEAT-ML benchmark report."""
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
    """Render the current HEAT-ML benchmark report as Markdown."""
    benchmark = validate_report(report)
    lines = [
        "# HEAT-ML Magnetic-Shadow Design-Scanner Benchmark",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Runtime: `{benchmark['runtime_seconds']:.3f} s`",
        f"- Seed: `{benchmark['seed']}`",
        "",
        "## Metrics",
        "",
        f"- RMSE (%): `{benchmark['rmse_pct']:.3f}%` "
        f"(threshold `<= {benchmark['rmse_threshold_pct']:.1f}%`)",
        f"- Inference time (200k samples): `{benchmark['inference_seconds_200k']:.4f} s` "
        f"(threshold `<= {benchmark['inference_threshold_seconds']:.1f} s`)",
        f"- Mean divertor load reduction: `{benchmark['mean_divertor_reduction_pct']:.2f}%` "
        f"(threshold `>= {benchmark['reduction_threshold_pct']:.1f}%`)",
        f"- Threshold pass: `{'YES' if benchmark['passes_thresholds'] else 'NO'}`",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the HEAT-ML benchmark CLI and write current report artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--eval-samples", type=int, default=512)
    parser.add_argument("--scan-samples", type=int, default=600)
    parser.add_argument("--rmse-threshold-pct", type=float, default=10.0)
    parser.add_argument("--inference-threshold-seconds", type=float, default=1.0)
    parser.add_argument("--reduction-threshold-pct", type=float, default=8.0)
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / "heat_ml_magnetic_shadow_benchmark.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "heat_ml_magnetic_shadow_benchmark.md"),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    report = generate_report(
        seed=args.seed,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        scan_samples=args.scan_samples,
        rmse_threshold_pct=args.rmse_threshold_pct,
        inference_threshold_seconds=args.inference_threshold_seconds,
        reduction_threshold_pct=args.reduction_threshold_pct,
    )

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    benchmark = validate_report(report)
    print("HEAT-ML magnetic-shadow design-scanner benchmark complete.")
    print(
        f"rmse_pct={benchmark['rmse_pct']:.3f}, "
        f"inference_seconds_200k={benchmark['inference_seconds_200k']:.4f}, "
        f"mean_divertor_reduction_pct={benchmark['mean_divertor_reduction_pct']:.2f}, "
        f"passes_thresholds={benchmark['passes_thresholds']}"
    )

    if args.strict and not benchmark["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
