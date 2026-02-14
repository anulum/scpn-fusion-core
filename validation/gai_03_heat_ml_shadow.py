# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — GAI-03 HEAT-ML Shadow Validation
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""GAI-03: HEAT-ML magnetic-shadow surrogate validation for MVR scanner."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

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
) -> dict[str, Any]:
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

    rmse_threshold_pct = 10.0
    infer_threshold_s = 1.0
    reduction_threshold_pct = 8.0
    passes = bool(
        rmse_pct_val <= rmse_threshold_pct
        and inference_seconds <= infer_threshold_s
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
        "inference_threshold_seconds": infer_threshold_s,
        "mean_divertor_reduction_pct": float(mean_reduction_pct),
        "reduction_threshold_pct": reduction_threshold_pct,
        "passes_thresholds": passes,
        "runtime_seconds": float(time.perf_counter() - t0),
    }


def generate_report(**kwargs: Any) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gai_03": run_campaign(**kwargs),
    }


def render_markdown(report: dict[str, Any]) -> str:
    g = report["gai_03"]
    lines = [
        "# GAI-03 HEAT-ML Shadow Validation",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Runtime: `{g['runtime_seconds']:.3f} s`",
        f"- Seed: `{g['seed']}`",
        "",
        "## Metrics",
        "",
        f"- RMSE (%): `{g['rmse_pct']:.3f}%` (threshold `<= {g['rmse_threshold_pct']:.1f}%`)",
        f"- Inference time (200k samples): `{g['inference_seconds_200k']:.4f} s` (threshold `<= {g['inference_threshold_seconds']:.1f} s`)",
        f"- Mean divertor load reduction: `{g['mean_divertor_reduction_pct']:.2f}%` (threshold `>= {g['reduction_threshold_pct']:.1f}%`)",
        f"- Threshold pass: `{'YES' if g['passes_thresholds'] else 'NO'}`",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--eval-samples", type=int, default=512)
    parser.add_argument("--scan-samples", type=int, default=600)
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / "gai_03_heat_ml_shadow.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "gai_03_heat_ml_shadow.md"),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    report = generate_report(
        seed=args.seed,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        scan_samples=args.scan_samples,
    )

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    g = report["gai_03"]
    print("GAI-03 HEAT-ML validation complete.")
    print(
        f"rmse_pct={g['rmse_pct']:.3f}, "
        f"inference_seconds_200k={g['inference_seconds_200k']:.4f}, "
        f"mean_divertor_reduction_pct={g['mean_divertor_reduction_pct']:.2f}, "
        f"passes_thresholds={g['passes_thresholds']}"
    )

    if args.strict and not g["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
