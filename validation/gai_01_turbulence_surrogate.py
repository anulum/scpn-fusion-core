# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — GAI-01 Turbulence Surrogate Validation
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""GAI-01: deterministic GyroSwin-like surrogate benchmark (synthetic v1)."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

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
) -> dict[str, Any]:
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
    rmse_threshold_pct = 10.0
    speedup_threshold = 1000.0
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
        "passes_thresholds": bool(err_pct <= rmse_threshold_pct and speed.speedup >= speedup_threshold),
        "runtime_seconds": float(elapsed),
    }


def generate_report(**kwargs: Any) -> dict[str, Any]:
    campaign = run_campaign(**kwargs)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gai_01": campaign,
    }


def render_markdown(report: dict[str, Any]) -> str:
    g = report["gai_01"]
    lines = [
        "# GAI-01 Turbulence Surrogate Validation",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Runtime: `{g['runtime_seconds']:.3f} s`",
        f"- Seed: `{g['seed']}`",
        "",
        "## Metrics",
        "",
        f"- RMSE (% of mean target): `{g['rmse_pct']:.3f}%` (threshold `{g['rmse_threshold_pct']:.1f}%`)",
        f"- Speedup vs GENE-like proxy: `{g['speedup_vs_gene_proxy']:.1f}x` (threshold `{g['speedup_threshold']:.1f}x`)",
        f"- Baseline time/sample: `{g['gene_proxy_s_per_sample']:.3e} s`",
        f"- Surrogate time/sample: `{g['surrogate_s_per_sample']:.3e} s`",
        f"- Threshold pass: `{'YES' if g['passes_thresholds'] else 'NO'}`",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--eval-samples", type=int, default=384)
    parser.add_argument("--benchmark-samples", type=int, default=96)
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / "gai_01_turbulence_surrogate.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "gai_01_turbulence_surrogate.md"),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    report = generate_report(
        seed=args.seed,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        benchmark_samples=args.benchmark_samples,
    )

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    g = report["gai_01"]
    print("GAI-01 turbulence surrogate validation complete.")
    print(
        f"rmse_pct={g['rmse_pct']:.3f}, "
        f"speedup_vs_gene_proxy={g['speedup_vs_gene_proxy']:.1f}x, "
        f"passes_thresholds={g['passes_thresholds']}"
    )

    if args.strict and not g["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
