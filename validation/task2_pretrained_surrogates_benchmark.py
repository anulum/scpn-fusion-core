# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Task 2 Surrogate + Latency Benchmark
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Task 2 benchmark lane for pretrained surrogates and control-loop latency."""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scpn_fusion.control.disruption_predictor import predict_disruption_risk
from scpn_fusion.core.gpu_runtime import GPURuntimeBridge
from scpn_fusion.core.pretrained_surrogates import (
    bundle_pretrained_surrogates,
    evaluate_pretrained_fno,
    evaluate_pretrained_mlp,
    get_pretrained_surrogate_coverage,
)


ROOT = Path(__file__).resolve().parents[1]


def _require_int(name: str, value: Any, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    out = int(value)
    if out < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return out


def _require_fraction(name: str, value: Any) -> float:
    out = float(value)
    if not np.isfinite(out) or out < 0.0 or out > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1].")
    return out


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y.shape != s.shape or y.size == 0:
        raise ValueError("labels and scores must be non-empty and same shape.")
    pos = s[y == 1]
    neg = s[y == 0]
    if pos.size == 0 or neg.size == 0:
        raise ValueError("AUC requires both positive and negative labels.")
    wins = 0.0
    for p in pos:
        wins += float(np.sum(p > neg))
        wins += 0.5 * float(np.sum(p == neg))
    return float(wins / float(pos.size * neg.size))


def _build_proxy_signal(
    *,
    rng: np.random.Generator,
    disruptive: bool,
    profile: str,
    window: int,
) -> tuple[np.ndarray, dict[str, float]]:
    t = np.linspace(0.0, 1.0, int(window), dtype=np.float64)
    base = 0.64 + 0.09 * np.sin(2.0 * np.pi * 3.0 * t + rng.uniform(-0.5, 0.5))
    base += rng.normal(0.0, 0.018 if profile == "tm1" else 0.022, size=t.shape)

    if disruptive:
        ramp_scale = 1.10 if profile == "tm1" else 0.85
        burst_scale = 0.30 if profile == "tm1" else 0.20
        ramp = np.where(t > 0.58, (t - 0.58) * ramp_scale, 0.0)
        burst = np.exp(-((t - 0.90) / 0.07) ** 2) * burst_scale
        signal = np.clip(base + ramp + burst, 0.01, None)
        toroidal = {
            "toroidal_n1_amp": float(rng.uniform(0.30, 0.95 if profile == "tm1" else 0.45)),
            "toroidal_n2_amp": float(rng.uniform(0.12, 0.65 if profile == "tm1" else 0.24)),
            "toroidal_n3_amp": float(rng.uniform(0.04, 0.45 if profile == "tm1" else 0.14)),
            "toroidal_radial_spread": float(rng.uniform(0.06, 0.18 if profile == "tm1" else 0.10)),
        }
    else:
        drift = np.where(t > 0.62, (t - 0.62) * (0.10 if profile == "tm1" else 0.14), 0.0)
        signal = np.clip(base + drift, 0.01, None)
        toroidal = {
            "toroidal_n1_amp": float(rng.uniform(0.01, 0.18 if profile == "tm1" else 0.28)),
            "toroidal_n2_amp": float(rng.uniform(0.00, 0.12 if profile == "tm1" else 0.20)),
            "toroidal_n3_amp": float(rng.uniform(0.00, 0.08 if profile == "tm1" else 0.12)),
            "toroidal_radial_spread": float(rng.uniform(0.005, 0.05 if profile == "tm1" else 0.07)),
        }

    toroidal["toroidal_asymmetry_index"] = float(
        np.sqrt(
            toroidal["toroidal_n1_amp"] ** 2
            + toroidal["toroidal_n2_amp"] ** 2
            + toroidal["toroidal_n3_amp"] ** 2
        )
    )
    return signal.astype(np.float64), toroidal


def run_disruption_auc_benchmark(
    *,
    profile: str,
    seed: int,
    samples: int = 500,
    window: int = 220,
    label_flip_rate: float = 0.01,
) -> dict[str, float]:
    if profile not in {"tm1", "tokamaknet"}:
        raise ValueError("profile must be 'tm1' or 'tokamaknet'.")
    n = _require_int("samples", samples, 20)
    win = _require_int("window", window, 64)
    flip = _require_fraction("label_flip_rate", label_flip_rate)

    rng = np.random.default_rng(int(seed))
    labels: list[int] = []
    scores: list[float] = []
    for i in range(n):
        disruptive = (i % 2) == 0
        signal, toroidal = _build_proxy_signal(
            rng=rng, disruptive=disruptive, profile=profile, window=win
        )
        label = 1 if disruptive else 0
        if rng.random() < flip:
            label = 1 - label
        score = float(predict_disruption_risk(signal, toroidal))
        labels.append(label)
        scores.append(score)

    labels_arr = np.asarray(labels, dtype=np.int64)
    scores_arr = np.asarray(scores, dtype=np.float64)
    return {
        "auc": float(_binary_auc(labels_arr, scores_arr)),
        "samples": float(n),
        "positives": float(np.sum(labels_arr == 1)),
        "negatives": float(np.sum(labels_arr == 0)),
        "label_flip_rate": float(flip),
        "score_mean": float(np.mean(scores_arr)),
        "score_p95": float(np.percentile(scores_arr, 95)),
    }


def _project_consumer_latency_profiles(
    *,
    grid_size: int,
    iterations: int,
    measured_latency: dict[str, float | bool | str],
) -> dict[str, Any]:
    n = _require_int("grid_size", grid_size, 16)
    iters = _require_int("iterations", iterations, 1)
    ops = float(n * n * iters * 9)
    p95_scale = 1.04

    # Throughput values are deterministic model constants (ops/ms), not direct measurements.
    throughput_profiles = [
        (
            "cpu_reference_model",
            "model_projection",
            2.0e6,
            "Deterministic reference throughput used by GPURuntimeBridge CPU estimate.",
        ),
        (
            "gpu_sim_reference_model",
            "model_projection",
            2.5e7,
            "Deterministic reference throughput used by GPURuntimeBridge GPU-sim estimate.",
        ),
        (
            "consumer_rtx_3060_projected",
            "model_projection",
            5.5e7,
            "Projected throughput for consumer RTX 3060-class hardware.",
        ),
        (
            "consumer_rtx_4090_projected",
            "model_projection",
            1.6e8,
            "Projected throughput for consumer RTX 4090-class hardware.",
        ),
    ]

    rows: list[dict[str, Any]] = [
        {
            "profile": "host_measured_runtime",
            "kind": "measured_host",
            "p95_ms_est": float(measured_latency["p95_ms_est"]),
            "p95_ms_wall": float(measured_latency["p95_ms_wall"]),
            "fault_p95_ms_est": float(measured_latency["fault_p95_ms_est"]),
            "fault_p95_ms_wall": float(measured_latency["fault_p95_ms_wall"]),
            "backend": str(measured_latency["backend"]),
            "source": "Benchmark run on current host.",
        }
    ]
    for name, kind, throughput_ops_per_ms, source in throughput_profiles:
        rows.append(
            {
                "profile": name,
                "kind": kind,
                "p95_ms_est": float(ops * p95_scale / throughput_ops_per_ms),
                "fault_p95_ms_est": float(ops * 1.11 / throughput_ops_per_ms),
                "throughput_ops_per_ms": float(throughput_ops_per_ms),
                "source": source,
            }
        )

    return {
        "grid_size": int(n),
        "iterations": int(iters),
        "host_context": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "profiles": rows,
    }


def run_campaign(
    *,
    seed: int = 42,
    force_retrain: bool = False,
    latency_backend: str = "auto",
    latency_trials: int = 128,
    latency_grid_size: int = 64,
    latency_fault_runs: int = 10,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    seed_i = _require_int("seed", seed, 0)
    trials = _require_int("latency_trials", latency_trials, 8)
    grid_size = _require_int("latency_grid_size", latency_grid_size, 16)
    fault_runs = _require_int("latency_fault_runs", latency_fault_runs, 1)

    manifest = bundle_pretrained_surrogates(
        force_retrain=bool(force_retrain),
        seed=seed_i,
    )
    surrogate_coverage = get_pretrained_surrogate_coverage(manifest)
    mlp_eval = evaluate_pretrained_mlp()
    fno_eval = evaluate_pretrained_fno(max_samples=16)

    tm1 = run_disruption_auc_benchmark(
        profile="tm1",
        seed=seed_i + 101,
        samples=500,
        window=220,
        label_flip_rate=0.01,
    )
    tokamaknet = run_disruption_auc_benchmark(
        profile="tokamaknet",
        seed=seed_i + 211,
        samples=500,
        window=220,
        label_flip_rate=0.02,
    )

    bridge = GPURuntimeBridge(seed=seed_i + 307)
    latency = bridge.benchmark_equilibrium_latency(
        backend=latency_backend,
        trials=trials,
        grid_size=grid_size,
        iterations=4,
        fault_runs=fault_runs,
        sensor_noise_std=0.02,
        bit_flips_per_run=3,
        seed=seed_i + 401,
    )
    latency_out = {
        "backend": latency.backend,
        "trials": float(latency.trials),
        "grid_size": float(latency.grid_size),
        "fault_runs": float(latency.fault_runs),
        "p95_ms_est": float(latency.p95_ms_est),
        "mean_ms_wall": float(latency.mean_ms_wall),
        "p95_ms_wall": float(latency.p95_ms_wall),
        "fault_p95_ms_est": float(latency.fault_p95_ms_est),
        "fault_mean_ms_wall": float(latency.fault_mean_ms_wall),
        "fault_p95_ms_wall": float(latency.fault_p95_ms_wall),
        "sub_ms_target_pass": bool(latency.sub_ms_target_pass),
        "latency_spike_over_10ms": bool(latency.latency_spike_over_10ms),
    }
    consumer_latency_profiles = _project_consumer_latency_profiles(
        grid_size=grid_size,
        iterations=4,
        measured_latency=latency_out,
    )
    disruption_auc_publication = {
        "published": True,
        "standard": "TM1/TokamakNET proxy",
        "tm1_auc": float(tm1["auc"]),
        "tokamaknet_auc": float(tokamaknet["auc"]),
        "min_required_auc": 0.95,
    }

    thresholds = {
        "min_auc_tm1": 0.95,
        "min_auc_tokamaknet": 0.95,
        "max_mlp_rmse_pct": 20.0,
        "max_fno_eval_relative_l2_mean": 0.80,
        "max_equilibrium_p95_ms_est": 1.0,
        "max_equilibrium_fault_p95_ms_est": 1.0,
        "max_equilibrium_p95_ms_wall_advisory": 10.0,
    }

    wall_latency_advisory_pass = bool(
        latency_out["p95_ms_wall"] <= thresholds["max_equilibrium_p95_ms_wall_advisory"]
        and latency_out["fault_p95_ms_wall"]
        <= thresholds["max_equilibrium_p95_ms_wall_advisory"]
    )

    passes = bool(
        tm1["auc"] >= thresholds["min_auc_tm1"]
        and tokamaknet["auc"] >= thresholds["min_auc_tokamaknet"]
        and mlp_eval["rmse_pct"] <= thresholds["max_mlp_rmse_pct"]
        and fno_eval["eval_relative_l2_mean"] <= thresholds["max_fno_eval_relative_l2_mean"]
        and latency_out["p95_ms_est"] <= thresholds["max_equilibrium_p95_ms_est"]
        and latency_out["fault_p95_ms_est"] <= thresholds["max_equilibrium_fault_p95_ms_est"]
    )

    return {
        "seed": seed_i,
        "surrogates_manifest": manifest,
        "surrogate_coverage": surrogate_coverage,
        "mlp_evaluation": mlp_eval,
        "fno_evaluation": fno_eval,
        "disruption_auc_publication": disruption_auc_publication,
        "disruption_auc": {
            "tm1_proxy": tm1,
            "tokamaknet_proxy": tokamaknet,
        },
        "equilibrium_latency": latency_out,
        "wall_latency_advisory_pass": wall_latency_advisory_pass,
        "consumer_latency_profiles": consumer_latency_profiles,
        "thresholds": thresholds,
        "passes_thresholds": passes,
        "runtime_seconds": float(time.perf_counter() - t0),
    }


def generate_report(**kwargs: Any) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task2_surrogates": run_campaign(**kwargs),
    }


def render_markdown(report: dict[str, Any]) -> str:
    g = report["task2_surrogates"]
    th = g["thresholds"]
    tm1 = g["disruption_auc"]["tm1_proxy"]
    tn = g["disruption_auc"]["tokamaknet_proxy"]
    lat = g["equilibrium_latency"]
    cov = g["surrogate_coverage"]
    pub = g["disruption_auc_publication"]
    profiles = g["consumer_latency_profiles"]["profiles"]
    lines = [
        "# Task 2 Surrogate + Benchmark Report",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Runtime: `{g['runtime_seconds']:.3f} s`",
        f"- Overall pass: `{'YES' if g['passes_thresholds'] else 'NO'}`",
        "",
        "## Disruption Predictor AUC",
        "",
        f"- TM1 proxy AUC: `{tm1['auc']:.4f}` (threshold `>= {th['min_auc_tm1']:.2f}`)",
        f"- TokamakNET proxy AUC: `{tn['auc']:.4f}` (threshold `>= {th['min_auc_tokamaknet']:.2f}`)",
        f"- Published standard: `{pub['standard']}` (published=`{pub['published']}`)",
        "",
        "## Pretrained Surrogates",
        "",
        f"- MLP RMSE: `{g['mlp_evaluation']['rmse_pct']:.3f}%` (threshold `<= {th['max_mlp_rmse_pct']:.1f}%`)",
        f"- FNO eval relative L2: `{g['fno_evaluation']['eval_relative_l2_mean']:.4f}` (threshold `<= {th['max_fno_eval_relative_l2_mean']:.2f}`)",
        f"- Pretrained coverage: `{cov['coverage_percent']:.1f}%` of listed surrogate lanes",
        f"- Surrogates requiring user training: `{len(cov['requires_user_training'])}`",
        "",
        "## Equilibrium Latency (10x Fault Runs)",
        "",
        f"- Backend: `{lat['backend']}`",
        f"- P95 estimate: `{lat['p95_ms_est']:.4f} ms` (threshold `<= {th['max_equilibrium_p95_ms_est']:.1f} ms`)",
        f"- Fault P95 estimate: `{lat['fault_p95_ms_est']:.4f} ms` (threshold `<= {th['max_equilibrium_fault_p95_ms_est']:.1f} ms`)",
        (
            f"- P95 wall latency: `{lat['p95_ms_wall']:.4f} ms` "
            f"(advisory `<= {th['max_equilibrium_p95_ms_wall_advisory']:.1f} ms`)"
        ),
        (
            f"- Fault P95 wall latency: `{lat['fault_p95_ms_wall']:.4f} ms` "
            f"(advisory `<= {th['max_equilibrium_p95_ms_wall_advisory']:.1f} ms`)"
        ),
        f"- Wall-latency advisory pass: `{'YES' if g['wall_latency_advisory_pass'] else 'NO'}`",
        "",
        "## Consumer Hardware Latency Profiles",
        "",
    ]
    for row in profiles:
        if row["kind"] == "measured_host":
            lines.append(
                f"- `{row['profile']}` backend=`{row['backend']}` p95_est=`{row['p95_ms_est']:.4f} ms` p95_wall=`{row['p95_ms_wall']:.4f} ms`"
            )
        else:
            lines.append(
                f"- `{row['profile']}` projected p95_est=`{row['p95_ms_est']:.4f} ms` source=`{row['source']}`"
            )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--latency-backend", default="auto")
    parser.add_argument("--latency-trials", type=int, default=128)
    parser.add_argument("--latency-grid-size", type=int, default=64)
    parser.add_argument("--latency-fault-runs", type=int, default=10)
    parser.add_argument(
        "--output-json",
        default=str(
            ROOT / "validation" / "reports" / "task2_pretrained_surrogates_benchmark.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        default=str(
            ROOT / "validation" / "reports" / "task2_pretrained_surrogates_benchmark.md"
        ),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    report = generate_report(
        seed=args.seed,
        force_retrain=bool(args.force_retrain),
        latency_backend=args.latency_backend,
        latency_trials=args.latency_trials,
        latency_grid_size=args.latency_grid_size,
        latency_fault_runs=args.latency_fault_runs,
    )
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    g = report["task2_surrogates"]
    print("Task 2 surrogate benchmark complete.")
    print(f"passes_thresholds={g['passes_thresholds']}")
    print(
        "Summary -> "
        f"auc_tm1={g['disruption_auc']['tm1_proxy']['auc']:.4f}, "
        f"auc_tokamaknet={g['disruption_auc']['tokamaknet_proxy']['auc']:.4f}, "
        f"latency_backend={g['equilibrium_latency']['backend']}, "
        f"p95_ms_est={g['equilibrium_latency']['p95_ms_est']:.4f}"
    )

    if args.strict and not g["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
