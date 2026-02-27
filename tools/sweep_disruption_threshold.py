#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Disruption Threshold & Bias Sweep
# Sweeps risk_threshold and sigmoid bias to optimise recall vs FPR
# on the 16 NPZ reference disruption shots.
# ──────────────────────────────────────────────────────────────────────
"""
Usage:
    python tools/sweep_disruption_threshold.py

Outputs:
    artifacts/disruption_threshold_sweep.json   — full grid results + optimum
    artifacts/disruption_roc_curve.json         — ROC data (threshold-only at best bias)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scpn_fusion.control.disruption_predictor import build_disruption_feature_vector

# ── Configuration ─────────────────────────────────────────────────────

DISRUPTION_DIR = ROOT / "validation" / "reference_data" / "diiid" / "disruption_shots"
ARTIFACTS_DIR = ROOT / "artifacts"

# Sweep ranges
THRESHOLD_MIN = 0.05
THRESHOLD_MAX = 0.99
THRESHOLD_STEP = 0.01

BIAS_VALUES = np.arange(-4.0, -8.5, -0.5).tolist()  # -4.0, -4.5, ..., -8.0

# Targets
TARGET_FPR_MAX = 0.30
TARGET_RECALL_MIN = 0.80

WINDOW_SIZE = 128


# ── Core logic ────────────────────────────────────────────────────────

def load_shots(disruption_dir: Path) -> list[dict[str, Any]]:
    """Load all NPZ shots and extract metadata."""
    npz_files = sorted(disruption_dir.glob("*.npz"))
    shots = []
    for npz_path in npz_files:
        with np.load(str(npz_path), allow_pickle=False) as data:
            is_disruption = (
                bool(np.asarray(data["is_disruption"]).reshape(()).item())
                if "is_disruption" in data
                else False
            )
            disruption_time_idx = (
                int(np.asarray(data["disruption_time_idx"]).reshape(()).item())
                if "disruption_time_idx" in data
                else -1
            )
            if "dBdt_gauss_per_s" in data:
                signal = np.asarray(data["dBdt_gauss_per_s"], dtype=np.float64).reshape(-1)
            elif "n1_amp" in data:
                signal = np.asarray(data["n1_amp"], dtype=np.float64).reshape(-1)
            else:
                continue
            if signal.size == 0:
                continue
            n1_amp = (
                np.asarray(data["n1_amp"], dtype=np.float64).reshape(-1)
                if "n1_amp" in data
                else None
            )
            n2_amp = (
                np.asarray(data["n2_amp"], dtype=np.float64).reshape(-1)
                if "n2_amp" in data
                else None
            )
        if n1_amp is not None and n1_amp.size != signal.size:
            continue
        if n2_amp is not None and n2_amp.size != signal.size:
            continue
        shots.append({
            "file": npz_path.name,
            "is_disruption": is_disruption,
            "disruption_time_idx": disruption_time_idx,
            "signal": signal,
            "n1_amp": n1_amp,
            "n2_amp": n2_amp,
        })
    return shots


def precompute_unbiased_logits(shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Pre-compute unbiased logits (thermal + asym + state, without bias term)
    for each sliding window position of each shot. This avoids redundant
    feature extraction in the bias/threshold sweep.

    The logit at a given bias is: logit = bias + unbiased_logit
    The risk is: sigmoid(logit)
    Detection happens when sigmoid(logit) > threshold, i.e., logit > log(threshold/(1-threshold))
    """
    precomputed = []
    for shot in shots:
        signal = shot["signal"]
        n1_amp = shot["n1_amp"]
        n2_amp = shot["n2_amp"]
        window_size = min(WINDOW_SIZE, signal.size)

        unbiased_logits = []
        for t in range(window_size, signal.size):
            window = signal[t - window_size:t]
            n1 = float(n1_amp[t]) if n1_amp is not None else 0.1
            n2 = float(n2_amp[t]) if n2_amp is not None else 0.05
            toroidal = {
                "toroidal_n1_amp": n1,
                "toroidal_n2_amp": n2,
                "toroidal_n3_amp": 0.02,
            }
            features = build_disruption_feature_vector(window, toroidal)
            mean, std, max_val, slope, energy, last, fn1, fn2, fn3, asym, spread = features

            thermal_term = 0.03 * max_val + 0.55 * std + 0.005 * energy + 0.50 * slope
            asym_term = 1.10 * fn1 + 0.70 * fn2 + 0.45 * fn3 + 0.50 * asym + 0.15 * spread
            state_term = 0.02 * mean + 0.02 * last
            unbiased_logits.append(thermal_term + asym_term + state_term)

        precomputed.append({
            "file": shot["file"],
            "is_disruption": shot["is_disruption"],
            "disruption_time_idx": shot["disruption_time_idx"],
            "unbiased_logits": np.array(unbiased_logits),
        })
    return precomputed


def evaluate_fast(
    precomputed: list[dict[str, Any]],
    bias: float,
    risk_threshold: float,
) -> dict[str, Any]:
    """
    Fast evaluation using pre-computed unbiased logits.

    Detection occurs when sigmoid(bias + unbiased_logit) > threshold.
    Equivalently: bias + unbiased_logit > log(threshold / (1 - threshold))
    i.e., unbiased_logit > logit_threshold - bias
    """
    # Convert threshold to logit space to avoid sigmoid computation
    if risk_threshold >= 1.0:
        logit_threshold = float("inf")
    elif risk_threshold <= 0.0:
        logit_threshold = float("-inf")
    else:
        logit_threshold = np.log(risk_threshold / (1.0 - risk_threshold))

    # The detection condition: bias + unbiased_logit > logit_threshold
    # i.e., unbiased_logit > logit_threshold - bias
    detect_above = logit_threshold - bias

    true_positives = 0
    false_negatives = 0
    false_positives = 0
    true_negatives = 0

    for shot in precomputed:
        is_disruption = shot["is_disruption"]
        disruption_time_idx = shot["disruption_time_idx"]
        unbiased_logits = shot["unbiased_logits"]

        # Check if any window exceeds the detection threshold
        detected = bool(np.any(unbiased_logits > detect_above))

        if is_disruption and disruption_time_idx > 0:
            if detected:
                true_positives += 1
            else:
                false_negatives += 1
        elif not is_disruption:
            if detected:
                false_positives += 1
            else:
                true_negatives += 1

    n_disruptions = true_positives + false_negatives
    n_safe = true_negatives + false_positives
    recall = true_positives / max(n_disruptions, 1)
    fpr = false_positives / max(n_safe, 1)

    return {
        "bias": round(bias, 2),
        "threshold": round(risk_threshold, 2),
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "n_disruptions": n_disruptions,
        "n_safe": n_safe,
        "recall": round(recall, 4),
        "fpr": round(fpr, 4),
    }


def find_optimal(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    Find the operating point that maximises recall while keeping FPR <= TARGET.
    If no point meets both constraints, find the Pareto-optimal compromise.
    """
    # First: try to find points meeting both constraints
    feasible = [
        r for r in results
        if r["fpr"] <= TARGET_FPR_MAX and r["recall"] >= TARGET_RECALL_MIN
    ]

    if feasible:
        # Among feasible, maximise recall, then minimise FPR, then prefer higher threshold
        feasible.sort(key=lambda r: (-r["recall"], r["fpr"], -r["threshold"]))
        best = dict(feasible[0])
        best["selection"] = "feasible"
        return best

    # No feasible point: find Pareto-optimal
    # Score = recall - 2.0 * max(0, fpr - TARGET_FPR_MAX)
    scored = []
    for r in results:
        fpr_penalty = max(0.0, r["fpr"] - TARGET_FPR_MAX)
        score = r["recall"] - 2.0 * fpr_penalty
        scored.append((score, r))

    scored.sort(key=lambda x: (-x[0], x[1]["fpr"], -x[1]["threshold"]))
    if scored:
        best = dict(scored[0][1])
        best["selection"] = "pareto"
        best["pareto_score"] = round(scored[0][0], 4)
        return best

    return None


def build_roc_curve(
    precomputed: list[dict[str, Any]],
    bias: float,
) -> list[dict[str, float]]:
    """Build ROC curve data at fixed bias, sweeping threshold."""
    roc = []
    for thresh_int in range(int(THRESHOLD_MIN * 100), int(THRESHOLD_MAX * 100) + 1, 1):
        threshold = thresh_int / 100.0
        result = evaluate_fast(precomputed, bias, threshold)
        roc.append({
            "threshold": result["threshold"],
            "fpr": result["fpr"],
            "recall": result["recall"],
        })
    return roc


# ── Main ──────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 60)
    print("SCPN Fusion Core — Disruption Threshold & Bias Sweep")
    print("=" * 60)

    # Load shots
    if not DISRUPTION_DIR.exists():
        print(f"ERROR: {DISRUPTION_DIR} not found")
        return 1

    shots = load_shots(DISRUPTION_DIR)
    n_dis = sum(1 for s in shots if s["is_disruption"])
    n_safe = sum(1 for s in shots if not s["is_disruption"])
    print(f"\nLoaded {len(shots)} shots: {n_dis} disruptions, {n_safe} safe")

    # Pre-compute unbiased logits (expensive step, done once)
    print("\nPre-computing feature logits for all window positions...")
    t0 = time.perf_counter()
    precomputed = precompute_unbiased_logits(shots)
    t_precompute = time.perf_counter() - t0
    total_windows = sum(len(p["unbiased_logits"]) for p in precomputed)
    print(f"  Done: {total_windows} window evaluations in {t_precompute:.1f}s")

    # Print per-shot logit statistics
    print(f"\n{'File':42s} {'Label':5s} {'logit_min':>10s} {'logit_mean':>10s} {'logit_max':>10s}")
    print("-" * 85)
    for p in precomputed:
        label = "DIS" if p["is_disruption"] else "SAFE"
        lmin = float(np.min(p["unbiased_logits"]))
        lmean = float(np.mean(p["unbiased_logits"]))
        lmax = float(np.max(p["unbiased_logits"]))
        print(f"  {p['file']:40s} {label:5s} {lmin:10.2f} {lmean:10.2f} {lmax:10.2f}")

    # Run full grid sweep (fast: just arithmetic comparisons)
    print(f"\nSweeping bias={BIAS_VALUES[0]}..{BIAS_VALUES[-1]} x threshold={THRESHOLD_MIN}..{THRESHOLD_MAX}")
    t1 = time.perf_counter()

    all_results = []
    best_per_bias = {}

    for bias in BIAS_VALUES:
        bias_results = []
        for thresh_int in range(int(THRESHOLD_MIN * 100), int(THRESHOLD_MAX * 100) + 1, 1):
            threshold = thresh_int / 100.0
            result = evaluate_fast(precomputed, bias, threshold)
            all_results.append(result)
            bias_results.append(result)

        # Find best for this bias
        feasible = [r for r in bias_results if r["fpr"] <= TARGET_FPR_MAX and r["recall"] >= TARGET_RECALL_MIN]
        if feasible:
            feasible.sort(key=lambda r: (-r["recall"], r["fpr"]))
            best_per_bias[bias] = feasible[0]

    t_sweep = time.perf_counter() - t1
    print(f"  Sweep done: {len(all_results)} configurations in {t_sweep:.1f}s")

    # Print per-bias summary
    print(f"\n{'Bias':>6s} | {'Best Threshold':>14s} | {'Recall':>7s} | {'FPR':>7s} | {'TP':>3s} | {'FP':>3s} | {'TN':>3s} | {'FN':>3s} | Status")
    print("-" * 85)
    for bias in BIAS_VALUES:
        if bias in best_per_bias:
            r = best_per_bias[bias]
            status = "MEETS TARGET" if r["recall"] >= TARGET_RECALL_MIN and r["fpr"] <= TARGET_FPR_MAX else "partial"
            print(f"{bias:6.1f} | {r['threshold']:14.2f} | {r['recall']:7.2%} | {r['fpr']:7.2%} | {r['true_positives']:3d} | {r['false_positives']:3d} | {r['true_negatives']:3d} | {r['false_negatives']:3d} | {status}")
        else:
            # Find best recall at this bias with FPR info
            bias_results = [r for r in all_results if r["bias"] == round(bias, 2)]
            if bias_results:
                bias_results.sort(key=lambda r: (-r["recall"], r["fpr"]))
                r = bias_results[0]
                print(f"{bias:6.1f} | {r['threshold']:14.2f} | {r['recall']:7.2%} | {r['fpr']:7.2%} | {r['true_positives']:3d} | {r['false_positives']:3d} | {r['true_negatives']:3d} | {r['false_negatives']:3d} | NO FEASIBLE POINT")

    # Find global optimum
    optimal = find_optimal(all_results)

    if optimal:
        print(f"\n{'=' * 60}")
        print(f"OPTIMAL OPERATING POINT ({optimal.get('selection', '?')})")
        print(f"{'=' * 60}")
        print(f"  Bias:      {optimal['bias']}")
        print(f"  Threshold: {optimal['threshold']}")
        print(f"  Recall:    {optimal['recall']:.2%} (target >= {TARGET_RECALL_MIN:.0%})")
        print(f"  FPR:       {optimal['fpr']:.2%} (target <= {TARGET_FPR_MAX:.0%})")
        print(f"  TP={optimal['true_positives']} FP={optimal['false_positives']} TN={optimal['true_negatives']} FN={optimal['false_negatives']}")
    else:
        print("\nWARNING: No operating point found!")

    # Build ROC curve at optimal bias
    best_bias = optimal["bias"] if optimal else BIAS_VALUES[0]
    roc_data = build_roc_curve(precomputed, best_bias)

    # Write results
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    sweep_output = {
        "config": {
            "threshold_range": [THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEP],
            "bias_values": BIAS_VALUES,
            "target_fpr_max": TARGET_FPR_MAX,
            "target_recall_min": TARGET_RECALL_MIN,
            "n_shots": len(shots),
            "n_disruptions": n_dis,
            "n_safe": n_safe,
        },
        "optimal": optimal,
        "best_per_bias": {str(k): v for k, v in best_per_bias.items()},
        "all_results_count": len(all_results),
    }

    sweep_path = ARTIFACTS_DIR / "disruption_threshold_sweep.json"
    sweep_path.write_text(json.dumps(sweep_output, indent=2, default=str), encoding="utf-8")
    print(f"\nSweep results: {sweep_path}")

    roc_path = ARTIFACTS_DIR / "disruption_roc_curve.json"
    roc_output = {
        "bias": best_bias,
        "n_points": len(roc_data),
        "roc": roc_data,
    }
    roc_path.write_text(json.dumps(roc_output, indent=2, default=str), encoding="utf-8")
    print(f"ROC curve:     {roc_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
