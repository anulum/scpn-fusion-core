# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Disruption Threshold Sweep Tests
# Verifies the v2.1 weight tuning achieves FPR < 30% and recall > 80%
# on the reference DIII-D disruption shot dataset.
# ──────────────────────────────────────────────────────────────────────
"""Tests that the disruption predictor achieves target operating point.

The v2.1 weight update shifted from amplitude-heavy coefficients to
instability indicators (std, slope) so that high-power safe shots no
longer trigger false alarms.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scpn_fusion.control.disruption_predictor import (
    HybridAnomalyDetector,
    build_disruption_feature_vector,
    predict_disruption_risk,
)

ROOT = Path(__file__).resolve().parents[1]
DISRUPTION_DIR = ROOT / "validation" / "reference_data" / "diiid" / "disruption_shots"


def _has_reference_data() -> bool:
    return DISRUPTION_DIR.exists() and any(DISRUPTION_DIR.glob("*.npz"))


def _run_sliding_window_detection(
    risk_threshold: float = 0.50,
    window_size: int = 128,
) -> dict[str, int | float]:
    """Replicate the validate_real_shots.py sliding-window logic."""
    npz_files = sorted(DISRUPTION_DIR.glob("*.npz"))
    true_positives = 0
    false_negatives = 0
    false_positives = 0
    true_negatives = 0

    for npz_path in npz_files:
        data = np.load(str(npz_path), allow_pickle=True)
        is_disruption = bool(data.get("is_disruption", False))
        disruption_time_idx = int(data.get("disruption_time_idx", -1))
        signal = np.asarray(
            data.get("dBdt_gauss_per_s", data.get("n1_amp", []))
        )
        if signal.size == 0:
            continue

        ws = min(window_size, signal.size)
        detection_idx = -1

        for t in range(ws, signal.size):
            window = signal[t - ws : t]
            n1 = float(data["n1_amp"][t]) if "n1_amp" in data else 0.1
            n2 = float(data["n2_amp"][t]) if "n2_amp" in data else 0.05
            toroidal = {
                "toroidal_n1_amp": n1,
                "toroidal_n2_amp": n2,
                "toroidal_n3_amp": 0.02,
            }
            risk = predict_disruption_risk(window, toroidal)
            if risk > risk_threshold:
                detection_idx = t
                break

        detected = detection_idx >= 0
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
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "n_disruptions": n_disruptions,
        "n_safe": n_safe,
        "recall": recall,
        "fpr": fpr,
    }


# ── Tests ─────────────────────────────────────────────────────────────


@pytest.mark.skipif(not _has_reference_data(), reason="No DIII-D NPZ files")
def test_disruption_recall_above_80_pct() -> None:
    """Recall must be >= 80% on the reference disruption shots."""
    result = _run_sliding_window_detection(risk_threshold=0.50)
    assert result["recall"] >= 0.80, (
        f"Recall {result['recall']:.2%} < 80%: "
        f"TP={result['true_positives']} FN={result['false_negatives']}"
    )


@pytest.mark.skipif(not _has_reference_data(), reason="No DIII-D NPZ files")
def test_disruption_fpr_below_30_pct() -> None:
    """FPR must be <= 30% on the reference safe shots."""
    result = _run_sliding_window_detection(risk_threshold=0.50)
    assert result["fpr"] <= 0.30, (
        f"FPR {result['fpr']:.2%} > 30%: "
        f"FP={result['false_positives']} TN={result['true_negatives']}"
    )


@pytest.mark.skipif(not _has_reference_data(), reason="No DIII-D NPZ files")
def test_disruption_operating_point_recall_100_fpr_0() -> None:
    """With v2.1 weights, the operating point should achieve recall=100% FPR=0%."""
    result = _run_sliding_window_detection(risk_threshold=0.50)
    assert result["recall"] == 1.0, (
        f"Expected recall=100%, got {result['recall']:.2%}"
    )
    assert result["fpr"] == 0.0, (
        f"Expected FPR=0%, got {result['fpr']:.2%}"
    )


def test_v21_weights_safe_signal_below_threshold() -> None:
    """A flat high-power signal (high mean, low std) should not trigger."""
    # Simulate a safe high-power shot: constant dBdt=12 with small noise
    rng = np.random.default_rng(42)
    signal = 12.0 + rng.normal(0.0, 0.2, size=200)
    toroidal = {
        "toroidal_n1_amp": 0.03,
        "toroidal_n2_amp": 0.015,
        "toroidal_n3_amp": 0.02,
    }
    risk = predict_disruption_risk(signal, toroidal)
    assert risk < 0.50, (
        f"Safe high-power signal should have risk < 0.50, got {risk:.4f}"
    )


def test_v21_weights_unstable_signal_above_threshold() -> None:
    """A signal with growing instability (high std, slope) should trigger."""
    rng = np.random.default_rng(42)
    t = np.linspace(0, 1, 200)
    # Exponential growth -> high std, slope, max
    signal = 2.0 + 50.0 * np.exp(5.0 * (t - 0.8))
    signal += rng.normal(0.0, 0.5, size=200)
    toroidal = {
        "toroidal_n1_amp": 0.5,
        "toroidal_n2_amp": 0.2,
        "toroidal_n3_amp": 0.02,
    }
    risk = predict_disruption_risk(signal, toroidal)
    assert risk > 0.50, (
        f"Unstable signal should have risk > 0.50, got {risk:.4f}"
    )


def test_hybrid_anomaly_detector_default_threshold_is_050() -> None:
    """Verify the default threshold was updated to 0.50."""
    detector = HybridAnomalyDetector()
    assert detector.threshold == 0.50


def test_predict_disruption_risk_monotonic_in_std() -> None:
    """Risk should increase as signal std (instability) increases."""
    base = np.full(128, 5.0)
    toroidal = {"toroidal_n1_amp": 0.1, "toroidal_n2_amp": 0.05}
    risks = []
    for noise_scale in [0.01, 0.1, 1.0, 10.0, 100.0]:
        rng = np.random.default_rng(0)
        signal = base + rng.normal(0, noise_scale, size=128)
        risks.append(predict_disruption_risk(signal, toroidal))
    # Risk should be non-decreasing (approximately, given fixed rng)
    for i in range(len(risks) - 1):
        assert risks[i + 1] >= risks[i] - 0.01, (
            f"Risk not monotonic in std: {risks}"
        )
