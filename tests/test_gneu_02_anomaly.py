# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — GNEU-02 Anomaly Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for hybrid anomaly detector in disruption predictor."""

from __future__ import annotations

import numpy as np

from scpn_fusion.control.disruption_predictor import (
    HybridAnomalyDetector,
    run_anomaly_alarm_campaign,
)


def test_hybrid_detector_score_is_bounded() -> None:
    det = HybridAnomalyDetector(threshold=0.65, ema=0.05)
    signal = np.linspace(0.2, 1.0, 64)
    out = det.score(
        signal,
        {
            "toroidal_n1_amp": 0.2,
            "toroidal_n2_amp": 0.1,
            "toroidal_n3_amp": 0.05,
            "toroidal_asymmetry_index": 0.25,
            "toroidal_radial_spread": 0.03,
        },
    )
    assert 0.0 <= out["supervised_score"] <= 1.0
    assert 0.0 <= out["unsupervised_score"] <= 1.0
    assert 0.0 <= out["anomaly_score"] <= 1.0


def test_hybrid_detector_deterministic_given_same_sequence() -> None:
    sig = np.linspace(0.3, 0.9, 80)
    obs = {
        "toroidal_n1_amp": 0.18,
        "toroidal_n2_amp": 0.09,
        "toroidal_n3_amp": 0.04,
        "toroidal_asymmetry_index": 0.205,
        "toroidal_radial_spread": 0.02,
    }
    d1 = HybridAnomalyDetector()
    d2 = HybridAnomalyDetector()
    o1 = [d1.score(sig[: i + 1], obs)["anomaly_score"] for i in range(32)]
    o2 = [d2.score(sig[: i + 1], obs)["anomaly_score"] for i in range(32)]
    assert np.allclose(o1, o2)


def test_anomaly_alarm_campaign_outputs_expected_metrics() -> None:
    report = run_anomaly_alarm_campaign(seed=23, episodes=8, window=64)
    for key in (
        "true_positive_rate",
        "false_positive_rate",
        "p95_alarm_latency_steps",
        "passes_thresholds",
    ):
        assert key in report
    assert 0.0 <= report["true_positive_rate"] <= 1.0
    assert 0.0 <= report["false_positive_rate"] <= 1.0
