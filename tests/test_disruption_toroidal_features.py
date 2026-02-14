# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Disruption Toroidal Feature Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for toroidal-asymmetry-aware disruption feature path."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.control.disruption_predictor import (
    apply_bit_flip_fault,
    build_disruption_feature_vector,
    predict_disruption_risk,
    run_fault_noise_campaign,
)


def test_build_disruption_feature_vector_includes_toroidal_terms() -> None:
    signal = np.linspace(0.2, 1.0, 100)
    observables = {
        "toroidal_n1_amp": 0.4,
        "toroidal_n2_amp": 0.2,
        "toroidal_n3_amp": 0.1,
        "toroidal_asymmetry_index": 0.6,
        "toroidal_radial_spread": 0.05,
    }
    features = build_disruption_feature_vector(signal, observables)
    assert features.shape == (11,)
    assert features[6] == 0.4
    assert features[7] == 0.2
    assert features[8] == 0.1
    assert features[9] == 0.6
    assert features[10] == 0.05


def test_predict_disruption_risk_increases_with_toroidal_asymmetry() -> None:
    signal = np.full(128, 0.7, dtype=float)
    low = predict_disruption_risk(signal, {"toroidal_n1_amp": 0.0})
    high = predict_disruption_risk(
        signal,
        {
            "toroidal_n1_amp": 1.2,
            "toroidal_n2_amp": 0.8,
            "toroidal_n3_amp": 0.5,
            "toroidal_asymmetry_index": 1.6,
            "toroidal_radial_spread": 0.4,
        },
    )
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    assert high > low


def test_apply_bit_flip_fault_returns_finite_value() -> None:
    value = 0.75
    flipped = apply_bit_flip_fault(value, 7)
    assert np.isfinite(flipped)
    assert flipped != value


def test_fault_noise_campaign_metrics_and_thresholds() -> None:
    report = run_fault_noise_campaign(
        seed=123,
        episodes=8,
        window=48,
        noise_std=0.02,
        bit_flip_interval=9,
        recovery_window=6,
    )
    for key in (
        "mean_abs_risk_error",
        "p95_abs_risk_error",
        "recovery_steps_p95",
        "recovery_success_rate",
        "passes_thresholds",
    ):
        assert key in report
    assert 0.0 <= report["mean_abs_risk_error"]
    assert 0.0 <= report["p95_abs_risk_error"]
    assert 0.0 <= report["recovery_success_rate"] <= 1.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"episodes": 0}, "episodes"),
        ({"window": 8}, "window"),
        ({"noise_std": -0.1}, "noise_std"),
        ({"noise_std": float("nan")}, "noise_std"),
        ({"bit_flip_interval": 0}, "bit_flip_interval"),
        ({"recovery_window": 0}, "recovery_window"),
        ({"recovery_epsilon": 0.0}, "recovery_epsilon"),
    ],
)
def test_fault_noise_campaign_rejects_invalid_inputs(
    kwargs: dict[str, float | int], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        run_fault_noise_campaign(**kwargs)
