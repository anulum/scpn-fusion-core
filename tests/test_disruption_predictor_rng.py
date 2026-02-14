# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Disruption Predictor RNG Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Determinism tests for RNG-scoped disruption predictor simulation paths."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.control.disruption_predictor import (
    run_anomaly_alarm_campaign,
    simulate_tearing_mode,
)


def test_simulate_tearing_mode_is_deterministic_with_seeded_rng() -> None:
    r1 = np.random.default_rng(123)
    r2 = np.random.default_rng(123)
    sig1, lbl1, ttd1 = simulate_tearing_mode(steps=420, rng=r1)
    sig2, lbl2, ttd2 = simulate_tearing_mode(steps=420, rng=r2)
    assert lbl1 == lbl2
    assert ttd1 == ttd2
    np.testing.assert_allclose(sig1, sig2, rtol=0.0, atol=0.0)


def test_anomaly_campaign_is_independent_of_global_numpy_seed() -> None:
    kwargs = dict(seed=55, episodes=8, window=64, threshold=0.65)
    np.random.seed(1)
    a = run_anomaly_alarm_campaign(**kwargs)
    np.random.seed(999)
    b = run_anomaly_alarm_campaign(**kwargs)
    for key in (
        "true_positive_rate",
        "false_positive_rate",
        "p95_alarm_latency_steps",
        "passes_thresholds",
    ):
        assert a[key] == pytest.approx(b[key], rel=0.0, abs=0.0)


def test_simulate_tearing_mode_without_rng_still_returns_valid_shape() -> None:
    sig, lbl, ttd = simulate_tearing_mode(steps=128)
    assert sig.ndim == 1
    assert sig.size > 0
    assert lbl in (0, 1)
    assert isinstance(ttd, (int, np.integer))
