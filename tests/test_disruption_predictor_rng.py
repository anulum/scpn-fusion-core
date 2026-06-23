# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Disruption Predictor RNG Tests
"""Determinism tests for RNG-scoped disruption predictor simulation paths."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.control.disruption_predictor import (
    run_anomaly_alarm_campaign,
    simulate_tearing_mode,
)
from scpn_fusion.control.disruption_risk_runtime import (
    _require_int,
    predict_disruption_risk,
    run_anomaly_alarm_campaign as _run_anomaly_alarm_campaign,
    rutherford_island_growth,
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


def test_simulate_tearing_mode_without_rng_does_not_mutate_global_state() -> None:
    np.random.seed(8642)
    state = np.random.get_state()

    simulate_tearing_mode(steps=128)

    observed = float(np.random.random())
    np.random.set_state(state)
    expected = float(np.random.random())
    assert observed == expected


@pytest.mark.parametrize("steps", [0, 1.5, True])
def test_simulate_tearing_mode_rejects_invalid_steps(steps: object) -> None:
    with pytest.raises(ValueError, match="steps"):
        simulate_tearing_mode(steps=steps)


def test_rutherford_island_growth_matches_closed_form() -> None:
    """The deterministic step reproduces the Modified Rutherford expression."""
    w, delta_prime, beta_p, w_crit, dt = 0.2, -0.1, 0.8, 0.05, 0.01
    f_bs = beta_p * (w / (w**2 + w_crit**2))
    expected = (delta_prime + f_bs) * (1.0 - w / 12.0) * dt
    assert rutherford_island_growth(w, delta_prime, beta_p, w_crit, dt) == pytest.approx(
        expected, rel=1e-15
    )


def test_rutherford_island_growth_bootstrap_adds_to_bare_drive() -> None:
    """A positive beta_p raises dw above the bare-delta_prime increment; zero recovers it."""
    w, delta_prime, w_crit, dt = 0.2, -0.1, 0.05, 0.01
    bare = delta_prime * (1.0 - w / 12.0) * dt
    assert rutherford_island_growth(w, delta_prime, 0.8, w_crit, dt) > bare
    assert rutherford_island_growth(w, delta_prime, 0.0, w_crit, dt) == bare


@pytest.mark.parametrize(
    ("beta_p", "w_crit"),
    [(-1.0, 0.05), (float("nan"), 0.05), (0.8, 0.0), (0.8, -0.1), (0.8, float("inf"))],
)
def test_simulate_tearing_mode_rejects_invalid_shaping(beta_p: float, w_crit: float) -> None:
    with pytest.raises(ValueError, match="beta_p|w_crit"):
        simulate_tearing_mode(steps=10, beta_p=beta_p, w_crit=w_crit)


def test_require_int_rejects_non_integer_without_minimum() -> None:
    """Without a minimum, a non-integer value raises the plain message."""
    with pytest.raises(ValueError, match="must be an integer\\."):
        _require_int("widget", 1.5)
    assert _require_int("widget", 7) == 7


def test_simulate_tearing_mode_seeds_island_on_a_triggered_shot() -> None:
    """A triggered shot re-seeds a collapsed island to the seed width.

    With the bootstrap drive disabled (beta_p=0) the island width decays to its
    floor during the stable phase, so when the tearing trigger fires the post-
    trigger branch re-seeds it to 0.15 — the only way the width can exceed 0.1
    for this configuration. seed 0 is a disruptive shot.
    """
    sig, _label, _ttd = simulate_tearing_mode(
        steps=1000, rng=np.random.default_rng(0), beta_p=0.0, w_crit=0.05
    )
    assert float(np.max(sig)) >= 0.15 - 1e-9


def test_simulate_tearing_mode_labels_disruption_on_threshold_crossing() -> None:
    """A strong bootstrap drive grows the island past 8.0 and flags a disruption."""
    sig, label, ttd = simulate_tearing_mode(
        steps=3000, rng=np.random.default_rng(1), beta_p=5.0, w_crit=0.05
    )
    assert label == 1
    assert float(np.max(sig)) > 8.0
    assert isinstance(ttd, (int, np.integer))


@pytest.mark.parametrize("bias", [float("inf"), float("-inf"), float("nan")])
def test_predict_disruption_risk_rejects_non_finite_bias(bias: float) -> None:
    with pytest.raises(ValueError, match="bias_delta must be finite"):
        predict_disruption_risk(np.linspace(0.0, 1.0, 32), bias_delta=bias)


def test_anomaly_campaign_counts_true_and_false_positive_alarms() -> None:
    """A low alarm threshold exercises both the true- and false-positive branches."""
    result = _run_anomaly_alarm_campaign(seed=0, episodes=128, window=64, threshold=0.02)
    assert result["true_positive_rate"] > 0.0
    assert result["false_positive_rate"] > 0.0
    assert result["p95_alarm_latency_steps"] >= 0
