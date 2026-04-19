# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Hypothesis Property Tests for phase/ Modules
"""Property-based tests for knm.py, upde.py, and adaptive_knm.py."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from scpn_fusion.phase.knm import KnmSpec, build_knm_paper27
from scpn_fusion.phase.upde import UPDESystem
from scpn_fusion.phase.adaptive_knm import (
    AdaptiveKnmConfig,
    AdaptiveKnmEngine,
    DiagnosticSnapshot,
)


# ── Strategies ────────────────────────────────────────────────────────

small_L = st.integers(min_value=2, max_value=8)
K_base_st = st.floats(min_value=0.1, max_value=2.0)
K_alpha_st = st.floats(min_value=0.01, max_value=1.0)


# ── KnmSpec / build_knm_paper27 ──────────────────────────────────────


@given(L=small_L, K_base=K_base_st, K_alpha=K_alpha_st)
@settings(max_examples=30)
def test_knm_symmetric(L, K_base, K_alpha):
    spec = build_knm_paper27(L=L, K_base=K_base, K_alpha=K_alpha)
    K = np.asarray(spec.K)
    np.testing.assert_allclose(K, K.T, atol=1e-14)


@given(L=small_L, K_base=K_base_st, K_alpha=K_alpha_st)
@settings(max_examples=30)
def test_knm_non_negative(L, K_base, K_alpha):
    spec = build_knm_paper27(L=L, K_base=K_base, K_alpha=K_alpha)
    assert np.all(np.asarray(spec.K) >= 0.0)


@given(L=small_L)
@settings(max_examples=20)
def test_knm_shape(L):
    spec = build_knm_paper27(L=L)
    assert spec.L == L
    assert np.asarray(spec.K).shape == (L, L)


def test_knm_calibration_anchors():
    """Paper 27 Table 2 anchors present in 16-layer matrix."""
    spec = build_knm_paper27(L=16)
    K = np.asarray(spec.K)
    assert K[0, 1] == pytest.approx(0.302)
    assert K[1, 2] == pytest.approx(0.201)
    assert K[2, 3] == pytest.approx(0.252)
    assert K[3, 4] == pytest.approx(0.154)


def test_knm_rejects_non_square():
    with pytest.raises(ValueError, match="square"):
        KnmSpec(K=np.ones((3, 4)))


# ── UPDESystem ────────────────────────────────────────────────────────


@given(L=st.integers(min_value=2, max_value=4), N=st.integers(min_value=5, max_value=20))
@settings(max_examples=20)
def test_upde_phase_wrapping(L, N):
    """After step(), all phases stay in [0, 2π)."""
    rng = np.random.default_rng(0)
    spec = build_knm_paper27(L=L, K_base=0.45, K_alpha=0.3)
    sys = UPDESystem(spec=spec, dt=1e-3, psi_mode="external", wrap=True)

    theta_layers = [rng.uniform(0, 2 * np.pi, N) for _ in range(L)]
    omega_layers = [rng.uniform(0.5, 3.0, N) for _ in range(L)]

    out = sys.step(theta_layers, omega_layers, psi_driver=0.0)
    for th in out["theta1"]:
        assert np.all(th > -np.pi - 1e-12), f"min={th.min()}"
        assert np.all(th <= np.pi + 1e-12), f"max={th.max()}"


@given(L=st.integers(min_value=2, max_value=4), N=st.integers(min_value=5, max_value=20))
@settings(max_examples=20)
def test_upde_order_parameter_range(L, N):
    """R_global and R_layer values in [0, 1]."""
    rng = np.random.default_rng(1)
    spec = build_knm_paper27(L=L, K_base=0.45, K_alpha=0.3)
    sys = UPDESystem(spec=spec, dt=1e-3, psi_mode="external")

    theta_layers = [rng.uniform(0, 2 * np.pi, N) for _ in range(L)]
    omega_layers = [rng.uniform(0.5, 3.0, N) for _ in range(L)]

    out = sys.step(theta_layers, omega_layers, psi_driver=0.0)
    assert 0.0 <= out["R_global"] <= 1.0 + 1e-10
    for r in out["R_layer"]:
        assert 0.0 <= r <= 1.0 + 1e-10


def test_upde_run_r_monotonic_strong_coupling():
    """Strong coupling (K_base=5) drives R_global upward over 200 steps."""
    L, N = 3, 30
    rng = np.random.default_rng(42)
    spec = build_knm_paper27(L=L, K_base=5.0, K_alpha=0.1)
    sys = UPDESystem(spec=spec, dt=1e-3, psi_mode="external")
    theta = [rng.uniform(0, 2 * np.pi, N) for _ in range(L)]
    omega = [np.full(N, 1.0) for _ in range(L)]
    result = sys.run(200, theta, omega, psi_driver=0.0)
    assert result["R_global_hist"][-1] > result["R_global_hist"][0]


def test_upde_layer_count_mismatch():
    spec = build_knm_paper27(L=3)
    sys = UPDESystem(spec=spec)
    with pytest.raises(ValueError, match="Expected 3"):
        sys.step([np.zeros(5)], [np.zeros(5)], psi_driver=0.0)


# ── AdaptiveKnmEngine ────────────────────────────────────────────────


def _snap(L, beta_n=0.5, risk=0.2, R_val=0.4, guard=True):
    return DiagnosticSnapshot(
        R_layer=np.full(L, R_val),
        V_layer=np.zeros(L),
        lambda_exp=0.0,
        beta_n=beta_n,
        q95=3.0,
        disruption_risk=risk,
        mirnov_rms=0.01,
        guard_approved=guard,
    )


@given(L=st.integers(min_value=3, max_value=8))
@settings(max_examples=20)
def test_adaptive_symmetry_preserved(L):
    """Adapted K stays symmetric after update."""
    spec = build_knm_paper27(L=L)
    engine = AdaptiveKnmEngine(spec)
    K = engine.update(_snap(L))
    np.testing.assert_allclose(K, K.T, atol=1e-14)


@given(L=st.integers(min_value=3, max_value=8))
@settings(max_examples=20)
def test_adaptive_non_negative(L):
    """Adapted K stays non-negative after update."""
    spec = build_knm_paper27(L=L)
    engine = AdaptiveKnmEngine(spec)
    K = engine.update(_snap(L))
    assert np.all(K >= -1e-14)


@given(L=st.integers(min_value=3, max_value=8), n_ticks=st.integers(min_value=1, max_value=20))
@settings(max_examples=15)
def test_adaptive_bounded_after_many_ticks(L, n_ticks):
    """K elements don't grow unboundedly over repeated ticks."""
    spec = build_knm_paper27(L=L)
    cfg = AdaptiveKnmConfig(max_delta_per_tick=0.02)
    engine = AdaptiveKnmEngine(spec, config=cfg)

    baseline_max = np.asarray(spec.K).max()
    for _ in range(n_ticks):
        K = engine.update(_snap(L, beta_n=1.0, risk=0.5))

    # rate limiter at 0.02/tick → max drift is n_ticks * 0.02 + beta/risk deltas
    assert K.max() < baseline_max + n_ticks * cfg.max_delta_per_tick + 2.0


def test_adaptive_guard_revert():
    """Guard refusal reverts K to last known-good state."""
    L = 4
    spec = build_knm_paper27(L=L)
    engine = AdaptiveKnmEngine(spec)

    K_before = engine.update(_snap(L, guard=True))
    K_reverted = engine.update(_snap(L, guard=False))
    np.testing.assert_allclose(K_reverted, K_before, atol=1e-14)


def test_adaptive_reset():
    """reset() returns engine to baseline."""
    L = 4
    spec = build_knm_paper27(L=L)
    engine = AdaptiveKnmEngine(spec)
    engine.update(_snap(L, beta_n=2.0))
    engine.reset()
    np.testing.assert_allclose(engine.K_current, np.asarray(spec.K), atol=1e-14)
