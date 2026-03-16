# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Adaptive Knm Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.phase.adaptive_knm import (
    AdaptiveKnmConfig,
    AdaptiveKnmEngine,
    DiagnosticSnapshot,
)
from scpn_fusion.phase.plasma_knm import build_knm_plasma, plasma_omega
from scpn_fusion.phase.upde import UPDESystem

# ── Helpers ───────────────────────────────────────────────────────────


def _make_spec(L: int = 8):
    return build_knm_plasma(mode="baseline", L=L)


def _make_snap(
    L: int = 8,
    *,
    beta_n: float = 0.0,
    disruption_risk: float = 0.0,
    R_layer: np.ndarray | None = None,
    guard_approved: bool = True,
) -> DiagnosticSnapshot:
    return DiagnosticSnapshot(
        R_layer=R_layer if R_layer is not None else np.full(L, 0.5),
        V_layer=np.full(L, 0.3),
        lambda_exp=-0.01,
        beta_n=beta_n,
        q95=3.0,
        disruption_risk=disruption_risk,
        mirnov_rms=0.0,
        guard_approved=guard_approved,
    )


# ── DiagnosticSnapshot ───────────────────────────────────────────────


class TestDiagnosticSnapshot:
    def test_construction(self):
        snap = _make_snap()
        assert snap.beta_n == 0.0
        assert snap.guard_approved is True

    def test_field_access(self):
        snap = _make_snap(beta_n=1.5, disruption_risk=0.8)
        assert snap.beta_n == 1.5
        assert snap.disruption_risk == 0.8
        assert snap.R_layer.shape == (8,)


# ── AdaptiveKnmConfig ───────────────────────────────────────────────


class TestAdaptiveKnmConfig:
    def test_defaults(self):
        cfg = AdaptiveKnmConfig()
        assert cfg.beta_scale == 0.3
        assert cfg.max_delta_per_tick == 0.02

    def test_custom(self):
        cfg = AdaptiveKnmConfig(beta_scale=0.5, risk_gain=0.8)
        assert cfg.beta_scale == 0.5
        assert cfg.risk_gain == 0.8


# ── Beta channel ─────────────────────────────────────────────────────


class TestBetaChannel:
    def test_zero_beta_noop(self):
        spec = _make_spec()
        engine = AdaptiveKnmEngine(spec)
        K0 = engine.K_current.copy()
        snap = _make_snap(beta_n=0.0)
        K1 = engine.update(snap)
        # Rate limiter means K1 may not exactly equal K0, but
        # with zero beta the target is K_baseline, so delta~0
        assert np.allclose(K1, K0, atol=0.025)

    def test_high_beta_strengthens(self):
        spec = _make_spec()
        engine = AdaptiveKnmEngine(spec)
        K_base = engine.K_current.copy()
        # Run several ticks with high beta to let rate limiter converge
        for _ in range(100):
            K = engine.update(_make_snap(beta_n=1.5))
        assert K.mean() > K_base.mean()

    def test_beta_clamped(self):
        spec = _make_spec()
        cfg = AdaptiveKnmConfig(beta_scale=0.3, beta_max_boost=0.5)
        engine = AdaptiveKnmEngine(spec, cfg)
        # Even with extreme beta, boost capped at beta_max_boost
        for _ in range(200):
            K = engine.update(_make_snap(beta_n=100.0))
        baseline = np.asarray(spec.K, dtype=np.float64)
        max_ratio = (K / np.where(baseline > 0, baseline, 1.0)).max()
        # The maximum element ratio should not exceed 1 + beta_max_boost + coherence_max_boost + risk
        assert max_ratio < 3.0


# ── Risk channel ─────────────────────────────────────────────────────


class TestRiskChannel:
    def test_zero_risk_noop(self):
        spec = _make_spec()
        engine = AdaptiveKnmEngine(spec)
        K0 = engine.K_current.copy()
        K1 = engine.update(_make_snap(disruption_risk=0.0))
        assert np.allclose(K1, K0, atol=0.025)

    def test_high_risk_amplifies_mhd(self):
        spec = _make_spec()
        engine = AdaptiveKnmEngine(spec)
        K_base = engine.K_current.copy()
        for _ in range(100):
            K = engine.update(_make_snap(disruption_risk=0.9))
        # MHD pair (2,5) should be stronger
        assert K[2, 5] > K_base[2, 5]

    def test_risk_symmetric(self):
        spec = _make_spec()
        engine = AdaptiveKnmEngine(spec)
        for _ in range(50):
            K = engine.update(_make_snap(disruption_risk=0.7))
        assert np.allclose(K, K.T, atol=1e-12)


# ── Coherence PI ─────────────────────────────────────────────────────


class TestCoherencePI:
    def test_low_R_boosts_diagonal(self):
        spec = _make_spec()
        engine = AdaptiveKnmEngine(spec)
        K_base = engine.K_current.copy()
        low_R = np.full(8, 0.2)
        for _ in range(100):
            K = engine.update(_make_snap(R_layer=low_R))
        diag_increase = np.diag(K) - np.diag(K_base)
        assert (diag_increase > 0).all()

    def test_high_R_no_boost(self):
        spec = _make_spec()
        engine = AdaptiveKnmEngine(spec)
        K_base = engine.K_current.copy()
        high_R = np.full(8, 0.9)
        K = engine.update(_make_snap(R_layer=high_R))
        # Diagonal should not increase when R > R_target
        diag_diff = np.diag(K) - np.diag(K_base)
        assert np.all(diag_diff <= 0.025)

    def test_integral_accumulates(self):
        spec = _make_spec()
        engine = AdaptiveKnmEngine(spec)
        low_R = np.full(8, 0.3)
        for _ in range(20):
            engine.update(_make_snap(R_layer=low_R))
        assert engine._integral.sum() > 0

    def test_integral_clamped(self):
        spec = _make_spec()
        cfg = AdaptiveKnmConfig(coherence_max_boost=0.3)
        engine = AdaptiveKnmEngine(spec, cfg)
        low_R = np.full(8, 0.1)
        for _ in range(500):
            engine.update(_make_snap(R_layer=low_R))
        assert np.all(engine._integral <= cfg.coherence_max_boost + 1e-12)


# ── Invariants ───────────────────────────────────────────────────────


class TestInvariants:
    def test_always_symmetric(self):
        spec = _make_spec()
        engine = AdaptiveKnmEngine(spec)
        rng = np.random.default_rng(7)
        for _ in range(50):
            snap = _make_snap(
                beta_n=rng.uniform(0, 2),
                disruption_risk=rng.uniform(0, 1),
                R_layer=rng.uniform(0, 1, 8),
            )
            K = engine.update(snap)
            assert np.allclose(K, K.T, atol=1e-12)

    def test_always_nonneg(self):
        spec = _make_spec()
        engine = AdaptiveKnmEngine(spec)
        rng = np.random.default_rng(8)
        for _ in range(50):
            snap = _make_snap(
                beta_n=rng.uniform(0, 2),
                disruption_risk=rng.uniform(0, 1),
                R_layer=rng.uniform(0, 1, 8),
            )
            K = engine.update(snap)
            assert np.all(K >= -1e-12)


# ── Rate limiting ────────────────────────────────────────────────────


class TestRateLimiting:
    def test_large_jump_clamped(self):
        spec = _make_spec()
        cfg = AdaptiveKnmConfig(max_delta_per_tick=0.02)
        engine = AdaptiveKnmEngine(spec, cfg)
        K0 = engine.K_current.copy()
        snap = _make_snap(beta_n=5.0, disruption_risk=1.0)
        K1 = engine.update(snap)
        delta = np.abs(K1 - K0)
        assert delta.max() <= cfg.max_delta_per_tick + 1e-12

    def test_small_change_passes(self):
        spec = _make_spec()
        cfg = AdaptiveKnmConfig(max_delta_per_tick=1.0)
        engine = AdaptiveKnmEngine(spec, cfg)
        K0 = engine.K_current.copy()
        snap = _make_snap(beta_n=0.01)
        K1 = engine.update(snap)
        # With generous rate limit, small perturbation applies freely
        delta = np.abs(K1 - K0).max()
        assert delta < 0.5


# ── Guard veto ───────────────────────────────────────────────────────


class TestGuardVeto:
    def test_refusal_reverts(self):
        spec = _make_spec()
        engine = AdaptiveKnmEngine(spec)
        # First do a normal update to change K
        engine.update(_make_snap(beta_n=0.5))
        K_before = engine.K_current.copy()
        # Now simulate guard refusal
        K_after = engine.update(_make_snap(guard_approved=False))
        # Should revert to last known-good (which was K before the refused tick)
        assert np.allclose(K_after, K_before, atol=1e-12)

    def test_approval_saves(self):
        spec = _make_spec()
        engine = AdaptiveKnmEngine(spec)
        K0 = engine.K_current.copy()
        engine.update(_make_snap(beta_n=0.5))
        K1 = engine.K_current.copy()
        # K1 should differ from K0 (update was approved)
        assert not np.allclose(K1, K0)


# ── Reset ────────────────────────────────────────────────────────────


class TestReset:
    def test_returns_to_baseline(self):
        spec = _make_spec()
        engine = AdaptiveKnmEngine(spec)
        baseline = np.asarray(spec.K, dtype=np.float64)
        for _ in range(20):
            engine.update(_make_snap(beta_n=1.0, disruption_risk=0.5))
        engine.reset()
        assert np.allclose(engine.K_current, baseline)

    def test_clears_integral(self):
        spec = _make_spec()
        engine = AdaptiveKnmEngine(spec)
        for _ in range(20):
            engine.update(_make_snap(R_layer=np.full(8, 0.1)))
        assert engine._integral.sum() > 0
        engine.reset()
        assert engine._integral.sum() == 0.0


# ── Integration ──────────────────────────────────────────────────────


class TestIntegration:
    def test_ten_tick_stable(self):
        spec = _make_spec()
        engine = AdaptiveKnmEngine(spec)
        for _ in range(10):
            K = engine.update(_make_snap(beta_n=0.3, disruption_risk=0.1))
            assert np.isfinite(K).all()
            assert np.allclose(K, K.T, atol=1e-12)
            assert np.all(K >= -1e-12)

    def test_adaptation_summary_keys(self):
        spec = _make_spec()
        engine = AdaptiveKnmEngine(spec)
        engine.update(_make_snap(beta_n=0.5))
        s = engine.adaptation_summary
        expected = {"L", "K_mean", "K_max", "delta_frobenius", "delta_max_element", "integral_sum"}
        assert set(s.keys()) == expected


# ── Hypothesis-style property tests ──────────────────────────────────


class TestHypothesis:
    @pytest.mark.parametrize("seed", range(10))
    def test_symmetric_under_random(self, seed):
        rng = np.random.default_rng(seed)
        spec = _make_spec()
        engine = AdaptiveKnmEngine(spec)
        snap = _make_snap(
            beta_n=rng.uniform(0, 3),
            disruption_risk=rng.uniform(0, 1),
            R_layer=rng.uniform(0, 1, 8),
        )
        K = engine.update(snap)
        assert np.allclose(K, K.T, atol=1e-12)

    @pytest.mark.parametrize("seed", range(10))
    def test_nonneg_under_random(self, seed):
        rng = np.random.default_rng(seed)
        spec = _make_spec()
        engine = AdaptiveKnmEngine(spec)
        snap = _make_snap(
            beta_n=rng.uniform(0, 3),
            disruption_risk=rng.uniform(0, 1),
            R_layer=rng.uniform(0, 1, 8),
        )
        K = engine.update(snap)
        assert np.all(K >= -1e-12)


# ── End-to-end UPDE integration ──────────────────────────────────────


class TestE2E:
    def test_50_tick_adaptive_loop(self):
        """Adaptive Knm feeds into UPDE for 50 ticks without divergence."""
        spec = _make_spec()
        engine = AdaptiveKnmEngine(spec)
        upde = UPDESystem(spec=spec, dt=1e-3, psi_mode="external")

        rng = np.random.default_rng(99)
        L = spec.L
        N = 30
        omega_base = plasma_omega(L)
        theta = [rng.uniform(-np.pi, np.pi, N) for _ in range(L)]
        omega = [omega_base[m] + rng.normal(0, 0.1, N) for m in range(L)]

        for t in range(50):
            out = upde.step(
                theta,
                omega,
                psi_driver=0.0,
                K_override=engine.K_current,
            )
            theta = out["theta1"]

            snap = DiagnosticSnapshot(
                R_layer=out["R_layer"],
                V_layer=out["V_layer"],
                lambda_exp=-0.01,
                beta_n=0.3 + 0.01 * t,
                q95=3.0,
                disruption_risk=min(0.01 * t, 0.8),
                mirnov_rms=0.0,
                guard_approved=True,
            )
            engine.update(snap)

        assert np.isfinite(out["R_global"])
        assert 0.0 <= out["R_global"] <= 1.0
        assert np.isfinite(out["V_global"])

    def test_guard_veto_recovery(self):
        """After guard veto, system recovers when approval resumes."""
        spec = _make_spec()
        engine = AdaptiveKnmEngine(spec)
        upde = UPDESystem(spec=spec, dt=1e-3, psi_mode="external")

        rng = np.random.default_rng(77)
        L = spec.L
        N = 30
        omega_base = plasma_omega(L)
        theta = [rng.uniform(-np.pi, np.pi, N) for _ in range(L)]
        omega = [omega_base[m] + rng.normal(0, 0.1, N) for m in range(L)]

        # 20 normal ticks
        for _ in range(20):
            out = upde.step(theta, omega, psi_driver=0.0, K_override=engine.K_current)
            theta = out["theta1"]
            engine.update(_make_snap(beta_n=0.5))

        K_before_veto = engine.K_current.copy()

        # 5 refused ticks
        for _ in range(5):
            out = upde.step(theta, omega, psi_driver=0.0, K_override=engine.K_current)
            theta = out["theta1"]
            engine.update(_make_snap(guard_approved=False))

        # K should have reverted
        assert np.allclose(engine.K_current, K_before_veto, atol=1e-12)

        # 20 recovery ticks
        for _ in range(20):
            out = upde.step(theta, omega, psi_driver=0.0, K_override=engine.K_current)
            theta = out["theta1"]
            engine.update(_make_snap(beta_n=0.3))

        assert np.isfinite(out["R_global"])
        assert 0.0 <= out["R_global"] <= 1.0
