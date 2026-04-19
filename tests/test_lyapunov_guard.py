# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Lyapunov Guard Tests
"""Tests for LyapunovGuard, LyapunovVerdict, and TrajectoryRecorder."""

from __future__ import annotations

import numpy as np

from scpn_fusion.phase.lyapunov_guard import LyapunovGuard, LyapunovVerdict


class TestLyapunovVerdict:
    def test_score_near_zero_lambda(self):
        v = LyapunovVerdict(v=0.5, lambda_exp=0.0, approved=True, consecutive_violations=0)
        assert 0.4 < v.score < 0.6

    def test_score_negative_lambda_high(self):
        v = LyapunovVerdict(v=0.1, lambda_exp=-5.0, approved=True, consecutive_violations=0)
        assert v.score > 0.99

    def test_score_positive_lambda_low(self):
        v = LyapunovVerdict(v=0.9, lambda_exp=5.0, approved=False, consecutive_violations=3)
        assert v.score < 0.01

    def test_score_bounded_01(self):
        for lam in [-100.0, -1.0, 0.0, 1.0, 100.0]:
            v = LyapunovVerdict(v=0.5, lambda_exp=lam, approved=True, consecutive_violations=0)
            assert 0.0 <= v.score <= 1.0


class TestLyapunovGuard:
    def test_first_check_always_approved(self):
        guard = LyapunovGuard(window=10, dt=0.01)
        theta = np.zeros(50)
        verdict = guard.check(theta, 0.0)
        assert verdict.approved is True
        assert verdict.lambda_exp == 0.0

    def test_stable_trajectory_stays_approved(self):
        guard = LyapunovGuard(window=20, dt=0.01, max_violations=3)
        rng = np.random.default_rng(42)
        for _ in range(30):
            theta = rng.uniform(-0.1, 0.1, size=50)
            verdict = guard.check(theta, 0.0)
        assert verdict.approved is True

    def test_diverging_trajectory_triggers_refusal(self):
        guard = LyapunovGuard(window=10, dt=0.01, lambda_threshold=0.0, max_violations=3)
        ever_refused = False
        for i in range(50):
            spread = 0.1 * (i + 1)
            theta = np.linspace(-spread, spread, 50)
            verdict = guard.check(theta, 0.0)
            if not verdict.approved:
                ever_refused = True
        assert ever_refused, "guard should have refused at least once during diverging trajectory"

    def test_reset_clears_state(self):
        guard = LyapunovGuard(window=10, dt=0.01, max_violations=2)
        for i in range(20):
            theta = np.linspace(-0.5 * i, 0.5 * i, 50)
            guard.check(theta, 0.0)
        guard.reset()
        theta = np.zeros(50)
        verdict = guard.check(theta, 0.0)
        assert verdict.approved is True
        assert verdict.consecutive_violations == 0

    def test_check_trajectory_stable(self):
        guard = LyapunovGuard(window=20, dt=0.01)
        v_hist = [0.5, 0.4, 0.35, 0.3, 0.28, 0.25]
        verdict = guard.check_trajectory(v_hist)
        assert verdict.approved is True
        assert verdict.lambda_exp < 0.0

    def test_check_trajectory_growing(self):
        guard = LyapunovGuard(window=20, dt=0.01)
        v_hist = [0.1, 0.2, 0.4, 0.8, 1.6]
        verdict = guard.check_trajectory(v_hist)
        assert verdict.lambda_exp > 0.0
        assert verdict.approved is False

    def test_check_trajectory_single_sample(self):
        guard = LyapunovGuard()
        verdict = guard.check_trajectory([0.5])
        assert verdict.approved is True
        assert verdict.v == 0.5

    def test_check_trajectory_empty(self):
        guard = LyapunovGuard()
        verdict = guard.check_trajectory([])
        assert verdict.approved is True
        assert verdict.v == 0.0

    def test_to_director_ai_dict_approved(self):
        guard = LyapunovGuard()
        verdict = LyapunovVerdict(v=0.1, lambda_exp=-0.5, approved=True, consecutive_violations=0)
        d = guard.to_director_ai_dict(verdict)
        assert d["approved"] is True
        assert d["halt_reason"] == ""
        assert "lyapunov" in d["query"]

    def test_to_director_ai_dict_refused(self):
        guard = LyapunovGuard()
        verdict = LyapunovVerdict(v=0.9, lambda_exp=1.5, approved=False, consecutive_violations=5)
        d = guard.to_director_ai_dict(verdict)
        assert d["approved"] is False
        assert "5 windows" in d["halt_reason"]
        assert d["h_factual"] == 1.5


class TestTrajectoryRecorder:
    def test_record_and_n_ticks(self):
        from scpn_fusion.phase.realtime_monitor import TrajectoryRecorder

        rec = TrajectoryRecorder()
        snap = {
            "R_global": 0.5,
            "R_layer": [0.5],
            "V_global": 0.1,
            "V_layer": [0.1],
            "lambda_exp": -0.01,
            "guard_approved": True,
            "latency_us": 12.0,
            "Psi_global": 0.0,
        }
        rec.record(snap)
        rec.record(snap)
        assert rec.n_ticks == 2

    def test_clear(self):
        from scpn_fusion.phase.realtime_monitor import TrajectoryRecorder

        rec = TrajectoryRecorder()
        snap = {
            "R_global": 0.5,
            "R_layer": [0.5],
            "V_global": 0.1,
            "V_layer": [0.1],
            "lambda_exp": -0.01,
            "guard_approved": True,
            "latency_us": 12.0,
            "Psi_global": 0.0,
        }
        rec.record(snap)
        rec.clear()
        assert rec.n_ticks == 0
