# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — PyO3 Control Bridge Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""Tests for PyO3 bindings: fusion-control crate → Python.

Covers: MPC controller, Digital Twin (Plasma2D).
"""

import numpy as np
import pytest

try:
    import scpn_fusion_rs

    HAS_RUST = True
except ImportError:
    HAS_RUST = False

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="scpn_fusion_rs not compiled")


# ── WP-PY3: MPC Controller ──────────────────────────────────────────


class TestPyMpcController:
    """Tests for PyMpcController binding (fusion-control/mpc.rs)."""

    @pytest.fixture
    def mpc(self):
        """Create a simple 4-state, 2-coil MPC controller."""
        b_matrix = np.array(
            [[0.1, 0.0], [0.0, 0.1], [0.05, 0.05], [0.02, -0.02]],
            dtype=np.float64,
        )
        target = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return scpn_fusion_rs.PyMpcController(b_matrix, target)

    def test_plan_returns_action(self, mpc):
        state = np.array([0.5, -0.3, 0.1, 0.0], dtype=np.float64)
        action = mpc.plan(state)
        assert isinstance(action, np.ndarray)
        assert len(action) == 2  # n_coils

    def test_action_bounded(self, mpc):
        state = np.array([10.0, -10.0, 5.0, -5.0], dtype=np.float64)
        action = mpc.plan(state)
        assert np.all(np.abs(action) <= 2.1)  # ACTION_CLIP=2.0 + tolerance

    def test_tracks_target_over_10_steps(self, mpc):
        state = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        b_matrix = np.array(
            [[0.1, 0.0], [0.0, 0.1], [0.05, 0.05], [0.02, -0.02]],
            dtype=np.float64,
        )
        for _ in range(10):
            action = mpc.plan(state)
            state = state + b_matrix @ action
        # After 10 steps, state should be closer to zero target
        assert np.linalg.norm(state) < 1.3  # started at norm=2.0

    def test_rejects_nan_state(self, mpc):
        state = np.array([float("nan"), 0.0, 0.0, 0.0], dtype=np.float64)
        with pytest.raises((ValueError, RuntimeError)):
            mpc.plan(state)


# ── WP-PY6: Digital Twin (Plasma2D) ─────────────────────────────────


class TestPyPlasma2D:
    """Tests for Plasma2D binding (fusion-control/digital_twin.rs)."""

    def test_construction(self):
        plasma = scpn_fusion_rs.PyPlasma2D()
        assert plasma is not None

    def test_step_returns_tuple(self):
        plasma = scpn_fusion_rs.PyPlasma2D()
        temp, position = plasma.step(0.0)
        assert isinstance(temp, float) and np.isfinite(temp)
        assert isinstance(position, float) and np.isfinite(position)

    def test_measure_core_temp(self):
        plasma = scpn_fusion_rs.PyPlasma2D()
        plasma.step(0.5)
        temp = plasma.measure_core_temp(0.01)
        assert isinstance(temp, float) and np.isfinite(temp)

    def test_100_steps_stable(self):
        plasma = scpn_fusion_rs.PyPlasma2D()
        for _ in range(100):
            temp, pos = plasma.step(0.1)
            assert np.isfinite(temp) and np.isfinite(pos)

    def test_heating_raises_temperature(self):
        plasma = scpn_fusion_rs.PyPlasma2D()
        # Zero heating baseline
        temp_cold, _ = plasma.step(0.0)
        # Apply heating
        for _ in range(20):
            temp_hot, _ = plasma.step(1.0)
        assert temp_hot > temp_cold


# ── High-speed Flight Simulator ─────────────────────────────────────


class TestPyRustFlightSim:
    """Tests for RustFlightSim binding (fusion-control/flight_sim.rs)."""

    def test_construction(self):
        sim = scpn_fusion_rs.PyRustFlightSim(6.2, 0.0, 10000.0)
        assert sim is not None

    def test_run_shot_returns_valid_report(self):
        sim = scpn_fusion_rs.PyRustFlightSim(6.2, 0.0, 10000.0)
        report = sim.run_shot(0.1)  # 0.1s @ 10kHz = 1000 steps
        assert report.steps == 1000
        assert report.duration_s == pytest.approx(0.1)
        assert report.wall_time_ms > 0
        assert np.isfinite(report.mean_abs_r_error)
        assert np.isfinite(report.mean_abs_z_error)
        assert isinstance(report.disrupted, bool)
        assert len(report.r_history) == 1000
        assert len(report.z_history) == 1000
        assert 0 <= report.vessel_contact_events <= report.steps
        assert 0 <= report.pf_constraint_events <= report.steps
        assert 0 <= report.heating_constraint_events <= report.steps
        assert report.retained_steps == report.steps
        assert report.history_truncated is False

    def test_30khz_stability(self):
        sim = scpn_fusion_rs.PyRustFlightSim(6.2, 0.0, 30000.0)
        report = sim.run_shot(0.05)
        assert report.steps == 1500
        assert not report.disrupted  # Baseline should be stable
        assert report.max_step_time_us > 0

    def test_report_exposes_heating_beta_contracts(self):
        sim = scpn_fusion_rs.PyRustFlightSim(6.2, 0.0, 10000.0)
        report = sim.run_shot(0.02)

        assert np.isfinite(report.final_beta)
        assert np.isfinite(report.final_heating_mw)
        assert np.isfinite(report.max_beta)
        assert np.isfinite(report.max_heating_mw)

        assert 0.2 <= report.final_beta <= 10.0
        assert 0.2 <= report.max_beta <= 10.0
        assert report.max_beta >= report.final_beta

        assert 0.0 <= report.final_heating_mw <= 100.0
        assert 0.0 <= report.max_heating_mw <= 100.0
        assert report.max_heating_mw >= report.final_heating_mw
        assert 0 <= report.pf_constraint_events <= report.steps
        assert 0 <= report.heating_constraint_events <= report.steps

    def test_prepare_and_step_once_api(self):
        sim = scpn_fusion_rs.PyRustFlightSim(6.2, 0.0, 10000.0)
        steps = sim.prepare_shot(0.01)
        assert steps == 100

        first = sim.step_once(0, 0.01)
        mid = sim.step_once(50, 0.01)
        last = sim.step_once(99, 0.01)

        for step in (first, mid, last):
            assert np.isfinite(step.r_error)
            assert np.isfinite(step.z_error)
            assert np.isfinite(step.step_time_us)
            assert np.isfinite(step.beta)
            assert np.isfinite(step.heating_mw)
            assert 0.2 <= step.beta <= 10.0
            assert 0.0 <= step.heating_mw <= 100.0
            assert isinstance(step.vessel_contact, bool)
            assert isinstance(step.pf_constraint_active, bool)
            assert isinstance(step.heating_constraint_active, bool)

        assert first.heating_mw <= mid.heating_mw <= last.heating_mw
        with pytest.raises(RuntimeError):
            sim.step_once(steps, 0.01)

    def test_reset_plasma_state_restores_nominal(self):
        sim = scpn_fusion_rs.PyRustFlightSim(6.2, 0.0, 10000.0)
        sim.run_shot(0.01)
        sim.reset_plasma_state()
        state = sim.plasma_state()
        assert state.r == pytest.approx(6.2)
        assert state.z == pytest.approx(0.0)
        assert state.ip_ma == pytest.approx(5.0)
        assert state.beta == pytest.approx(1.0)
        assert state.heating_mw == pytest.approx(20.0)
