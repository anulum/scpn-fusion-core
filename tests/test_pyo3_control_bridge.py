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
        assert np.linalg.norm(state) < 1.0  # started at norm=2.0

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
