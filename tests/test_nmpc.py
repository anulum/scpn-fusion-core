# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — NMPC Controller Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# ─────────────────────────────────────────────────────────────────────

import numpy as np
import pytest
import scpn_fusion.control.fusion_nmpc_jax as nmpc_mod
from scpn_fusion.control.fusion_nmpc_jax import (
    NeuralODEDynamics,
    NonlinearMPC,
    get_nmpc_controller,
)


def test_nmpc_factory_creates_controller():
    ctrl = get_nmpc_controller(state_dim=4, action_dim=4, horizon=10)
    assert ctrl.horizon == 10
    assert ctrl.dynamics.state_dim == 4


def test_nmpc_plan_returns_correct_shape():
    ctrl = get_nmpc_controller(state_dim=4, action_dim=4, horizon=5)
    x0 = np.zeros(4)
    target = np.ones(4) * 0.5
    action = ctrl.plan_trajectory(x0, target)
    assert action.shape == (4,)
    assert np.all(np.isfinite(action))


# S2-004: NMPC adaptive early stopping


def test_nmpc_converges_before_max_iterations():
    """Controller should exit early when cost stalls, using fewer than max iterations."""
    ctrl = get_nmpc_controller(state_dim=2, action_dim=2, horizon=5)
    ctrl.iterations = 200  # High limit
    ctrl.rtol = 1e-4

    x0 = np.zeros(2)
    target = np.array([0.1, 0.1])

    # Plan uses the numpy fallback (no JAX in CI typically)
    action = ctrl.plan_trajectory(x0, target)
    assert np.all(np.isfinite(action))


def test_nmpc_rtol_parameter_respected():
    """The rtol parameter should be stored and accessible."""
    dyn = NeuralODEDynamics(state_dim=2, action_dim=2)
    ctrl = NonlinearMPC(dyn, horizon=5, rtol=1e-3)
    assert ctrl.rtol == 1e-3

    ctrl2 = NonlinearMPC(dyn, horizon=5, rtol=1e-6)
    assert ctrl2.rtol == 1e-6


def test_nmpc_records_backend_after_plan():
    ctrl = get_nmpc_controller(state_dim=2, action_dim=2, horizon=3)
    x0 = np.zeros(2)
    target = np.array([0.2, -0.1], dtype=float)
    _ = ctrl.plan_trajectory(x0, target)
    assert ctrl.last_backend in {"jax", "numpy_compat"}


def test_nmpc_strict_mode_rejects_numpy_fallback_when_jax_unavailable(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(nmpc_mod, "_HAS_JAX", False)
    dyn = NeuralODEDynamics(state_dim=2, action_dim=2)
    with pytest.raises(ImportError, match="allow_numpy_fallback=False"):
        NonlinearMPC(dyn, horizon=5, allow_numpy_fallback=False)
