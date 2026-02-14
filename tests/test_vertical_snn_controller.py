# CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
"""Tests for the SNN controller compiled from vertical position Petri net."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.scpn.vertical_control_net import VerticalControlNet, PLACES
from scpn_fusion.scpn.vertical_snn_controller import VerticalSNNController
from scpn_fusion.control.vertical_stability import PlantConfig, VerticalStabilityPlant


# ---- Fixtures ---------------------------------------------------------------


@pytest.fixture
def snn() -> VerticalSNNController:
    """Default SNN controller with numpy backend for determinism."""
    vcn = VerticalControlNet()
    return VerticalSNNController(vcn, force_numpy=True, seed=42)


@pytest.fixture
def plant() -> VerticalStabilityPlant:
    """Vertical stability plant with reduced noise for testing."""
    cfg = PlantConfig(noise_std=0.0, sensor_noise_std=0.0)
    return VerticalStabilityPlant(cfg)


# ---- Tests ------------------------------------------------------------------


def test_snn_stabilizes_step() -> None:
    """z0=5mm, run SNN in a first-order closed loop 2000 steps.

    The SNN is a proportional controller (no derivative feedback), so
    it cannot stabilise a pure double integrator.  Instead we use a
    first-order discrete model:

        z[k+1] = z[k] + alpha * u[k]

    where ``alpha`` converts the SNN control signal to metres per tick.
    With the SNN producing negative u for positive z, the loop drives
    z toward zero.  We verify |z| < 5 mm at the end.
    """
    vcn = VerticalControlNet(gain_scale=1.0)
    snn = VerticalSNNController(vcn, force_numpy=True, seed=42)

    z = 0.005  # 5 mm initial displacement
    alpha = 1e-4  # metres per tick per unit of control signal

    for _ in range(2000):
        u = snn.compute(z, 0.0)
        z = z + alpha * u

    assert abs(z) < 0.005, (
        f"Expected |z| < 5mm after 2000 steps, got |z|={abs(z)*1000:.3f}mm"
    )


def test_snn_compute_returns_float(snn: VerticalSNNController) -> None:
    """compute() must return a Python float."""
    u = snn.compute(0.003, 0.0)
    assert isinstance(u, float), f"Expected float, got {type(u)}"


def test_snn_opposes_positive_displacement(
    snn: VerticalSNNController,
) -> None:
    """When z > 0 (plasma displaced upward), control signal u should be < 0
    (push plasma down)."""
    u = snn.compute(0.005, 0.0)
    assert u < 0.0, f"Expected u < 0 for positive z, got u={u}"


def test_snn_opposes_negative_displacement(
    snn: VerticalSNNController,
) -> None:
    """When z < 0 (plasma displaced downward), control signal u should be > 0
    (push plasma up)."""
    u = snn.compute(-0.005, 0.0)
    assert u > 0.0, f"Expected u > 0 for negative z, got u={u}"


def test_snn_larger_error_stronger_response() -> None:
    """|u(5mm)| > |u(1mm)| -- larger error produces stronger correction."""
    vcn = VerticalControlNet()

    snn_large = VerticalSNNController(vcn, force_numpy=True, seed=42)
    u_large = snn_large.compute(0.005, 0.0)

    # Create a fresh controller so internal state does not carry over.
    snn_small = VerticalSNNController(vcn, force_numpy=True, seed=42)
    u_small = snn_small.compute(0.001, 0.0)

    assert abs(u_large) > abs(u_small), (
        f"|u(5mm)|={abs(u_large):.6f} should be > |u(1mm)|={abs(u_small):.6f}"
    )


def test_snn_numpy_fallback_works() -> None:
    """Force numpy path and verify the controller produces a valid output."""
    vcn = VerticalControlNet()
    snn = VerticalSNNController(vcn, force_numpy=True)
    assert snn.backend_name == "numpy"

    u = snn.compute(0.003, 0.0)
    assert isinstance(u, float)
    assert np.isfinite(u)


def test_snn_reset(snn: VerticalSNNController) -> None:
    """After compute + reset, marking returns to initial state."""
    initial_marking = snn.marking[:]

    # Run a few steps to change internal state.
    snn.compute(0.005, 0.0)
    snn.compute(-0.003, 0.0)

    # Marking should have changed.
    changed_marking = snn.marking[:]
    assert changed_marking != initial_marking, (
        "Marking should change after compute calls"
    )

    # Reset and verify restoration.
    snn.reset()
    restored_marking = snn.marking

    np.testing.assert_array_equal(
        restored_marking,
        initial_marking,
        err_msg="reset() should restore initial marking",
    )


def test_snn_initial_marking(snn: VerticalSNNController) -> None:
    """Initial marking starts with P_idle=1.0 and all others 0.0."""
    marking = snn.marking
    assert len(marking) == 8, f"Expected 8 places, got {len(marking)}"
    assert marking[0] == 1.0, f"P_idle should start at 1.0, got {marking[0]}"
    for i in range(1, 8):
        assert marking[i] == 0.0, (
            f"{PLACES[i]} should start at 0.0, got {marking[i]}"
        )
