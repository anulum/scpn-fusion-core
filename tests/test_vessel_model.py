# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Vessel Model Tests
"""
Tests for the vacuum vessel eddy current model.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.vessel_model import VesselElement, VesselModel


def test_vessel_current_decay():
    """Verify that currents decay exponentially with the L/R time constant."""
    # Single element for simplicity
    L = 1e-6
    R_res = 1e-3
    tau = L / R_res  # 1ms

    el = VesselElement(R=1.0, Z=0.0, resistance=R_res, cross_section=0.01, inductance=L)
    model = VesselModel([el])

    # Initial current
    model.I = np.array([1000.0])

    # Use small dt to minimize Euler error
    dt = 1e-6  # 0.01ms
    # Evolve 500 steps (0.5ms)
    for _ in range(500):
        model.step(dt, np.array([0.0]))

    # Expected: I = I0 * exp(-t/tau) = 1000 * exp(-0.5) approx 606.5
    expected = 1000.0 * np.exp(-0.5e-3 / tau)
    # With dt=1e-6, Euler error should be small enough for rel=0.01
    assert model.I[0] == pytest.approx(expected, rel=0.01)


def test_vessel_symmetry():
    """Verify that a symmetric vessel responds symmetrically to symmetric drive."""
    # Two symmetric elements at +/- Z
    el1 = VesselElement(R=1.5, Z=0.5, resistance=1e-3, cross_section=0.01, inductance=1e-6)
    el2 = VesselElement(R=1.5, Z=-0.5, resistance=1e-3, cross_section=0.01, inductance=1e-6)

    model = VesselModel([el1, el2])

    # Symmetric drive (e.g. from plasma current change)
    drive = np.array([10.0, 10.0])
    model.step(0.01, drive)

    assert model.I[0] == pytest.approx(model.I[1], rel=1e-10)


def test_psi_vessel_axisymmetry():
    """Verify that the flux contribution is axisymmetric (constant in theta).

    Actually, our model is already 2D (R, Z), so we check that psi(R, Z)
    is calculated correctly at multiple points.
    """
    el = VesselElement(R=1.0, Z=0.0, resistance=1e-3, cross_section=0.01, inductance=1e-6)
    model = VesselModel([el])
    model.I = np.array([100.0])

    # Check at two points with same R but different Z (should differ)
    # and check at two points with same Z but different R (should differ)
    # This just ensures the Green's function is working.
    R_obs = np.array([1.5, 1.5, 2.0])
    Z_obs = np.array([0.0, 0.5, 0.0])

    psi = model.psi_vessel(R_obs, Z_obs)

    assert psi[0] != psi[1]
    assert psi[0] != psi[2]
    assert np.all(psi > 0)  # Current is positive, R is positive -> psi should be positive
