# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Pedestal Model Tests
"""
Tests for the mtanh pedestal profile and EPED1 width scaling.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.pedestal import PedestalParams, PedestalProfile, pedestal_width_eped1


def test_pedestal_monotonicity():
    """Verify that the pedestal profile is monotonically decreasing."""
    # Use parameters that ensure mtanh is well-behaved near edges
    params = PedestalParams(f_ped=5.0, f_sep=0.1, x_ped=0.96, delta=0.02, a=0.1)
    profile = PedestalProfile(params)

    rho = np.linspace(0, 1, 100)
    vals = profile.evaluate(rho)

    # Check monotonicity: f(rho_i) >= f(rho_{i+1})
    diffs = np.diff(vals)
    # Allow 1e-12 for floating point precision
    assert np.all(diffs <= 1e-12), f"Pedestal profile is not monotonic. Max diff: {np.max(diffs)}"

    # Check boundary values approx
    assert vals[0] > params.f_ped
    # At rho=1, y = (0.96-1.0)/0.02 = -2.0. mtanh(-2.0) approx -0.964.
    # mid = 2.55, half = 2.45. vals[-1] = 2.55 + 2.45*(-0.964) = 2.55 - 2.36 = 0.19.
    # Closer to f_sep=0.1 than 0.29.
    assert vals[-1] < 0.25


def test_eped1_scaling_values():
    """Verify EPED1 scaling matches known JET/ITER-like values.

    JET baseline: beta_p_ped ~ 0.25 -> Delta ~ 0.038
    ITER baseline: beta_p_ped ~ 0.35 -> Delta ~ 0.045
    """
    # JET
    delta_jet = pedestal_width_eped1(0.25)
    assert delta_jet == pytest.approx(0.038, rel=0.01)

    # ITER
    delta_iter = pedestal_width_eped1(0.35)
    assert delta_iter == pytest.approx(0.045, rel=0.01)

    # 20% tolerance check as requested
    assert 0.03 <= delta_jet <= 0.05
    assert 0.035 <= delta_iter <= 0.055


def test_pedestal_evaluate_shape():
    """Verify evaluate handles array inputs and returns correct shape."""
    params = PedestalParams(f_ped=1.0)
    profile = PedestalProfile(params)

    rho = np.array([0.5, 0.95])
    vals = profile.evaluate(rho)
    assert vals.shape == (2,)
    assert vals[0] > vals[1]
