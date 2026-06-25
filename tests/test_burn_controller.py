# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Burn Control Tests
from __future__ import annotations

import numpy as np

from scpn_fusion.control.burn_controller import (
    AlphaHeating,
    BurnController,
    BurnStabilityAnalysis,
    SubignitedBurnPoint,
)


def test_zero_temperature():
    alpha = AlphaHeating(R0=6.2, a=2.0)
    P = alpha.power(np.array([1.0]), np.array([0.0]), np.array([0.0]), np.array([0.5]))
    assert P == 0.0


def test_iter_alpha_power():
    alpha = AlphaHeating(R0=6.2, a=2.0, kappa=1.7)

    # 20 keV, 1e20 m-3 flat
    rho = np.linspace(0, 1, 50)
    ne = np.ones(50) * 1.0
    Te = np.ones(50) * 20.0
    Ti = np.ones(50) * 20.0

    P_alpha = alpha.power(ne, Te, Ti, rho)

    # A completely flat 20 keV profile yields much higher power than a peaked profile.
    # ~ 500 MW is physically correct for a uniform 800 m^3 plasma at 20 keV.
    assert 400.0 < P_alpha < 600.0


def test_Q_definition():
    alpha = AlphaHeating(R0=6.2, a=2.0)
    Q = alpha.Q(P_alpha_MW=20.0, P_aux_MW=10.0)
    assert Q == 10.0

    # Ignition
    Q_ign = alpha.Q(P_alpha_MW=20.0, P_aux_MW=0.0)
    assert Q_ign == float("inf")


def test_stability_boundary():
    alpha = AlphaHeating(R0=6.2, a=2.0)
    analysis = BurnStabilityAnalysis(alpha)

    # Exponent < 2 for T > 15
    assert analysis.reactivity_exponent(20.0) < 2.0
    assert analysis.is_thermally_stable(20.0)

    # Exponent > 2 for T < 10
    assert analysis.reactivity_exponent(8.0) > 2.0
    assert not analysis.is_thermally_stable(8.0)

    T_bound = analysis.stability_boundary_keV()
    assert 12.0 < T_bound < 16.0


def test_burn_controller():
    ctrl = BurnController(Q_target=10.0, T_target_keV=20.0, P_aux_max_MW=50.0)

    # Test emergency cooling
    u_emerg = ctrl.step(Q_meas=15.0, T_meas_keV=35.0, P_alpha_MW=100.0, dt=0.1)
    assert u_emerg == 0.0

    # Test normal response (T too low -> increase power)
    ctrl.integral_T = 0.0
    u_norm = ctrl.step(Q_meas=5.0, T_meas_keV=10.0, P_alpha_MW=20.0, dt=0.1)

    # K_T_p = -5, e_T = -10 => +50 MW. Base is 25. Total > 50 -> clipped to 50
    assert u_norm == 50.0

    # T too high -> decrease power
    ctrl.integral_T = 0.0
    u_high = ctrl.step(Q_meas=10.0, T_meas_keV=25.0, P_alpha_MW=80.0, dt=0.1)

    # Feed-forward is 5 * P_alpha / Q_target = 40 MW.
    # K_T_p = -5, e_T = 5 => -25 MW, plus integral correction -0.5 MW.
    assert u_high == 14.5


def test_burn_controller_uses_alpha_power_feedforward() -> None:
    ctrl = BurnController(Q_target=10.0, T_target_keV=20.0, P_aux_max_MW=50.0)
    assert ctrl.feedforward_power_MW(40.0) == 20.0

    u_target = ctrl.step(Q_meas=10.0, T_meas_keV=20.0, P_alpha_MW=40.0, dt=0.1)
    assert u_target == 20.0

    u_clipped = ctrl.feedforward_power_MW(500.0)
    assert u_clipped == 50.0


def test_subignited_burn_point():
    alpha = AlphaHeating(R0=6.2, a=2.0, kappa=1.7)
    sbp = SubignitedBurnPoint(alpha)

    # Find operating points
    # Need high enough confinement for ignition or subignition with a flat profile
    tau_E = 3.0  # From simple sweep, Pa crosses Ploss around T=10 keV with tau_E=3
    ne_20 = 1.0
    P_aux = 10.0

    pts = sbp.find_operating_point(ne_20, P_aux, tau_E)

    # For these parameters, we typically get 1 or 2 intersection points
    assert len(pts) > 0

    # The higher T point should be stable (if two exist),
    # but finding any physically valid equilibrium point is sufficient.
    pts.sort(key=lambda p: p.Te_keV)
    assert pts[-1].P_alpha_MW > 0.0


import pytest

from scpn_fusion.control import burn_controller as burn_controller_mod


def test_reactivity_exponent_clamps_for_low_temperature():
    """Below 0.1 keV the finite-difference exponent is clamped to the cap."""
    analysis = BurnStabilityAnalysis(AlphaHeating(R0=6.2, a=2.0))
    assert analysis.reactivity_exponent(0.05) == 10.0


def test_reactivity_exponent_clamps_when_reactivity_underflows(monkeypatch):
    """A non-positive reactivity sample falls back to the capped exponent."""
    analysis = BurnStabilityAnalysis(AlphaHeating(R0=6.2, a=2.0))
    monkeypatch.setattr(
        burn_controller_mod, "_bosch_hale_reactivity", lambda arr: np.zeros_like(arr)
    )
    assert analysis.reactivity_exponent(20.0) == 10.0


def test_reactivity_exponent_is_positive_for_relevant_regime():
    """In the 10-30 keV burn regime D-T reactivity rises with temperature."""
    analysis = BurnStabilityAnalysis(AlphaHeating(R0=6.2, a=2.0))
    assert analysis.reactivity_exponent(15.0) > 0.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"Q_target": 0.0}, "Q_target must be positive"),
        ({"Q_target": -5.0}, "Q_target must be positive"),
        ({"P_aux_max_MW": 0.0}, "P_aux_max_MW must be positive"),
        ({"P_aux_max_MW": -1.0}, "P_aux_max_MW must be positive"),
    ],
)
def test_burn_controller_rejects_nonpositive_limits(kwargs, match):
    with pytest.raises(ValueError, match=match):
        BurnController(**kwargs)
