# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tests
"""Pytest tests for fusion ignition simulation (DynamicBurnModel)."""

from __future__ import annotations

import numpy as np

from scpn_fusion.core.fusion_ignition_sim import DynamicBurnModel
from scpn_fusion.core.uncertainty import _dt_reactivity


def test_iter98y2_tau_e_power_degradation():
    """Confinement time must decrease with increasing heating power (IPB98y2)."""
    model = DynamicBurnModel()
    tau_prev = float("inf")
    for p in [10.0, 30.0, 60.0, 100.0]:
        tau = model.iter98y2_tau_e(p)
        assert tau < tau_prev
        tau_prev = tau


def test_bosch_hale_dt_positivity():
    """Reactivity must be positive for T in [1, 100] keV."""
    for t in np.linspace(1.0, 100.0, 20):
        sv = _dt_reactivity(t)
        val = float(np.asarray(sv).ravel()[0])
        assert val > 0.0, f"Reactivity non-positive at {t} keV"


def test_calculate_thermodynamics_finite():
    """DynamicBurnModel.simulate() output must contain all-finite values."""
    model = DynamicBurnModel()
    result = model.simulate(
        P_aux_mw=50.0,
        duration_s=1.0,
        dt_s=0.1,
        warn_on_temperature_cap=False,
    )
    for key in (
        "P_fus_MW",
        "P_alpha_MW",
        "P_loss_MW",
        "Q",
        "T_keV",
        "tau_E_s",
        "W_MJ",
    ):
        arr = np.asarray(result[key])
        assert np.all(np.isfinite(arr)), f"{key} contains non-finite values"
    assert result["Q_final"] >= 0.0
    assert result["T_final_keV"] > 0.0


def test_h_mode_threshold():
    model = DynamicBurnModel()
    P_thr = model.h_mode_threshold_mw()
    assert P_thr > 0
    assert np.isfinite(P_thr)


def test_h_mode_threshold_scales_with_field():
    m_low = DynamicBurnModel(B_t=3.0)
    m_high = DynamicBurnModel(B_t=12.0)
    assert m_high.h_mode_threshold_mw() > m_low.h_mode_threshold_mw()


def test_bosch_hale_peak_near_67_kev():
    """D-T reactivity peaks near ~67 keV."""
    model = DynamicBurnModel()
    svs = [model.bosch_hale_dt(T) for T in np.linspace(10, 100, 50)]
    peak_T = np.linspace(10, 100, 50)[np.argmax(svs)]
    assert 50 < peak_T < 80


def test_simulate_returns_q_and_power():
    model = DynamicBurnModel()
    result = model.simulate(P_aux_mw=50.0, duration_s=2.0, dt_s=0.1, warn_on_temperature_cap=False)
    assert "Q" in result
    assert "P_fus_MW" in result
    assert result["Q_final"] >= 0.0


def test_simulate_custom_params():
    model = DynamicBurnModel(R0=1.85, a=0.6, B_t=12.2, I_p=8.7, kappa=1.97)
    result = model.simulate(P_aux_mw=25.0, duration_s=0.5, dt_s=0.05, warn_on_temperature_cap=False)
    assert result["Q_final"] >= 0.0


def test_plasma_volume():
    model = DynamicBurnModel(R0=6.2, a=2.0, kappa=1.7)
    V = model.V_plasma
    # V = 2 pi^2 R0 a^2 kappa = 2 * 9.87 * 6.2 * 4.0 * 1.7 ≈ 832
    assert 800 < V < 900


def test_rejects_nonpositive_params():
    import pytest as _pt

    with _pt.raises(ValueError):
        DynamicBurnModel(R0=0.0)
    with _pt.raises(ValueError):
        DynamicBurnModel(B_t=-1.0)
