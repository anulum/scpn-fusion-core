# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3
# ──────────────────────────────────────────────────────────────────────
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
