# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Volt-Second Tests
from __future__ import annotations

import numpy as np

from scpn_fusion.control.volt_second_manager import (
    BootstrapCurrentEstimate,
    FluxBudget,
    FluxConsumptionMonitor,
    ScenarioFluxAnalysis,
    VoltSecondOptimizer,
)


def test_flux_budget():
    fb = FluxBudget(Phi_CS_Vs=280.0, L_plasma_uH=10.0, R_plasma_uOhm=10.0)

    ind = fb.inductive_flux(15.0)
    assert np.isclose(ind, 150.0)

    Ip_trace = np.linspace(0, 15.0, 100)
    dt = 1.0  # 100s ramp
    ramp_res = fb.resistive_flux_ramp(Ip_trace, dt)
    assert ramp_res > 0.0

    t_flat = fb.max_flattop_duration(Ip_MA=15.0, I_bs_MA=5.0, ramp_flux=30.0)
    # Remaining = 280 - 150 - 30 = 100 Vs
    # V_loop = R * I_driven = 10e-6 * 10e6 = 100 V ? No, 10e-6 * 10e6 = 10 V
    # Ah, 100 Vs / 100 V = 1s?
    # R_plasma = 10 uOhm = 1e-5. I_driven = 10 MA = 1e7.
    # V_loop = 1e-5 * 1e7 = 100 V. Wait, standard R_plasma is ~10 nOhm, so 0.01 uOhm.
    # We used 10 uOhm -> V_loop is 100V. So flat top is short. But the logic works.
    assert t_flat > 0.0


def test_optimizer():
    fb = FluxBudget(280.0, 10.0, 10.0)
    opt = VoltSecondOptimizer(fb)

    ramp = opt.optimize_ramp(15.0, 100.0, 20)
    assert len(ramp) == 20
    assert ramp[-1] == 15.0


def test_bootstrap():
    r = np.linspace(0, 1, 50)
    ne = np.ones(50) * 5.0
    Te = np.linspace(10.0, 0.1, 50)

    I_bs = BootstrapCurrentEstimate.from_profiles(ne, Te, ne, ne, r, 6.2, 2.0)
    assert I_bs > 0.0


def test_monitor():
    fb = FluxBudget(280.0, 10.0, 10.0)
    mon = FluxConsumptionMonitor(fb)

    st = mon.step(15.0, 1.0, 10.0)  # Consumed 10 Vs
    assert st.flux_consumed_Vs == 10.0
    assert st.flux_remaining_Vs == 270.0


def test_scenario():
    fb = FluxBudget(280.0, 10.0, 0.01)  # Realistic R_plasma = 10 nOhm
    an = ScenarioFluxAnalysis(fb)

    rep = an.analyze(ramp_dur=100.0, flat_dur=400.0, down_dur=100.0, Ip_MA=15.0, I_bs_MA=5.0)

    assert rep.ramp_flux > 150.0
    assert rep.within_budget
