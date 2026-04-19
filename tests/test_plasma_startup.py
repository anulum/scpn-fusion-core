# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Plasma Startup Tests
from __future__ import annotations


from scpn_fusion.core.plasma_startup import (
    BurnThrough,
    PaschenBreakdown,
    StartupController,
    StartupPhase,
    StartupSequence,
    TownsendAvalanche,
)


def test_paschen_breakdown():
    pb = PaschenBreakdown("D2")

    # Check optimal pressure
    p_opt = pb.optimal_prefill_pressure(V_loop_max=10.0)
    assert p_opt > 0.0

    # Check curve shape
    V_opt = pb.breakdown_voltage(p_opt, 100.0)
    V_high = pb.breakdown_voltage(p_opt * 10.0, 100.0)
    V_low = pb.breakdown_voltage(p_opt * 0.1, 100.0)

    assert V_high > V_opt
    assert V_low > V_opt

    assert pb.is_breakdown(V_loop=20.0, p_Pa=p_opt)
    assert not pb.is_breakdown(V_loop=0.01, p_Pa=p_opt)


def test_townsend_avalanche():
    pb = PaschenBreakdown()
    p_opt = pb.optimal_prefill_pressure(20.0)

    # Above threshold
    ava = TownsendAvalanche(V_loop=20.0, p_Pa=p_opt, R0=6.2, a=2.0)
    res = ava.evolve(1e-4, 100)

    assert res.time_to_full_ionization_ms > 0.0
    assert res.ne_trace[-1] > 1e16


def test_burn_through():
    bt = BurnThrough(R0=6.2, a=2.0, B0=5.3, V_loop=15.0)

    # Clean plasma
    res_clean = bt.evolve(ne_19=0.1, f_imp=1e-4, dt=1e-3, n_steps=100, impurity="C")
    assert res_clean.success
    assert res_clean.time_to_burn_through_ms > 0.0

    # Very Dirty plasma -> fails to burn through
    res_dirty = bt.evolve(ne_19=0.1, f_imp=0.9, dt=1e-3, n_steps=100, impurity="C")
    assert not res_dirty.success


def test_critical_impurity_fraction():
    bt = BurnThrough(R0=6.2, a=2.0, B0=5.3, V_loop=15.0)

    f_crit_C = bt.critical_impurity_fraction(Te_eV=10.0, ne_19=0.1, Ip_kA=100.0, impurity="C")
    f_crit_W = bt.critical_impurity_fraction(Te_eV=50.0, ne_19=0.1, Ip_kA=100.0, impurity="W")

    # W should be much lower than C
    assert f_crit_W < f_crit_C


def test_startup_sequence():
    pb = PaschenBreakdown("D2")
    p_opt = pb.optimal_prefill_pressure(V_loop_max=15.0)
    seq = StartupSequence(R0=6.2, a=2.0, B0=5.3, V_loop=15.0, p_prefill_Pa=p_opt, f_imp=1e-4)
    res = seq.run()

    assert res.success
    assert res.breakdown_time_ms > 0.0
    assert res.burn_through_time_ms > 0.0
    assert res.Ip_at_100ms_kA > 100.0


def test_startup_controller():
    ctrl = StartupController(V_loop_max=15.0, gas_puff_max=10.0)

    c1 = ctrl.step(ne=0.0, Te=0.0, Ip=0.0, t=0.05, dt=0.01)
    assert c1.phase == StartupPhase.GAS_PUFF
    assert c1.V_loop == 0.0

    c2 = ctrl.step(ne=0.0, Te=0.0, Ip=0.0, t=0.15, dt=0.01)
    assert c2.phase == StartupPhase.BREAKDOWN
    assert c2.V_loop == 15.0

    c3 = ctrl.step(ne=1e19, Te=5.0, Ip=100.0, t=0.2, dt=0.01)
    assert c3.phase == StartupPhase.BURN_THROUGH

    c4 = ctrl.step(ne=1e19, Te=100.0, Ip=200.0, t=0.3, dt=0.01)
    assert c4.phase == StartupPhase.RAMP
    assert c4.V_loop == 7.5
