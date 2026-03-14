# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Plasma Wall Interaction Tests
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np

from scpn_fusion.core.plasma_wall_interaction import (
    DivertorLifetimeAssessment,
    ErosionModel,
    SputteringYield,
    TransientThermalLoad,
    WallThermalModel,
)
from scpn_fusion.core.sol_model import TwoPointSOL


def test_sputtering_threshold():
    sputt = SputteringYield()

    assert sputt.yield_at_energy(100.0) == 0.0

    Y_1keV = sputt.yield_at_energy(1000.0)
    assert 0.001 < Y_1keV < 0.05


def test_sputtering_angular_dependence():
    sputt = SputteringYield()

    Y_0 = sputt.yield_at_energy(1000.0, 0.0)
    Y_60 = sputt.yield_at_energy(1000.0, 60.0)

    assert Y_60 > Y_0


def test_erosion_model():
    erosion = ErosionModel()

    flux = 1e24  # High flux
    gross = erosion.gross_erosion_rate(flux, 1000.0)

    assert gross > 0.0

    net = erosion.net_erosion_rate(gross, f_redeposition=0.99)
    assert np.isclose(net, gross * 0.01)


def test_wall_thermal_steady_state():
    wall = WallThermalModel()

    T_steady = wall.step(10.0, 10.0)  # 10 MW/m2

    assert T_steady > 400.0
    assert T_steady < wall.T_melt
    assert not wall.is_melted()


def test_wall_thermal_melting():
    wall = WallThermalModel()

    T_steady = wall.step(10.0, 100.0)  # 100 MW/m2 -> guaranteed to melt W

    assert T_steady > wall.T_melt
    assert wall.is_melted()


def test_transient_thermal_load():
    wall = WallThermalModel()
    trans = TransientThermalLoad(wall)

    delta_T = trans.elm_load(delta_W_MJ=20.0, A_wet_m2=2.0)

    # 20 MJ over 2 m2 in 0.25 ms is a massive load
    assert delta_T > 1000.0


def test_divertor_lifetime_assessment():
    sol = TwoPointSOL(R0=6.2, a=2.0, q95=3.0, B_pol=0.56)
    sputt = SputteringYield()
    eros = ErosionModel()
    wall = WallThermalModel()

    assessment = DivertorLifetimeAssessment(sol, sputt, eros, wall)

    rep = assessment.assess(P_SOL_MW=100.0, n_u_19=4.0, f_ELM_Hz=5.0, delta_W_ELM_MJ=1.0)

    assert rep.limiting_factor in ["Erosion", "Fatigue", "Melting"]
    if rep.limiting_factor == "Melting":
        assert rep.lifetime_years == 0.0
    else:
        assert rep.lifetime_years > 0.0
