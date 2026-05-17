# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Plasma Wall Interaction Tests
from __future__ import annotations

import numpy as np
import pytest

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


def test_sputtering_rejects_unsupported_species_and_invalid_inputs():
    with pytest.raises(ValueError, match="unsupported"):
        SputteringYield(target="Unobtainium", projectile="D")

    sputt = SputteringYield(target="W", projectile="D")
    with pytest.raises(ValueError, match="finite"):
        sputt.yield_at_energy(float("nan"))
    with pytest.raises(ValueError, match="theta_deg"):
        sputt.yield_at_energy(1000.0, theta_deg=-1.0)
    with pytest.raises(ValueError, match="theta_deg"):
        sputt.yield_at_energy(1000.0, theta_deg=90.0)


def test_sputtering_material_projectile_database_changes_thresholds():
    tungsten_d = SputteringYield(target="W", projectile="D")
    tungsten_he = SputteringYield(target="W", projectile="He")
    carbon_d = SputteringYield(target="C", projectile="D")

    thresholds = {
        round(tungsten_d.threshold_energy(), 6),
        round(tungsten_he.threshold_energy(), 6),
        round(carbon_d.threshold_energy(), 6),
    }

    assert min(thresholds) > 0.0
    assert len(thresholds) == 3


def test_erosion_model():
    erosion = ErosionModel()

    flux = 1e24  # High flux
    gross = erosion.gross_erosion_rate(flux, 1000.0)

    assert gross > 0.0

    net = erosion.net_erosion_rate(gross, f_redeposition=0.99)
    assert np.isclose(net, gross * 0.01)


def test_erosion_model_rejects_invalid_domain_values():
    erosion = ErosionModel()

    with pytest.raises(ValueError, match="ion_flux"):
        erosion.gross_erosion_rate(-1.0, 1000.0)
    with pytest.raises(ValueError, match="f_redeposition"):
        erosion.net_erosion_rate(1.0e20, f_redeposition=1.2)
    with pytest.raises(ValueError, match="wall_thickness_mm"):
        erosion.lifetime_estimate(-1.0, 1.0e-9)


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


def test_fatigue_life_decreases_with_swing_and_base_temperature():
    trans = TransientThermalLoad(WallThermalModel())

    mild = trans.n_elm_cycles_to_fatigue(delta_T_K=300.0, T_base_K=500.0)
    severe_swing = trans.n_elm_cycles_to_fatigue(delta_T_K=900.0, T_base_K=500.0)
    hot_base = trans.n_elm_cycles_to_fatigue(delta_T_K=300.0, T_base_K=1200.0)

    assert severe_swing < mild
    assert hot_base < mild

    with pytest.raises(ValueError, match="delta_T_K"):
        trans.n_elm_cycles_to_fatigue(delta_T_K=float("nan"))
    with pytest.raises(ValueError, match="T_base_K"):
        trans.n_elm_cycles_to_fatigue(delta_T_K=300.0, T_base_K=-1.0)


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
