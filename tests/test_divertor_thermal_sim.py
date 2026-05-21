# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Divertor Thermal Sim Tests
from __future__ import annotations

import pytest

from scpn_fusion.core.divertor_thermal_sim import DivertorLab


def test_2point_target_temperature_decreases_with_radiative_fraction() -> None:
    lab = DivertorLab(P_sol_MW=80.0, R_major=2.1, B_pol=2.5)
    t_u0, t_t0 = lab.solve_2point_transport(expansion_factor=15.0, f_rad=0.0)
    t_u1, t_t1 = lab.solve_2point_transport(expansion_factor=15.0, f_rad=0.8)
    assert t_u0 == pytest.approx(t_u1)
    assert t_t1 < t_t0
    assert 1.0 <= t_t1 <= t_u1


def test_2point_target_temperature_decreases_with_flux_expansion() -> None:
    lab = DivertorLab(P_sol_MW=80.0, R_major=2.1, B_pol=2.5)
    _, t_t_small = lab.solve_2point_transport(expansion_factor=5.0, f_rad=0.3)
    _, t_t_large = lab.solve_2point_transport(expansion_factor=20.0, f_rad=0.3)
    assert t_t_large < t_t_small


def test_2point_transport_rejects_invalid_operating_inputs() -> None:
    lab = DivertorLab(P_sol_MW=80.0, R_major=2.1, B_pol=2.5)
    with pytest.raises(ValueError, match="expansion_factor"):
        lab.solve_2point_transport(expansion_factor=0.0, f_rad=0.2)
    with pytest.raises(ValueError, match="f_rad"):
        lab.solve_2point_transport(expansion_factor=10.0, f_rad=1.0)
