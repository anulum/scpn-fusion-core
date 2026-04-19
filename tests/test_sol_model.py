# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — SOL Two-Point Model Tests
from __future__ import annotations

import numpy as np

from scpn_fusion.core.sol_model import (
    TwoPointSOL,
    eich_heat_flux_width,
    peak_target_heat_flux,
)


def test_iter_like_sol():
    R0 = 6.2
    a = 2.0
    q95 = 3.0
    B_pol = 5.3 * (2.0 / 6.2) / 3.0

    sol = TwoPointSOL(R0, a, q95, B_pol)

    P_SOL = 100.0
    n_u = 4.0
    res = sol.solve(P_SOL, n_u, f_rad=0.9)

    assert res.T_upstream_eV > 100.0
    assert res.T_target_eV < res.T_upstream_eV

    p_u = n_u * res.T_upstream_eV
    p_t = res.n_target_19 * res.T_target_eV
    assert np.isclose(p_u, 2.0 * p_t, rtol=1e-2)


def test_detachment_density_scan():
    R0 = 6.2
    a = 2.0
    q95 = 3.0
    B_pol = 0.56

    sol = TwoPointSOL(R0, a, q95, B_pol)

    res_low = sol.solve(100.0, 3.0, f_rad=0.8)
    res_high = sol.solve(100.0, 10.0, f_rad=0.8)

    assert res_high.T_target_eV < res_low.T_target_eV


def test_eich_scaling():
    lam = eich_heat_flux_width(P_SOL_MW=100.0, R0=6.2, B_pol=0.56, epsilon=2.0 / 6.2)
    assert 0.5 < lam < 2.5


def test_peak_heat_flux():
    q_peak = peak_target_heat_flux(
        P_SOL_MW=100.0, R0=6.2, lambda_q_m=0.001, f_expansion=5.0, alpha_deg=3.0
    )
    assert q_peak > 10.0


def test_power_scan():
    R0 = 6.2
    a = 2.0
    q95 = 3.0
    B_pol = 0.56

    sol = TwoPointSOL(R0, a, q95, B_pol)

    res_low_p = sol.solve(10.0, 3.0)
    res_high_p = sol.solve(100.0, 3.0)

    assert res_high_p.T_upstream_eV > res_low_p.T_upstream_eV
    assert res_high_p.T_target_eV > res_low_p.T_target_eV
