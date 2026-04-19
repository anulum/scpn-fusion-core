# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Current Drive Physics Tests
from __future__ import annotations

import numpy as np

from scpn_fusion.core.current_drive import (
    CurrentDriveMix,
    ECCDSource,
    LHCDSource,
    NBISource,
)


def test_zero_power():
    rho = np.linspace(0, 1, 50)
    ne = np.ones(50)
    Te = np.ones(50)
    Ti = np.ones(50)

    eccd = ECCDSource(P_ec_MW=0.0, rho_dep=0.5, sigma_rho=0.05)
    assert np.allclose(eccd.j_cd(rho, ne, Te), 0.0)

    nbi = NBISource(P_nbi_MW=0.0, E_beam_keV=100.0, rho_tangency=0.0)
    assert np.allclose(nbi.j_cd(rho, ne, Te, Ti), 0.0)


def test_eccd_peaked_deposition():
    rho = np.linspace(0, 1, 100)
    ne = np.ones(100)
    Te = np.ones(100)

    eccd = ECCDSource(P_ec_MW=10.0, rho_dep=0.4, sigma_rho=0.05)
    j_cd = eccd.j_cd(rho, ne, Te)

    peak_idx = np.argmax(j_cd)
    assert 0.35 < rho[peak_idx] < 0.45


def test_eccd_total_driven_current():
    rho = np.linspace(0, 1, 100)
    ne = np.ones(100)
    Te = np.ones(100)
    Ti = np.ones(100)

    eccd = ECCDSource(P_ec_MW=10.0, rho_dep=0.4, sigma_rho=0.05)

    mix = CurrentDriveMix(a=1.0)
    mix.add_source(eccd)

    I_cd = mix.total_driven_current(rho, ne, Te, Ti)
    assert I_cd > 0.0


def test_nbi_tangential_vs_radial():
    rho = np.linspace(0, 1, 100)
    ne = np.ones(100)
    Te = np.ones(100)
    Ti = np.ones(100)

    nbi_tangential = NBISource(P_nbi_MW=10.0, E_beam_keV=100.0, rho_tangency=0.2)
    nbi_radial = NBISource(P_nbi_MW=10.0, E_beam_keV=100.0, rho_tangency=0.8)

    j_tang = nbi_tangential.j_cd(rho, ne, Te, Ti)
    j_rad = nbi_radial.j_cd(rho, ne, Te, Ti)

    assert np.argmax(j_tang) < np.argmax(j_rad)


def test_lhcd_off_axis():
    rho = np.linspace(0, 1, 100)
    ne = np.ones(100)
    Te = np.ones(100)

    lhcd = LHCDSource(P_lh_MW=5.0, rho_dep=0.7, sigma_rho=0.1)
    j_cd = lhcd.j_cd(rho, ne, Te)

    assert rho[np.argmax(j_cd)] > 0.5


def test_current_drive_mix():
    rho = np.linspace(0, 1, 100)
    ne = np.ones(100)
    Te = np.ones(100)
    Ti = np.ones(100)

    eccd = ECCDSource(10.0, 0.4, 0.05)
    lhcd = LHCDSource(5.0, 0.7, 0.1)

    mix = CurrentDriveMix()
    mix.add_source(eccd)
    mix.add_source(lhcd)

    j_tot = mix.total_j_cd(rho, ne, Te, Ti)
    p_tot = mix.total_heating_power(rho)

    j_eccd = eccd.j_cd(rho, ne, Te)
    j_lhcd = lhcd.j_cd(rho, ne, Te)

    assert np.allclose(j_tot, j_eccd + j_lhcd)
    assert np.allclose(p_tot, eccd.P_absorbed(rho) + lhcd.P_absorbed(rho))


def test_current_drive_efficiency():
    rho = np.linspace(0, 1, 200)
    ne = np.ones(200) * 1.0
    Te = np.ones(200) * 1.0
    Ti = np.ones(200) * 1.0

    eccd = ECCDSource(P_ec_MW=10.0, rho_dep=0.5, sigma_rho=0.05, eta_cd=0.03)
    mix = CurrentDriveMix(a=1.0)
    mix.add_source(eccd)

    I_cd = mix.total_driven_current(rho, ne, Te, Ti)
    assert I_cd > 1000.0
