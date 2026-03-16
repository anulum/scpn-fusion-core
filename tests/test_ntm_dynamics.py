# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — NTM Dynamics Tests
from __future__ import annotations

import numpy as np

from scpn_fusion.core.ntm_dynamics import (
    NTMIslandDynamics,
    find_rational_surfaces,
)


def test_find_rational_surfaces():
    rho = np.linspace(0, 1, 100)
    q = np.linspace(1.0, 3.5, 100)
    a = 1.0

    surfaces = find_rational_surfaces(q, rho, a, m_max=5, n_max=3)
    q_vals = [s.q for s in surfaces]
    assert 1.5 in q_vals
    assert 2.0 in q_vals
    assert 3.0 in q_vals


def test_classical_stable_decay():
    ntm = NTMIslandDynamics(r_s=0.5, m=2, n=1, a=1.0, R0=3.0, B0=2.0)
    t, w = ntm.evolve(
        w0=0.05,
        t_span=(0.0, 1.0),
        dt=0.01,
        j_bs=0.0,
        j_phi=1e6,
        j_cd=0.0,
        eta=1e-7,
    )
    assert w[-1] < w[0]


def test_bootstrap_drive_growth():
    ntm = NTMIslandDynamics(r_s=0.5, m=2, n=1, a=1.0, R0=3.0, B0=2.0)
    t, w = ntm.evolve(
        w0=0.01,
        t_span=(0.0, 1.0),
        dt=0.01,
        j_bs=1e5,
        j_phi=1e6,
        j_cd=0.0,
        eta=1e-7,
    )
    assert w[-1] > w[0]


def test_island_saturation():
    ntm = NTMIslandDynamics(r_s=0.5, m=2, n=1, a=1.0, R0=3.0, B0=2.0)
    t, w = ntm.evolve(
        w0=0.01,
        t_span=(0.0, 10.0),
        dt=0.1,
        j_bs=5e4,
        j_phi=1e6,
        j_cd=0.0,
        eta=1e-7,
    )
    dw_dt = ntm.dw_dt(w[-1], j_bs=5e4, j_phi=1e6, j_cd=0.0, eta=1e-7)
    assert abs(dw_dt) < 1e-2
    assert w[-1] > 0.01


def test_eccd_stabilization():
    ntm = NTMIslandDynamics(r_s=0.5, m=2, n=1, a=1.0, R0=3.0, B0=2.0)

    w0 = 0.05
    j_bs = 1e5
    j_phi = 1e6
    eta = 1e-7

    dw_dt_no_cd = ntm.dw_dt(w0, j_bs, j_phi, 0.0, eta)
    assert dw_dt_no_cd > 0

    dw_dt_with_cd = ntm.dw_dt(w0, j_bs, j_phi, 2e5, eta, d_cd=0.05)
    assert dw_dt_with_cd < 0


def test_eccd_misalignment():
    ntm = NTMIslandDynamics(r_s=0.5, m=2, n=1, a=1.0, R0=3.0, B0=2.0)

    w0 = 0.05
    j_bs = 1e5
    j_phi = 1e6
    eta = 1e-7
    j_cd = 2e5

    dw_dt_aligned = ntm.dw_dt(w0, j_bs, j_phi, j_cd, eta, d_cd=0.05)
    dw_dt_broad = ntm.dw_dt(w0, j_bs, j_phi, j_cd, eta, d_cd=0.5)

    assert dw_dt_aligned < dw_dt_broad


def test_polarization_threshold():
    ntm = NTMIslandDynamics(r_s=0.5, m=2, n=1, a=1.0, R0=3.0, B0=2.0)

    w0_small = 1e-4
    j_bs = 1e5
    j_phi = 1e6
    eta = 1e-7

    dw_dt_small = ntm.dw_dt(w0_small, j_bs, j_phi, 0.0, eta, w_d=1e-3, w_pol=5e-4)
    assert dw_dt_small <= 0.0
