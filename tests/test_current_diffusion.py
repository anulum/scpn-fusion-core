# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Current Diffusion Tests
from __future__ import annotations

import numpy as np

from scpn_fusion.core.current_diffusion import (
    CurrentDiffusionSolver,
    neoclassical_resistivity,
    q_from_psi,
    resistive_diffusion_time,
)


def test_neoclassical_resistivity():
    eta = neoclassical_resistivity(Te_keV=1.0, ne_19=1.0, Z_eff=1.5, epsilon=0.1)
    assert eta > 0.0
    assert 1e-8 < eta < 1e-7


def test_q_from_psi():
    rho = np.linspace(0, 1, 50)
    R0 = 2.0
    a = 0.5
    B0 = 1.0
    psi = -(a**2) * B0 / R0 * rho**2 / 2.0

    q = q_from_psi(rho, psi, R0, a, B0)
    assert np.allclose(q, 1.0, atol=0.1)


def test_resistive_diffusion_time():
    tau = resistive_diffusion_time(a=2.0, eta=1e-8)
    assert tau > 100.0


def test_pure_ohmic_relaxation():
    rho = np.linspace(0, 1, 50)
    solver = CurrentDiffusionSolver(rho, R0=2.0, a=0.5, B0=1.0)

    Te = np.ones(50)
    ne = np.ones(50)
    j_bs = np.zeros(50)
    j_cd = np.zeros(50)

    psi_initial = solver.psi.copy()

    dt = 10.0
    for _ in range(10):
        solver.step(dt, Te, ne, 1.5, j_bs, j_cd)

    assert not np.allclose(solver.psi, psi_initial)
    q = q_from_psi(rho, solver.psi, 2.0, 0.5, 1.0)
    assert np.all(q > 0)


def test_conservation_and_steady_state():
    rho = np.linspace(0, 1, 50)
    solver = CurrentDiffusionSolver(rho, R0=2.0, a=0.5, B0=1.0)

    Te = np.ones(50)
    ne = np.ones(50)
    j_bs = np.zeros(50)
    j_cd = np.ones(50) * 1e5

    dt = 1.0
    for _ in range(100):
        solver.step(dt, Te, ne, 1.5, j_bs, j_cd)

    psi_new = solver.psi.copy()
    solver.step(dt, Te, ne, 1.5, j_bs, j_cd)
    assert np.allclose(psi_new, solver.psi, rtol=1e-3, atol=1e-3)


def test_bootstrap_steepens_q():
    rho = np.linspace(0, 1, 50)
    solver = CurrentDiffusionSolver(rho, R0=2.0, a=0.5, B0=1.0)

    Te = np.ones(50)
    ne = np.ones(50)
    j_bs = np.exp(-(((rho - 0.5) / 0.1) ** 2)) * 1e6
    j_cd = np.zeros(50)

    for _ in range(5):
        solver.step(10.0, Te, ne, 1.5, j_bs, j_cd)

    q = q_from_psi(rho, solver.psi, 2.0, 0.5, 1.0)
    assert q[0] > 0
