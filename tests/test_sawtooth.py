# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Sawtooth Model Tests
from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid

from scpn_fusion.core.sawtooth import SawtoothCycler, SawtoothMonitor, kadomtsev_crash


def test_no_crash_q_above_1():
    rho = np.linspace(0, 1, 50)
    q = np.linspace(1.1, 3.0, 50)
    shear = np.linspace(0.1, 1.0, 50)

    monitor = SawtoothMonitor(rho)
    assert monitor.find_q1_radius(q) is None
    assert not monitor.check_trigger(q, shear)


def test_trigger_q_below_1():
    rho = np.linspace(0, 1, 50)
    q = np.linspace(0.8, 3.0, 50)

    shear_low = np.full(50, 0.05)
    monitor = SawtoothMonitor(rho, s_crit=0.1)
    assert not monitor.check_trigger(q, shear_low)

    shear_high = np.full(50, 0.2)
    assert monitor.check_trigger(q, shear_high)


def test_kadomtsev_crash():
    rho = np.linspace(0, 1, 100)
    q = 0.8 + 2.0 * rho**2
    T = 5.0 * (1 - rho**2) ** 2
    n = 1.0 * (1 - rho**2)

    T_new, n_new, q_new, rho_1, rho_mix = kadomtsev_crash(rho, T, n, q, R0=2.0, a=0.5)

    assert rho_1 > 0.0
    assert rho_mix > rho_1

    idx_1 = int(rho_1 * 100)
    assert np.allclose(T_new[0], T_new[idx_1 - 1])
    assert T_new[0] < T[0]

    idx_mix = int(rho_mix * 100)
    assert np.allclose(q_new[0], 1.01)
    assert np.allclose(q_new[idx_mix - 1], 1.01)


def test_density_conservation():
    rho = np.linspace(0, 1, 100)
    q = 0.8 + 2.0 * rho**2
    T = 5.0 * (1 - rho**2)
    n = 2.0 * (1 - rho**2)

    T_new, n_new, q_new, rho_1, rho_mix = kadomtsev_crash(rho, T, n, q, R0=2.0, a=0.5)

    def total_particles(n_prof):
        return trapezoid(n_prof * rho, rho)

    N_before = total_particles(n)
    N_after = total_particles(n_new)

    assert np.isclose(N_before, N_after, rtol=1e-2)


def test_sawtooth_cycler():
    rho = np.linspace(0, 1, 50)
    q = 0.8 + 2.0 * rho**2
    shear = np.full(50, 0.2)
    T = np.linspace(5.0, 0.1, 50)
    n = np.linspace(1.0, 0.1, 50)

    cycler = SawtoothCycler(rho, R0=2.0, a=0.5, s_crit=0.1)
    event = cycler.step(0.1, q, shear, T, n)

    assert event is not None
    assert event.T_drop > 0
    assert event.seed_energy > 0

    event2 = cycler.step(0.1, q, shear, T, n)
    assert event2 is None
