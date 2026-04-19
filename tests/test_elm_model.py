# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — ELM Model Tests
from __future__ import annotations

import numpy as np

from scpn_fusion.core.elm_model import (
    ELMCrashModel,
    ELMCycler,
    PeelingBallooningBoundary,
    RMPSuppression,
    elm_power_balance_frequency,
)


def test_pb_boundary_subcritical():
    pb = PeelingBallooningBoundary(q95=3.0, kappa=1.7, delta=0.3, a=2.0, R0=6.2)
    # Very low alpha and j
    assert not pb.is_unstable(alpha_edge=0.01, j_edge=1e4, s_edge=2.0)
    assert pb.stability_margin(alpha_edge=0.01, j_edge=1e4, s_edge=2.0) > 0.0


def test_pb_boundary_supercritical():
    pb = PeelingBallooningBoundary(q95=3.0, kappa=1.7, delta=0.3, a=2.0, R0=6.2)
    # High alpha
    assert pb.is_unstable(alpha_edge=25.0, j_edge=1e6, s_edge=2.0)
    assert pb.stability_margin(alpha_edge=25.0, j_edge=1e6, s_edge=2.0) < 0.0


def test_elm_crash_model():
    crash = ELMCrashModel(f_elm_fraction=0.1)

    W_ped = 100.0  # MJ
    T_ped = 5.0
    n_ped = 5.0

    res = crash.crash(T_ped, n_ped, W_ped)

    # 10% of W_ped
    assert np.isclose(res.delta_W_MJ, 10.0)
    assert res.T_ped_post < T_ped
    assert res.n_ped_post < n_ped


def test_crash_profile_flattening():
    crash = ELMCrashModel(f_elm_fraction=0.1)

    rho = np.linspace(0, 1, 100)
    Te = np.ones(100) * 5.0
    ne = np.ones(100) * 5.0

    Te_new, ne_new = crash.apply_to_profiles(rho, Te, ne, rho_ped=0.9)

    # Core unchanged
    assert Te_new[0] == 5.0
    # Edge drops
    assert Te_new[95] < 5.0


def test_elm_frequency_power_balance():
    f = elm_power_balance_frequency(P_SOL_MW=100.0, W_ped_MJ=100.0, f_elm_fraction=0.1)
    # 100 / (0.1 * 100) = 100 / 10 = 10 Hz
    assert np.isclose(f, 10.0)


def test_rmp_suppression():
    rmp = RMPSuppression()
    q = np.linspace(1, 3, 50)
    rho = np.linspace(0, 1, 50)

    # Very weak B_r
    chir = rmp.chirikov_parameter(q, rho, delta_B_r=1e-6, B0=5.3, R0=6.2)
    assert not rmp.suppressed(chir)
    assert rmp.density_pump_out(chir) == 0.0

    # Strong B_r (overlap > 1)
    chir2 = rmp.chirikov_parameter(q, rho, delta_B_r=1e-1, B0=5.3, R0=6.2)
    assert rmp.suppressed(chir2)
    assert rmp.density_pump_out(chir2) > 0.0


def test_elm_cycler():
    pb = PeelingBallooningBoundary(q95=3.0, kappa=1.7, delta=0.3, a=2.0, R0=6.2)
    crash = ELMCrashModel(f_elm_fraction=0.1)

    cycler = ELMCycler(pb, crash)

    # Step 1: Stable
    ev = cycler.step(0.1, alpha_edge=0.1, j_edge=1e4, s_edge=2.0, T_ped=5.0, n_ped=5.0, W_ped=100.0)
    assert ev is None

    # Step 2: Unstable
    ev2 = cycler.step(
        0.1, alpha_edge=10.0, j_edge=1e6, s_edge=2.0, T_ped=5.0, n_ped=5.0, W_ped=100.0
    )
    assert ev2 is not None
    assert ev2.delta_W_MJ == 10.0
