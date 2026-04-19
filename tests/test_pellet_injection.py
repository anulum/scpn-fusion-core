# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Pellet Injection Tests
from __future__ import annotations

import numpy as np

from scpn_fusion.core.pellet_injection import (
    PelletFuelingController,
    PelletParams,
    PelletTrajectory,
    ngs_ablation_rate,
    pellet_pacing_elm_control,
)


def test_ablation_rate():
    # Zero T
    assert ngs_ablation_rate(0.002, 1e20, 0.0, 2.0) == 0.0

    # Scaling tests
    rate1 = ngs_ablation_rate(0.002, 1e20, 1000.0, 2.0)
    rate2 = ngs_ablation_rate(0.002, 1e20, 2000.0, 2.0)

    # Te^(5/3)
    assert np.isclose(rate2 / rate1, 2.0 ** (5.0 / 3.0))


def test_pellet_trajectory_penetration():
    params1 = PelletParams(r_p_mm=4.0, v_p_m_s=300.0)  # Slower
    params2 = PelletParams(r_p_mm=4.0, v_p_m_s=1000.0)  # Faster

    rho = np.linspace(0, 1, 50)
    ne = np.ones(50) * 5.0  # 5e19
    Te = np.ones(50) * 10000.0  # High temp so they ablate

    traj1 = PelletTrajectory(params1, R0=6.2, a=2.0, B0=5.3)
    res1 = traj1.simulate(rho, ne, Te)

    traj2 = PelletTrajectory(params2, R0=6.2, a=2.0, B0=5.3)
    res2 = traj2.simulate(rho, ne, Te)

    # A faster pellet will always reach a *smaller* or equal final rho.
    assert res2.penetration_depth <= res1.penetration_depth


def test_pellet_drift():
    params_hfs = PelletParams(r_p_mm=4.0, v_p_m_s=300.0, injection_side="HFS")
    params_lfs = PelletParams(r_p_mm=4.0, v_p_m_s=300.0, injection_side="LFS")

    rho = np.linspace(0, 1, 50)
    ne = np.ones(50) * 5.0
    Te = np.ones(50) * 2000.0

    t_hfs = PelletTrajectory(params_hfs, 6.2, 2.0, 5.3)
    res_hfs = t_hfs.simulate(rho, ne, Te)

    t_lfs = PelletTrajectory(params_lfs, 6.2, 2.0, 5.3)
    res_lfs = t_lfs.simulate(rho, ne, Te)

    assert res_hfs.drift_displacement < 0.0  # Inward
    assert res_lfs.drift_displacement > 0.0  # Outward


def test_pellet_pacing():
    # Natural: 5 Hz, 20 MJ
    # Pacing: 20 Hz
    f, w = pellet_pacing_elm_control(20.0, 5.0, 20.0)

    assert f == 20.0
    # W should be reduced by 5/20 = 1/4 -> 5 MJ
    assert np.isclose(w, 5.0)


def test_fueling_controller():
    params = PelletParams(4.0, 300.0)
    ctrl = PelletFuelingController(target_density=10.0, pellet_params=params)

    rho = np.linspace(0, 1, 50)
    ne_low = np.ones(50) * 5.0
    Te = np.ones(50) * 5000.0
    V = 800.0

    # Wait for period
    for _ in range(10):
        ctrl.step(ne_low, Te, 0.1, V)

    cmd = ctrl.step(ne_low, Te, 1.0, V)
    assert cmd is not None
    assert cmd.pellet_params.r_p_mm == 4.0
