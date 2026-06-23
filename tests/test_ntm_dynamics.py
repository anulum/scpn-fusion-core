# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
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


def test_rational_surface_record_preserves_mode_numbers_and_shear() -> None:
    """Rational-surface records keep the q=m/n geometry used by NTM control."""
    from scpn_fusion.core.ntm_dynamics import RationalSurface

    surface = RationalSurface(rho=0.5, r_s=1.0, m=2, n=1, q=2.0, shear=1.5)

    assert surface.q == surface.m / surface.n
    assert surface.r_s == surface.rho * 2.0
    assert surface.shear > 0.0


def test_ntm_controller_latches_and_clears_eccd_request_at_width_thresholds() -> None:
    """ECCD request activates above onset and clears below target island width."""
    from scpn_fusion.core.ntm_dynamics import NTMController

    controller = NTMController(w_onset=0.02, w_target=0.005)

    assert controller.step(w=0.01, rho_rs=0.4, max_power=12.0) == 0.0
    assert controller.active is False
    assert np.isclose(controller.step(w=0.03, rho_rs=0.45, max_power=12.0), 12.0)
    assert controller.active is True
    assert np.isclose(controller.target_rho, 0.45)
    assert controller.step(w=0.004, rho_rs=0.5, max_power=12.0) == 0.0
    assert controller.active is False


def test_eccd_stabilization_factor_zero_for_nonpositive_inputs() -> None:
    from scpn_fusion.core.ntm_dynamics import eccd_stabilization_factor
    assert eccd_stabilization_factor(1.0, 0.0) == 0.0
    assert eccd_stabilization_factor(0.0, 0.05) == 0.0


def test_find_rational_surfaces_skips_flat_q_segment() -> None:
    rho = np.linspace(0.0, 1.0, 6)
    q = np.array([1.0, 2.0, 2.0, 2.0, 3.0, 4.0])  # flat q1==q2 segment
    surfaces = find_rational_surfaces(q, rho, a=2.0)
    assert isinstance(surfaces, list)


def test_ntm_controller_step_toggles_eccd_power() -> None:
    from scpn_fusion.core.ntm_dynamics import NTMController
    ctrl = NTMController(w_onset=0.02, w_target=0.005)
    p_off = ctrl.step(w=0.001, rho_rs=0.5, max_power=20.0)  # below onset -> inactive
    p_on = ctrl.step(w=0.05, rho_rs=0.5, max_power=20.0)   # above onset -> active (max_power)
    p_sustain = ctrl.step(w=0.05, rho_rs=0.5, max_power=20.0)  # active, w>=w_target -> else max_power
    p_clear = ctrl.step(w=0.001, rho_rs=0.5, max_power=20.0)   # active, w<w_target -> deactivate, 0
    assert p_off == 0.0
    assert p_on == 20.0 and p_sustain == 20.0
    assert p_clear == 0.0
