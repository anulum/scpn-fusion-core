# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — NMPC Controller Tests
from __future__ import annotations

import numpy as np

from scpn_fusion.control.nmpc_controller import NMPCConfig, NonlinearMPC


def mock_tokamak_plant(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Very simple linearish mock.
    x = [Ip, beta_N, q95, li, T_axis, n_bar]
    u = [P_aux, I_p_ref, n_gas_puff]
    """
    x_next = x.copy()
    dt = 0.1

    # Ip tracks I_p_ref
    x_next[0] += dt * (u[1] - x[0]) * 0.5

    # beta_N driven by P_aux, heavily to force controller action
    x_next[1] += dt * (u[0] - x[1]) * 0.5

    # q95 ~ 1 / Ip
    if x_next[0] > 0.1:
        x_next[2] = 15.0 / x_next[0]
    else:
        x_next[2] = 150.0

    # li relaxes
    x_next[3] += dt * (1.0 - x[3]) * 0.1

    # T_axis driven by P_aux / n_bar
    if x[5] > 0.1:
        x_next[4] += dt * (2.0 * u[0] / x[5] - x[4]) * 0.5

    # n_bar driven by gas puff
    x_next[5] += dt * (u[2] - 0.5 * x[5])

    return x_next


def test_unconstrained_nmpc():
    cfg = NMPCConfig(horizon=5, max_sqp_iter=3)
    # Widen bounds
    cfg.u_max = np.array([1000.0, 1000.0, 1000.0])
    cfg.du_max = np.array([1000.0, 1000.0, 1000.0])

    nmpc = NonlinearMPC(mock_tokamak_plant, cfg)

    x0 = np.array([1.0, 1.0, 15.0, 1.0, 2.0, 1.0])
    # Target high beta_N to force P_aux
    x_ref = np.array([5.0, 20.0, 3.0, 1.0, 5.0, 2.0])
    u_prev = np.array([10.0, 1.0, 1.0])

    u_opt = nmpc.step(x0, x_ref, u_prev)

    # Should aggressively command I_p_ref and P_aux
    assert u_opt[0] > 5.0  # P_aux
    assert u_opt[1] > 1.0  # I_p_ref


def test_input_constrained_nmpc():
    cfg = NMPCConfig(horizon=5, max_sqp_iter=3)
    cfg.u_max = np.array([10.0, 10.0, 10.0])  # Restrict P_aux to 10
    cfg.du_max = np.array([1000.0, 1000.0, 1000.0])

    nmpc = NonlinearMPC(mock_tokamak_plant, cfg)

    x0 = np.array([1.0, 1.0, 15.0, 1.0, 2.0, 1.0])
    x_ref = np.array([5.0, 5.0, 3.0, 1.0, 10.0, 2.0])  # High targets
    u_prev = np.array([10.0, 1.0, 1.0])

    u_opt = nmpc.step(x0, x_ref, u_prev)

    assert u_opt[0] <= 10.0


def test_slew_rate():
    cfg = NMPCConfig(horizon=5, max_sqp_iter=3)
    cfg.u_max = np.array([100.0, 100.0, 100.0])
    cfg.du_max = np.array([1.0, 1.0, 1.0])  # Tight slew rate

    nmpc = NonlinearMPC(mock_tokamak_plant, cfg)

    x0 = np.array([1.0, 1.0, 15.0, 1.0, 2.0, 1.0])
    x_ref = np.array([15.0, 5.0, 3.0, 1.0, 10.0, 2.0])
    u_prev = np.array([5.0, 5.0, 5.0])

    u_opt = nmpc.step(x0, x_ref, u_prev)

    assert np.all(np.abs(u_opt - u_prev) <= 1.0 + 1e-6)


def test_infeasibility_recovery():
    cfg = NMPCConfig(horizon=5, max_sqp_iter=3)
    # Contradictory state constraints: beta_N < 1, but we command it to go high
    cfg.x_max[1] = 0.5

    nmpc = NonlinearMPC(mock_tokamak_plant, cfg)

    x0 = np.array([1.0, 1.0, 15.0, 1.0, 2.0, 1.0])  # Already violating beta_N
    x_ref = np.array([5.0, 5.0, 3.0, 1.0, 10.0, 2.0])
    u_prev = np.array([10.0, 1.0, 1.0])

    u_opt = nmpc.step(x0, x_ref, u_prev)

    # Should run and log infeasibility
    assert nmpc.infeasibility_count > 0
    assert u_opt.shape == (3,)
