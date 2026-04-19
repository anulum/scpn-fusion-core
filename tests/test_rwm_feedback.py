# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — RWM Feedback Tests
from __future__ import annotations

import numpy as np

from scpn_fusion.control.rwm_feedback import (
    RWMFeedbackController,
    RWMPhysics,
    RWMStabilityAnalysis,
)


def test_rwm_stable_below_nowall():
    rwm = RWMPhysics(beta_n=2.0, beta_n_nowall=2.8, beta_n_wall=3.5, tau_wall=0.01)
    assert not rwm.is_unstable()
    assert rwm.growth_rate() == 0.0


def test_rwm_unstable_between_limits():
    rwm = RWMPhysics(beta_n=3.0, beta_n_nowall=2.8, beta_n_wall=3.5, tau_wall=0.01)
    assert rwm.is_unstable()
    gamma = rwm.growth_rate()
    assert gamma > 0.0
    # gamma = 1/0.01 * (3.0 - 2.8)/(3.5 - 3.0) = 100 * 0.2 / 0.5 = 40
    assert np.isclose(gamma, 40.0)


def test_rwm_ideal_kink():
    rwm = RWMPhysics(beta_n=3.6, beta_n_nowall=2.8, beta_n_wall=3.5, tau_wall=0.01)
    # Above wall limit -> ideal kink
    assert rwm.growth_rate() >= 1e6


def test_rwm_feedback_stabilization():
    rwm = RWMPhysics(beta_n=3.0, beta_n_nowall=2.8, beta_n_wall=3.5, tau_wall=0.01)

    # Needs G_p > (1 + 40 * 1e-4) = 1.004
    ctrl_weak = RWMFeedbackController(n_sensors=1, n_coils=1, G_p=0.5, G_d=0.0, M_coil=1.0)
    assert ctrl_weak.effective_growth_rate(rwm) > 0.0
    assert not ctrl_weak.is_stabilized(rwm)

    ctrl_strong = RWMFeedbackController(n_sensors=1, n_coils=1, G_p=2.0, G_d=0.0, M_coil=1.0)
    assert ctrl_strong.effective_growth_rate(rwm) < 0.0
    assert ctrl_strong.is_stabilized(rwm)


def test_rwm_step():
    ctrl = RWMFeedbackController(n_sensors=2, n_coils=2, G_p=2.0, G_d=0.1)

    B_r = np.array([1.0, -1.0])
    I_coil = ctrl.step(B_r, dt=0.01)

    # First step dB_dt = B_r / dt = [100, -100]
    # I = 2*B_r + 0.1*dB_dt = [2 + 10, -2 - 10] = [12, -12]
    assert np.allclose(I_coil, [12.0, -12.0])


def test_wall_time_limits():
    rwm_ideal_wall = RWMPhysics(
        beta_n=3.0, beta_n_nowall=2.8, beta_n_wall=3.5, tau_wall=float("inf")
    )
    assert rwm_ideal_wall.growth_rate() == 0.0  # Actually 1/inf -> 0

    rwm_no_wall = RWMPhysics(beta_n=3.0, beta_n_nowall=2.8, beta_n_wall=3.5, tau_wall=0.0)
    assert rwm_no_wall.growth_rate() >= 1e6


def test_required_gain():
    gain = RWMStabilityAnalysis.required_feedback_gain(
        beta_n=3.0, beta_n_nowall=2.8, beta_n_wall=3.5, tau_wall=0.01, tau_controller=1e-4
    )
    # gamma = 40. 1 + 40*1e-4 = 1.004
    assert np.isclose(gain, 1.004)
