# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Super-Twisting SMC Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import math

import numpy as np

from scpn_fusion.control.sliding_mode_vertical import (
    SuperTwistingSMC,
    VerticalStabilizer,
    estimate_convergence_time,
    lyapunov_certificate,
)


def test_gain_verification():
    L_max = 10.0
    alpha = math.sqrt(2.0 * L_max) + 1.0
    beta = L_max + 1.0

    assert lyapunov_certificate(alpha, beta, L_max)

    assert not lyapunov_certificate(1.0, beta, L_max)
    assert not lyapunov_certificate(alpha, 1.0, L_max)


def test_convergence_time_estimate():
    s0 = 4.0
    L_max = 2.0
    alpha = 5.0
    beta = 5.0

    t_conv = estimate_convergence_time(alpha, beta, L_max, s0)
    assert t_conv > 0.0
    assert t_conv < 10.0


def test_no_disturbance_convergence():
    smc = SuperTwistingSMC(alpha=50.0, beta=100.0, c=5.0, u_max=500.0)

    dt = 0.001
    x = 1.0
    x_dot = 0.0

    for _ in range(5000):
        u = smc.step(x, x_dot, dt)
        x_dot += u * dt
        x += x_dot * dt

    assert abs(x) < 0.5
    assert abs(x_dot) < 0.5


def test_constant_disturbance_rejection():
    smc = SuperTwistingSMC(alpha=50.0, beta=100.0, c=5.0, u_max=500.0)

    dt = 0.001
    x = 0.0
    x_dot = 0.0
    dist = 50.0

    for _ in range(5000):
        u = smc.step(x, x_dot, dt)
        x_dot += (u + dist) * dt
        x += x_dot * dt

    assert abs(x) < 0.5
    assert abs(x_dot) < 0.5
    assert np.isclose(smc.v, -dist, atol=2.0)


def test_actuator_saturation():
    smc = SuperTwistingSMC(alpha=100.0, beta=100.0, c=1.0, u_max=10.0)
    u = smc.step(10.0, 10.0, 0.01)

    assert abs(u) <= 10.0


def test_vertical_stabilizer_wrapper():
    smc = SuperTwistingSMC(alpha=10.0, beta=20.0, c=1.0, u_max=100.0)
    vs = VerticalStabilizer(n_index=-1.0, Ip_MA=15.0, R0=6.2, m_eff=1.0, tau_wall=0.01, smc=smc)

    u = vs.step(Z_meas=0.1, Z_ref=0.0, dZ_dt_meas=0.0, dt=0.01)
    assert u != 0.0
