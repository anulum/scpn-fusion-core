# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Super-Twisting Sliding Mode Control (Vertical)
from __future__ import annotations

import math

import numpy as np


class SuperTwistingSMC:
    """Second-order sliding mode controller (super-twisting algorithm)."""

    def __init__(self, alpha: float, beta: float, c: float, u_max: float):
        self.alpha = alpha
        self.beta = beta
        self.c = c
        self.u_max = u_max
        self.v = 0.0

    def sliding_surface(self, e: float, de_dt: float) -> float:
        """s = e + c * de/dt."""
        return e + self.c * de_dt

    def step(self, e: float, de_dt: float, dt: float) -> float:
        """u = -alpha * abs(s)^0.5 * sign(s) + v, v_dot = -beta * sign(s)."""
        s = self.sliding_surface(e, de_dt)

        if dt > 0:
            self.v -= self.beta * np.sign(s) * dt

        self.v = np.clip(self.v, -self.u_max, self.u_max)

        u = -self.alpha * math.sqrt(abs(s)) * np.sign(s) + self.v

        return float(np.clip(u, -self.u_max, self.u_max))


class VerticalStabilizer:
    """Vertical stability controller wrapper for a tokamak."""

    def __init__(
        self,
        n_index: float,
        Ip_MA: float,
        R0: float,
        m_eff: float,
        tau_wall: float,
        smc: SuperTwistingSMC,
    ):
        self.n_index = n_index
        self.Ip = Ip_MA * 1e6
        self.R0 = R0
        self.m_eff = m_eff
        self.tau_wall = tau_wall
        self.smc = smc

    def step(self, Z_meas: float, Z_ref: float, dZ_dt_meas: float, dt: float) -> float:
        """Compute vertical stabilization command from position error and velocity."""
        e = Z_meas - Z_ref
        u = self.smc.step(e, dZ_dt_meas, dt)
        return u


def lyapunov_certificate(alpha: float, beta: float, L_max: float) -> bool:
    """Verify gain conditions: alpha > sqrt(2 L_max), beta > L_max."""
    cond1 = alpha > math.sqrt(2.0 * max(L_max, 1e-12))
    cond2 = beta > max(L_max, 1e-12)
    return cond1 and cond2


def estimate_convergence_time(alpha: float, beta: float, L_max: float, s0: float) -> float:
    """Upper bound on time to reach s=0."""
    if L_max < 0 or alpha <= math.sqrt(2.0 * L_max):
        return float("inf")

    denom = alpha - math.sqrt(2.0 * L_max)
    if denom <= 0:
        return float("inf")

    return 2.0 * math.sqrt(abs(s0)) / denom
