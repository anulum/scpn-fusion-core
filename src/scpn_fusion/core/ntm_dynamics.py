# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Modified Rutherford Equation & NTM Dynamics
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MU_0 = 4.0 * np.pi * 1e-7


@dataclass
class RationalSurface:
    """Represents a q=m/n rational surface."""

    rho: float
    r_s: float
    m: int
    n: int
    q: float
    shear: float


def eccd_stabilization_factor(d_cd: float, w: float) -> float:
    """ECCD stabilization efficiency: f = (w/d_cd) exp(-w^2/(4 d_cd^2))."""
    if w <= 0.0 or d_cd <= 0.0:
        return 0.0
    return float((w / d_cd) * np.exp(-(w**2) / (4.0 * d_cd**2)))


def find_rational_surfaces(
    q: np.ndarray, rho: np.ndarray, a: float, m_max: int = 5, n_max: int = 3
) -> list[RationalSurface]:
    """Locate radii where q(rho) = m/n."""
    surfaces = []
    dq_drho = np.gradient(q, rho)

    for m in range(1, m_max + 1):
        for n in range(1, n_max + 1):
            q_target = m / n
            q_diff = q - q_target
            crossings = np.where(np.diff(np.sign(q_diff)))[0]

            for idx in crossings:
                r1, r2 = rho[idx], rho[idx + 1]
                q1, q2 = q[idx], q[idx + 1]
                if q1 == q2:
                    continue

                frac = (q_target - q1) / (q2 - q1)
                rho_s = r1 + frac * (r2 - r1)

                dq_s = dq_drho[idx] + frac * (dq_drho[idx + 1] - dq_drho[idx])
                shear_s = (rho_s / q_target) * dq_s

                surfaces.append(
                    RationalSurface(
                        rho=float(rho_s),
                        r_s=float(rho_s * a),
                        m=m,
                        n=n,
                        q=float(q_target),
                        shear=float(shear_s),
                    )
                )

    surfaces.sort(key=lambda x: x.rho)
    return surfaces


class NTMIslandDynamics:
    """Solves the Modified Rutherford Equation (MRE) for an NTM island."""

    def __init__(
        self,
        r_s: float,
        m: int,
        n: int,
        a: float,
        R0: float,
        B0: float,
        Delta_prime_0: float | None = None,
    ):
        self.r_s = r_s
        self.m = m
        self.n = n
        self.a = a
        self.R0 = R0
        self.B0 = B0
        self.Delta_prime_0 = (
            Delta_prime_0 if Delta_prime_0 is not None else -2.0 * m / max(r_s, 1e-3)
        )

        # MRE coefficients — La Haye 2006
        self.a1 = 6.35
        self.a2 = 1.2
        self.a3 = 9.36

    def delta_prime_model(self, w: float) -> float:
        """Classical stability index with finite-width correction."""
        c = 0.5
        return self.Delta_prime_0 * self.r_s / (self.r_s + c * w)

    def dw_dt(
        self,
        w: float,
        j_bs: float,
        j_phi: float,
        j_cd: float,
        eta: float,
        w_d: float = 1e-3,
        w_pol: float = 5e-4,
        d_cd: float = 0.05,
    ) -> float:
        """Evaluate the RHS of the Modified Rutherford Equation."""
        if w <= 1e-6:
            return 0.0

        tau_R = MU_0 * self.r_s**2 / max(eta, 1e-12)

        term_classical = self.r_s * self.delta_prime_model(w)

        j_ratio = j_bs / max(j_phi, 1e-6)
        term_bs = self.a1 * j_ratio * (self.r_s / w) * (1.0 / (1.0 + (w_d / w) ** 2))

        term_pol = -self.a2 * (w_pol**2 / w**3)

        j_cd_ratio = j_cd / max(j_phi, 1e-6)
        f_eccd = eccd_stabilization_factor(d_cd, w)
        term_cd = -self.a3 * j_cd_ratio * (self.r_s / w) * f_eccd

        dw_dt_val = (self.r_s / tau_R) * (term_classical + term_bs + term_pol + term_cd)
        return float(dw_dt_val)

    def evolve(
        self,
        w0: float,
        t_span: tuple[float, float],
        dt: float,
        j_bs: float,
        j_phi: float,
        j_cd: float,
        eta: float,
        w_d: float = 1e-3,
        w_pol: float = 5e-4,
        d_cd: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Integrate w(t) using RK4."""
        t_start, t_end = t_span
        n_steps = int(np.ceil((t_end - t_start) / dt))
        t_arr = np.linspace(t_start, t_end, n_steps + 1)
        w_arr = np.zeros(n_steps + 1)
        w_arr[0] = max(w0, 1e-6)

        for i in range(n_steps):
            w_curr = w_arr[i]

            k1 = self.dw_dt(w_curr, j_bs, j_phi, j_cd, eta, w_d, w_pol, d_cd)
            w2 = max(w_curr + 0.5 * dt * k1, 1e-6)
            k2 = self.dw_dt(w2, j_bs, j_phi, j_cd, eta, w_d, w_pol, d_cd)
            w3 = max(w_curr + 0.5 * dt * k2, 1e-6)
            k3 = self.dw_dt(w3, j_bs, j_phi, j_cd, eta, w_d, w_pol, d_cd)
            w4 = max(w_curr + dt * k3, 1e-6)
            k4 = self.dw_dt(w4, j_bs, j_phi, j_cd, eta, w_d, w_pol, d_cd)

            w_next = w_curr + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            w_arr[i + 1] = max(w_next, 1e-6)

        return t_arr, w_arr


class NTMController:
    """Monitors and controls NTM islands via ECCD."""

    def __init__(self, w_onset: float = 0.02, w_target: float = 0.005):
        self.w_onset = w_onset
        self.w_target = w_target
        self.active = False
        self.target_rho = 0.0
        self.eccd_power_request = 0.0

    def step(self, w: float, rho_rs: float, max_power: float = 20.0) -> float:
        """Returns requested ECCD power in MW."""
        if not self.active and w > self.w_onset:
            self.active = True
            self.target_rho = rho_rs
            self.eccd_power_request = max_power

        elif self.active:
            self.target_rho = rho_rs
            if w < self.w_target:
                self.active = False
                self.eccd_power_request = 0.0
            else:
                self.eccd_power_request = max_power

        return self.eccd_power_request
