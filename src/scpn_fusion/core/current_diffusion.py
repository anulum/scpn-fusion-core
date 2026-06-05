# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Current Diffusion Equation
"""One-dimensional current-diffusion and safety-factor profile evolution tools."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_banded

# Sauter, Angioni & Lin-Liu (1999/2002) neoclassical resistivity
MU_0 = 4.0 * np.pi * 1e-7
FloatArray: TypeAlias = NDArray[np.float64]
ProfileCallable: TypeAlias = Callable[[float, FloatArray], FloatArray | float]
EtaCallable: TypeAlias = Callable[[FloatArray], FloatArray | float]
ScalarCallable: TypeAlias = Callable[[float], float]
TauCallable: TypeAlias = Callable[..., FloatArray | float]


@dataclass(frozen=True)
class FluxEvolutionTrajectory:
    """Time history for the non-adiabatic MIF/FRC flux carrier equation.

    Arrays use SI units. ``psi`` has shape ``(n_steps + 1, n_rho)`` and stores
    the evolved poloidal-flux profile. ``hall_drive`` stores
    ``R_null E_theta``, ``resistive_loss`` stores ``eta J_theta``, and
    ``damping_rate`` stores ``1 / tau_psi`` at the same time/grid points.
    ``source_increment`` and ``damping_decrement`` have shape
    ``(n_steps, n_rho)`` and close the exact discrete balance
    ``psi[n+1] = psi[n] - damping_decrement[n] + source_increment[n]``.
    """

    time_s: FloatArray
    rho: FloatArray
    psi: FloatArray
    hall_drive: FloatArray
    resistive_loss: FloatArray
    source: FloatArray
    damping_rate: FloatArray
    source_increment: FloatArray
    damping_decrement: FloatArray
    update_residual: FloatArray
    dt_s: float


def solve_flux_evolution_nonadiabatic(
    rho: FloatArray,
    psi0: FloatArray,
    *,
    tau_psi_fn: TauCallable,
    R_null_t: ScalarCallable,
    E_theta_t: ProfileCallable,
    eta_spitzer_fn: EtaCallable,
    J_theta_t: ProfileCallable,
    dt: float,
    n_steps: int,
) -> FluxEvolutionTrajectory:
    """Evolve the pulsed non-adiabatic flux constraint.

    The implemented carrier is the Ono-style flattened FRC flux equation

    ``dpsi/dt = -psi / tau_psi + R_null E_theta - eta J_theta``.

    Each time step samples the source at both endpoints and advances the local
    linear damping term analytically using the midpoint damping rate. This is
    exact for constant damping/source, second-order for smooth time-dependent
    drive terms, and reduces exactly to constant ``psi`` when Hall drive,
    resistive loss, and damping are all zero.
    """
    rho_grid = _validate_radial_grid(rho)
    psi_initial = _as_profile(psi0, rho_grid.size, "psi0")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")

    time_s = np.linspace(0.0, dt * n_steps, n_steps + 1, dtype=np.float64)
    psi = np.zeros((n_steps + 1, rho_grid.size), dtype=np.float64)
    hall_drive = np.zeros_like(psi)
    resistive_loss = np.zeros_like(psi)
    source = np.zeros_like(psi)
    damping_rate = np.zeros_like(psi)
    source_increment = np.zeros((n_steps, rho_grid.size), dtype=np.float64)
    damping_decrement = np.zeros_like(source_increment)
    update_residual = np.zeros_like(source_increment)
    psi[0] = psi_initial

    for step_index, time_value in enumerate(time_s):
        tau_profile = _as_profile(
            _call_tau(tau_psi_fn, time_value, rho_grid),
            rho_grid.size,
            "tau",
            allow_infinite=True,
        )
        invalid_tau = np.isneginf(tau_profile) | ((tau_profile <= 0.0) & ~np.isposinf(tau_profile))
        if np.any(invalid_tau):
            raise ValueError("tau_psi_fn must return positive finite values or np.inf")
        damping_rate[step_index] = np.where(np.isinf(tau_profile), 0.0, 1.0 / tau_profile)

        r_null = float(R_null_t(float(time_value)))
        if not np.isfinite(r_null) or r_null < 0.0:
            raise ValueError("R_null_t must return a finite non-negative radius")

        e_theta = _as_profile(E_theta_t(float(time_value), rho_grid), rho_grid.size, "E_theta_t")
        eta = _as_profile(eta_spitzer_fn(rho_grid), rho_grid.size, "eta_spitzer_fn")
        if np.any(eta < 0.0):
            raise ValueError("eta_spitzer_fn must return non-negative resistivity")
        j_theta = _as_profile(J_theta_t(float(time_value), rho_grid), rho_grid.size, "J_theta_t")

        hall_drive[step_index] = r_null * e_theta
        resistive_loss[step_index] = eta * j_theta
        source[step_index] = hall_drive[step_index] - resistive_loss[step_index]

    for step_index in range(n_steps):
        gamma = 0.5 * (damping_rate[step_index] + damping_rate[step_index + 1])
        source_midpoint = 0.5 * (source[step_index] + source[step_index + 1])
        decay = np.exp(-gamma * dt)
        driven_increment = np.empty(rho_grid.size, dtype=np.float64)
        damped = gamma > 0.0
        driven_increment[damped] = source_midpoint[damped] * (1.0 - decay[damped]) / gamma[damped]
        driven_increment[~damped] = dt * source_midpoint[~damped]
        damping_drop = psi[step_index] * (1.0 - decay)
        expected_next = psi[step_index] - damping_drop + driven_increment
        psi[step_index + 1] = expected_next
        source_increment[step_index] = driven_increment
        damping_decrement[step_index] = damping_drop
        update_residual[step_index] = psi[step_index + 1] - expected_next

    return FluxEvolutionTrajectory(
        time_s=time_s,
        rho=rho_grid,
        psi=psi,
        hall_drive=hall_drive,
        resistive_loss=resistive_loss,
        source=source,
        damping_rate=damping_rate,
        source_increment=source_increment,
        damping_decrement=damping_decrement,
        update_residual=update_residual,
        dt_s=float(dt),
    )


def neoclassical_resistivity(
    Te_keV: float, ne_19: float, Z_eff: float, epsilon: float, q: float = 1.0, R0: float = 2.0
) -> float:
    """Sauter neoclassical parallel resistivity [Ohm-m]."""
    Te_keV = max(Te_keV, 1e-3)
    ne_19 = max(ne_19, 1e-3)
    epsilon = max(epsilon, 1e-6)

    ln_Lambda = 17.0
    eta_Spitzer = 1.65e-9 * Z_eff * ln_Lambda / (Te_keV**1.5)

    # Trapped fraction — Sauter 2002 Eq. 14
    f_t = 1.0 - (1.0 - epsilon) ** 2 / (np.sqrt(1.0 - epsilon**2) * (1.0 + 1.46 * np.sqrt(epsilon)))
    f_t = max(0.0, min(f_t, 1.0))

    e_charge = 1.602e-19
    m_e = 9.109e-31
    v_te = np.sqrt(2.0 * Te_keV * 1e3 * e_charge / m_e)
    nu_ei = (
        (ne_19 * 1e19)
        * Z_eff
        * e_charge**4
        * ln_Lambda
        / (12.0 * np.pi**1.5 * (8.854e-12) ** 2 * np.sqrt(m_e) * (Te_keV * 1e3 * e_charge) ** 1.5)
    )
    # Sauter 2002 Eq. 13 — electron collisionality (needed for banana/plateau regime selection)
    nu_star_e = nu_ei * max(q, 0.5) * R0 / (epsilon**1.5 * v_te)  # noqa: F841

    # Neoclassical correction — Sauter 2002 Eq. 15
    C_R = 1.0 - (1.0 + 0.36 / Z_eff) * f_t + (0.59 / Z_eff) * f_t**2

    eta_neo = eta_Spitzer / (1.0 - f_t) * C_R
    return float(max(eta_neo, eta_Spitzer))


def q_from_psi(rho: FloatArray, psi: FloatArray, R0: float, a: float, B0: float) -> FloatArray:
    """q(rho) = -rho a^2 B0 / (R0 dpsi/drho), L'Hopital at axis."""
    nr = len(rho)
    q = np.zeros(nr)
    drho = rho[1] - rho[0]

    dpsi_drho = np.gradient(psi, drho)

    for i in range(1, nr):
        denom = R0 * dpsi_drho[i]
        if abs(denom) < 1e-12:
            q[i] = q[i - 1] if i > 1 else 1.0
        else:
            q[i] = -rho[i] * a**2 * B0 / denom

    # L'Hopital at rho=0: q(0) = -a^2 B0 / (R0 d^2psi/drho^2)
    d2psi = (psi[2] - 2 * psi[1] + psi[0]) / (drho**2)
    if abs(d2psi) > 1e-12:
        q[0] = -(a**2) * B0 / (R0 * d2psi)
    else:
        q[0] = q[1]

    q = np.abs(q)
    return cast(FloatArray, q)


def resistive_diffusion_time(a: float, eta: float) -> float:
    """tau_R = mu0 a^2 / eta [seconds]."""
    return MU_0 * a**2 / max(eta, 1e-12)


def _validate_radial_grid(rho: FloatArray) -> FloatArray:
    rho_grid = cast(FloatArray, np.asarray(rho, dtype=np.float64))
    if rho_grid.ndim != 1:
        raise ValueError("rho must be one-dimensional")
    if rho_grid.size < 2:
        raise ValueError("rho must contain at least two points")
    if not np.all(np.isfinite(rho_grid)):
        raise ValueError("rho must contain finite values")
    if not np.all(np.diff(rho_grid) > 0.0):
        raise ValueError("rho must be strictly increasing")
    return rho_grid


def _as_profile(
    values: FloatArray | float,
    expected_size: int,
    name: str,
    *,
    allow_infinite: bool = False,
) -> FloatArray:
    profile = np.asarray(values, dtype=np.float64)
    if profile.ndim == 0:
        profile = np.full(expected_size, float(profile), dtype=np.float64)
    if profile.shape != (expected_size,):
        raise ValueError(f"{name} must be scalar or shape ({expected_size},)")
    valid_values = (
        np.isfinite(profile) | np.isposinf(profile) if allow_infinite else np.isfinite(profile)
    )
    if not np.all(valid_values):
        suffix = "finite values or np.inf" if allow_infinite else "finite values"
        raise ValueError(f"{name} must contain {suffix}")
    return cast(FloatArray, profile)


def _call_tau(tau_psi_fn: TauCallable, time_s: float, rho: FloatArray) -> FloatArray | float:
    try:
        return tau_psi_fn(time_s, rho)
    except TypeError:
        return tau_psi_fn(time_s)


class CurrentDiffusionSolver:
    """Implicit Crank-Nicolson solver for 1D poloidal flux diffusion."""

    def __init__(self, rho: FloatArray, R0: float, a: float, B0: float):
        self.rho = rho
        self.R0 = R0
        self.a = a
        self.B0 = B0
        self.nr = len(rho)
        self.drho = rho[1] - rho[0]
        self.psi = np.zeros(self.nr)
        for i in range(1, self.nr):
            r = self.rho[i]
            q_r = 1.0 + 2.0 * r**2
            dpsi = -r * self.a**2 * self.B0 / (self.R0 * q_r)
            self.psi[i] = self.psi[i - 1] + dpsi * self.drho
        self.psi -= self.psi[-1]

    def step(
        self,
        dt: float,
        Te: FloatArray,
        ne: FloatArray,
        Z_eff: float,
        j_bs: FloatArray,
        j_cd: FloatArray,
        j_ext: FloatArray | None = None,
    ) -> FloatArray:
        """Advance poloidal flux psi by one timestep dt."""
        if j_ext is None:
            j_ext = np.zeros(self.nr)

        j_tot_source = j_bs + j_cd + j_ext

        q_prof = q_from_psi(self.rho, self.psi, self.R0, self.a, self.B0)
        eta_prof = np.zeros(self.nr)
        for i in range(self.nr):
            epsilon = self.rho[i] * self.a / self.R0
            eta_prof[i] = neoclassical_resistivity(
                float(Te[i]),
                float(ne[i]),
                Z_eff,
                epsilon,
                float(q_prof[i]),
                self.R0,
            )

        D = eta_prof / (MU_0 * self.a**2)

        alpha = dt / 2.0
        drho2 = self.drho**2

        sub = np.zeros(self.nr)
        diag = np.zeros(self.nr)
        sup = np.zeros(self.nr)
        rhs = np.zeros(self.nr)

        # Axis BC: L(psi)_0 = 4 D_0 (psi_1 - psi_0) / drho^2
        diag[0] = 1.0 + alpha * 4.0 * D[0] / drho2
        sup[0] = -alpha * 4.0 * D[0] / drho2
        rhs[0] = (
            self.psi[0]
            + alpha * 4.0 * D[0] * (self.psi[1] - self.psi[0]) / drho2
            + dt * self.R0 * eta_prof[0] * j_tot_source[0]
        )

        for i in range(1, self.nr - 1):
            r = self.rho[i]
            coeff_prev = D[i] * (1.0 / drho2 - 1.0 / (2.0 * r * self.drho))
            coeff_curr = D[i] * (-2.0 / drho2)
            coeff_next = D[i] * (1.0 / drho2 + 1.0 / (2.0 * r * self.drho))

            sub[i] = -alpha * coeff_prev
            diag[i] = 1.0 - alpha * coeff_curr
            sup[i] = -alpha * coeff_next

            L_psi_n = (
                coeff_prev * self.psi[i - 1]
                + coeff_curr * self.psi[i]
                + coeff_next * self.psi[i + 1]
            )
            rhs[i] = self.psi[i] + alpha * L_psi_n + dt * self.R0 * eta_prof[i] * j_tot_source[i]

        # Edge Dirichlet BC
        diag[-1] = 1.0
        sub[-1] = 0.0
        rhs[-1] = self.psi[-1]

        ab = np.zeros((3, self.nr))
        ab[0, 1:] = sup[:-1]
        ab[1, :] = diag
        ab[2, :-1] = sub[1:]

        self.psi = cast(FloatArray, solve_banded((1, 1), ab, rhs))
        return self.psi
