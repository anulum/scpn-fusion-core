# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Pellet Injection Physics
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PelletParams:
    r_p_mm: float  # [mm]
    v_p_m_s: float  # [m/s]
    M_p: float = 2.0
    injection_side: str = "HFS"
    injection_angle_deg: float = 0.0


@dataclass
class PelletResult:
    penetration_depth: float
    deposition_profile: np.ndarray
    total_particles: float
    drift_displacement: float


@dataclass
class PelletInjectionCommand:
    inject_time: float
    pellet_params: PelletParams


def ngs_ablation_rate(r_p: float, ne: float, Te_eV: float, M_p: float) -> float:
    """
    Parks & Turnbull (1978) NGS ablation rate [atoms/s].
    """
    if r_p <= 0.0 or Te_eV <= 0.0:
        return 0.0

    C_abl = 1.12e6
    rate = (
        C_abl
        * (ne ** (1.0 / 3.0))
        * (Te_eV ** (5.0 / 3.0))
        * (r_p ** (4.0 / 3.0))
        * (M_p ** (-1.0 / 3.0))
    )
    return float(max(0.0, rate))


class PelletTrajectory:
    def __init__(self, params: PelletParams, R0: float, a: float, B0: float):
        self.params = params
        self.R0 = R0
        self.a = a
        self.B0 = B0

        # Solid density of deuterium ~ 5e28 atoms/m^3
        self.n_solid = 5.0e28
        volume_m3 = (4.0 / 3.0) * np.pi * (params.r_p_mm * 1e-3) ** 3
        self.N_initial = volume_m3 * self.n_solid

    def simulate(self, rho: np.ndarray, ne: np.ndarray, Te_eV: np.ndarray) -> PelletResult:
        N_p = self.N_initial
        r_p_m = self.params.r_p_mm * 1e-3

        # Path simulation
        # Assume it travels strictly radially for simplicity: s = a * (1 - rho)
        # So v_p = ds/dt = -a * drho/dt

        # We integrate over time steps
        dt = 1e-5

        deposition = np.zeros_like(rho)

        current_rho = 1.0  # start at edge
        dr_dt = -self.params.v_p_m_s / self.a

        while N_p > 0 and current_rho > 0.0:
            # Interpolate plasma parameters
            idx = np.searchsorted(rho, current_rho)
            if idx == 0:
                n_local = ne[0]
                T_local = Te_eV[0]
            elif idx >= len(rho):
                n_local = ne[-1]
                T_local = Te_eV[-1]
            else:
                frac = (current_rho - rho[idx - 1]) / (rho[idx] - rho[idx - 1])
                n_local = ne[idx - 1] + frac * (ne[idx] - ne[idx - 1])
                T_local = Te_eV[idx - 1] + frac * (Te_eV[idx] - Te_eV[idx - 1])

            rate = ngs_ablation_rate(r_p_m, n_local * 1e19, T_local, self.params.M_p)

            dN = rate * dt
            if dN > N_p:
                dN = N_p

            N_p -= dN

            # Distribute dN to profile
            if idx < len(rho):
                dep_idx = max(idx - 1, 0)
                r1 = rho[dep_idx] if dep_idx > 0 else 0.0
                r2 = rho[min(dep_idx + 1, len(rho) - 1)]
                vol = 2.0 * np.pi**2 * self.R0 * self.a**2 * (r2**2 - r1**2)
                deposition[dep_idx] += dN / max(vol, 1e-6)

            # Update radius based on lost mass
            # N = 4/3 pi r^3 n_solid => r = (3N / 4 pi n_solid)^(1/3)
            if N_p > 0:
                r_p_m = (0.75 * N_p / (np.pi * self.n_solid)) ** (1.0 / 3.0)
            else:
                r_p_m = 0.0

            current_rho += dr_dt * dt

        # ∇B drift displacement — Pegourie, Plasma Phys. Control. Fusion 47, 17 (2005)
        T_avg = np.mean(Te_eV)
        n_dep = np.mean(deposition[deposition > 0]) if np.any(deposition > 0) else 1.0
        n_e_avg = np.mean(ne) * 1e19
        # Parametric fit: displacement ~ 0.1-0.2 a for ITER
        drift_m = (
            0.1
            * self.a
            * np.sqrt(n_dep / max(n_e_avg, 1e-3))
            * (T_avg / max(self.B0**2, 1e-3))
            / 1000.0
        )

        if self.params.injection_side == "HFS":
            drift_rho = -drift_m / self.a
        else:
            drift_rho = drift_m / self.a

        # Shift profile
        shifted_dep = np.zeros_like(deposition)
        shift_idx = int(round(drift_rho / (rho[1] - rho[0])))

        if shift_idx > 0:
            shifted_dep[shift_idx:] = deposition[:-shift_idx]
        elif shift_idx < 0:
            shifted_dep[:shift_idx] = deposition[-shift_idx:]
        else:
            shifted_dep = deposition.copy()

        return PelletResult(
            penetration_depth=float(max(0.0, current_rho)),
            deposition_profile=shifted_dep,
            total_particles=float(self.N_initial - N_p),
            drift_displacement=float(drift_rho),
        )


class PelletFuelingController:
    def __init__(self, target_density: float, pellet_params: PelletParams):
        self.target_density = target_density
        self.pellet_params = pellet_params

        volume_m3 = (4.0 / 3.0) * np.pi * (pellet_params.r_p_mm * 1e-3) ** 3
        self.N_pellet = volume_m3 * 5.0e28
        self.time = 0.0
        self.last_injection = -100.0

    def required_frequency(self, ne_current: float, tau_p: float, V_plasma: float) -> float:
        # N_target = target * V, N_current = current * V
        # dN/dt = -N/tau_p + S_pellet
        # S_pellet = f * N_pellet
        # To maintain target: f = N_target / (tau_p * N_pellet)
        N_targ = self.target_density * 1e19 * V_plasma
        f = N_targ / (max(tau_p, 0.1) * self.N_pellet)
        return float(f)

    def step(
        self, ne_profile: np.ndarray, Te_profile: np.ndarray, dt: float, V_plasma: float
    ) -> PelletInjectionCommand | None:
        self.time += dt

        current_density = np.mean(ne_profile)

        if current_density < self.target_density * 0.95:
            # Need fueling
            freq = self.required_frequency(current_density, 5.0, V_plasma)
            period = 1.0 / max(freq, 0.1)

            if self.time - self.last_injection > period:
                self.last_injection = self.time
                return PelletInjectionCommand(self.time, self.pellet_params)

        return None


def pellet_pacing_elm_control(
    f_pellet_Hz: float, f_elm_natural_Hz: float, w_elm_natural_MJ: float
) -> tuple[float, float]:
    """
    Returns (f_ELM, delta_W_ELM).
    """
    if f_pellet_Hz > 1.5 * f_elm_natural_Hz:
        f_elm = f_pellet_Hz
        w_elm = w_elm_natural_MJ * (f_elm_natural_Hz / f_pellet_Hz)
        return float(f_elm), float(w_elm)
    return float(f_elm_natural_Hz), float(w_elm_natural_MJ)
