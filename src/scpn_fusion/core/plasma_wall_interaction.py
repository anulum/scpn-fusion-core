# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Plasma Wall Interaction
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from scpn_fusion.core.sol_model import TwoPointSOL


class SputteringYield:
    """
    Physical sputtering yield [atoms/ion] based on Eckstein fits.
    """

    def __init__(self, target: str = "W", projectile: str = "D"):
        self.target = target
        self.projectile = projectile

        # Eckstein parameters for D on W
        # D -> W: M1=2, M2=183.8, Z1=1, Z2=74
        self.E_s = 8.68  # eV (Surface binding energy for W)
        self.M_ratio = 2.0 / 183.8

        # Simplified threshold energy
        self.E_th_val = 8.0 * self.E_s * (2.0 / 183.8) ** (-0.5)  # Heuristic approx
        self.E_th_val = 220.0  # Common accepted value for D -> W

        # Eckstein fit parameters for D -> W
        self.Q = 0.09
        self.Gamma = 1.0

    def threshold_energy(self) -> float:
        return float(self.E_th_val)

    def yield_at_energy(self, E_ion_eV: float, theta_deg: float = 0.0) -> float:
        if E_ion_eV <= self.threshold_energy():
            return 0.0

        # Reduced energy epsilon (proportional to E_ion)
        # Using a simplified parameterized curve from Eckstein
        eps = E_ion_eV / 5000.0  # Heuristic scaling

        Y_0 = (
            self.Q
            * eps
            / (1.0 + self.Gamma * eps**0.3)
            * (1.0 - (self.E_th_val / E_ion_eV) ** (2.0 / 3.0))
            * (1.0 - self.E_th_val / E_ion_eV) ** 2
        )

        # Angular dependence
        theta_rad = math.radians(min(theta_deg, 89.0))
        cos_theta = math.cos(theta_rad)

        f_theta = cos_theta ** (-1.5) * math.exp(0.5 * (1.0 - 1.0 / cos_theta))

        return max(0.0, float(Y_0 * f_theta))


class ErosionModel:
    def __init__(self, material: str = "W", n_atom: float = 6.31e28):
        self.material = material
        self.n_atom = n_atom

    def gross_erosion_rate(self, ion_flux: float, E_ion_eV: float, theta_deg: float = 0.0) -> float:
        """Rate in atoms/m^2/s."""
        sputt = SputteringYield(target=self.material)
        Y = sputt.yield_at_energy(E_ion_eV, theta_deg)
        return float(Y * ion_flux)

    def net_erosion_rate(self, gross_rate: float, f_redeposition: float = 0.97) -> float:
        """Rate in atoms/m^2/s."""
        return float(gross_rate * (1.0 - f_redeposition))

    def depth_rate(self, net_rate: float) -> float:
        """Rate in m/s."""
        return float(net_rate / self.n_atom)

    def lifetime_estimate(self, wall_thickness_mm: float, net_rate_m_s: float) -> float:
        """Lifetime in years."""
        if net_rate_m_s <= 0.0:
            return float("inf")
        seconds_per_year = 365.25 * 24 * 3600
        return float((wall_thickness_mm * 1e-3) / (net_rate_m_s * seconds_per_year))


class WallThermalModel:
    """1D heat diffusion into the wall."""

    def __init__(self, material: str = "W", thickness_mm: float = 10.0, n_nodes: int = 50):
        self.material = material
        self.thickness = thickness_mm * 1e-3
        self.n_nodes = n_nodes
        self.T_coolant = 400.0

        # Tungsten thermal properties
        self.kappa = 173.0  # W/(m K)
        self.rho = 19300.0  # kg/m^3
        self.cp = 132.0  # J/(kg K)
        self.T_melt = 3695.0  # K

        self.T_nodes = np.ones(n_nodes) * self.T_coolant
        self.dx = self.thickness / (n_nodes - 1)
        self.alpha = self.kappa / (self.rho * self.cp)

    def step(self, dt: float, q_surface_MW_m2: float) -> float:
        # q in W/m^2
        q_W = q_surface_MW_m2 * 1e6

        # Euler explicit step (ensure dt < dx^2 / (2 alpha))
        dt_max = self.dx**2 / (2.0 * self.alpha)
        steps = int(math.ceil(dt / dt_max))
        dt_sub = dt / steps

        for _ in range(steps):
            T_new = self.T_nodes.copy()

            # Surface boundary condition (q_in - q_rad)
            # -kappa * dT/dx = q_W - eps * sigma * T^4
            sigma_sb = 5.67e-8
            eps = 0.3
            q_rad = eps * sigma_sb * self.T_nodes[0] ** 4
            q_net = q_W - q_rad

            # Ghost node to enforce gradient
            T_ghost = self.T_nodes[1] + 2.0 * self.dx * q_net / self.kappa

            T_new[0] = self.T_nodes[0] + self.alpha * dt_sub / self.dx**2 * (
                T_ghost - 2.0 * self.T_nodes[0] + self.T_nodes[1]
            )

            # Interior nodes
            for i in range(1, self.n_nodes - 1):
                T_new[i] = self.T_nodes[i] + self.alpha * dt_sub / self.dx**2 * (
                    self.T_nodes[i + 1] - 2.0 * self.T_nodes[i] + self.T_nodes[i - 1]
                )

            # Coolant boundary
            T_new[-1] = self.T_coolant

            self.T_nodes = T_new

        return float(self.T_nodes[0])

    def is_melted(self) -> bool:
        return bool(self.T_nodes[0] > self.T_melt)


class TransientThermalLoad:
    def __init__(self, wall: WallThermalModel):
        self.wall = wall

    def elm_load(self, delta_W_MJ: float, A_wet_m2: float, tau_IR_ms: float = 0.25) -> float:
        """Peak surface Delta T [K] from an ELM."""
        if A_wet_m2 <= 0.0 or tau_IR_ms <= 0.0:
            return 0.0

        q_peak_MW_m2 = (delta_W_MJ / A_wet_m2) / (tau_IR_ms * 1e-3)
        q_W = q_peak_MW_m2 * 1e6
        tau = tau_IR_ms * 1e-3

        # Delta T = 2 * q * sqrt(tau / (pi * kappa * rho * c_p))
        thermal_effusivity = math.sqrt(math.pi * self.wall.kappa * self.wall.rho * self.wall.cp)
        delta_T = 2.0 * q_W * math.sqrt(tau) / thermal_effusivity
        return float(delta_T)

    def disruption_load(self, W_th_MJ: float, A_wet_m2: float, tau_TQ_ms: float = 1.0) -> float:
        return self.elm_load(W_th_MJ, A_wet_m2, tau_TQ_ms)

    def n_elm_cycles_to_fatigue(self, delta_T_K: float, T_base_K: float = 600.0) -> int:
        """Coffin-Manson empirical rule for Tungsten."""
        if delta_T_K < 100.0:
            return 10000000
        # Highly simplified fatigue life mapping
        # E.g., at Delta T = 1000 K, cycles ~ 10^4
        cycles = 10**4 * (1000.0 / delta_T_K) ** 3
        return int(max(1, cycles))


@dataclass
class LifetimeReport:
    erosion_mm_per_year: float
    peak_T_steady_K: float
    peak_T_elm_K: float
    n_cycles_to_fatigue: int
    lifetime_years: float
    limiting_factor: str


class DivertorLifetimeAssessment:
    def __init__(
        self,
        sol: TwoPointSOL,
        sputtering: SputteringYield,
        erosion: ErosionModel,
        wall: WallThermalModel,
    ):
        self.sol = sol
        self.sputtering = sputtering
        self.erosion = erosion
        self.wall = wall
        self.transient = TransientThermalLoad(wall)

    def assess(
        self, P_SOL_MW: float, n_u_19: float, f_ELM_Hz: float, delta_W_ELM_MJ: float
    ) -> LifetimeReport:
        # Steady state
        sol_res = self.sol.solve(P_SOL_MW, n_u_19)
        T_t = sol_res.T_target_eV
        q_steady = sol_res.q_parallel_MW_m2  # Approx

        # Thermal
        # Evolve wall to steady state
        T_steady = self.wall.step(10.0, q_steady)

        # Sputtering (assume 1:1 D/T flux)
        # ion_flux = q_steady / (gamma_sh * T_t * e)
        gamma_sh = 7.0
        e_charge = 1.602e-19
        ion_flux = (q_steady * 1e6) / (gamma_sh * max(T_t, 1e-1) * e_charge)

        gross = self.erosion.gross_erosion_rate(ion_flux, E_ion_eV=3.0 * T_t)
        net = self.erosion.net_erosion_rate(gross, 0.97)
        depth_rate = self.erosion.depth_rate(net)
        mm_yr = depth_rate * 1e3 * 365.25 * 24 * 3600

        # Transients
        A_wet = 4.0 * math.pi * self.sol.R0 * (sol_res.lambda_q_mm * 1e-3) * 5.0  # f_expansion=5
        delta_T_elm = self.transient.elm_load(delta_W_ELM_MJ, A_wet)
        T_elm_peak = T_steady + delta_T_elm

        cycles = self.transient.n_elm_cycles_to_fatigue(delta_T_elm, T_steady)

        # Lifetimes
        life_erosion = float("inf") if mm_yr <= 0 else self.wall.thickness * 1e3 / mm_yr
        life_fatigue = cycles / max(f_ELM_Hz, 1e-6) / (365.25 * 24 * 3600)

        limit_years = min(life_erosion, life_fatigue)
        factor = "Erosion" if life_erosion < life_fatigue else "Fatigue"
        if T_elm_peak > self.wall.T_melt:
            limit_years = 0.0
            factor = "Melting"

        return LifetimeReport(
            erosion_mm_per_year=float(mm_yr),
            peak_T_steady_K=float(T_steady),
            peak_T_elm_K=float(T_elm_peak),
            n_cycles_to_fatigue=int(cycles),
            lifetime_years=float(limit_years),
            limiting_factor=factor,
        )
