# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Divertor Thermal Sim
"""Reduced divertor heat-exhaust model with tungsten and liquid-lithium lanes."""

from __future__ import annotations

import logging
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class DivertorLab:
    """Simulate Heat Exhaust in a Compact Fusion Reactor.

    Compares Solid Tungsten vs. Liquid Lithium Vapor Shielding.
    """

    def __init__(self, P_sol_MW: float = 50.0, R_major: float = 2.1, B_pol: float = 2.0) -> None:
        self.P_sol = P_sol_MW
        self.R = R_major
        self.B_pol = B_pol
        self.q_parallel = 0.0
        self.q_target_solid = 0.0

        # Eich scaling: lambda_q [mm] = 0.63 * B_pol^(-1.19)
        self.lambda_q_mm = 0.63 * (B_pol ** (-1.19))
        self.lambda_q = self.lambda_q_mm / 1000.0

        logger.info(
            "Divertor physics initialized: P_sol_MW=%.3f lambda_q_mm=%.3f",
            self.P_sol,
            self.lambda_q_mm,
        )

    def solve_2point_transport(
        self, expansion_factor: float = 10.0, f_rad: float = 0.5
    ) -> tuple[float, float]:
        """Two-Point Model (2PM) for SOL Transport.

        Balances upstream pressure with target flux constraints.
        T_u = (7/2 * L_c * q_par / kappa0)^(2/7)
        n_u determines if we are in sheath-limited or conduction-limited regime.
        """
        if not np.isfinite(expansion_factor) or expansion_factor <= 0.0:
            raise ValueError("expansion_factor must be finite and > 0.")
        if not np.isfinite(f_rad) or f_rad < 0.0 or f_rad >= 1.0:
            raise ValueError("f_rad must be finite and in [0, 1).")

        q95 = 3.0
        L_c = np.pi * self.R * q95

        # q_par for single-null: P_sol / (2π R λ_q)
        self.q_parallel = (self.P_sol * 1e6) / (2 * np.pi * self.R * self.lambda_q)

        # Upstream T: T_u = (3.5 q_par L_c / κ_0)^(2/7)
        k0 = 2000.0  # Spitzer conductivity
        T_u_eV = (3.5 * self.q_parallel * L_c / k0) ** (2.0 / 7.0)

        q_target = self.q_parallel * (1.0 - f_rad) / expansion_factor

        # Conduction-limited 2PM at target: T_t = (3.5 q_t L_c / kappa0)^(2/7)
        T_t_eV = (3.5 * q_target * L_c / k0) ** (2.0 / 7.0)
        T_t_eV = float(np.clip(T_t_eV, 1.0, T_u_eV))

        self.q_target_solid = q_target
        return float(T_u_eV), float(T_t_eV)

    def calculate_heat_load(self, expansion_factor: float = 10.0) -> float:
        """Calculate Peak Heat Flux using 2-Point Model Physics."""
        T_u, T_t = self.solve_2point_transport(expansion_factor, f_rad=0.0)  # Unmitigated

        logger.info(
            "Divertor heat load: q_parallel_GW_m2=%.1f T_u_eV=%.1f T_t_eV=%.1f q_target_MW_m2=%.1f",
            self.q_parallel / 1e9,
            T_u,
            T_t,
            self.q_target_solid / 1e6,
        )

        return float(self.q_target_solid)

    def simulate_tungsten(self) -> tuple[float, str]:
        """1D Thermal limit of Tungsten Monoblock.

        Simple conduction model: T_surf = q * d / k + T_coolant.
        """
        k_W = 100.0  # W/(m·K)
        d_block = 0.01  # 1 cm to cooling channel
        T_coolant = 100.0  # °C (water)

        q = self.q_target_solid

        delta_T = (q * d_block) / k_W
        T_surf = T_coolant + delta_T

        status = "MELTED" if T_surf > 3422 else "OK"
        return T_surf, status

    def simulate_lithium_vapor(
        self,
        *,
        relaxation: float = 0.7,
        max_iter: int = 50,
        tol: float = 0.1,
    ) -> tuple[float, float, float]:
        """
        Self-Consistent Vapor Shielding Physics.

        Parameters
        ----------
        relaxation : float
            Under-relaxation factor in (0, 1) for iterative convergence.
        max_iter : int
            Maximum Picard iterations.
        tol : float
            Absolute temperature convergence tolerance [°C].
        """
        if not (0.0 < relaxation < 1.0):
            raise ValueError("relaxation must be in (0, 1).")
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1.")
        if not np.isfinite(tol) or tol <= 0.0:
            raise ValueError("tol must be finite and > 0.")
        T_surf = 500.0

        # Li vapour pressure (Alcock et al. 1984): log10(P) = A − B/T_K
        A_li, B_li = 10.0, 8000.0

        current_relaxation = float(relaxation)
        prev_residual = float("inf")
        q_surf = float(self.q_target_solid)
        f_rad = 0.0
        best = (float("inf"), T_surf, q_surf, f_rad)

        for _ in range(max_iter):
            T_K = T_surf + 273.15
            P_sat = 10 ** (A_li - B_li / T_K)

            tau = P_sat / 10.0
            f_rad = 0.98 * (1.0 - np.exp(-tau))
            q_surf = self.q_target_solid * (1.0 - f_rad)

            k_eff = 150.0
            d = 0.005
            T_back = 300.0

            T_new = T_back + (q_surf * d) / k_eff
            residual = abs(T_new - T_surf)
            if residual < best[0]:
                best = (residual, T_new, q_surf, f_rad)

            if residual < tol:
                T_surf = T_new
                break

            if residual > prev_residual:
                current_relaxation = min(0.97, current_relaxation + 0.05)
            else:
                current_relaxation = max(float(relaxation), current_relaxation - 0.01)

            delta = np.clip(T_new - T_surf, -1200.0, 1200.0)
            T_surf = T_surf + (1.0 - current_relaxation) * delta
            prev_residual = residual
        else:
            residual, t_best, q_best, f_best = best
            T_surf = float(t_best)
            q_surf = float(q_best)
            f_rad = float(f_best)
            logger.warning(
                "Li vapor shielding did not converge after %d iterations (residual=%.2f °C)",
                max_iter,
                residual,
            )

        return T_surf, q_surf, f_rad

    def calculate_mhd_pressure_loss(
        self,
        flow_velocity_m_s: float,
        channel_length_m: float = 1.2,
        channel_half_gap_m: float = 0.012,
        density_kg_m3: float = 510.0,
        viscosity_pa_s: float = 2.5e-3,
        conductivity_s_m: float = 8.0e5,
    ) -> dict[str, float]:
        """
        Reduced TEMHD pressure-loss model using a Hartmann-flow correction.

        Returns pressure-loss summary for the provided channel flow speed.
        """
        v = max(float(flow_velocity_m_s), 1e-6)
        b_field = max(float(self.B_pol), 1e-6)
        a = max(float(channel_half_gap_m), 1e-5)
        l = max(float(channel_length_m), 1e-3)
        rho = max(float(density_kg_m3), 1.0)
        mu = max(float(viscosity_pa_s), 1e-6)
        sigma = max(float(conductivity_s_m), 1e3)

        nu = mu / rho
        ha = b_field * a * np.sqrt(sigma / max(rho * nu, 1e-12))
        dp_viscous = 12.0 * mu * l * v / (a**2)
        dp_total = dp_viscous * (1.0 + ha / 6.0)

        return {
            "flow_velocity_m_s": v,
            "hartmann_number": float(ha),
            "pressure_loss_pa": float(dp_total),
        }

    def estimate_evaporation_rate(self, surface_temp_c: float, flow_velocity_m_s: float) -> float:
        """Velocity-dependent lithium evaporation estimate [kg m^-2 s^-1]."""
        t_c = float(surface_temp_c)
        v = max(float(flow_velocity_m_s), 1e-6)
        thermal_drive = np.exp(np.clip((t_c - 500.0) / 260.0, -8.0, 8.0))
        flow_relief = 1.0 / (1.0 + 0.45 * np.sqrt(v))
        return float(2.0e-6 * thermal_drive * flow_relief)

    def simulate_temhd_liquid_metal(
        self, flow_velocity_m_s: float, expansion_factor: float = 15.0
    ) -> dict[str, object]:
        """Reduced TEMHD divertor state including MHD pressure loss and evaporation."""
        self.calculate_heat_load(expansion_factor=expansion_factor)
        t_li_c, q_surface_w_m2, shielding = self.simulate_lithium_vapor()
        mhd = self.calculate_mhd_pressure_loss(flow_velocity_m_s)
        evap_rate = self.estimate_evaporation_rate(t_li_c, flow_velocity_m_s)

        stability_index = (
            q_surface_w_m2 / 45.0e6 + mhd["pressure_loss_pa"] / 8.0e5 + evap_rate / 1.0e-3
        )
        is_stable = bool(stability_index <= 1.0)

        return {
            "flow_velocity_m_s": float(flow_velocity_m_s),
            "surface_temperature_c": float(t_li_c),
            "surface_heat_flux_w_m2": float(q_surface_w_m2),
            "shielding_fraction": float(shielding),
            "pressure_loss_pa": float(mhd["pressure_loss_pa"]),
            "hartmann_number": float(mhd["hartmann_number"]),
            "evaporation_rate_kg_m2_s": float(evap_rate),
            "stability_index": float(stability_index),
            "is_stable": is_stable,
        }


def run_divertor_sim() -> None:
    """Run the standalone divertor comparison and write the diagnostic plot."""
    logger.info("SCPN heat exhaust lithium divertor comparison start")

    lab = DivertorLab(P_sol_MW=80.0, R_major=2.1, B_pol=2.5)  # Compact Pilot parameters

    q_solid = lab.calculate_heat_load(expansion_factor=15.0)

    Tw, status_w = lab.simulate_tungsten()
    logger.info("Tungsten divertor state: surface_temp_c=%.0f status=%s", Tw, status_w)

    Tli, q_li, shielding = lab.simulate_lithium_vapor()
    logger.info(
        "Liquid lithium divertor state: surface_temp_c=%.0f shielding_percent=%.1f",
        Tli,
        shielding * 100,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    materials = ["Tungsten (Solid)", "Lithium (Vapor Shield)"]
    temps = [Tw, Tli]
    limits = [3422, 1342]
    colors = ["gray", "purple"]

    bars = ax.bar(materials, temps, color=colors)
    ax.axhline(3422, color="red", linestyle="--", label="W Melting Point")
    ax.set_ylabel("Surface Temperature (°C)")
    ax.set_title("Divertor Material Performance")

    # Add Heat Flux annotations
    ax.text(
        0, Tw / 2, f"Flux: {q_solid / 1e6:.0f} MW/m2", ha="center", color="white", fontweight="bold"
    )
    ax.text(
        1, Tli / 2, f"Flux: {q_li / 1e6:.1f} MW/m2", ha="center", color="white", fontweight="bold"
    )

    ax.legend()
    plt.savefig("Divertor_Solution.png")
    logger.info("Divertor comparison figure saved: Divertor_Solution.png")


if __name__ == "__main__":
    run_divertor_sim()
