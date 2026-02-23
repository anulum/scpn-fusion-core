# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Divertor Thermal Sim
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import sys

class DivertorLab:
    """
    Simulates Heat Exhaust in a Compact Fusion Reactor.
    Compares Solid Tungsten vs. Liquid Lithium Vapor Shielding.
    """
    def __init__(self, P_sol_MW=50.0, R_major=2.1, B_pol=2.0):
        self.P_sol = P_sol_MW
        self.R = R_major
        self.B_pol = B_pol

        # Eich scaling: lambda_q [mm] = 0.63 * B_pol^(-1.19)
        self.lambda_q_mm = 0.63 * (B_pol**(-1.19))
        self.lambda_q = self.lambda_q_mm / 1000.0
        
        print(f"--- DIVERTOR PHYSICS ---")
        print(f"Power to Divertor: {self.P_sol} MW")
        print(f"Eich Width (lambda_q): {self.lambda_q_mm:.3f} mm")
        
    def solve_2point_transport(self, expansion_factor=10.0, f_rad=0.5):
        """
        Two-Point Model (2PM) for SOL Transport.
        Balances upstream pressure with target flux constraints.
        T_u = (7/2 * L_c * q_par / kappa0)^(2/7)
        n_u determines if we are in sheath-limited or conduction-limited regime.
        """
        q95 = 3.0
        L_c = np.pi * self.R * q95
        
        # q_par for single-null: P_sol / (2π R λ_q)
        self.q_parallel = (self.P_sol * 1e6) / (2 * np.pi * self.R * self.lambda_q)

        # Upstream T: T_u = (3.5 q_par L_c / κ_0)^(2/7)
        k0 = 2000.0  # Spitzer conductivity
        T_u_eV = (3.5 * self.q_parallel * L_c / k0)**(2.0/7.0)
        
        q_target = self.q_parallel * (1.0 - f_rad) / expansion_factor

        # Stangeby 2PM heuristic: T_t = T_u * ((1-f_rad) * 0.1)^2
        T_t_eV = T_u_eV * ((1.0 - f_rad) * 0.1)**2
        T_t_eV = max(T_t_eV, 1.0)
        
        self.q_target_solid = q_target
        return T_u_eV, T_t_eV

    def calculate_heat_load(self, expansion_factor=10.0):
        """
        Calculates Peak Heat Flux using 2-Point Model Physics.
        """
        T_u, T_t = self.solve_2point_transport(expansion_factor, f_rad=0.0) # Unmitigated
        
        print(f"Parallel Heat Flux: {self.q_parallel/1e9:.1f} GW/m2")
        print(f"Upstream Temp (T_u): {T_u:.1f} eV")
        print(f"Target Temp (T_t): {T_t:.1f} eV")
        print(f"Unmitigated Target Flux: {self.q_target_solid/1e6:.1f} MW/m2")
        
        return self.q_target_solid

    def simulate_tungsten(self):
        """
        1D Thermal limit of Tungsten Monoblock.
        Simple conduction model: T_surf = q * d / k + T_coolant
        """
        k_W = 100.0      # W/(m·K)
        d_block = 0.01   # 1 cm to cooling channel
        T_coolant = 100.0 # °C (water)
        
        q = self.q_target_solid
        
        delta_T = (q * d_block) / k_W
        T_surf = T_coolant + delta_T
        
        status = "MELTED" if T_surf > 3422 else "OK"
        return T_surf, status

    def simulate_lithium_vapor(self, *, relaxation: float = 0.7):
        """
        Self-Consistent Vapor Shielding Physics.
        Lithium evaporates based on P_sat(T), forming a dense cloud.
        The cloud radiates energy back to the SOL and walls, shielding the target.

        Parameters
        ----------
        relaxation : float
            Under-relaxation factor in (0, 1) for iterative convergence.
        """
        if not (0.0 < relaxation < 1.0):
            raise ValueError("relaxation must be in (0, 1).")
        T_surf = 500.0  # initial guess [°C]

        # Li vapour pressure (Alcock et al. 1984): log10(P) = A − B/T_K
        A_li, B_li = 10.0, 8000.0
        
        for i in range(50):
            T_K = T_surf + 273.15
            P_sat = 10**(A_li - B_li / T_K)

            tau = P_sat / 10.0  # optical depth ∝ P_sat
            f_rad = 0.98 * (1.0 - np.exp(-tau))
            q_surf = self.q_target_solid * (1.0 - f_rad)

            k_eff = 150.0
            d = 0.005   # 5 mm layer
            T_back = 300.0
            
            T_new = T_back + (q_surf * d) / k_eff
            
            if abs(T_new - T_surf) < 0.1:
                T_surf = T_new
                break
                
            T_surf = relaxation * T_surf + (1.0 - relaxation) * T_new
            
        return T_surf, q_surf, f_rad

    def calculate_mhd_pressure_loss(
        self,
        flow_velocity_m_s,
        channel_length_m=1.2,
        channel_half_gap_m=0.012,
        density_kg_m3=510.0,
        viscosity_pa_s=2.5e-3,
        conductivity_s_m=8.0e5,
    ):
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

    def estimate_evaporation_rate(self, surface_temp_c, flow_velocity_m_s):
        """
        Velocity-dependent lithium evaporation estimate [kg m^-2 s^-1].
        """
        t_c = float(surface_temp_c)
        v = max(float(flow_velocity_m_s), 1e-6)
        thermal_drive = np.exp(np.clip((t_c - 500.0) / 260.0, -8.0, 8.0))
        flow_relief = 1.0 / (1.0 + 0.45 * np.sqrt(v))
        return float(2.0e-6 * thermal_drive * flow_relief)

    def simulate_temhd_liquid_metal(self, flow_velocity_m_s, expansion_factor=15.0):
        """
        Reduced TEMHD divertor state including MHD pressure loss and evaporation.
        """
        self.calculate_heat_load(expansion_factor=expansion_factor)
        t_li_c, q_surface_w_m2, shielding = self.simulate_lithium_vapor()
        mhd = self.calculate_mhd_pressure_loss(flow_velocity_m_s)
        evap_rate = self.estimate_evaporation_rate(t_li_c, flow_velocity_m_s)

        stability_index = (
            q_surface_w_m2 / 45.0e6
            + mhd["pressure_loss_pa"] / 8.0e5
            + evap_rate / 1.0e-3
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

def run_divertor_sim():
    print("\n--- SCPN HEAT EXHAUST: The Lithium Solution ---")
    
    lab = DivertorLab(P_sol_MW=80.0, R_major=2.1, B_pol=2.5) # Compact Pilot parameters
    
    q_solid = lab.calculate_heat_load(expansion_factor=15.0)

    Tw, status_w = lab.simulate_tungsten()
    print(f"Tungsten Divertor: {Tw:.0f} degC -> {status_w}!")
    
    Tli, q_li, shielding = lab.simulate_lithium_vapor()
    print(f"Liquid Li Divertor: {Tli:.0f} degC (Shielding: {shielding*100:.1f}%)")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    materials = ['Tungsten (Solid)', 'Lithium (Vapor Shield)']
    temps = [Tw, Tli]
    limits = [3422, 1342]
    colors = ['gray', 'purple']
    
    bars = ax.bar(materials, temps, color=colors)
    ax.axhline(3422, color='red', linestyle='--', label='W Melting Point')
    ax.set_ylabel("Surface Temperature (°C)")
    ax.set_title("Divertor Material Performance")
    
    # Add Heat Flux annotations
    ax.text(0, Tw/2, f"Flux: {q_solid/1e6:.0f} MW/m2", ha='center', color='white', fontweight='bold')
    ax.text(1, Tli/2, f"Flux: {q_li/1e6:.1f} MW/m2", ha='center', color='white', fontweight='bold')
    
    ax.legend()
    plt.savefig("Divertor_Solution.png")
    print("Saved: Divertor_Solution.png")

if __name__ == "__main__":
    run_divertor_sim()
