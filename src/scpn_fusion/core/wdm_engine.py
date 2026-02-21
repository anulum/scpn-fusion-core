# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — WDM Engine
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd

try:
    from scpn_fusion.core._rust_compat import FusionKernel
except ImportError:
    from scpn_fusion.core.fusion_kernel import FusionKernel
from scpn_fusion.core.integrated_transport_solver import TransportSolver
from scpn_fusion.nuclear.pwi_erosion import SputteringPhysics

class WholeDeviceModel:
    """
    SCPN WDM: Coupled Multi-Physics Simulation.
    Loops: Equilibrium <-> Transport <-> Wall <-> Radiation
    """
    def __init__(self, config_path):
        self.transport = TransportSolver(config_path)
        self.pwi = SputteringPhysics("Tungsten")
        
        # Initialize
        self.transport.solve_equilibrium()
        
    def thomas_fermi_pressure(self, n_e_m3, T_eV):
        """
        Hardened Thomas-Fermi Equation of State (EOS) heuristic.
        Accounts for electron degeneracy pressure in the WDM regime.
        P_total = P_ideal + P_deg
        """
        # 1. Ideal Gas Term (Boltzmann)
        # P = n * k * T
        p_ideal = n_e_m3 * (T_eV * 1.602e-19)
        
        # 2. Degeneracy Term (Thomas-Fermi / Fermi-Dirac limit)
        # P_deg ~ (h_bar^2 / m_e) * n_e^(5/3)
        h_bar = 1.054e-34
        m_e = 9.109e-31
        p_deg = (h_bar**2 / m_e) * (n_e_m3**(5.0/3.0))
        
        # 3. Interpolation (Smooth transition)
        # We use a simple additive model which is a common WDM baseline
        return p_ideal + p_deg

    def calculate_redeposition_fraction(self, T_edge_eV, B_field_T):
        """
        Estimates the fraction of sputtered atoms that are promptly redeposited.
        Redeposition fraction f_redep ~ 1 - (lambda_ion / rho_L)
        For heavy impurities (W) in high B-field, redeposition can exceed 90%.
        """
        # Ionization mean free path lambda_ion ~ v_thermal / (n_e * <sigma_v>_ion)
        # Larmor radius rho_L = m*v / qB
        # Heuristic for W: f_redep increases with density and B-field
        n_e_edge = self.transport.ne[-1] * 1e19
        
        # Scaling based on impurity transport benchmarks
        f_redep = 0.95 * (1.0 - np.exp(-(B_field_T / 5.0) * (n_e_edge / 1e19)))
        return float(np.clip(f_redep, 0.0, 0.99))

    def run_discharge(self, duration_sec=10.0):
        print(f"--- SCPN WDM: WHOLE DEVICE SIMULATION ({duration_sec}s) ---")
        
        dt = 0.01
        steps = int(duration_sec / dt)
        
        history = []
        
        # Scenario: Constant Heating, but Wall degrades
        P_aux = 50.0 # MW
        
        print(f"{'Time':<6} | {'Te_core':<8} | {'Impurity':<8} | {'Status'}")
        print("-" * 50)
        
        for t in range(steps):
            # 1. Transport Step
            avg_T, core_T = self.transport.evolve_profiles(dt, P_aux)
            
            # 2. PWI Step (Calculate Erosion)
            n_edge = self.transport.ne[-1] * 1e19
            T_edge = self.transport.Te[-1] * 1000 # eV
            B_edge = 5.0 # Tesla approx
            
            # Sound speed at edge
            cs = np.sqrt((T_edge + T_edge) / (2 * 1.67e-27))
            flux_wall = n_edge * cs
            
            # Gross Erosion
            erosion = self.pwi.calculate_erosion_rate(flux_wall, T_edge)
            gross_impurity_flux = erosion['Impurity_Source']
            
            # Net Erosion (Hardened with Redeposition)
            f_redep = self.calculate_redeposition_fraction(T_edge, B_edge)
            net_impurity_flux = gross_impurity_flux * (1.0 - f_redep)
            
            # 3. Impurity Transport (Inject into Plasma)
            total_atoms_sec = net_impurity_flux * 500.0
            self.transport.inject_impurities(total_atoms_sec * 1e-4, dt)
            
            # 4. Equilibrium Coupling (Rare update)
            if t % 100 == 0:
                self.transport.map_profiles_to_2d()
                self.transport.solve_equilibrium()
                
            # 5. Check Survival
            # If Core Temp drops too low -> Radiative Collapse
            status = "OK"
            if core_T < 0.5: status = "COLLAPSE"
            
            # Log
            state = {
                'time': t*dt,
                'Te_core': core_T,
                'W_impurity': np.sum(self.transport.n_impurity),
                'P_rad': np.max(self.transport.n_impurity) * 100 # Approx metric
            }
            history.append(state)
            
            if t % 50 == 0:
                print(f"{t*dt:<6.2f} | {core_T:<8.2f} | {state['W_impurity']:<8.2e} | {status}")
                
            if status == "COLLAPSE":
                print(f"!!! RADIATIVE COLLAPSE DETECTED AT t={t*dt:.2f}s !!!")
                break
                
        self.plot_results(history)
        
    def plot_results(self, history):
        df = pd.DataFrame(history)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.plot(df['time'], df['Te_core'], 'r-', label='Core Temp (keV)')
        ax1.set_ylabel("Temperature")
        ax1.set_title("Thermal Quench due to Impurity Accumulation")
        ax1.grid(True)
        ax1.legend()
        
        ax2.plot(df['time'], df['W_impurity'], 'k-', label='Total Impurities')
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Accumulation (a.u.)")
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig("WDM_Simulation_Result.png")
        print("Analysis Saved: WDM_Simulation_Result.png")

if __name__ == "__main__":
    cfg = "03_CODE/SCPN-Fusion-Core/validation/iter_validated_config.json"
    wdm = WholeDeviceModel(cfg)
    wdm.run_discharge(duration_sec=2.0) # Short run to see collapse
