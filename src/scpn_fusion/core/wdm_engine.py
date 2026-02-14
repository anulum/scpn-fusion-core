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
            # 1. Transport Step (Evolve T, n)
            # This now includes Radiation Cooling!
            avg_T, core_T = self.transport.evolve_profiles(dt, P_aux)
            
            # 2. PWI Step (Calculate Erosion)
            # Flux to wall ~ Particle Density at Edge * Sound Speed
            n_edge = self.transport.ne[-1] * 1e19
            T_edge = self.transport.Te[-1] * 1000 # eV
            flux_wall = n_edge * np.sqrt((T_edge + T_edge)/ (2*1.67e-27)) # Gamma * n * cs
            
            # Erosion
            erosion = self.pwi.calculate_erosion_rate(flux_wall, T_edge)
            impurity_flux = erosion['Impurity_Source'] # Atoms/s/m2
            
            # 3. Impurity Transport (Inject into Plasma)
            # Total atoms = Flux * Area (approx 500 m2)
            total_atoms_sec = impurity_flux * 500.0
            # Scale down for demo stability (real W is deadly)
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
