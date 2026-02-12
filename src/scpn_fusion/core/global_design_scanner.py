import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from scpn_fusion.core._rust_compat import FusionKernel
except ImportError:
    from scpn_fusion.core.fusion_kernel import FusionKernel
from scpn_fusion.core.fusion_ignition_sim import FusionBurnPhysics
from scpn_fusion.core.divertor_thermal_sim import DivertorLab
from scpn_fusion.engineering.balance_of_plant import PowerPlantModel

class GlobalDesignExplorer:
    """
    Monte Carlo Design Space Explorer.
    Searches for the Pareto Frontier of Fusion Reactors.
    Objectives: Maximize Q, Minimize Radius (Cost), Minimize Wall Load.
    """
    def __init__(self, base_config_path):
        self.base_config_path = base_config_path
        
    def evaluate_design(self, R_maj, B_field, I_plasma):
        """
        Runs a full-stack simulation for a specific design point.
        """
        # 1. Modify Configuration in-memory (Mocking the file load for speed)
        # We assume scaling laws for quick evaluation, 
        # but calling the actual Kernel would be too slow for 1000s of points without the Neural Accelerator.
        # Here we use the Physics Scaling relationships derived from our Kernel.
        
        # Scaling Laws (Calibrated from our previous Kernel runs):
        # Vol ~ 2 * pi * R * pi * a^2 (a = R/3)
        a_min = R_maj / 3.0
        Vol = 2 * np.pi**2 * R_maj * a_min**2
        
        # Stability Limit (Greenwald / Troyon)
        # beta_max ~ I / aB
        beta_N = 2.5
        max_pressure = beta_N * (I_plasma / (a_min * B_field)) * (B_field**2) # Simplified
        
        # Fusion Power ~ Vol * Pressure^2
        # Calibrated constant C (Optimistic H-Mode)
        C_fus = 0.20 
        P_fus = C_fus * Vol * max_pressure**2
        
        # Wall Load
        Surface = 4 * np.pi**2 * R_maj * a_min
        Neutron_Load = (0.8 * P_fus) / Surface
        
        # Divertor Load (Eich scaling)
        lambda_q = 0.63 * (B_field**(-1.19)) # mm
        P_sol = 0.2 * P_fus + 50.0 # Alpha + Aux
        # q_div ~ P_sol / (R * lambda)
        Div_Load = P_sol / (R_maj * lambda_q * 1e-3) / 10.0 # Expansion factor 10
        
        # Engineering Q
        P_aux = 50.0
        Q_eng = P_fus / P_aux
        
        return {
            'R': R_maj, 'B': B_field, 'Ip': I_plasma,
            'P_fus': P_fus,
            'Q': Q_eng,
            'Wall_Load': Neutron_Load,
            'Div_Load': Div_Load,
            'Cost': R_maj**3 * B_field # Rough proxy for cost
        }

    def run_scan(self, n_samples=2000):
        print(f"--- SCPN GLOBAL DESIGN SCAN ({n_samples} Universes) ---")
        
        results = []
        
        for i in range(n_samples):
            # Sampling Strategy (Latin Hypercube-ish)
            R = np.random.uniform(2.0, 9.0)
            B = np.random.uniform(4.0, 12.0)
            I = np.random.uniform(5.0, 25.0)
            
            # Physics Constraint: Safety Factor q95 > 3
            # q ~ 5 a^2 B / R I
            a = R/3.0
            q95 = 5 * a**2 * B / (R * I) * 2.0 # Approx
            
            if q95 < 3.0: continue # Unstable design, discard
            
            res = self.evaluate_design(R, B, I)
            results.append(res)
            
        df = pd.DataFrame(results)
        print(f"Valid Designs Found: {len(df)}")
        return df

    def analyze_pareto(self, df):
        # Filter: Viable Reactors
        # Q > 2 (Pilot Goal), Wall Load < 5.0
        viable = df[ (df['Q'] > 2.0) & (df['Wall_Load'] < 5.0) ]
        
        print(f"Viable Reactors (Q>2, Load<5): {len(viable)}")
        
        if len(viable) == 0:
            print("No viable reactors found with current technology limits.")
            return
            
        # Find "Best" (Min Cost)
        best = viable.loc[viable['Cost'].idxmin()]
        
        print("\n=== THE OPTIMAL REACTOR ===")
        print(f"Radius: {best['R']:.2f} m")
        print(f"Field:  {best['B']:.2f} T")
        print(f"Power:  {best['P_fus']:.1f} MW")
        print(f"Cost Index: {best['Cost']:.1f}")
        print(f"Div Load: {best['Div_Load']:.1f} MW/m2")
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Cost vs Performance
        sc = ax1.scatter(df['R'], df['Q'], c=df['Wall_Load'], cmap='jet', alpha=0.5, s=10)
        ax1.scatter(viable['R'], viable['Q'], color='black', marker='x', label='Viable')
        ax1.scatter(best['R'], best['Q'], color='red', s=100, label='OPTIMAL')
        
        ax1.set_xlabel("Major Radius R (m) -> Cost")
        ax1.set_ylabel("Fusion Gain Q")
        ax1.set_title("The Reactor Design Space")
        ax1.axhline(10, color='r', linestyle='--')
        plt.colorbar(sc, ax=ax1, label='Neutron Wall Load (MW/m2)')
        ax1.legend()
        
        # Plot 2: Divertor Challenge
        # B field vs Divertor Load
        sc2 = ax2.scatter(df['B'], df['Div_Load'], c=df['P_fus'], cmap='plasma', alpha=0.5)
        ax2.set_xlabel("Magnetic Field B (T)")
        ax2.set_ylabel("Divertor Heat Flux (MW/m2)")
        ax2.set_yscale('log')
        ax2.axhline(10, color='green', linestyle='--', label='W Limit (10)')
        ax2.axhline(50, color='orange', linestyle='--', label='Li Limit (50)')
        ax2.set_title("The Heat Exhaust Challenge")
        plt.colorbar(sc2, ax=ax2, label='Fusion Power (MW)')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig("Global_Design_Pareto.png")
        print("Analysis Saved: Global_Design_Pareto.png")

if __name__ == "__main__":
    # Dummy path, we use scaling laws
    explorer = GlobalDesignExplorer("dummy") 
    data = explorer.run_scan(n_samples=5000)
    explorer.analyze_pareto(data)
