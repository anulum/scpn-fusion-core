# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Global Design Scanner
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
from scpn_fusion.core.fusion_ignition_sim import FusionBurnPhysics
from scpn_fusion.core.divertor_thermal_sim import DivertorLab
from scpn_fusion.core.heat_ml_shadow_surrogate import HeatMLShadowSurrogate
from scpn_fusion.engineering.balance_of_plant import PowerPlantModel

class GlobalDesignExplorer:
    """
    Monte Carlo Design Space Explorer.
    Searches for the Pareto Frontier of Fusion Reactors.
    Objectives: Maximize Q, Minimize Radius (Cost), Minimize Wall Load.
    """
    def __init__(
        self,
        base_config_path,
        *,
        divertor_flux_cap_mw_m2=45.0,
        zeff_cap=0.4,
        hts_peak_cap_t=21.0,
    ):
        self.base_config_path = base_config_path
        self.heat_ml_shadow = HeatMLShadowSurrogate()
        self.heat_ml_shadow.fit_synthetic(seed=42, samples=1536)
        self.divertor_flux_cap_mw_m2 = float(divertor_flux_cap_mw_m2)
        self.zeff_cap = float(zeff_cap)
        self.hts_peak_cap_t = float(hts_peak_cap_t)
        
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
        # q_div ~ P_sol / (2*pi*R*lambda) with compact-device calibration.
        expansion_factor = 12.0 + 0.6 * B_field
        Div_Load = (
            P_sol / (2.0 * np.pi * R_maj * lambda_q * 1e-3) / expansion_factor
        ) * 1e-4

        # HEAT-ML magnetic-shadow attenuation (GAI-03).
        b_pol_equiv = max(0.4, 0.22 * B_field)
        shadow_features = np.array([R_maj, b_pol_equiv, P_sol, 10.0, 1.65, 0.35, -1.8])
        shadow_fraction = float(self.heat_ml_shadow.predict_shadow_fraction(shadow_features)[0])
        Div_Load_Optimized = float(
            self.heat_ml_shadow.predict_divertor_flux(Div_Load, shadow_features)[0]
        )

        b_peak_hts_t = float(1.72 * B_field + 0.6)
        zeff_est = float(
            np.clip(0.18 + 0.0035 * Div_Load_Optimized + 0.015 * max(0.0, 1.6 - R_maj), 0.15, 0.8)
        )
        constraint_ok = bool(
            Div_Load_Optimized <= self.divertor_flux_cap_mw_m2
            and zeff_est <= self.zeff_cap
            and b_peak_hts_t <= self.hts_peak_cap_t
        )
        
        # Engineering Q
        P_aux = 50.0
        Q_eng = P_fus / P_aux
        
        return {
            'R': R_maj, 'B': B_field, 'Ip': I_plasma,
            'P_fus': P_fus,
            'Q': Q_eng,
            'Wall_Load': Neutron_Load,
            'Div_Load_Baseline': Div_Load,
            'Shadow_Fraction': shadow_fraction,
            'Div_Load_Optimized': Div_Load_Optimized,
            'Div_Load': Div_Load_Optimized,
            'B_peak_HTS_T': b_peak_hts_t,
            'Zeff_Est': zeff_est,
            'Constraint_OK': constraint_ok,
            'Cost': R_maj**3 * B_field # Rough proxy for cost
        }

    def run_scan(
        self,
        n_samples=2000,
        *,
        r_bounds=(2.0, 9.0),
        b_bounds=(4.0, 12.0),
        i_bounds=(5.0, 25.0),
        seed=None,
        q95_min=3.0,
    ):
        print(f"--- SCPN GLOBAL DESIGN SCAN ({n_samples} Universes) ---")
        if seed is not None:
            np.random.seed(int(seed))
        
        results = []
        
        for i in range(n_samples):
            # Sampling Strategy (Latin Hypercube-ish)
            R = np.random.uniform(r_bounds[0], r_bounds[1])
            B = np.random.uniform(b_bounds[0], b_bounds[1])
            I = np.random.uniform(i_bounds[0], i_bounds[1])
            
            # Physics Constraint: Safety Factor q95 > 3
            # q ~ 5 a^2 B / R I
            a = R/3.0
            q95 = 5 * a**2 * B / (R * I) * 2.0 # Approx
            
            if q95 < q95_min:
                continue  # Unstable design, discard
            
            res = self.evaluate_design(R, B, I)
            results.append(res)
            
        if results:
            df = pd.DataFrame(results)
        else:
            df = pd.DataFrame(
                columns=[
                    "R",
                    "B",
                    "Ip",
                    "P_fus",
                    "Q",
                    "Wall_Load",
                    "Div_Load_Baseline",
                    "Shadow_Fraction",
                    "Div_Load_Optimized",
                    "Div_Load",
                    "B_peak_HTS_T",
                    "Zeff_Est",
                    "Constraint_OK",
                    "Cost",
                ]
            )
        print(f"Valid Designs Found: {len(df)}")
        return df

    def run_compact_scan(self, n_samples=2000, seed=42):
        return self.run_scan(
            n_samples=n_samples,
            r_bounds=(1.1, 1.9),
            b_bounds=(8.8, 12.2),
            i_bounds=(2.0, 9.0),
            seed=seed,
            q95_min=1.2,
        )

    def analyze_pareto(self, df):
        # Filter: Viable Reactors
        # Q > 2 (Pilot Goal), Wall Load < 5.0 + active engineering constraints.
        viable = df[(df['Q'] > 2.0) & (df['Wall_Load'] < 5.0) & (df['Constraint_OK'])]
        
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
        print(f"Div Load (baseline): {best['Div_Load_Baseline']:.1f} MW/m2")
        print(f"Div Load (HEAT-ML optimized): {best['Div_Load_Optimized']:.1f} MW/m2")
        print(f"Zeff estimate: {best['Zeff_Est']:.3f}")
        print(f"HTS peak field: {best['B_peak_HTS_T']:.2f} T")
        
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
        ax2.set_ylabel("Divertor Heat Flux (MW/m2, HEAT-ML optimized)")
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
