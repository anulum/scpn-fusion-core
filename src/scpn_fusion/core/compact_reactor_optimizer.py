# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Compact Reactor Optimizer
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sys

class CompactReactorArchitect:
    """
    Optimizes Fusion Reactor Geometry for MINIMUM SIZE.
    """
    def __init__(self):
        self.J_crit_base = 1500.0 # MA/m2 
        self.B_max_coil = 30.0   # Tesla 
        self.lambda_shield = 0.10 
        self.fluence_limit = 5e22 
        
    def plasma_physics_model(self, R, a, B0):
        Vol = 2 * np.pi * R * np.pi * a**2 
        kappa = 2.0
        I_p = (5 * a**2 * B0 / R) * ((1 + kappa**2)/2) / 3.0
        beta_limit = 4.0 * (I_p / (a * B0)) / 100.0 
        pressure = beta_limit * (B0**2 / (2 * 1.25e-6))
        p_fus_density = 0.25 * (pressure / 1e6)**2 
        P_fusion = p_fus_density * Vol
        return P_fusion, I_p, Vol

    def radial_build_constraints(self, R, a, B0):
        d_shield = 0.10 
        gap = 0.02
        d_coil = 0.2
        R_post = R - a - d_shield - gap
        if R_post < 0.05: return False, 0
        B_coil = B0 * (R / R_post)
        I_total_MA = 5.0 * R * B0 
        Area_coil = np.pi * (R_post**2 - (R_post-d_coil)**2)
        J_real = I_total_MA / (Area_coil + 1e-9)
        J_limit = self.J_crit_base * (20.0 / B_coil)
        magnet_ok = (J_real < J_limit) and (B_coil < self.B_max_coil)
        return magnet_ok, B_coil

    def visualize_space(self, designs, label=""):
        if not designs: return
        Rs = [d['R'] for d in designs]
        Bs = [d['B0'] for d in designs]
        Ps = [d['P_fus'] for d in designs]
        plt.figure(figsize=(10, 6))
        sc = plt.scatter(Rs, Bs, c=Ps, cmap='viridis', s=50, alpha=0.8)
        plt.colorbar(sc, label='Fusion Power (MW)')
        plt.xlabel('Major Radius R (m)')
        plt.ylabel('Magnetic Field B0 (Tesla)')
        plt.title(f'Compact Fusion Design Space - {label}')
        plt.savefig(f"Compact_Space_{label}.png")

    def report_design(self, d):
        print("\n=== MINIMUM VIABLE REACTOR FOUND ===")
        print(f"Geometry:      R = {d['R']:.3f} m, a = {d['a']:.3f} m (A={d['R']/d['a']:.1f})")
        print(f"Magnetics:     B0 = {d['B0']:.1f} T (Plasma), B_max = {d['B_coil']:.1f} T (Coil)")
        print(f"Performance:   P_fusion = {d['P_fus']:.1f} MW")
        print(f"Heat Loads:    Divertor = {d['q_div']:.1f} MW/m2, Wall = {d['q_wall']:.2f} MW/m2")
        print(f"Plasma Vol:    {d['Vol']:.1f} m3")
        print("Technology:    REBCO HTS, TEMHD Liquid Divertor, Detached Mode")
        print("====================================")

    def find_minimum_reactor(self, target_power_MW=1.0, use_temhd=True):
        label = "TEMHD" if use_temhd else "Solid"
        print(f"--- SCPN COMPACT OPTIMIZER (Target: >{target_power_MW} MW, {label}, Detached) ---")
        best_R = 999.0
        best_design = None
        radii = np.linspace(0.3, 5.0, 100)
        fields = np.linspace(5.0, 20.0, 30)
        valid_designs = []
        max_div_load = 100.0 if use_temhd else 10.0 
        for R in radii:
            for B0 in fields:
                for A in [2.0, 2.5, 3.0]:
                    a = R / A
                    P_fus, Ip, Vol = self.plasma_physics_model(R, a, B0)
                    if P_fus < target_power_MW: continue
                    ok, B_coil = self.radial_build_constraints(R, a, B0)
                    f_rad = 0.90
                    P_sep = (0.2 * P_fus + 5.0) * (1.0 - f_rad)
                    lambda_q = 0.63 * (B0**(-1.19)) * 1e-3
                    Area_div = 2 * np.pi * R * lambda_q * 20.0
                    q_div = P_sep / Area_div 
                    Area_wall = 4 * np.pi**2 * R * a
                    q_wall = (0.8 * P_fus) / Area_wall 
                    if ok and q_div < max_div_load and q_wall < 5.0:
                        design = {'R': R, 'a': a, 'B0': B0, 'B_coil': B_coil,
                                  'P_fus': P_fus, 'Vol': Vol, 'Ip': Ip,
                                  'q_div': q_div, 'q_wall': q_wall}
                        valid_designs.append(design)
                        if R < best_R:
                            best_R = R
                            best_design = design
        if best_design:
            self.report_design(best_design)
            self.visualize_space(valid_designs, label)
        else:
            print(f"No viable design found for {label}.")

if __name__ == "__main__":
    architect = CompactReactorArchitect()
    architect.find_minimum_reactor(target_power_MW=5.0, use_temhd=True)
