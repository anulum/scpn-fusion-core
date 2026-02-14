# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Force Balance
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
import numpy as np
import sys
import os
import json

try:
    from scpn_fusion.core._rust_compat import FusionKernel
except ImportError:
    from scpn_fusion.core.fusion_kernel import FusionKernel
from scpn_fusion.core.stability_analyzer import StabilityAnalyzer

class ForceBalanceSolver:
    """
    Automatic Newton-Raphson Solver to find Coil Currents that result in 
    ZERO Radial Force on the plasma at the target Radius.
    This guarantees static equilibrium at R_target.
    """
    def __init__(self, config_path):
        self.config_path = config_path
        self.analyzer = StabilityAnalyzer(config_path)
        
    def solve_for_equilibrium(self, target_R=6.2, target_Z=0.0):
        print(f"--- FORCE BALANCE SOLVER (Newton-Raphson) ---")
        print(f"Target Equilibrium: R={target_R}m, Z={target_Z}m")
        
        # We control PF3 and PF4 (Outer coils) to control Radial Position
        # We assume Up-Down symmetry for now (PF3=PF4)
        control_indices = [2, 3] # PF3, PF4 indexes in coil list
        
        # Load Physics Params
        Ip = self.analyzer.kernel.cfg['physics']['plasma_current_target']
        
        # Newton Loop
        for iter in range(10):
            # 1. Calculate Current Force
            Fr, Fz, n_idx = self.analyzer.calculate_forces(target_R, target_Z, Ip)
            print(f"Iter {iter}: Radial Force = {Fr/1e6:.2f} MN")
            
            if abs(Fr) < 1e4: # Tolerance 10 kN
                print("  -> CONVERGED. Force Balance Achieved.")
                break
                
            # 2. Calculate Jacobian dF/dI (Numerical Derivative)
            # We perturb PF3/PF4 together
            dI = 0.1 # MA perturbation
            
            # Apply perturbation
            currents = [c['current'] for c in self.analyzer.kernel.cfg['coils']]
            original_I3 = currents[2]
            
            # Modify config in memory
            self.analyzer.kernel.cfg['coils'][2]['current'] += dI
            self.analyzer.kernel.cfg['coils'][3]['current'] += dI
            
            # Recalculate Vacuum Field (Expensive part, but necessary)
            self.analyzer.Psi_vac = self.analyzer.kernel.calculate_vacuum_field()
            
            # Calculate new Force
            Fr_new, _, _ = self.analyzer.calculate_forces(target_R, target_Z, Ip)
            
            # Jacobian J = dF/dI (MN / MA)
            J = (Fr_new - Fr) / dI
            print(f"  Jacobian dF/dI_PF34: {J/1e6:.2f} MN/MA")
            
            # 3. Newton Step: I_new = I_old - F / J
            # We want F_target = 0
            # 0 = F_old + J * delta_I
            # delta_I = - F_old / J
            
            delta_I = - Fr / J
            
            # Safety Clamp (don't jump more than 5 MA at once)
            delta_I = np.clip(delta_I, -5.0, 5.0)
            
            print(f"  Correction: Delta I = {delta_I:.3f} MA")
            
            # Apply correction
            new_I = original_I3 + delta_I
            
            # Reset Config to new values
            self.analyzer.kernel.cfg['coils'][2]['current'] = new_I
            self.analyzer.kernel.cfg['coils'][3]['current'] = new_I
            
            # Update Vacuum field for next iteration base
            self.analyzer.Psi_vac = self.analyzer.kernel.calculate_vacuum_field()
            
        # Save Result
        self.save_config()
        
    def save_config(self):
        # Save to validation folder
        out_path = os.path.join(os.path.dirname(__file__), "../../../validation/iter_force_balanced.json")
        with open(out_path, "w") as f:
            json.dump(self.analyzer.kernel.cfg, f, indent=4)
        print(f"Balanced Config Saved: {out_path}")

if __name__ == "__main__":
    # Start from the last "best" config (Genetic or Validated)
    base_cfg = "03_CODE/SCPN-Fusion-Core/validation/iter_validated_config.json"
    
    solver = ForceBalanceSolver(base_cfg)
    solver.solve_for_equilibrium(target_R=6.2)
