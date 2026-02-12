import numpy as np
import json
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from scpn_fusion.core._rust_compat import FusionKernel
except ImportError:
    from scpn_fusion.core.fusion_kernel import FusionKernel

class AnalyticEquilibriumSolver:
    """
    Calculates the exact Coil Currents required to hold plasma at a specific Radius.
    Uses the Shafranov formula for Vertical Field required for radial equilibrium.
    """
    def __init__(self, config_path):
        self.kernel = FusionKernel(config_path)
        
    def calculate_required_Bv(self, R_geo, a_min, Ip_MA, beta_p=0.5, li=0.8):
        """
        Calculates the Vertical Magnetic Field (Bv) required to hold the plasma.
        Shafranov Equation (1966).
        """
        mu0 = 4 * np.pi * 1e-7
        Ip = Ip_MA * 1e6 # Convert to Amps
        
        # Term 1: Hoop Force (Expansion force of the current ring)
        # Term 2: Pressure force (beta_p)
        # Term 3: Internal Inductance (li)
        
        term_log = np.log(8 * R_geo / a_min)
        term_physics = beta_p + (li / 2.0) - 1.5
        
        Bv = - (mu0 * Ip) / (4 * np.pi * R_geo) * (term_log + term_physics)
        
        print(f"--- SHAFRANOV EQUILIBRIUM CHECK ---")
        print(f"Target Radius: {R_geo} m")
        print(f"Plasma Current: {Ip_MA} MA")
        print(f"Required Vertical Field (Bv): {Bv:.4f} Tesla")
        
        return Bv

    def solve_coil_currents(self, target_Bv, target_R):
        """
        Finds coil currents [I1, I2, ...] such that:
        1. Sum(B_z_i * I_i) = Target_Bv (at R_geo)
        2. Minimize Total Current (Energy efficiency)
        """
        coils = self.kernel.cfg['coils']
        n_coils = len(coils)
        
        # Build Green's Matrix G [1, n_coils] representing B_z per Ampere at Target
        G = np.zeros(n_coils)
        
        print("\nCalculating Coil Influence Matrix (Green's Functions)...")
        for i, coil in enumerate(coils):
            # Biot-Savart Law for a Loop at (Rc, Zc) affecting (Rt, 0)
            # Simplified for B_z on the midplane
            Rc, Zc = coil['r'], coil['z']
            Rt = target_R
            
            # Distance vectors
            # This requires Elliptic Integrals for precision, 
            # but we use the dipole approx for far-field estimation in this script
            # B_z ~ mu0 * I * R^2 / (2 * (R^2 + Z^2)^1.5)
            
            dist_sq = (Rc - Rt)**2 + (Zc - 0)**2
            dist = np.sqrt(dist_sq)
            
            # Using our kernel's Green function logic logic (Logarithmic approx)
            # dPsi/dR gives B_Z. Psi ~ ln(1/dist). dPsi/dR ~ -1/dist * d(dist)/dR
            # Actually, let's use the Kernel's existing function to be consistent!
            
            # Temporarily set coil to 1A, others 0
            # This is "Virtual Experiment"
            pass # We will do it properly below
            
        # Better approach: Use the Kernel to compute B-field map for unitary currents
        # Because our Kernel uses specific approximations, we must be consistent with it.
        
        for i in range(n_coils):
            # Reset
            for c in self.kernel.cfg['coils']: c['current'] = 0.0
            
            # Set Unit Current
            self.kernel.cfg['coils'][i]['current'] = 1.0 # 1 MA
            
            # Compute Field
            Psi_vac = self.kernel.calculate_vacuum_field()
            
            # Extract Bz at Target R
            # Bz = 1/R * dPsi/dR
            # Find grid index for Target R
            idx_R = np.abs(self.kernel.R - target_R).argmin()
            idx_Z = np.abs(self.kernel.Z - 0.0).argmin() # Midplane
            
            # Gradient dPsi/dR
            dPsi = (Psi_vac[idx_Z, idx_R+1] - Psi_vac[idx_Z, idx_R-1]) / (2 * self.kernel.dR)
            Bz_unit = (1.0 / target_R) * dPsi
            
            G[i] = Bz_unit
            print(f"  Coil {coil['name']} Efficiency: {Bz_unit:.4f} T/MA")
            
        # NOW SOLVE: G . I = Target_Bv
        # We have 1 equation, N unknowns. Underdetermined.
        # We add regularization: Minimize norm(I)
        # Solution: I = pinv(G) * Target_Bv
        
        # Weighted solution? 
        # We prefer Outer Coils (PF3, PF4) for Vertical Field.
        # We prefer CS for Flux (but here we solve only for Equilibrium Bv)
        
        I_solution = np.linalg.pinv(G.reshape(1, -1)).dot(np.array([target_Bv]))
        I_solution = I_solution.flatten()
        
        print("\n--- ANALYTIC SOLUTION (Least Norm) ---")
        for i, val in enumerate(I_solution):
            name = coils[i]['name']
            print(f"  {name}: {val:.3f} MA")
            
        return I_solution

    def apply_and_save(self, currents):
        # Update Config
        for i, val in enumerate(currents):
            self.kernel.cfg['coils'][i]['current'] = val
            
        # Save
        out_path = os.path.join(os.path.dirname(__file__), "../../../validation/iter_analytic_config.json")
        with open(out_path, "w") as f:
            json.dump(self.kernel.cfg, f, indent=4)
        print(f"Saved Analytic Configuration: {out_path}")
if __name__ == "__main__":
    # Load template
    cfg_path = "03_CODE/SCPN-Fusion-Core/calibration/iter_genetic_temp.json" 
    # Use temp template from previous step if exists, or create new
    if not os.path.exists(cfg_path):
        cfg_path = "03_CODE/SCPN-Fusion-Core/validation/iter_validated_config.json"
        
    solver = AnalyticEquilibriumSolver(cfg_path)
    
    # ITER Specs
    R_target = 6.2
    a_minor = 2.0
    Ip_target = 15.0 # MA
    
    # 1. Calc Bv
    B_req = solver.calculate_required_Bv(R_target, a_minor, Ip_target)
    
    # 2. Solve Currents
    I_opt = solver.solve_coil_currents(B_req, R_target)
    
    # 3. Save
    solver.apply_and_save(I_opt)
