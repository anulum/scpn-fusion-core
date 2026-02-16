# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Stability Analyzer
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

try:
    from scpn_fusion.core._rust_compat import FusionKernel
except ImportError:
    from scpn_fusion.core.fusion_kernel import FusionKernel

class StabilityAnalyzer:
    """
    Performs Linear Stability Analysis (Eigenvalue analysis) 
    of the plasma rigid-body motion (n=0 mode).
    """
    def __init__(self, config_path):
        self.kernel = FusionKernel(config_path)
        # Pre-calculate vacuum field once
        self.Psi_vac = self.kernel.calculate_vacuum_field()
        
    def get_vacuum_field_at(self, R, Z):
        """
        Interpolates Bz and Br from vacuum coils at position (R,Z).
        Bz = 1/R * dPsi/dR
        Br = -1/R * dPsi/dZ
        """
        # Simple Bilinear interpolation or Grid lookup
        # Map R,Z to grid indices
        ir = (R - self.kernel.R[0]) / self.kernel.dR
        iz = (Z - self.kernel.Z[0]) / self.kernel.dZ
        
        ir0 = int(np.clip(ir, 0, self.kernel.NR-2))
        iz0 = int(np.clip(iz, 0, self.kernel.NZ-2))
        
        # Gradients on grid
        dPsi_dR = (self.Psi_vac[iz0, ir0+1] - self.Psi_vac[iz0, ir0-1]) / (2*self.kernel.dR)
        dPsi_dZ = (self.Psi_vac[iz0+1, ir0] - self.Psi_vac[iz0-1, ir0]) / (2*self.kernel.dZ)
        
        Bz = (1.0 / R) * dPsi_dR
        Br = -(1.0 / R) * dPsi_dZ
        
        # Calculate Decay Index 'n' (Field curvature)
        # n = - (R/Bz) * (dBz/dR)
        # We need dBz/dR
        # d2Psi_dR2 needed
        d2Psi_dR2 = (self.Psi_vac[iz0, ir0+1] - 2*self.Psi_vac[iz0, ir0] + self.Psi_vac[iz0, ir0-1]) / (self.kernel.dR**2)
        dBz_dR = (1.0/R)*d2Psi_dR2 - (1.0/R**2)*dPsi_dR
        
        n_index = - (R / (Bz + 1e-9)) * dBz_dR
        
        return Bz, Br, n_index

    def calculate_forces(self, R, Z, Ip):
        """
        Calculates radial and vertical forces acting on the plasma ring.
        F_R = F_Hoop + F_Lorentz_R
        F_Z = F_Lorentz_Z
        """
        Bz, Br, n_idx = self.get_vacuum_field_at(R, Z)
        
        # 1. Hoop Force (Expansion)
        # Shafranov Formula
        mu0 = 4 * np.pi * 1e-7
        # Need 'a' (minor radius) and 'li' (inductance). 
        # Approx: a ~ R/3, li ~ 0.8
        a = R / 3.0
        li = 0.8
        beta_p = 0.5 # Assumed poloidal beta
        
        # F_hoop = (mu0 * I^2 / 2) * (ln(8R/a) + beta_p + li/2 - 1.5) / R
        # Note: Ip is in MA in kernel, convert to Amps
        Ip_A = Ip * 1e6
        
        term = np.log(8*R/a) + beta_p + (li/2.0) - 1.5
        F_hoop = (mu0 * Ip_A**2 / 2.0) * term / R
        
        # 2. Lorentz Force (Confining)
        # F = I x B
        # F_R = I_phi * B_z (length 2*pi*R)
        F_Lorentz_R = Ip_A * Bz * (2 * np.pi * R)
        
        # F_Z = - I_phi * B_r
        F_Lorentz_Z = - Ip_A * Br * (2 * np.pi * R)
        
        F_total_R = F_hoop + F_Lorentz_R
        F_total_Z = F_Lorentz_Z
        
        return F_total_R, F_total_Z, n_idx

    def analyze_stability(self, R_target=6.2, Z_target=0.0):
        print(f"--- EIGENVALUE STABILITY ANALYSIS ---")
        print(f"Checking Point: R={R_target}m, Z={Z_target}m")
        
        Ip = self.kernel.cfg['physics']['plasma_current_target']
        
        # 1. Base Forces
        Fr0, Fz0, n0 = self.calculate_forces(R_target, Z_target, Ip)
        print(f"Equilibrium Check:")
        print(f"  Radial Force:   {Fr0/1e6:.2f} MN (Should be 0)")
        print(f"  Vertical Force: {Fz0/1e6:.2f} MN (Should be 0)")
        print(f"  Field Index n:  {n0:.3f}")
        
        # 2. Build Stiffness Matrix K (Jacobian)
        # We perturb R and Z slightly to get derivatives dF/dx
        dR = 0.01 # 1 cm perturbation
        dZ = 0.01
        
        Fr_pR, _, _ = self.calculate_forces(R_target + dR, Z_target, Ip)
        Fr_mR, _, _ = self.calculate_forces(R_target - dR, Z_target, Ip)
        K_RR = -(Fr_pR - Fr_mR) / (2*dR) # Negative gradient = Stiffness
        
        _, Fz_pZ, _ = self.calculate_forces(R_target, Z_target + dZ, Ip)
        _, Fz_mZ, _ = self.calculate_forces(R_target, Z_target - dZ, Ip)
        K_ZZ = -(Fz_pZ - Fz_mZ) / (2*dZ)
        
        # Cross terms (usually small for symmetric symmetric)
        K_RZ = 0.0 
        K_ZR = 0.0
        
        K_matrix = np.array([[K_RR, K_RZ], [K_ZR, K_ZZ]])
        
        print("\nStiffness Matrix K (MN/m):")
        print(f"  [[ {K_RR/1e6:.2f}, {K_RZ/1e6:.2f} ]")
        print(f"   [ {K_ZR/1e6:.2f}, {K_ZZ/1e6:.2f} ]]")
        
        # 3. Eigenvalues
        eigvals, eigvecs = np.linalg.eig(K_matrix)
        
        print("\nEigenvalues (Stability):")
        for i, ev in enumerate(eigvals):
            status = "STABLE (Restoring Force)" if ev > 0 else "UNSTABLE (Exp Growth)"
            mode = "Radial" if abs(eigvecs[0,i]) > abs(eigvecs[1,i]) else "Vertical"
            print(f"  Mode {mode}: Lambda={ev/1e6:.2f} -> {status}")
            
        # Physics interpretation of n-index
        # Radial Stability requires: n < 1.5
        # Vertical Stability requires: n > 0
        if 0 < n0 < 1.5:
            print(f"\n[PASS] Field Index {n0:.2f} is in stable region (0 < n < 1.5).")
        else:
            print(f"\n[FAIL] Field Index {n0:.2f} violates stability limits!")
            if n0 < 0: print("  -> Vertical Instability (VDE Risk)")
            if n0 > 1.5: print("  -> Radial Instability")

        self.plot_stability_landscape(R_target, Z_target)

    def analyze_mhd_stability(
        self,
        R0: float = 6.2,
        a: float = 2.0,
        B0: float = 5.3,
        Ip_MA: float = 15.0,
        transport_solver=None,
    ) -> dict:
        """Run Mercier and ballooning stability analysis.

        Uses either profiles from *transport_solver* (if provided) or
        default parabolic profiles.

        Parameters
        ----------
        R0 : float — major radius [m]
        a : float — minor radius [m]
        B0 : float — toroidal field [T]
        Ip_MA : float — plasma current [MA]
        transport_solver : TransportSolver, optional
            If given, uses its rho/ne/Ti/Te profiles.

        Returns
        -------
        dict with keys ``q_profile``, ``mercier``, ``ballooning``
        """
        from scpn_fusion.core.stability_mhd import (
            compute_q_profile,
            mercier_stability,
            ballooning_stability,
        )

        if transport_solver is not None:
            rho = transport_solver.rho
            ne = transport_solver.ne
            Ti = transport_solver.Ti
            Te = transport_solver.Te
        else:
            nr = 50
            rho = np.linspace(0, 1, nr)
            ne = 10.0 * (1 - rho**2) ** 0.5
            Ti = 10.0 * (1 - rho**2) ** 1.5
            Te = Ti.copy()

        qp = compute_q_profile(rho, ne, Ti, Te, R0, a, B0, Ip_MA)
        mr = mercier_stability(qp)
        br = ballooning_stability(qp)

        return {"q_profile": qp, "mercier": mr, "ballooning": br}

    def plot_stability_landscape(self, R0, Z0):
        # Scan grid around target
        r_range = np.linspace(R0-2, R0+2, 50)
        z_range = np.linspace(Z0-2, Z0+2, 50)
        RR, ZZ = np.meshgrid(r_range, z_range)
        
        Potential = np.zeros_like(RR)
        
        # Integrate Force to get Potential Surface (-Integral F dot dl) 
        # Approximate W ~ - (Fr*R + Fz*Z) ... locally
        # Better: Calculate Force Magnitude map
        
        F_radial = np.zeros_like(RR)
        
        Ip = self.kernel.cfg['physics']['plasma_current_target']
        
        for i in range(50):
            for j in range(50):
                fr, fz, _ = self.calculate_forces(RR[i,j], ZZ[i,j], Ip)
                # We plot Radial Force map. Zero contour = Equilibrium.
                F_radial[i,j] = fr
                
        fig, ax = plt.subplots(figsize=(8, 6))
        # Contour of Radial Force
        # Where Fr = 0 is the equilibrium radius
        cs = ax.contour(RR, ZZ, F_radial / 1e6, levels=[-50, -20, -10, -5, 0, 5, 10, 20, 50], cmap='RdBu')
        ax.clabel(cs, inline=1, fontsize=10)
        ax.set_title("Radial Force Landscape (MN)")
        ax.set_xlabel("R (m)")
        ax.set_ylabel("Z (m)")
        ax.axvline(R0, color='k', linestyle='--', alpha=0.3)
        ax.axhline(Z0, color='k', linestyle='--', alpha=0.3)
        ax.scatter([R0], [Z0], color='green', label='Target')
        
        plt.savefig("Stability_Eigenanalysis.png")
        print("Analysis Plot saved: Stability_Eigenanalysis.png")

if __name__ == "__main__":
    # Use the analytic config which gave reasonable Bv but wrong position
    cfg = "03_CODE/SCPN-Fusion-Core/validation/iter_analytic_config.json"
    if not os.path.exists(cfg):
        cfg = "03_CODE/SCPN-Fusion-Core/validation/iter_validated_config.json"
        
    analyzer = StabilityAnalyzer(cfg)
    analyzer.analyze_stability(R_target=6.2, Z_target=0.0)
