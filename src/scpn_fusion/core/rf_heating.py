import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from scpn_fusion.core._rust_compat import FusionKernel
except ImportError:
    from scpn_fusion.core.fusion_kernel import FusionKernel

class RFHeatingSystem:
    """
    Simulates Ion Cyclotron Resonance Heating (ICRH).
    Uses Ray-Tracing to track EM waves launching from the antenna 
    and absorbing at the resonance layer.
    """
    def __init__(self, config_path):
        self.kernel = FusionKernel(config_path)
        self.kernel.solve_equilibrium() # Get B-field map
        
        # Physics Constants
        self.q_D = 1.602e-19 # Charge (Deuterium)
        self.m_D = 3.34e-27  # Mass (Deuterium)
        self.freq = 50e6     # 50 MHz (Standard ICRH freq)
        self.omega_wave = 2 * np.pi * self.freq
        
    def get_plasma_params(self, R, Z):
        """
        Returns B_mod, density, and derivatives at (R,Z).
        """
        # 1. Magnetic Field (Toroidal dominates)
        # B_tor ~ B0 * R0 / R
        B0 = 5.3 # Tesla at axis (ITER)
        R0 = 6.2
        B_tor = B0 * R0 / R
        
        # Poloidal field from kernel
        # We need grid lookup
        ir = int((R - self.kernel.R[0]) / self.kernel.dR)
        iz = int((Z - self.kernel.Z[0]) / self.kernel.dZ)
        
        if 0 <= ir < self.kernel.NR and 0 <= iz < self.kernel.NZ:
            B_R = self.kernel.B_R[iz, ir]
            B_Z = self.kernel.B_Z[iz, ir]
            psi_val = self.kernel.Psi[iz, ir]
        else:
            B_R, B_Z, psi_val = 0, 0, 0
            
        B_mod = np.sqrt(B_tor**2 + B_R**2 + B_Z**2)
        
        # 2. Density Profile (Parabolic)
        # n = n0 * (1 - psi_norm)
        # Simplified: Gaussian blob
        dist_sq = (R - R0)**2 + Z**2
        n_e = 1e20 * np.exp(-dist_sq / 2.0)
        
        # Derivatives of density (for refraction)
        dn_dR = -n_e * (R - R0) / 1.0
        dn_dZ = -n_e * Z / 1.0
        
        return B_mod, n_e, dn_dR, dn_dZ

    def dispersion_relation(self, R, Z, k_R, k_Z):
        """
        Calculates the local dispersion D(w, k) = 0.
        Simplified Cold Plasma (Alfven wave approx).
        k^2 = (omega/v_A)^2
        """
        B_mod, n_e, _, _ = self.get_plasma_params(R, Z)
        
        if n_e < 1e18: return 1.0 # Vacuum
        
        # Alfven speed
        mu0 = 4*np.pi*1e-7
        v_A = B_mod / np.sqrt(mu0 * n_e * self.m_D)
        
        k_sq = k_R**2 + k_Z**2
        
        # Dispersion: omega^2 = k^2 * v_A^2
        # D = k^2 * v_A^2 - omega^2
        D = k_sq * v_A**2 - self.omega_wave**2
        
        return D

    def ray_equations(self, state, t):
        """
        Hamiltonian Ray Tracing equations.
        dr/dt = dD/dk
        dk/dt = -dD/dr
        """
        R, Z, kR, kZ = state
        
        # Finite differences for derivatives of D
        # This implicitly handles refraction and reflection
        eps = 1e-3
        
        # dD/dk
        D_pkR = self.dispersion_relation(R, Z, kR + eps, kZ)
        D_mkR = self.dispersion_relation(R, Z, kR - eps, kZ)
        dD_dkR = (D_pkR - D_mkR) / (2*eps)
        
        D_pkZ = self.dispersion_relation(R, Z, kR, kZ + eps)
        D_mkZ = self.dispersion_relation(R, Z, kR, kZ - eps)
        dD_dkZ = (D_pkZ - D_mkZ) / (2*eps)
        
        # dD/dr
        D_pR = self.dispersion_relation(R + eps, Z, kR, kZ)
        D_mR = self.dispersion_relation(R - eps, Z, kR, kZ)
        dD_dR = (D_pR - D_mR) / (2*eps)
        
        D_pZ = self.dispersion_relation(R, Z + eps, kR, kZ)
        D_mZ = self.dispersion_relation(R, Z - eps, kR, kZ)
        dD_dZ = (D_pZ - D_mZ) / (2*eps)
        
        # Group Velocity (dr/dt)
        dR_dt = -dD_dkR
        dZ_dt = -dD_dkZ
        
        # Wavevector change (dk/dt)
        dkR_dt = dD_dR
        dkZ_dt = dD_dZ
        
        return [dR_dt, dZ_dt, dkR_dt, dkZ_dt]

    def trace_rays(self, n_rays=10):
        print("--- RF HEATING RAY TRACING ---")
        print(f"Frequency: {self.freq/1e6} MHz")
        
        # Antenna Position (Outboard midplane)
        R_ant = 9.0
        Z_ant_spread = np.linspace(-1.0, 1.0, n_rays)
        
        trajectories = []
        
        for i in range(n_rays):
            # Initial condition: Launch inward (kR < 0)
            k0 = 10.0 # Initial wavenumber
            init_state = [R_ant, Z_ant_spread[i], -k0, 0.0]
            
            t_span = np.linspace(0, 0.5, 100) # Short time (normalized)
            
            # Solve ODE
            sol = odeint(self.ray_equations, init_state, t_span)
            
            # Check Resonance
            # Resonance condition: omega = omega_ci = qB/m
            # B_res = omega * m / q
            B_res = self.omega_wave * self.m_D / self.q_D
            
            # Find where B matches B_res along path
            # (Post-processing check)
            
            trajectories.append(sol)
            
        print(f"Resonance Field B_res: {B_res:.2f} Tesla")
        return trajectories, B_res

    def plot_heating(self, trajectories, B_res):
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 1. Plasma Geometry
        ax.contour(self.kernel.RR, self.kernel.ZZ, self.kernel.Psi, colors='gray', alpha=0.3)
        
        # 2. Resonance Layer (Vertical Line approx)
        # B_tor ~ B0*R0/R. B ~ B_res => R_res ~ B0*R0/B_res
        R_res = (5.3 * 6.2) / B_res
        ax.axvline(R_res, color='green', linestyle='--', linewidth=2, label='Cyclotron Resonance Layer')
        
        # 3. Rays
        for i, sol in enumerate(trajectories):
            R = sol[:, 0]
            Z = sol[:, 1]
            ax.plot(R, Z, 'r-', alpha=0.6)
            
            # Draw absorption point (intersection with Resonance)
            # Simple geometric check
            idx = np.abs(R - R_res).argmin()
            if np.abs(R[idx] - R_res) < 0.1:
                ax.plot(R[idx], Z[idx], 'y*', markersize=10, label='Energy Dump' if i==0 else "")
        
        # Antenna
        ax.plot([9.0]*len(trajectories), [t[0,1] for t in trajectories], 'ko', label='Antenna Array')
        
        ax.set_title(f"ICRH Wave Propagation ({self.freq/1e6} MHz)")
        ax.set_xlabel("R (m)")
        ax.set_ylabel("Z (m)")
        ax.set_xlim(2, 10)
        ax.set_ylim(-6, 6)
        ax.legend()
        
        plt.savefig("RF_Heating_Rays.png")
        print("Saved: RF_Heating_Rays.png")

if __name__ == "__main__":
    cfg = "03_CODE/SCPN-Fusion-Core/validation/iter_validated_config.json"
    rf = RFHeatingSystem(cfg)
    rays, B_res = rf.trace_rays()
    rf.plot_heating(rays, B_res)
