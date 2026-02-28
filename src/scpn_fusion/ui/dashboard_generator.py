# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Dashboard Generator
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging

try:
    from scpn_fusion.core.fusion_kernel import FusionKernel
except ImportError:
    FusionKernel = None

logger = logging.getLogger(__name__)

class DashboardGenerator:
    """
    Advanced Visualization Suite for Fusion Core.
    Generates Poincaré plots, stability landscapes, and performance metrics.
    """
    def __init__(self, kernel):
        self.kernel = kernel
        
    def generate_poincare_plot(self, n_lines=20, n_transits=500):
        """
        Trace magnetic field lines to visualize topology (Poincaré Map).
        Reveals islands, stochastic regions, and X-points.
        """
        logger.info("Generating Poincaré plot...")
        
        # Seed points along the midplane (Z=0, R=R_min..R_max)
        R_start = np.linspace(self.kernel.cfg['dimensions']['R_min'] + 0.1, 
                              self.kernel.cfg['dimensions']['R_max'] - 0.1, n_lines)
        Z_start = np.zeros_like(R_start)
        r_min = float(self.kernel.cfg["dimensions"]["R_min"])
        r_max = float(self.kernel.cfg["dimensions"]["R_max"])
        z_min = float(self.kernel.cfg["dimensions"]["Z_min"])
        z_max = float(self.kernel.cfg["dimensions"]["Z_max"])
        r0 = 0.5 * (r_min + r_max)
        z0 = 0.5 * (z_min + z_max)
        max_psi = max(float(np.max(np.abs(self.kernel.Psi))), 1.0e-9)
        max_radius = max(r_max - r0, r0 - r_min, z_max - z0, z0 - z_min)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot separatrix reference
        ax.contour(self.kernel.RR, self.kernel.ZZ, self.kernel.Psi, levels=20, colors='gray', alpha=0.2)
        
        # Field line integrator (Symplectic-like or RK4)
        # dR/dphi = R * Br / Bphi
        # dZ/dphi = R * Bz / Bphi
        
        for i in range(n_lines):
            r, z = R_start[i], Z_start[i]
            path_r = [r]
            path_z = [z]
            
            # Trace
            for t in range(n_transits):
                # Simple map approximation (Standard Map-like) for demonstration
                # Real implementation needs 3D field tracing. 
                # Here we simulate the effect of magnetic perturbations.
                
                # q-profile proxy
                psi_val = self._get_psi(r, z)
                q = 1.0 + 3.0 * (float(psi_val) / max_psi)**2
                 
                # Perturbation (m=2, n=1 mode)
                k = 2e-4
                theta = np.arctan2(z - z0, r - r0) # Poloidal angle around magnetic axis
                 
                # Map: theta_n+1 = theta_n + 2*pi/q + k*sin(theta_n)
                # r_n+1 = r_n + k*sin(theta_n) (Radial kick)
                 
                theta_new = theta + 2*np.pi/q + k*np.sin(2*theta)
                radial_kick = (k / 10.0) * np.sin(2 * theta)
                 
                # Convert back to R,Z (assuming circular for plot)
                # This is a heuristic visualizer for topology health
                # Real tracing requires B_vector
                radius = np.hypot(r - r0, z - z0)
                radius = float(np.clip(radius + radial_kick, 0.02, max_radius))
                r_new = float(np.clip(r0 + radius * np.cos(theta_new), r_min, r_max))
                z_new = float(np.clip(z0 + radius * np.sin(theta_new), z_min, z_max))

                r = r_new
                z = z_new
                path_r.append(r)
                path_z.append(z)

            ax.plot(path_r, path_z, ".", ms=0.8, alpha=0.35)
                 
            # Fallback: Plot contours of Psi which represent the unperturbed surfaces
            # Real Poincaré requires 3D code (e.g. VENUS/NEMATO integration)
            
        # High-fidelity Contour Plot (surrogate for full Poincaré in 2D axisymmetry)
        cs = ax.contour(self.kernel.RR, self.kernel.ZZ, self.kernel.Psi, levels=50, cmap='nipy_spectral')
        ax.set_title("Magnetic Topology (Flux Surfaces)")
        ax.set_xlabel("R (m)")
        ax.set_ylabel("Z (m)")
        ax.set_aspect('equal')
        
        return fig

    def _get_psi(self, r, z):
        # Nearest neighbor lookup
        ir = int((r - self.kernel.R[0]) / self.kernel.dR)
        iz = int((z - self.kernel.Z[0]) / self.kernel.dZ)
        ir = np.clip(ir, 0, self.kernel.NR-1)
        iz = np.clip(iz, 0, self.kernel.NZ-1)
        return self.kernel.Psi[iz, ir]

def run_dashboard(config_path):
    if FusionKernel is None:
        print("FusionKernel not available.")
        return
        
    kernel = FusionKernel(config_path)
    kernel.solve_equilibrium()
    
    gen = DashboardGenerator(kernel)
    fig = gen.generate_poincare_plot()
    
    fig.savefig("Poincare_Topology.png")
    print("Dashboard saved: Poincare_Topology.png")

if __name__ == "__main__":
    cfg = str(Path(__file__).resolve().parents[3] / "iter_config.json")
    run_dashboard(cfg)
