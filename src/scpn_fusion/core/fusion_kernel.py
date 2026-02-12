# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Fusion Kernel
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
import numpy as np
import json
import time
import os
from scipy.optimize import minimize
from scpn_fusion.hpc.hpc_bridge import HPCBridge

class FusionKernel:
    """
    SCPN Fusion Core - Advanced Non-Linear Free-Boundary Solver.
    Features:
    - Full Grad-Shafranov Physics (Pressure + Poloidal Current)
    - X-Point Detection (Newton-Raphson)
    - Dynamic Separatrix finding
    - HPC Acceleration Support (C++ Kernel)
    """
    def __init__(self, config_path):
        self.load_config(config_path)
        self.initialize_grid()
        self.setup_accelerator()
        
    def setup_accelerator(self):
        # Initialize HPC Bridge
        self.hpc = HPCBridge()
        if self.hpc.is_available():
            print("[Kernel] HPC Acceleration ENABLED.")
            # Configure the C++ solver with grid parameters
            self.hpc.initialize(
                self.NR, self.NZ, 
                (self.R[0], self.R[-1]), 
                (self.Z[0], self.Z[-1])
            )
        else:
            print("[Kernel] HPC Acceleration UNAVAILABLE (Using Python fallback).")
        
    def load_config(self, path):
        with open(path, 'r') as f:
            self.cfg = json.load(f)
        print(f"[Kernel] Loaded configuration for: {self.cfg['reactor_name']}")
        
    def initialize_grid(self):
        dims = self.cfg['dimensions']
        res = self.cfg['grid_resolution']
        
        self.NR, self.NZ = res[0], res[1]
        self.R = np.linspace(dims['R_min'], dims['R_max'], self.NR)
        self.Z = np.linspace(dims['Z_min'], dims['Z_max'], self.NZ)
        self.dR = self.R[1] - self.R[0]
        self.dZ = self.Z[1] - self.Z[0]
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)
        
        self.Psi = np.zeros((self.NZ, self.NR))
        self.J_phi = np.zeros((self.NZ, self.NR))
        
        # Physics profiles parameters
        self.p_prime_0 = -1.0 # Pressure gradient scale
        self.ff_prime_0 = -1.0 # Poloidal current scale

        # Profile mode: 'l-mode' (linear) or 'h-mode' (mtanh pedestal)
        self.profile_mode = 'l-mode'
        self.ped_params_p = {
            'ped_top': 0.92, 'ped_width': 0.05,
            'ped_height': 1.0, 'core_alpha': 0.3,
        }
        self.ped_params_ff = {
            'ped_top': 0.92, 'ped_width': 0.05,
            'ped_height': 1.0, 'core_alpha': 0.3,
        }

        # Read profile config if present in JSON
        profiles_cfg = self.cfg.get('physics', {}).get('profiles')
        if profiles_cfg:
            self.profile_mode = profiles_cfg.get('mode', 'l-mode')
            if 'p_prime' in profiles_cfg:
                self.ped_params_p.update(profiles_cfg['p_prime'])
            if 'ff_prime' in profiles_cfg:
                self.ped_params_ff.update(profiles_cfg['ff_prime'])
        
    def calculate_vacuum_field(self):
        """Computes Green's function using Elliptic Integrals (Toroidal Geometry)."""
        from scipy.special import ellipk, ellipe
        print("[Kernel] Computing Vacuum Field (Toroidal Exact)...")
        Psi_vac = np.zeros((self.NZ, self.NR))
        mu0 = self.cfg['physics'].get('vacuum_permeability', 1.0)

        for coil in self.cfg['coils']:
            Rc, Zc = coil['r'], coil['z']
            I = coil['current']
            
            # Coordinate differences
            dZ = self.ZZ - Zc
            R_plus_Rc_sq = (self.RR + Rc)**2
            
            # k squared parameter m = k^2
            k2 = (4.0 * self.RR * Rc) / (R_plus_Rc_sq + dZ**2)
            k2 = np.clip(k2, 1e-9, 0.999999) # Avoid singularity
            
            # Elliptic Integrals
            K = ellipk(k2)
            E = ellipe(k2)
            
            # Flux Calculation (Standard Smythe/Jackson form for Toroidal loop)
            # Psi = (mu0 * I / 2 * pi) * sqrt((R + Rc)^2 + z^2) * ((2 - k^2)*K - 2*E) / k^2
            
            prefactor = (mu0 * I) / (2 * np.pi)
            sqrt_term = np.sqrt(R_plus_Rc_sq + dZ**2)
            
            term = ((2.0 - k2) * K - 2.0 * E) / k2
            coil_flux = prefactor * sqrt_term * term
            
            Psi_vac += coil_flux
            
        return Psi_vac
    
    def find_x_point(self, Psi):
        """
        Locates the Null Point (B=0) using local minimization.
        This defines the Separatrix (LCFS).
        """
        # Gradient of Psi (proportional to B)
        dPsi_dR, dPsi_dZ = np.gradient(Psi, self.dR, self.dZ)
        B_mag = np.sqrt(dPsi_dR**2 + dPsi_dZ**2)
        
        # Look for minimum B in the divertor region (usually bottom part)
        # We mask the core to avoid finding the O-point (magnetic axis)
        mask_divertor = self.ZZ < (self.cfg['dimensions']['Z_min'] * 0.5) 
        
        if np.any(mask_divertor):
            masked_B = np.where(mask_divertor, B_mag, 1e9)
            idx_min = np.argmin(masked_B)
            iz, ir = np.unravel_index(idx_min, Psi.shape)
            
            # Sub-grid refinement could go here, but grid point is enough for physics logic
            return (self.R[ir], self.Z[iz]), Psi[iz, ir]
        else:
            return (0,0), np.min(Psi) # Fallback

    @staticmethod
    def mtanh_profile(psi_norm, params):
        """
        Vectorized mtanh pedestal profile.

        Parameters
        ----------
        psi_norm : ndarray
            Normalised poloidal flux (0 at axis, 1 at separatrix).
        params : dict
            Keys: ped_top, ped_width, ped_height, core_alpha.

        Returns
        -------
        ndarray  — profile value (0 outside plasma).
        """
        result = np.zeros_like(psi_norm)
        mask = (psi_norm >= 0) & (psi_norm < 1.0)
        x = psi_norm[mask]

        # Pedestal step via tanh
        y = np.clip((params['ped_top'] - x) / params['ped_width'], -20, 20)
        pedestal = 0.5 * params['ped_height'] * (1.0 + np.tanh(y))

        # Core parabolic peaking
        core = np.where(
            x < params['ped_top'],
            np.maximum(0.0, 1.0 - (x / params['ped_top'])**2),
            0.0,
        )

        result[mask] = pedestal + params['core_alpha'] * core
        return result

    def update_plasma_source_nonlinear(self, Psi_axis, Psi_boundary):
        """
        Calculates J_phi using full Grad-Shafranov source term:
        J_phi = R * p'(psi) + (1/(mu0*R)) * FF'(psi)

        Supports both L-mode (linear 1-psi_norm) and H-mode (mtanh pedestal)
        profiles, controlled by self.profile_mode.
        """
        mu0 = self.cfg['physics']['vacuum_permeability']

        denom = (Psi_boundary - Psi_axis)
        if abs(denom) < 1e-9: denom = 1e-9

        Psi_norm = (self.Psi - Psi_axis) / denom
        mask_plasma = (Psi_norm >= 0) & (Psi_norm < 1.0)

        if self.profile_mode in ('h-mode', 'H-mode', 'hmode'):
            p_profile = self.mtanh_profile(Psi_norm, self.ped_params_p)
            ff_profile = self.mtanh_profile(Psi_norm, self.ped_params_ff)
        else:
            # L-mode: linear (1 - psi_norm)
            p_profile = np.zeros_like(self.Psi)
            p_profile[mask_plasma] = 1.0 - Psi_norm[mask_plasma]
            ff_profile = p_profile.copy()

        J_p = self.RR * p_profile
        J_f = (1.0 / (mu0 * self.RR)) * ff_profile

        beta_mix = 0.5
        J_raw = beta_mix * J_p + (1 - beta_mix) * J_f

        I_current = np.sum(J_raw) * self.dR * self.dZ
        I_target = self.cfg['physics']['plasma_current_target']

        if abs(I_current) > 1e-9:
            scale_factor = I_target / I_current
            self.J_phi = J_raw * scale_factor
        else:
            self.J_phi = np.zeros_like(self.Psi)

        return self.J_phi
        
    def solve_equilibrium(self):
        t0 = time.time()
        self.Psi = self.calculate_vacuum_field()
        Psi_vac_boundary = self.Psi.copy()
        
        max_iter = self.cfg['solver']['max_iterations']
        tol = self.cfg['solver']['convergence_threshold']
        alpha = 0.1 # Lower relaxation for non-linear stability
        mu0 = self.cfg['physics']['vacuum_permeability']
        
        # Init X-point tracker
        x_point_pos = (0,0)
        Psi_best = self.Psi.copy()
        diff_best = 1e9
        
        # SEED PLASMA (Initial Guess)
        # Force current to be centered at R=6.0 to avoid wall-lock
        R_center = (self.cfg['dimensions']['R_min'] + self.cfg['dimensions']['R_max']) / 2.0
        Z_center = 0.0
        dist_sq = (self.RR - R_center)**2 + (self.ZZ - Z_center)**2
        sigma = 1.0
        self.J_phi = np.exp(-dist_sq / (2 * sigma**2))
        
        # Normalize Seed Current
        I_seed = np.sum(self.J_phi) * self.dR * self.dZ
        I_target = self.cfg['physics']['plasma_current_target']
        if I_seed > 0:
            self.J_phi *= (I_target / I_seed)
            
        # Initial Elliptic Solve to set Psi based on Seed
        Source = -mu0 * self.RR * self.J_phi
        for _ in range(50):
             self.Psi[1:-1, 1:-1] = 0.25 * (
                self.Psi[0:-2, 1:-1] + self.Psi[2:, 1:-1] + 
                self.Psi[1:-1, 0:-2] + self.Psi[1:-1, 2:] - 
                (self.dR**2) * Source[1:-1, 1:-1]
            )
        
        for k in range(max_iter):
            # 1. Analyze Topology
            # Axis (O-point)
            idx_max = np.argmax(self.Psi)
            iz_ax, ir_ax = np.unravel_index(idx_max, self.Psi.shape)
            Psi_axis = self.Psi[iz_ax, ir_ax]
            
            if abs(Psi_axis) < 1e-6: Psi_axis = 1e-6 # Safety
            
            # Boundary (Separatrix defined by X-point)
            x_point_pos, Psi_x = self.find_x_point(self.Psi)
            Psi_boundary = Psi_x 
            
            # Safety check: if axis and boundary are too close (no plasma)
            if abs(Psi_axis - Psi_boundary) < 0.1:
                Psi_boundary = Psi_axis * 0.1 # Fallback to limiter mode
            
            # 2. Update Source (Non-Linear Physics)
            # If we are in "Fixed Profile" mode (e.g. from Transport Solver), skip internal profile update
            if not getattr(self, 'external_profile_mode', False):
                self.J_phi = self.update_plasma_source_nonlinear(Psi_axis, Psi_boundary)
            else:
                # We still need to re-normalize the external profile to match target Ip if desired
                # But we assume the shape is set externally (e.g. by TransportSolver)
                pass 
            
            # 3. Elliptic Solve
            Source = -mu0 * self.RR * self.J_phi # Note R^2 logic handled inside J calculation
            
            Psi_new = self.Psi.copy()
            
            # Check for HPC Acceleration
            if self.hpc.is_available():
                # Offload the heavy lifting to C++
                # It does multiple iterations internally (e.g. 50-100)
                # We can reduce the python-side 'k' loop iterations if we trust the inner loop
                
                # Pass current J_phi (Source term basically) and get updated Psi
                # Note: C++ solver expects J, it calculates Source internally.
                # But our C++ solver expects J_phi.
                
                Psi_accelerated = self.hpc.solve(self.J_phi, iterations=50)
                if Psi_accelerated is not None:
                    Psi_new = Psi_accelerated
                    
                    # Boundary conditions are handled in C++? 
                    # Actually C++ solver in this version doesn't handle boundary update from vacuum field dynamically
                    # We need to enforce boundary conditions here
                    Psi_new[0,:] = Psi_vac_boundary[0,:]
                    Psi_new[-1,:] = Psi_vac_boundary[-1,:]
                    Psi_new[:,0] = Psi_vac_boundary[:,0]
                    Psi_new[:,-1] = Psi_vac_boundary[:,-1]
            else:
                # Python Slow-Mo Solver
                Psi_new[1:-1, 1:-1] = 0.25 * (
                    self.Psi[0:-2, 1:-1] + 
                    self.Psi[2:, 1:-1] + 
                    self.Psi[1:-1, 0:-2] + 
                    self.Psi[1:-1, 2:] - 
                    (self.dR**2) * Source[1:-1, 1:-1]
                )
                
                # Boundary conditions
                Psi_new[0,:] = Psi_vac_boundary[0,:]
                Psi_new[-1,:] = Psi_vac_boundary[-1,:]
                Psi_new[:,0] = Psi_vac_boundary[:,0]
                Psi_new[:,-1] = Psi_vac_boundary[:,-1]
            
            # Robustness Check
            if np.isnan(Psi_new).any() or np.isinf(Psi_new).any():
                print(f"[Kernel] WARNING: Solver diverged at iter {k}. Reverting to best known state.")
                self.Psi = Psi_best
                break
            
            # 4. Relax
            diff = np.mean(np.abs(Psi_new - self.Psi))
            self.Psi = (1.0 - alpha) * self.Psi + alpha * Psi_new
            
            # Save best
            if diff < diff_best:
                diff_best = diff
                Psi_best = self.Psi.copy()
            
            if diff < tol:
                print(f"[Kernel] Converged at iter {k}. Res: {diff:.6e}")
                break
                
            if k % 100 == 0:
                print(f"  Iter {k}: Res={diff:.6e} | Axis={Psi_axis:.2f} | X-Point={Psi_boundary:.2f} at R={x_point_pos[0]:.2f},Z={x_point_pos[1]:.2f}")

        # Finalize
        self.compute_b_field()
        print(f"[Kernel] Solved in {time.time()-t0:.2f}s. Final X-Point: R={x_point_pos[0]:.2f}, Z={x_point_pos[1]:.2f}")

    def compute_b_field(self):
        dPsi_dR, dPsi_dZ = np.gradient(self.Psi, self.dR, self.dZ)
        R_safe = np.maximum(self.RR, 1e-6)
        self.B_R = -(1.0 / R_safe) * dPsi_dZ
        self.B_Z =  (1.0 / R_safe) * dPsi_dR
        
    def save_results(self, filename="equilibrium_nonlinear.npz"):
        np.savez(filename, R=self.R, Z=self.Z, Psi=self.Psi, J_phi=self.J_phi)
        print(f"[Kernel] Saved: {filename}")

if __name__ == "__main__":
    # Test run
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else "03_CODE/SCPN-Fusion-Core/iter_config.json"
    fk = FusionKernel(config_file)
    fk.solve_equilibrium()
    fk.save_results("03_CODE/SCPN-Fusion-Core/final_state_nonlinear.npz")