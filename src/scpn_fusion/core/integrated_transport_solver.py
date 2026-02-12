import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import copy

# Add src to path for direct execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from scpn_fusion.core._rust_compat import FusionKernel
except ImportError:
    from scpn_fusion.core.fusion_kernel import FusionKernel

class TransportSolver(FusionKernel):
    """
    1.5D Integrated Transport Code.
    Solves Heat and Particle diffusion equations on flux surfaces,
    coupled self-consistently with the 2D Grad-Shafranov equilibrium.
    """
    def __init__(self, config_path):
        super().__init__(config_path)
        self.external_profile_mode = True # Tell Kernel to respect our calculated profiles
        self.nr = 50 # Radial grid points (normalized radius rho)
        self.rho = np.linspace(0, 1, self.nr)
        self.drho = 1.0 / (self.nr - 1)
        
        # PROFILES (Evolving state variables)
        # Te = Electron Temp (keV), Ti = Ion Temp (keV), ne = Density (10^19 m-3)
        self.Te = 1.0 * (1 - self.rho**2) # Initial guess
        self.Ti = 1.0 * (1 - self.rho**2)
        self.ne = 5.0 * (1 - self.rho**2)**0.5
        
        # Transport Coefficients (Anomalous Transport Models)
        self.chi_e = np.ones(self.nr) # Electron diffusivity
        self.chi_i = np.ones(self.nr) # Ion diffusivity
        self.D_n = np.ones(self.nr)   # Particle diffusivity
        
        # Impurity Profile (Tungsten density)
        self.n_impurity = np.zeros(self.nr) 

    def inject_impurities(self, flux_from_wall_per_sec, dt):
        """
        Models impurity influx from PWI erosion.
        Simple diffusion model: Source at edge, diffuses inward.
        """
        # Source at edge (last grid point)
        # Flux is total particles. Volume of edge shell approx 20 m3.
        # Delta_n = Flux * dt / Vol_edge
        # Scaling factor adjusted for simulation stability
        d_n_edge = (flux_from_wall_per_sec * dt) / 20.0 * 1e-18 
        
        # Add to edge
        self.n_impurity[-1] += d_n_edge
        
        # Diffuse inward (Explicit step)
        D_imp = 1.0 # m2/s
        new_imp = self.n_impurity.copy()
        
        grad = np.gradient(self.n_impurity, self.drho)
        flux = -D_imp * grad
        div = np.gradient(flux, self.drho) / (self.rho + 1e-6)
        
        new_imp += (-div) * dt
        
        # Boundary
        new_imp[0] = new_imp[1] # Axis symmetry
        
        self.n_impurity = np.maximum(0, new_imp)

    def update_transport_model(self, P_aux):
        """
        Bohm / Gyro-Bohm Transport Model.
        """
        # 1. Critical Gradient Model
        grad_T = np.gradient(self.Ti, self.drho)
        threshold = 2.0
        
        # Base Level
        chi_base = 0.5
        
        # Turbulent Level
        chi_turb = 5.0 * np.maximum(0, -grad_T - threshold)
        
        # H-Mode Barrier Logic
        is_H_mode = P_aux > 30.0 # MW
        
        if is_H_mode:
            # Suppress turbulence at edge (rho > 0.9)
            edge_mask = self.rho > 0.9
            chi_turb[edge_mask] *= 0.1 # Transport Barrier
            
        self.chi_e = chi_base + chi_turb
        self.chi_i = chi_base + chi_turb
        self.D_n = 0.1 * self.chi_e

    def evolve_profiles(self, dt, P_aux):
        """
        Solves dW/dt = Diffusion + Source - Sink
        """
        # Sources
        # Gaussian heating centered at rho=0
        heating_profile = np.exp(-self.rho**2 / 0.1)
        S_heat = (P_aux / np.sum(heating_profile)) * heating_profile
        
        # SINKS: Radiation Cooling (Bremsstrahlung + Line)
        # P_rad = n_e * n_imp * L_z(T)
        cooling_factor = 5.0 
        S_rad = cooling_factor * self.ne * self.n_impurity * np.sqrt(self.Te + 0.1)
        
        # Explicit Step (Simplified for stability in this demo)
        new_Ti = self.Ti.copy()
        
        # Fluxes
        grad_Ti = np.gradient(self.Ti, self.drho)
        flux_Ti = -self.chi_i * grad_Ti
        
        # Divergence (Cylindrical approx)
        r_flux = self.rho * flux_Ti
        div_flux = np.gradient(r_flux, self.drho) / (self.rho + 1e-6)
        
        # Update
        dT_dt = -div_flux + S_heat - S_rad
        new_Ti += dT_dt * dt
        
        # Boundary Conditions
        new_Ti[0] = new_Ti[1] # Zero gradient at core
        new_Ti[-1] = 0.1      # Cold edge (Separatrix temp)
        
        self.Ti = np.maximum(0.01, new_Ti) # Prevent negative temp
        self.Te = self.Ti # Assume equilibrated
        
        return np.mean(self.Ti), self.Ti[0]

    def map_profiles_to_2d(self):
        """
        Projects the 1D radial profiles back onto the 2D Grad-Shafranov grid.
        """
        # 1. Get Flux Topology
        idx_max = np.argmax(self.Psi)
        iz_ax, ir_ax = np.unravel_index(idx_max, self.Psi.shape)
        Psi_axis = self.Psi[iz_ax, ir_ax]
        xp, psi_x = self.find_x_point(self.Psi)
        Psi_edge = psi_x
        if abs(Psi_edge - Psi_axis) < 1.0: Psi_edge = np.min(self.Psi)
        
        # 2. Calculate Rho for every 2D point
        Psi_norm = (self.Psi - Psi_axis) / (Psi_edge - Psi_axis)
        Psi_norm = np.clip(Psi_norm, 0, 1)
        # Rho is approx sqrt(Psi_norm)
        Rho_2D = np.sqrt(Psi_norm)
        
        # 3. Interpolate 1D profiles to 2D
        self.Pressure_2D = np.interp(Rho_2D.flatten(), self.rho, self.ne * (self.Ti + self.Te))
        self.Pressure_2D = self.Pressure_2D.reshape(self.Psi.shape)
        
        # 4. Update J_phi based on new Pressure
        self.J_phi = self.Pressure_2D * self.RR 
        
        # Normalize to target current
        I_curr = np.sum(self.J_phi) * self.dR * self.dZ
        I_target = self.cfg['physics']['plasma_current_target']
        if I_curr > 1e-9:
            self.J_phi *= (I_target / I_curr)
