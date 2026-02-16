# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Integrated Transport Solver
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import copy

try:
    from scpn_fusion.core._rust_compat import FusionKernel
except ImportError:
    from scpn_fusion.core.fusion_kernel import FusionKernel

def chang_hinton_chi_profile(rho, T_i, n_e_19, q, R0, a, B0, A_ion=2.0, Z_eff=1.5):
    """
    Chang-Hinton (1982) neoclassical ion thermal diffusivity profile [m²/s].

    Parameters
    ----------
    rho : array  — normalised radius [0,1]
    T_i : array  — ion temperature [keV]
    n_e_19 : array  — electron density [10^19 m^-3]
    q : array  — safety factor profile
    R0 : float  — major radius [m]
    a : float  — minor radius [m]
    B0 : float  — toroidal field [T]
    A_ion : float  — ion mass number (default 2 = deuterium)
    Z_eff : float  — effective charge

    Returns
    -------
    chi_nc : array  — neoclassical chi_i [m²/s]
    """
    e_charge = 1.602176634e-19
    eps0 = 8.854187812e-12
    m_p = 1.672621924e-27
    m_e = 9.10938370e-31
    m_i = A_ion * m_p

    chi_nc = np.zeros_like(rho)
    for i in range(len(rho)):
        r = rho[i]
        if r <= 0.0 or T_i[i] <= 0.0 or n_e_19[i] <= 0.0 or q[i] <= 0.0:
            chi_nc[i] = 0.01
            continue

        epsilon = r * a / R0
        if epsilon < 1e-6:
            chi_nc[i] = 0.01
            continue

        T_J = T_i[i] * 1.602176634e-16  # keV -> J
        v_ti = np.sqrt(2.0 * T_J / m_i)
        rho_i = m_i * v_ti / (e_charge * B0)

        # ion-ion collision frequency
        n_e = n_e_19[i] * 1e19
        ln_lambda = 17.0
        nu_ii = (n_e * Z_eff**2 * e_charge**4 * ln_lambda
                 / (12.0 * np.pi**1.5 * eps0**2 * m_i**0.5 * T_J**1.5))

        eps32 = epsilon**1.5
        nu_star = nu_ii * q[i] * R0 / (eps32 * v_ti)

        alpha_sh = epsilon
        chi_val = (0.66 * (1.0 + 1.54 * alpha_sh) * q[i]**2
                   * rho_i**2 * nu_ii
                   / (eps32 * (1.0 + 0.74 * nu_star**(2.0/3.0))))

        chi_nc[i] = max(chi_val, 0.01) if np.isfinite(chi_val) else 0.01

    return chi_nc


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

        # Neoclassical transport configuration (None = constant chi_base=0.5)
        self.neoclassical_params = None

    def set_neoclassical(self, R0, a, B0, A_ion=2.0, Z_eff=1.5, q0=1.0, q_edge=3.0):
        """Configure Chang-Hinton neoclassical transport model.

        When set, update_transport_model uses the Chang-Hinton formula instead
        of the constant chi_base = 0.5.
        """
        q_profile = q0 + (q_edge - q0) * self.rho**2
        self.neoclassical_params = {
            'R0': R0, 'a': a, 'B0': B0,
            'A_ion': A_ion, 'Z_eff': Z_eff,
            'q_profile': q_profile,
        }

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
        
        # Base Level — use Chang-Hinton if configured, else constant
        if self.neoclassical_params is not None:
            p = self.neoclassical_params
            chi_base = chang_hinton_chi_profile(
                self.rho, self.Ti, self.ne,
                p['q_profile'], p['R0'], p['a'], p['B0'],
                p['A_ion'], p['Z_eff']
            )
        else:
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

    # ── Confinement time ───────────────────────────────────────────────

    def compute_confinement_time(self, P_loss_MW: float) -> float:
        """Compute the energy confinement time from stored energy.

        τ_E = W_stored / P_loss, where W_stored = ∫ 3/2 n (Ti+Te) dV
        and the volume element is estimated from the 1D radial profiles
        using cylindrical approximation.

        Parameters
        ----------
        P_loss_MW : float
            Total loss power [MW].  Must be > 0.

        Returns
        -------
        float
            Energy confinement time [s].
        """
        if P_loss_MW <= 0:
            return float("inf")

        # Stored energy: W = ∫ 3/2 n_e (T_i + T_e) dV
        # In 1D with cylindrical approx: dV ≈ 2πR₀ · 2π · r · a² · dρ
        # Units: n_e is in 10^19 m^-3, T in keV → W in MJ
        e_keV = 1.602176634e-16  # J per keV
        dims = self.cfg["dimensions"]
        R0 = (dims["R_min"] + dims["R_max"]) / 2.0
        a = (dims["R_max"] - dims["R_min"]) / 2.0

        # Volume element per rho bin: dV = 2π R₀ · 2π ρ a² dρ
        rho_mid = self.rho
        dV = 2.0 * np.pi * R0 * 2.0 * np.pi * rho_mid * a**2 * self.drho

        # Energy density: 3/2 * n_e * (Ti + Te) [10^19 m^-3 * keV]
        energy_density = 1.5 * (self.ne * 1e19) * (self.Ti + self.Te) * e_keV
        W_stored_J = float(np.sum(energy_density * dV))
        W_stored_MW = W_stored_J / 1e6  # J → MJ → MW·s

        return W_stored_MW / P_loss_MW

    def run_to_steady_state(
        self,
        P_aux: float,
        n_steps: int = 500,
        dt: float = 0.01,
    ) -> dict:
        """Run transport evolution until approximate steady state.

        Parameters
        ----------
        P_aux : float
            Auxiliary heating power [MW].
        n_steps : int
            Number of evolution steps.
        dt : float
            Time step [s].

        Returns
        -------
        dict
            ``{"T_avg": float, "T_core": float, "tau_e": float,
            "n_steps": int, "Ti_profile": ndarray,
            "ne_profile": ndarray}``
        """
        for _ in range(n_steps):
            self.update_transport_model(P_aux)
            T_avg, T_core = self.evolve_profiles(dt, P_aux)

        tau_e = self.compute_confinement_time(P_aux)

        return {
            "T_avg": float(T_avg),
            "T_core": float(T_core),
            "tau_e": tau_e,
            "n_steps": n_steps,
            "Ti_profile": self.Ti.copy(),
            "ne_profile": self.ne.copy(),
        }
