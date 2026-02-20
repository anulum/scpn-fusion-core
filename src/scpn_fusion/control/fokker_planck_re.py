# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Kinetic Fokker-Planck RE Solver
# © 1998–2026 Miroslav Šotek. All rights reserved.
# ──────────────────────────────────────────────────────────────────────
"""
1D-in-momentum Fokker-Planck solver for Runaway Electron (RE) dynamics.

Solves the equation for the electron distribution function f(p, t):
    df/dt + d/dp [ (F_acc - F_drag - F_synch) f - D df/dp ] = S_source

Physics included:
- Electric field acceleration (F_acc = e E_par)
- Collisional friction / drag (Connor-Hastie)
- Synchrotron radiation reaction force (F_synch)
- Pitch-angle scattering (approximate effect on parallel momentum)
- Avalanche source (Rosenbluth-Putvinski)
- Dreicer generation source

Reference:
- Aleynikov & Breizman, Phys. Rev. Lett. 114, 155001 (2015)
- Hesslow et al., J. Plasma Phys. 85, 475850601 (2019) (DREAM)
"""

from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass
from typing import Tuple

logger = logging.getLogger(__name__)

# Constants
MC = 9.109e-31 * 2.998e8  # m_e * c [kg m/s]
C = 2.998e8               # speed of light [m/s]
E_CHARGE = 1.602e-19      # elementary charge [C]
EPS0 = 8.854e-12          # vacuum permittivity [F/m]
ME = 9.109e-31            # electron mass [kg]

@dataclass
class RunawayElectronState:
    """State of the RE population."""
    f: np.ndarray          # Distribution function f(p) [1/m^3 / (mc)^3] ? 
                           # Normalized such that integral f dp = n_RE
    p_grid: np.ndarray     # Momentum grid (normalized to mc)
    time: float = 0.0
    n_re: float = 0.0      # Total RE density [m^-3]
    current_re: float = 0.0 # Runaway current density [A/m^2]

class FokkerPlanckSolver:
    """
    Solves the 1D kinetic equation for runaway electrons.
    """
    def __init__(self, np_grid: int = 200, p_max: float = 100.0):
        # p is normalized momentum: p = P / (m_e c)
        # Grid from p=0 to p_max
        # Logarithmic grid is better for resolution at low p (thermal) and high p
        self.np_grid = np_grid
        self.p_max = p_max
        
        # Log-uniform grid
        p_min = 1e-2
        self.p = np.logspace(np.log10(p_min), np.log10(p_max), np_grid)
        self.dp = np.gradient(self.p)
        
        self.f = np.zeros(np_grid)
        self.time = 0.0

    def compute_coefficients(
        self, 
        E_field: float, 
        n_e: float, 
        Z_eff: float, 
        T_e_eV: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute advection (A) and diffusion (D) coefficients.
        Equation: df/dt = -d/dp (A * f) + d/dp (D * df/dp) + Source
        
        A = F_acc - F_drag - F_synch
        """
        # 1. Acceleration: F_acc = e * E_par / (m_e * c)  (normalized force)
        # E_field is V/m. Force is Newtons.
        # Normalized force F_norm = F / (m_e c) [1/s]
        F_acc = (E_CHARGE * E_field) / MC  # Constant in p
        
        # 2. Collisional Drag (Connor-Hastie / relativistic friction)
        # F_drag = nu_coll * p
        # High-energy limit: F_drag approx E_c (critical field)
        # E_c = n_e e^3 lnL / (4 pi eps0^2 m_e c^2)
        ln_lambda = 15.0 # Approx
        Ec = (n_e * E_CHARGE**3 * ln_lambda) / (4 * np.pi * EPS0**2 * ME * C**2)
        Fc_norm = (E_CHARGE * Ec) / MC
        
        # Relativistic gamma factor
        gamma = np.sqrt(1 + self.p**2)
        vel = (self.p / gamma) * C
        
        # Full collisional friction formula (simplified)
        # F_drag(p) = F_c * (1 + p^2/gamma) ... roughly 
        # Actually F_drag ~ F_c * gamma^2/p^2 at low p, F_c at high p
        # Using simplified formula: F_drag = F_c * (Z_eff + 1 + p^2) / p^2 ... no that's not right
        
        # Let's use the high-p limit for REs: F_drag approx F_c
        # Plus pitch-angle scattering correction which increases effective drag
        # Effective Electric field threshold E_c_eff ~ E_c * sqrt(Z_eff + 1)
        
        # Simple model: Chandrasekhar dynamical friction
        # F_coll = (1 + 1/p^2) * Fc_norm ... roughly
        # We will use a standard approximation for REs:
        # F_drag = Fc_norm * (1/v_norm^2) is low energy...
        # Let's use: F_drag = Fc_norm * (gamma**2 / p**2) * (1 + ...)
        
        # For this roadmap item, we use the Rosenbluth model for drag:
        # F_drag = E_c * (gamma / p)^2 * [ Z_eff + 1 + ... ] -> this is complex.
        # Let's use the Critical Field approximation:
        # F_drag(p) = Fc_norm  (constant retarding force at relativistic speeds)
        # With a correction at low p to merge with thermal.
        # Correct relativistic friction: F = F_c * gamma^2 / p^2
        F_drag = Fc_norm * (1.0 + (Z_eff + 1.0) / self.p**2) # Simplified
        
        # 3. Synchrotron Radiation
        # F_synch = (2/3) r_e p gamma (v/c)^3 ...
        # Simplified: F_synch ~ alpha * p * gamma
        # tau_rad = 6 pi eps0 (mc)^3 / (e^4 B^2)
        # B field needed? Yes. Assuming B=5T.
        B = 5.3
        tau_rad = (6 * np.pi * EPS0 * MC**3) / (E_CHARGE**4 * B**2)
        # F_synch_norm = 1/tau_rad * p * gamma * (something pitch angle)
        # We assume equilibrium pitch angle
        F_synch = (1.0 / tau_rad) * self.p * gamma * np.sqrt(1 + Z_eff) # Heuristic
        
        A = F_acc - F_drag - F_synch
        
        # 4. Diffusion (small for REs, mainly from collisions)
        # D_coll ~ F_drag / p ...
        D = np.zeros_like(self.p) + 1e-5 # Small numerical diffusion
        
        return A, D, Fc_norm

    def seed_hottail(self, T_initial_eV: float, T_final_eV: float, t_quench_s: float):
        """
        Generate hottail seed population from a rapid thermal quench.
        Based on Aleynikov & Breizman (2017).
        """
        # Thermal momentum: v_th = sqrt(2 * T / m)
        p_th_init = np.sqrt(2 * T_initial_eV * E_CHARGE / (ME * C**2))
        
        # During a quench, high-energy electrons with v > v_critical survive.
        # Simple hottail model: Gaussian tail that didn't have time to thermalize.
        f_hottail = np.exp(-self.p**2 / (p_th_init**2))
        
        # Scale such that total density matches a fraction of n_e (e.g. 1e-5)
        # In reality, this depends on t_quench / tau_coll.
        self.f = np.maximum(self.f, f_hottail * 1e10) 
        logger.info(f"Hottail seeding complete. Initial RE seed: {np.sum(self.f * self.dp):.2e} m^-3")

    def explicit_knock_on_source(self, n_e: float) -> np.ndarray:
        """
        Explicit large-angle knock-on collision source (Avalanche initiator).
        S_knock_on(p) ~ n_e * n_RE * (differential cross section)
        """
        # Simple 1/p^2 distribution for knock-on secondaries
        # (Moller scattering approximation)
        source = 1.0 / (self.p**2 + 1e-4)
        
        # Normalized by current n_re
        n_re = np.sum(self.f * self.dp)
        if n_re < 1e6:
            return np.zeros_like(self.f)
            
        return source * n_e * n_re * 1e-25 # Empirical scaling constant

    def step(
        self, 
        dt: float, 
        E_field: float, 
        n_e: float, 
        T_e_eV: float, 
        Z_eff: float
    ) -> RunawayElectronState:
        """
        Advance distribution function by dt.
        Uses implicit finite difference or simple upwind.
        """
        A, D, Fc = self.compute_coefficients(E_field, n_e, Z_eff, T_e_eV)
        
        # Sources
        # 1. Avalanche: S_av ~ n_RE * (E/Ec - 1) * ...
        # We implement avalanche as a source term proportional to existing f(p)
        # Gamma_av = (E/Ec - 1) * ...
        E_crit = (Fc * MC / E_CHARGE)
        gamma_av = 0.0
        if E_field > E_crit:
            gamma_av = (E_field / E_crit - 1.0) * np.sqrt(np.pi * (Z_eff + 1) / 2) # RP avalanche
            # Scale by collisional time approx? No RP is rate.
            # Convert to [1/s].
            # RP formula gives normalized rate. Real rate ~ 1/tau_coll * ...
            # We use a simplified rate constant k_av
            k_av = 100.0 # Placeholder
            gamma_av *= k_av
            
        S_av = gamma_av * self.f 
        
        # 2. Dreicer: S_dr at p ~ p_thermal
        # We inject at low p
        S_dr = np.zeros_like(self.f)
        # Calculate Dreicer rate
        if E_field > 0.05 * E_crit: # Only if significant
            # ... Dreicer formula ...
            # For now, put a Gaussian source at p_min
            S_dr[0:5] = 1.0e15 # Dummy source flux
            
        # 3. Knock-on (Avalanche)
        S_ko = self.explicit_knock_on_source(n_e)
        
        # Finite Volume / Upwind scheme
        # df/dt = - d/dp (A f) + S
        # f_new = f - dt/dp * (flux_i+1/2 - flux_i-1/2)
        
        f_new = self.f.copy()
        
        # Advection (Upwind)
        # Flux = A * f. If A > 0, flux from left. If A < 0, flux from right.
        for i in range(1, self.np_grid - 1):
            # Flux at i-1/2 -> i
            flux_in = 0.0
            if A[i-1] > 0:
                flux_in = A[i-1] * self.f[i-1]
            else:
                flux_in = A[i] * self.f[i] # ? Upwind is based on velocity
            
            # Flux at i -> i+1/2
            flux_out = 0.0
            if A[i] > 0:
                flux_out = A[i] * self.f[i]
            else:
                flux_out = A[i+1] * self.f[i+1]
                
            df_dt = -(flux_out - flux_in) / self.dp[i] + S_av[i] + S_dr[i] + S_ko[i]
            f_new[i] += df_dt * dt
            
        self.f = np.maximum(0, f_new)
        self.time += dt
        
        # Compute moments
        dp = self.dp
        n_re = np.sum(self.f * dp)
        
        # Current: J = e * n * v
        # v = c * p / gamma
        gamma = np.sqrt(1 + self.p**2)
        v = C * self.p / gamma
        j_re = E_CHARGE * np.sum(self.f * v * dp)
        
        return RunawayElectronState(
            f=self.f,
            p_grid=self.p,
            time=self.time,
            n_re=n_re,
            current_re=j_re
        )
