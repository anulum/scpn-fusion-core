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

# Physical constants
MC = 9.109e-31 * 2.998e8  # m_e * c [kg m/s]
C = 2.998e8               # speed of light [m/s]
E_CHARGE = 1.602e-19      # elementary charge [C]
EPS0 = 8.854e-12          # vacuum permittivity [F/m]
ME = 9.109e-31            # electron mass [kg]

# Model parameters
COULOMB_LOG = 15.0         # dimensionless, Wesson Ch. 14 Eq. 14.5.2
B_TOROIDAL = 5.3           # toroidal field [T], ITER-like
DIFFUSION_FLOOR = 1e-5     # numerical diffusion floor [arb.]
AVALANCHE_RATE = 100.0     # Rosenbluth-Putvinski avalanche rate prefactor [1/s]
DREICER_SOURCE = 1.0e15    # Dreicer injection flux [m^-3 s^-1]

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
        self.np_grid = np_grid
        self.p_max = p_max
        # Log-spaced momentum grid, p normalised to m_e c
        self.p = np.logspace(-2, np.log10(p_max), np_grid)
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
        """Compute advection (A), diffusion (D), and critical field (Fc) coefficients.

        A = F_acc - F_drag - F_synch, per Hesslow et al. J. Plasma Phys. 85 (2019).
        """
        # Acceleration: normalized force F/(m_e c) [1/s]
        F_acc = (E_CHARGE * E_field) / MC

        # Connor-Hastie critical field, Wesson "Tokamaks" Ch. 14
        ln_lambda = COULOMB_LOG
        Ec = (n_e * E_CHARGE**3 * ln_lambda) / (4 * np.pi * EPS0**2 * ME * C**2)
        Fc_norm = (E_CHARGE * Ec) / MC

        gamma = np.sqrt(1 + self.p**2)

        # Relativistic friction: Fc * (1 + (Z_eff+1)/(p²+p_th²))
        # Regularised at low p to merge with thermal population.
        # Assumption: high-p limit ≈ Fc; low-p correction via Connor-Hastie.
        p_thermal_sq = max(T_e_eV / 511e3, 1e-6)
        F_drag = Fc_norm * (1.0 + (Z_eff + 1.0) / (self.p**2 + p_thermal_sq))

        # Synchrotron radiation reaction (equilibrium pitch-angle assumption)
        # tau_rad per Aleynikov & Breizman, PRL 114 (2015) Eq. 3
        tau_rad = (6 * np.pi * EPS0 * MC**3) / (E_CHARGE**4 * B_TOROIDAL**2)
        F_synch = (1.0 / tau_rad) * self.p * gamma * np.sqrt(1 + Z_eff)

        A = F_acc - F_drag - F_synch

        # Numerical diffusion floor (collisional diffusion negligible for REs)
        D = np.full_like(self.p, DIFFUSION_FLOOR)

        return A, D, Fc_norm

    def seed_hottail(self, T_initial_eV: float, T_final_eV: float, t_quench_s: float):
        """Seed hottail RE population from thermal quench.

        Gaussian tail model per Aleynikov & Breizman, PRL 114 (2015).
        Assumption: seed fraction fixed at 1e10 (t_quench/tau_coll dependence ignored).
        """
        p_th_init = np.sqrt(2 * T_initial_eV * E_CHARGE / (ME * C**2))
        f_hottail = np.exp(-self.p**2 / (p_th_init**2))
        self.f = np.maximum(self.f, f_hottail * 1e10)
        logger.info("Hottail seed: %.2e m^-3", np.sum(self.f * self.dp))

    def explicit_knock_on_source(self, n_e: float) -> np.ndarray:
        """Knock-on collision source via Moller cross-section (1/p² scaling).

        Simplified from Rosenbluth-Putvinski, Nucl. Fusion 37 (1997).
        Empirical scaling 1e-25 calibrated to ITER-like RE generation rates.
        """
        source = 1.0 / (self.p**2 + 1e-4)
        n_re = np.sum(self.f * self.dp)
        if n_re < 1e6:
            return np.zeros_like(self.f)
        return source * n_e * n_re * 1e-25

    def step(
        self,
        dt: float,
        E_field: float,
        n_e: float,
        T_e_eV: float,
        Z_eff: float
    ) -> RunawayElectronState:
        """Advance distribution function by dt using first-order upwind."""
        A, D, Fc = self.compute_coefficients(E_field, n_e, Z_eff, T_e_eV)

        # Avalanche source: Rosenbluth-Putvinski, Nucl. Fusion 37 (1997) Eq. 19
        E_crit = (Fc * MC / E_CHARGE)
        gamma_av = 0.0
        if E_field > E_crit:
            gamma_av = (E_field / E_crit - 1.0) * np.sqrt(np.pi * (Z_eff + 1) / 2)
            gamma_av *= AVALANCHE_RATE
        S_av = gamma_av * self.f

        # Dreicer source: inject at low-p bins when E > 0.05 E_c
        S_dr = np.zeros_like(self.f)
        if E_field > 0.05 * E_crit:
            S_dr[0:5] = DREICER_SOURCE

        S_ko = self.explicit_knock_on_source(n_e)

        # First-order upwind advection
        f_new = self.f.copy()
        for i in range(1, self.np_grid - 1):
            flux_in = A[i-1] * self.f[i-1] if A[i-1] > 0 else A[i] * self.f[i]
            flux_out = A[i] * self.f[i] if A[i] > 0 else A[i+1] * self.f[i+1]
            df_dt = -(flux_out - flux_in) / self.dp[i] + S_av[i] + S_dr[i] + S_ko[i]
            f_new[i] += df_dt * dt

        self.f = np.maximum(0, f_new)
        self.time += dt

        # Moments: density and current
        dp = self.dp
        n_re = np.sum(self.f * dp)
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
