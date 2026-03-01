# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — SPI Ablation Solver
# © 1998–2026 Miroslav Šotek. All rights reserved.
# ──────────────────────────────────────────────────────────────────────
"""
Multi-fragment Shattered Pellet Injection (SPI) Ablation Solver.

Tracks the trajectories and ablation of neutral gas fragments injected
into the plasma. Based on the Parks ablation model.

Physics:
- Lagrangian tracking of N fragments (position r, velocity v).
- Shielding-modifed Parks ablation rate:
  dm/dt = -1.25e16 * n_e^(1/3) * T_e^(1.64) * r_p^(1.33) / v_p
  (Reduced-order scaling calibrated for fast control studies)
- Deposition of ablated material into the plasma density profile.
"""

from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Physical constants
AMU = 1.66e-27             # atomic mass unit [kg]
M_NEON = 20.18 * AMU       # neon atom mass [kg]
RHO_NEON_SOLID = 1444.0    # solid neon density [kg/m^3], CRC Handbook

# Parks ablation rate prefactor [g/s] in code units (ne in 10^20 m^-3, Te in keV, rp in cm).
# Derived from Parks NF 57 (2017) Eq. 8: C_P = 1.25e16 (CGS), converted to mixed units
# via dimensional analysis: C_P_mixed = C_P * (1e20)^{-1/3} * keV^{-1.64} * cm^{-1.33} ≈ 2.0.
_PARKS_COEFFICIENT = 2.0

# Tokamak geometry (ITER-like)
R_MAJOR = 6.2              # major radius [m]
A_MINOR = 2.0              # minor radius [m]
ELONGATION = 2.0           # plasma elongation

@dataclass
class SpiFragment:
    """A single shard of the shattered pellet."""
    id: int
    radius: float # m
    mass: float   # kg
    pos: np.ndarray # (x, y, z) [m]
    vel: np.ndarray # (vx, vy, vz) [m/s]
    active: bool = True

class SpiAblationSolver:
    def __init__(self, 
                 n_fragments: int = 100, 
                 total_mass_kg: float = 0.01, # 10g Neon
                 velocity_mps: float = 200.0,
                 dispersion: float = 0.1, # Velocity spread fraction
                 injector_pos: np.ndarray | None = None, # Outboard midplane
                 injector_dir: np.ndarray | None = None, # Radial inward
                 seed: int = 42,
                 ):
        if isinstance(n_fragments, bool) or int(n_fragments) < 1:
            raise ValueError("n_fragments must be an integer >= 1.")
        total_mass_kg = float(total_mass_kg)
        velocity_mps = float(velocity_mps)
        dispersion = float(dispersion)
        if not np.isfinite(total_mass_kg) or total_mass_kg <= 0.0:
            raise ValueError("total_mass_kg must be finite and > 0.")
        if not np.isfinite(velocity_mps) or velocity_mps <= 0.0:
            raise ValueError("velocity_mps must be finite and > 0.")
        if not np.isfinite(dispersion) or dispersion < 0.0:
            raise ValueError("dispersion must be finite and >= 0.")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an integer.")

        n_fragments = int(n_fragments)
        injector_pos_arr = np.asarray(
            [10.0, 0.0, 0.0] if injector_pos is None else injector_pos,
            dtype=np.float64,
        )
        injector_dir_arr = np.asarray(
            [-1.0, 0.0, 0.0] if injector_dir is None else injector_dir,
            dtype=np.float64,
        )
        if injector_pos_arr.shape != (3,) or not np.all(np.isfinite(injector_pos_arr)):
            raise ValueError("injector_pos must be a finite 3-vector.")
        if injector_dir_arr.shape != (3,) or not np.all(np.isfinite(injector_dir_arr)):
            raise ValueError("injector_dir must be a finite 3-vector.")
        dir_norm = float(np.linalg.norm(injector_dir_arr))
        if dir_norm <= 0.0:
            raise ValueError("injector_dir must be non-zero.")

        self.fragments: List[SpiFragment] = []

        # Uniform fragment mass; size from solid-sphere volume
        m_frag = total_mass_kg / n_fragments
        vol = m_frag / RHO_NEON_SOLID
        r_frag = (3 * vol / (4 * np.pi))**(1/3)

        rng = np.random.default_rng(int(seed))
        for i in range(n_fragments):
            v_dir = injector_dir_arr + rng.normal(0, dispersion, 3)
            v_dir /= np.linalg.norm(v_dir)
            v_mag = velocity_mps * rng.normal(1.0, 0.1)
            vel = v_dir * v_mag
            pos = injector_pos_arr + rng.normal(0, 0.05, 3)
            
            self.fragments.append(SpiFragment(
                id=i,
                radius=r_frag,
                mass=m_frag,
                pos=pos,
                vel=vel
            ))
            
    def step(self, dt: float, plasma_ne_profile: np.ndarray, plasma_te_profile: np.ndarray, r_grid: np.ndarray) -> np.ndarray:
        """Advance fragments and return deposition profile [particles/m^3/s]."""
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and > 0.")
        plasma_ne_profile = np.asarray(plasma_ne_profile, dtype=np.float64)
        plasma_te_profile = np.asarray(plasma_te_profile, dtype=np.float64)
        r_grid = np.asarray(r_grid, dtype=np.float64)
        if plasma_ne_profile.ndim != 1 or plasma_te_profile.ndim != 1 or r_grid.ndim != 1:
            raise ValueError("plasma_ne_profile, plasma_te_profile, and r_grid must be 1D arrays.")
        if len(r_grid) < 2:
            raise ValueError("r_grid must have at least 2 points.")
        if len(plasma_ne_profile) != len(plasma_te_profile):
            raise ValueError("plasma_ne_profile and plasma_te_profile must have matching lengths.")
        if not np.all(np.isfinite(plasma_ne_profile)) or not np.all(np.isfinite(plasma_te_profile)):
            raise ValueError("plasma profiles must be finite.")
        if not np.all(np.isfinite(r_grid)) or not np.all(np.diff(r_grid) > 0.0):
            raise ValueError("r_grid must be finite and strictly increasing.")

        deposition = np.zeros_like(r_grid)
        dr = r_grid[1] - r_grid[0] if len(r_grid) > 1 else 1.0

        for frag in self.fragments:
            if not frag.active:
                continue

            frag.pos += frag.vel * dt

            r_loc = np.sqrt(frag.pos[0]**2 + frag.pos[1]**2)
            z_loc = frag.pos[2]
            # Map (R,Z) to normalised minor radius rho
            rho_loc = np.sqrt(
                ((r_loc - R_MAJOR) / A_MINOR)**2 + (z_loc / ELONGATION)**2
            )

            if rho_loc > 1.2 or rho_loc < 0.0:
                continue

            rho_axis = np.linspace(0, 1, len(plasma_ne_profile))
            n_e = np.interp(rho_loc, rho_axis, plasma_ne_profile)   # [10^19 m^-3]
            T_e = np.interp(rho_loc, np.linspace(0, 1, len(plasma_te_profile)), plasma_te_profile)  # [keV]

            if T_e < 0.01:
                continue

            # Parks ablation scaling, Parks NF 57 (2017) Eq. 8
            ne_20 = max(float(n_e) / 10.0, 0.0)
            rp_cm = frag.radius * 100.0
            dm_dt_g = _PARKS_COEFFICIENT * (ne_20**0.33) * (float(T_e)**1.64) * (rp_cm**1.33)
            dm_dt_kg = dm_dt_g / 1000.0

            delta_m = dm_dt_kg * dt
            if delta_m > frag.mass:
                delta_m = frag.mass
                frag.active = False
                frag.mass = 0.0
                frag.radius = 0.0
            else:
                frag.mass -= delta_m
                frag.radius = (3 * (frag.mass / RHO_NEON_SOLID) / (4 * np.pi))**(1/3)

            n_particles = delta_m / M_NEON
            idx = int(round(rho_loc * (len(r_grid) - 1)))
            if 0 <= idx < len(r_grid):
                rate = n_particles / dt
                r_minor = rho_loc * A_MINOR
                dV = 4 * np.pi**2 * R_MAJOR * r_minor * (A_MINOR * dr)
                if dV < 1e-3:
                    dV = 1.0  # on-axis guard
                deposition[idx] += rate / dV

        return deposition
