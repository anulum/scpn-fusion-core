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
  (Simplified scaling)
- Deposition of ablated material into the plasma density profile.
"""

from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Constants
AMU = 1.66e-27 # kg
M_NEON = 20.18 * AMU
RHO_NEON_SOLID = 1444.0 # kg/m^3 (approx)

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
                 injector_pos: np.ndarray = np.array([10.0, 0.0, 0.0]), # Outboard midplane
                 injector_dir: np.ndarray = np.array([-1.0, 0.0, 0.0]) # Radial inward
                 ):
        self.fragments: List[SpiFragment] = []
        
        # Initialize fragments
        # Assume distribution of sizes (e.g. log-normal or uniform volume)
        # Uniform mass for now
        m_frag = total_mass_kg / n_fragments
        # Volume = m / rho
        vol = m_frag / RHO_NEON_SOLID
        r_frag = (3 * vol / (4 * np.pi))**(1/3)
        
        rng = np.random.default_rng(42)
        
        for i in range(n_fragments):
            # Velocity dispersion cone
            v_dir = injector_dir + rng.normal(0, dispersion, 3)
            v_dir /= np.linalg.norm(v_dir)
            v_mag = velocity_mps * rng.normal(1.0, 0.1)
            vel = v_dir * v_mag
            
            # Position dispersion (shatter head size)
            pos = injector_pos + rng.normal(0, 0.05, 3)
            
            self.fragments.append(SpiFragment(
                id=i,
                radius=r_frag,
                mass=m_frag,
                pos=pos,
                vel=vel
            ))
            
    def step(self, dt: float, plasma_ne_profile: np.ndarray, plasma_te_profile: np.ndarray, r_grid: np.ndarray) -> np.ndarray:
        """
        Advance fragments and calculate deposition.
        
        Returns:
            deposition_profile: Source term [particles/m^3/s] on the r_grid.
        """
        deposition = np.zeros_like(r_grid)
        
        # Grid spacing
        if len(r_grid) > 1:
            dr = r_grid[1] - r_grid[0]
        else:
            dr = 1.0
            
        for frag in self.fragments:
            if not frag.active:
                continue
                
            # 1. Move
            frag.pos += frag.vel * dt
            
            # 2. Map to Plasma Coordinates
            # Simple cylindrical R
            r_loc = np.sqrt(frag.pos[0]**2 + frag.pos[1]**2)
            z_loc = frag.pos[2]
            
            # Assume tokamak centered at R=6.2, a=2.0
            # Convert (R,Z) to minor radius rho (approx)
            R0 = 6.2
            a = 2.0
            rho_loc = np.sqrt(((r_loc - R0)/a)**2 + (z_loc/2.0)**2) # Elongation 2.0
            
            # Check bounds
            if rho_loc > 1.2 or rho_loc < 0.0:
                continue # Outside plasma
                
            # 3. Get local plasma parameters
            # Interp
            n_e = np.interp(rho_loc, np.linspace(0, 1, len(plasma_ne_profile)), plasma_ne_profile) # 10^19 m^-3
            T_e = np.interp(rho_loc, np.linspace(0, 1, len(plasma_te_profile)), plasma_te_profile) # keV
            
            if T_e < 0.01: # Too cold to ablate
                continue
                
            # 4. Ablation Rate (Parks Scaling simplified)
            # dm/dt [g/s] ~ ...
            # Parks 2017: G [g/s] = 1.25e16 * n_e^0.33 * T_e^1.64 * r_p^1.33
            # n_e in 10^20 m^-3, T_e in keV, r_p in cm
            
            ne_20 = n_e / 10.0 # 10^19 -> 10^20
            rp_cm = frag.radius * 100.0
            
            dm_dt_g = 2.0 * (ne_20**(0.33)) * (T_e**1.64) * (rp_cm**1.33) # Empirical scaling
            dm_dt_kg = dm_dt_g / 1000.0
            
            delta_m = dm_dt_kg * dt
            
            if delta_m > frag.mass:
                delta_m = frag.mass
                frag.active = False
                frag.mass = 0.0
                frag.radius = 0.0
            else:
                frag.mass -= delta_m
                # Update radius: r ~ m^(1/3)
                frag.radius = (3 * (frag.mass / RHO_NEON_SOLID) / (4 * np.pi))**(1/3)
                
            # 5. Deposition
            # Particles ablated = delta_m / m_atom
            n_particles = delta_m / M_NEON
            
            # Distribute to grid
            # Gaussian smear around rho_loc
            # bin index
            idx_float = rho_loc * (len(r_grid) - 1)
            idx = int(round(idx_float))
            
            if 0 <= idx < len(r_grid):
                # Simple single-bin deposition / dt
                # Rate [particles/s]
                rate = n_particles / dt
                
                # Volume element?
                # For 1D solver, source is density rate [m^-3 s^-1]
                # We need volume of the shell at rho_loc
                # dV = 2 pi R0 * 2 pi r dr
                r_minor = rho_loc * a
                dV = 4 * np.pi**2 * R0 * r_minor * (a * dr)
                if dV < 1e-3: dV = 1.0 # Axis guard
                
                deposition[idx] += rate / dV
                
        return deposition
