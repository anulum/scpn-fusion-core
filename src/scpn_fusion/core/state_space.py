# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Unified State Space
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Defines the standard FusionState object shared across solvers and controllers.
Reduces data copying overhead and ensures state consistency.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from typing import Dict, Optional

@dataclass
class FusionState:
    """
    Unified plasma state representation.
    """
    # Magnetic Axis (m)
    r_axis: float = 0.0
    z_axis: float = 0.0
    
    # Global Quantities
    ip_ma: float = 0.0
    beta_p: float = 0.0
    q95: float = 0.0
    li: float = 0.0
    
    # Topology
    x_point_r: float = 0.0
    x_point_z: float = 0.0
    
    # Instabilities (Resistive MHD Island Widths)
    island_widths: Dict[float, float] = field(default_factory=dict)
    
    # Bootstrap fraction
    f_bs: float = 0.0
    
    # Timestamp
    time_s: float = 0.0

    def compute_bootstrap_fraction(self, epsilon: float = 0.32) -> float:
        """
        Estimates the bootstrap current fraction using Sauter-like scaling.
        f_bs ~ 0.4 * beta_p * sqrt(epsilon)
        """
        self.f_bs = float(np.clip(0.4 * self.beta_p * np.sqrt(epsilon), 0.0, 0.9))
        return self.f_bs

    def to_vector(self) -> np.ndarray:
        """Convert scalar features to a vector for ML/MPC use."""
        return np.array([
            self.r_axis, self.z_axis, 
            self.x_point_r, self.x_point_z,
            self.ip_ma, self.q95
        ], dtype=np.float64)

    @classmethod
    def from_kernel(cls, kernel: any, time_s: float = 0.0) -> FusionState:
        """Construct state from a Physics Kernel instance."""
        # Find axis
        idx_max = int(np.argmax(kernel.Psi))
        iz, ir = np.unravel_index(idx_max, kernel.Psi.shape)
        r_ax = float(kernel.R[ir])
        z_ax = float(kernel.Z[iz])
        
        # Find X-point
        xp_pos, _ = kernel.find_x_point(kernel.Psi)
        
        # Get physics
        ip = float(kernel.cfg["physics"].get("plasma_current_target", 0.0))
        
        return cls(
            r_axis=r_ax,
            z_axis=z_ax,
            x_point_r=float(xp_pos[0]),
            x_point_z=float(xp_pos[1]),
            ip_ma=ip,
            time_s=time_s
        )
