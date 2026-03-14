# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Pedestal Model
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Pedestal profile model using the modified tanh (mtanh) parameterisation.

Provides H-mode pedestal structures for temperature and density profiles,
including width scaling based on the EPED1 model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PedestalParams:
    """Parameters for the mtanh pedestal profile.

    Parameters
    ----------
    f_ped : float
        Value at the pedestal top (e.g. T_ped in keV).
    f_sep : float
        Value at the separatrix (edge).
    x_ped : float
        Normalized radial position of the pedestal symmetry point (typically 0.94).
    delta : float
        Pedestal width in normalized radius (typically 0.04).
    a : float
        Slope parameter for the pedestal top (typically 0.1-0.3).
    """

    f_ped: float
    f_sep: float = 0.1
    x_ped: float = 0.94
    delta: float = 0.04
    a: float = 0.2


class PedestalProfile:
    """Modified tanh (mtanh) pedestal profile generator.

    Reference: Groebner, R.J. et al., Nucl. Fusion 41, 1789 (2001).
    """

    def __init__(self, params: PedestalParams) -> None:
        self.p = params

    def mtanh(self, y: np.ndarray) -> np.ndarray:
        """Core mtanh function: ((1+ay)e^y - e^-y) / (e^y + e^-y)."""
        exp_y = np.exp(np.clip(y, -20, 20))
        exp_ny = np.exp(np.clip(-y, -20, 20))
        result: np.ndarray = ((1.0 + self.p.a * y) * exp_y - exp_ny) / (exp_y + exp_ny)
        return result

    def evaluate(self, rho: np.ndarray) -> np.ndarray:
        """Evaluate the pedestal profile at given radial points.

        f(x) = (f_ped + f_sep)/2 + (f_ped - f_sep)/2 * mtanh((x_ped - x)/delta)
        """
        y = (self.p.x_ped - rho) / self.p.delta
        mid = (self.p.f_ped + self.p.f_sep) / 2.0
        half_width = (self.p.f_ped - self.p.f_sep) / 2.0
        return mid + half_width * self.mtanh(y)


def pedestal_width_eped1(beta_p_ped: float, delta_psi: float = 0.0) -> float:
    """Compute pedestal width using EPED1-like scaling.

    Scaling: Delta_psi = 0.076 * sqrt(beta_p_ped)
    Reference: Snyder, P.B. et al., Phys. Plasmas 16, 056118 (2009).

    Parameters
    ----------
    beta_p_ped : float
        Poloidal beta at the pedestal top.
    delta_psi : float
        Optional offset or alternative scaling parameter.

    Returns
    -------
    float — Predicted pedestal width in poloidal flux (or rho approx).
    """
    # Standard EPED1 width scaling (Delta_psi approx Delta_rho for simple geometry)
    # Snyder 2009, Eq. 1: Delta = 0.076 * sqrt(beta_p,ped)
    return float(0.076 * np.sqrt(max(beta_p_ped, 0.0)))
