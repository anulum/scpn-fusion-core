# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Vacuum Vessel Eddy Current Model
"""
Vacuum vessel eddy current model using a lumped-circuit approach.

Models the induction and decay of currents in the conducting vessel walls,
providing passive stability effects and flux perturbations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.special import ellipe, ellipk

logger = logging.getLogger(__name__)

# ─── Physical constants (CODATA 2018) ───────────────────────────────
_MU0 = 4.0 * np.pi * 1e-7  # H/m


@dataclass(frozen=True)
class VesselElement:
    """Discrete conducting element of the vacuum vessel.

    Parameters
    ----------
    R : float
        Major radius of the element center [m].
    Z : float
        Vertical position of the element center [m].
    resistance : float
        Electrical resistance of the toroidal loop [Ohm].
    cross_section : float
        Cross-sectional area of the element [m^2].
    inductance : float
        Self-inductance of the toroidal loop [H].
    """

    R: float
    Z: float
    resistance: float
    cross_section: float
    inductance: float


class VesselModel:
    """Lumped-circuit model for vessel eddy currents.

    Solves the circuit equation M * dI/dt + R * I = -V_ext, where M is
     the mutual inductance matrix and V_ext is the induced loop voltage.
    """

    def __init__(self, elements: list[VesselElement]) -> None:
        self.elements = elements
        self.n_elements = len(elements)
        self.I = np.zeros(self.n_elements)

        # Build mutual inductance matrix M
        self.M = np.zeros((self.n_elements, self.n_elements))
        for i in range(self.n_elements):
            for j in range(self.n_elements):
                if i == j:
                    self.M[i, j] = elements[i].inductance
                else:
                    self.M[i, j] = self._calculate_mutual_inductance(
                        elements[i].R, elements[i].Z, elements[j].R, elements[j].Z
                    )

        # Precompute resistance diagonal matrix
        self.R_mat = np.diag([el.resistance for el in elements])

        # LU decompose M for fast stepping
        # (Using direct inverse for simplicity in this module)
        try:
            self.M_inv = np.linalg.inv(self.M)
        except np.linalg.LinAlgError:
            logger.error("Vessel inductance matrix is singular.")
            self.M_inv = np.zeros_like(self.M)

    def _calculate_mutual_inductance(self, R1: float, Z1: float, R2: float, Z2: float) -> float:
        """Calculate mutual inductance between two toroidal loops."""
        denom = (R1 + R2) ** 2 + (Z1 - Z2) ** 2
        if denom < 1e-30:
            return 0.0
        k2 = 4.0 * R1 * R2 / denom
        k2 = np.clip(k2, 1e-9, 0.999999)

        K_val = ellipk(k2)
        E_val = ellipe(k2)

        # M = mu0 * sqrt(R1*R2) * [ (2/k - k)*K(k) - 2/k*E(k) ]
        # Note: standard formula uses k = sqrt(k2)
        prefactor = _MU0 * np.sqrt(R1 * R2)
        # Using k2 version to match fusion_kernel.py Green's function
        m = prefactor * ((2.0 - k2) * K_val - 2.0 * E_val) / np.sqrt(k2)
        return float(m)

    def step(self, dt: float, dphi_ext_dt: np.ndarray) -> np.ndarray:
        """Advance eddy currents by one time step.

        M * dI/dt + R * I = -dPhi_ext/dt
        dI/dt = M^-1 * (-R*I - dPhi_ext/dt)

        Parameters
        ----------
        dt : float
            Time step [s].
        dphi_ext_dt : np.ndarray
            Rate of change of external poloidal flux through each element [Wb/s].
            Length must match n_elements.

        Returns
        -------
        np.ndarray — Updated eddy currents [A].
        """
        if dt <= 0:
            return self.I

        # Explicit Euler for simplicity (or Trapezoidal if needed)
        # V_ind = -dphi_ext_dt
        dI_dt = self.M_inv @ (-self.R_mat @ self.I - dphi_ext_dt)
        self.I += dI_dt * dt
        return self.I

    def psi_vessel(self, R: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Compute poloidal flux contribution from vessel currents.

        Parameters
        ----------
        R, Z : np.ndarray
            Observation points (can be grid or arrays).

        Returns
        -------
        np.ndarray — Flux contribution [Wb/rad].
        """
        out_shape = R.shape
        R_flat = R.flatten()
        Z_flat = Z.flatten()
        psi = np.zeros_like(R_flat)

        # Prefactor for Green's function (mu0/2pi * sqrt(R_obs*R_src))
        # Wait, the Green's function in fusion_kernel returns psi per unit current.
        for i in range(self.n_elements):
            el = self.elements[i]
            if abs(self.I[i]) < 1e-6:
                continue

            # Vectorized Green's function eval
            denom = (R_flat + el.R) ** 2 + (Z_flat - el.Z) ** 2
            k2 = 4.0 * R_flat * el.R / np.maximum(denom, 1e-30)
            k2 = np.clip(k2, 1e-9, 0.999999)

            K_val = ellipk(k2)
            E_val = ellipe(k2)

            prefactor = (_MU0 / (2.0 * np.pi)) * np.sqrt(R_flat * el.R)
            g = prefactor * ((2.0 - k2) * K_val - 2.0 * E_val) / np.sqrt(k2)
            psi += g * self.I[i]

        return psi.reshape(out_shape)
