# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — 3D MHD Equilibrium (VMEC-lite Fixed-Boundary)
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VMECResult:
    R_mn: np.ndarray
    Z_mn: np.ndarray
    B_mn: np.ndarray
    force_residual: float
    iterations: int
    converged: bool


class SpectralBasis:
    def __init__(self, m_pol: int, n_tor: int, n_fp: int):
        self.m_pol = m_pol
        self.n_tor = n_tor
        self.n_fp = n_fp

        self.mn_modes = []
        for m in range(m_pol + 1):
            n_min = -n_tor if m > 0 else 0
            for n in range(n_min, n_tor + 1):
                self.mn_modes.append((m, n))

        self.n_modes = len(self.mn_modes)

    def evaluate(
        self, coeffs_mn: np.ndarray, theta: np.ndarray, zeta: np.ndarray, is_sin: bool = False
    ) -> np.ndarray:
        # Evaluate sum C_mn * cos(m*theta - n*N_fp*zeta) or sin(...)
        val = np.zeros_like(theta)
        for i, (m, n) in enumerate(self.mn_modes):
            arg = m * theta - n * self.n_fp * zeta
            basis = np.sin(arg) if is_sin else np.cos(arg)
            val += coeffs_mn[i] * basis
        return val


class VMECLiteSolver:
    """
    Highly simplified spectral 3D equilibrium solver mimicking VMEC principles.
    Uses steepest descent to minimize a mock MHD energy functional.
    """

    def __init__(self, n_s: int = 21, m_pol: int = 3, n_tor: int = 2, n_fp: int = 1):
        self.n_s = n_s
        self.basis = SpectralBasis(m_pol, n_tor, n_fp)

        self.R_mn = np.zeros((n_s, self.basis.n_modes))
        self.Z_mn = np.zeros((n_s, self.basis.n_modes))

        self.pressure = np.zeros(n_s)
        self.iota = np.zeros(n_s)

        self.s_grid = np.linspace(0.0, 1.0, n_s)

    def set_boundary(
        self, R_bound: dict[tuple[int, int], float], Z_bound: dict[tuple[int, int], float]
    ) -> None:
        """Set fixed boundary conditions at s=1."""
        for i, (m, n) in enumerate(self.basis.mn_modes):
            if (m, n) in R_bound:
                self.R_mn[-1, i] = R_bound[(m, n)]
            if (m, n) in Z_bound:
                self.Z_mn[-1, i] = Z_bound[(m, n)]

    def set_profiles(self, pressure: np.ndarray, iota: np.ndarray) -> None:
        # Interpolate onto s_grid
        self.pressure = np.interp(self.s_grid, np.linspace(0, 1, len(pressure)), pressure)
        self.iota = np.interp(self.s_grid, np.linspace(0, 1, len(iota)), iota)

    def _initial_guess(self) -> None:
        # Linear interpolation from magnetic axis to boundary
        R00_bound = 0.0
        # Find (0,0) index
        idx_00 = self.basis.mn_modes.index((0, 0)) if (0, 0) in self.basis.mn_modes else -1
        if idx_00 >= 0:
            R00_bound = self.R_mn[-1, idx_00]

        for i, (m, n) in enumerate(self.basis.mn_modes):
            if m == 0 and n == 0:
                self.R_mn[:, i] = R00_bound
                self.Z_mn[:, i] = 0.0
            else:
                self.R_mn[:, i] = self.s_grid ** (m / 2.0) * self.R_mn[-1, i]
                self.Z_mn[:, i] = self.s_grid ** (m / 2.0) * self.Z_mn[-1, i]

    def solve(self, max_iter: int = 100, tol: float = 1e-4) -> VMECResult:
        """
        Spectral steepest-descent equilibrium solver.
        Hirshman & Whitson, Phys. Fluids 26, 3553 (1983).

        Radial tension (finite-difference Laplacian) regularises the spectral
        coefficients while the pressure gradient drives the Shafranov shift
        on the R₀₀ mode via the q²(dp/ds) force.
        """
        self._initial_guess()

        converged = False
        residual = float("inf")
        lr = 0.1

        idx_00 = next((i for i, mn in enumerate(self.basis.mn_modes) if mn == (0, 0)), -1)
        dp_ds = np.gradient(self.pressure, self.s_grid)
        q_profile = 1.0 / np.maximum(np.abs(self.iota), 0.01)
        R_00_bound = max(abs(self.R_mn[-1, idx_00]), 1e-3) if idx_00 >= 0 else 1.0

        for it in range(max_iter):
            F_R = np.zeros_like(self.R_mn)
            F_Z = np.zeros_like(self.Z_mn)

            # Radial tension: d²/ds² keeps flux surfaces smooth
            for i in range(1, self.n_s - 1):
                F_R[i] = (self.R_mn[i + 1] - 2 * self.R_mn[i] + self.R_mn[i - 1]) * 2.0
                F_Z[i] = (self.Z_mn[i + 1] - 2 * self.Z_mn[i] + self.Z_mn[i - 1]) * 2.0

            # Pressure-driven Shafranov shift on R₀₀ mode
            # F_R₀₀ ∝ -q²(s) dp/ds / R₀ (Grad-Shafranov radial force balance)
            if idx_00 >= 0:
                for i in range(1, self.n_s - 1):
                    F_R[i, idx_00] -= q_profile[i] ** 2 * dp_ds[i] / R_00_bound * 1e-6

            residual = float(np.max(np.abs(F_R)) + np.max(np.abs(F_Z)))

            if residual < tol:
                converged = True
                break

            self.R_mn[1:-1] += lr * F_R[1:-1]
            self.Z_mn[1:-1] += lr * F_Z[1:-1]

        # B field from rotational transform ι and geometry
        # B_φ ∝ 1/R (toroidal), B_θ ∝ ι (poloidal)
        # Wesson, "Tokamaks" 4th ed., Ch. 3
        B_mn = np.zeros_like(self.R_mn)
        for s in range(self.n_s):
            R_00 = max(abs(self.R_mn[s, idx_00]), 1e-6) if idx_00 >= 0 else 1.0
            iota_s = self.iota[s]
            if idx_00 >= 0:
                B_mn[s, idx_00] = 1.0
            for k, (m, _n) in enumerate(self.basis.mn_modes):
                if k == idx_00:
                    continue
                # 1/R expansion of toroidal field
                B_mn[s, k] = -self.R_mn[s, k] / R_00
                # Poloidal field from ι (dominant for m=1 shaping modes)
                if m == 1:
                    B_mn[s, k] += iota_s * abs(self.Z_mn[s, k]) / R_00

        return VMECResult(
            self.R_mn.copy(), self.Z_mn.copy(), B_mn, float(residual), it + 1, converged
        )


class AxisymmetricTokamakBoundary:
    @staticmethod
    def from_parameters(
        R0: float, a: float, kappa: float, delta: float
    ) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
        # R = R0 + a * cos(theta + delta * sin(theta))
        # Z = a * kappa * sin(theta)
        # Approximate to low order Fourier
        b_R: dict[tuple[int, int], float] = {
            (0, 0): R0,
            (1, 0): a,
            (2, 0): -0.5 * a * delta,
        }
        b_Z: dict[tuple[int, int], float] = {(1, 0): a * kappa}
        return b_R, b_Z


class StellaratorBoundary:
    @staticmethod
    def w7x_standard() -> tuple[dict, dict]:
        # N_fp = 5
        b_R = {(0, 0): 5.5, (1, 0): 0.5, (1, 1): 0.1, (0, 1): 0.2}
        b_Z = {(1, 0): 0.6, (1, 1): -0.1}
        return b_R, b_Z
