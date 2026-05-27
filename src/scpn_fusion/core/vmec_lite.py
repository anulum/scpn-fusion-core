# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — 3D MHD Equilibrium (VMEC-lite Fixed-Boundary)
"""Lightweight VMEC-inspired fixed-boundary equilibrium helpers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VMECResult:
    """Container for a reduced-order VMEC-like equilibrium state."""
    R_mn: np.ndarray
    Z_mn: np.ndarray
    B_mn: np.ndarray
    force_residual: float
    iterations: int
    converged: bool
    residual_history: np.ndarray


class SpectralBasis:
    """Fourier basis over poloidal/toroidal mode pairs."""
    def __init__(self, m_pol: int, n_tor: int, n_fp: int):
        if not isinstance(m_pol, int) or m_pol < 0:
            raise ValueError("m_pol must be an integer >= 0.")
        if not isinstance(n_tor, int) or n_tor < 0:
            raise ValueError("n_tor must be an integer >= 0.")
        if not isinstance(n_fp, int) or n_fp < 1:
            raise ValueError("n_fp must be an integer >= 1.")
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
        """Evaluate a Fourier series on (theta, zeta)."""
        coeffs_mn = np.asarray(coeffs_mn, dtype=float)
        theta = np.asarray(theta, dtype=float)
        zeta = np.asarray(zeta, dtype=float)
        if coeffs_mn.shape != (self.n_modes,):
            raise ValueError(f"coeffs_mn must have shape ({self.n_modes},).")
        if theta.shape != zeta.shape:
            raise ValueError("theta and zeta must have the same shape.")
        if not (
            np.all(np.isfinite(coeffs_mn))
            and np.all(np.isfinite(theta))
            and np.all(np.isfinite(zeta))
        ):
            raise ValueError("spectral basis inputs must be finite.")
        # Evaluate sum C_mn * cos(m*theta - n*N_fp*zeta) or sin(...)
        val = np.zeros_like(theta)
        for i, (m, n) in enumerate(self.mn_modes):
            arg = m * theta - n * self.n_fp * zeta
            basis = np.sin(arg) if is_sin else np.cos(arg)
            val += coeffs_mn[i] * basis
        return val


class VMECLiteSolver:
    """
    Fixed-boundary spectral 3D equilibrium relaxation solver.

    The implementation follows the VMEC fixed-boundary representation at
    reduced order: radial spectral coefficients are relaxed under curvature
    tension, pressure-gradient drive on the magnetic-axis mode, fixed edge
    constraints, and adaptive residual-monotone steps.
    """

    def __init__(self, n_s: int = 21, m_pol: int = 3, n_tor: int = 2, n_fp: int = 1):
        if not isinstance(n_s, int) or n_s < 3:
            raise ValueError("n_s must be an integer >= 3.")
        self.n_s = n_s
        self.basis = SpectralBasis(m_pol, n_tor, n_fp)

        self.R_mn = np.zeros((n_s, self.basis.n_modes))
        self.Z_mn = np.zeros((n_s, self.basis.n_modes))

        self.pressure = np.zeros(n_s)
        self.iota = np.zeros(n_s)

        self.s_grid = np.linspace(0.0, 1.0, n_s)
        self._boundary_configured = False
        self._profiles_configured = False

    def set_boundary(
        self, R_bound: dict[tuple[int, int], float], Z_bound: dict[tuple[int, int], float]
    ) -> None:
        """Set fixed boundary conditions at s=1."""
        valid_modes = set(self.basis.mn_modes)
        unknown_modes = (set(R_bound) | set(Z_bound)) - valid_modes
        if unknown_modes:
            raise ValueError(
                f"boundary contains unsupported spectral mode(s): {sorted(unknown_modes)}"
            )

        for name, bound in (("R_bound", R_bound), ("Z_bound", Z_bound)):
            for mode, value in bound.items():
                if not np.isfinite(value):
                    raise ValueError(f"{name}{mode} must be finite.")

        for i, (m, n) in enumerate(self.basis.mn_modes):
            if (m, n) in R_bound:
                self.R_mn[-1, i] = R_bound[(m, n)]
            if (m, n) in Z_bound:
                self.Z_mn[-1, i] = Z_bound[(m, n)]
        idx_00 = self.basis.mn_modes.index((0, 0))
        if not np.isfinite(self.R_mn[-1, idx_00]) or self.R_mn[-1, idx_00] <= 0.0:
            raise ValueError("boundary must define a finite positive R(0,0) major-radius mode.")
        self._boundary_configured = True

    def set_profiles(self, pressure: np.ndarray, iota: np.ndarray) -> None:
        """Set pressure and rotational-transform profiles on the solver radial grid."""
        pressure = np.asarray(pressure, dtype=float)
        iota = np.asarray(iota, dtype=float)
        if pressure.ndim != 1 or pressure.size < 2:
            raise ValueError("pressure profile must be one-dimensional with at least two samples.")
        if iota.ndim != 1 or iota.size < 2:
            raise ValueError("iota profile must be one-dimensional with at least two samples.")
        if not np.all(np.isfinite(pressure)) or np.any(pressure < 0.0):
            raise ValueError("pressure profile must be finite and non-negative.")
        if not np.all(np.isfinite(iota)) or np.any(np.abs(iota) < 1e-6):
            raise ValueError("iota profile must be finite and non-zero.")
        # Interpolate onto s_grid
        self.pressure = np.interp(self.s_grid, np.linspace(0, 1, len(pressure)), pressure)
        self.iota = np.interp(self.s_grid, np.linspace(0, 1, len(iota)), iota)
        self._profiles_configured = True

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
        if not self._boundary_configured:
            raise ValueError("boundary must be configured before solve().")
        if not self._profiles_configured:
            raise ValueError("profiles must be configured before solve().")
        if not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be an integer >= 1.")
        if not np.isfinite(tol) or tol <= 0.0:
            raise ValueError("tol must be finite and positive.")

        self._initial_guess()

        converged = False
        residual = float("inf")
        lr = 0.1
        residual_history: list[float] = []

        idx_00 = next((i for i, mn in enumerate(self.basis.mn_modes) if mn == (0, 0)), -1)
        dp_ds = np.gradient(self.pressure, self.s_grid)
        q_profile = 1.0 / np.maximum(np.abs(self.iota), 0.01)
        R_00_bound = max(abs(self.R_mn[-1, idx_00]), 1e-3) if idx_00 >= 0 else 1.0

        def _forces() -> tuple[np.ndarray, np.ndarray, float]:
            f_r = np.zeros_like(self.R_mn)
            f_z = np.zeros_like(self.Z_mn)

            for j in range(1, self.n_s - 1):
                f_r[j] = (self.R_mn[j + 1] - 2 * self.R_mn[j] + self.R_mn[j - 1]) * 2.0
                f_z[j] = (self.Z_mn[j + 1] - 2 * self.Z_mn[j] + self.Z_mn[j - 1]) * 2.0

            if idx_00 >= 0:
                for j in range(1, self.n_s - 1):
                    f_r[j, idx_00] -= q_profile[j] ** 2 * dp_ds[j] / R_00_bound * 1e-6

            res = float(np.max(np.abs(f_r)) + np.max(np.abs(f_z)))
            return f_r, f_z, res

        for it in range(max_iter):
            F_R, F_Z, residual = _forces()
            residual_history.append(residual)

            if residual < tol:
                converged = True
                break

            old_R = self.R_mn[1:-1].copy()
            old_Z = self.Z_mn[1:-1].copy()
            step = lr
            accepted = False
            for _ in range(10):
                self.R_mn[1:-1] = old_R + step * F_R[1:-1]
                self.Z_mn[1:-1] = old_Z + step * F_Z[1:-1]
                _, _, trial_residual = _forces()
                if np.isfinite(trial_residual) and trial_residual <= residual:
                    lr = min(step * 1.1, 0.2)
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                self.R_mn[1:-1] = old_R
                self.Z_mn[1:-1] = old_Z
                lr *= 0.5

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
            self.R_mn.copy(),
            self.Z_mn.copy(),
            B_mn,
            float(residual),
            it + 1,
            converged,
            np.asarray(residual_history, dtype=float),
        )


class AxisymmetricTokamakBoundary:
    """Factory for common axisymmetric spectral boundary coefficients."""

    @staticmethod
    def from_parameters(
        R0: float, a: float, kappa: float, delta: float
    ) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
        """Build low-order Fourier boundary modes from geometry parameters."""
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
    """Preset helpers for representative stellarator boundaries."""

    @staticmethod
    def w7x_standard() -> tuple[dict, dict]:
        """Return a compact W7-X-like spectral boundary pair."""
        # N_fp = 5
        b_R = {(0, 0): 5.5, (1, 0): 0.5, (1, 1): 0.1, (0, 1): 0.2}
        b_Z = {(1, 0): 0.6, (1, 1): -0.1}
        return b_R, b_Z
