# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — 3D Equilibrium
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""VMEC-like reduced 3D equilibrium interface and flux coordinates.

This module provides a compact Fourier-parameterized equilibrium surrogate
that maps native flux coordinates ``(rho, theta, phi)`` into cylindrical and
Cartesian geometry. It is intentionally reduced-order (not a full VMEC solver),
but exposes a compatible interface for non-axisymmetric ``n != 0`` shaping.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FourierMode3D:
    """Single VMEC-like Fourier harmonic for 3D boundary shaping."""

    m: int
    n: int
    r_cos: float = 0.0
    r_sin: float = 0.0
    z_cos: float = 0.0
    z_sin: float = 0.0


class VMECStyleEquilibrium3D:
    """Reduced VMEC-like 3D equilibrium in native flux coordinates."""

    def __init__(
        self,
        *,
        r_axis: float,
        z_axis: float,
        a_minor: float,
        kappa: float = 1.0,
        triangularity: float = 0.0,
        nfp: int = 1,
        modes: Iterable[FourierMode3D] | None = None,
    ) -> None:
        if a_minor <= 0.0:
            raise ValueError("a_minor must be > 0.")
        if kappa <= 0.0:
            raise ValueError("kappa must be > 0.")
        if nfp < 1:
            raise ValueError("nfp must be >= 1.")

        self.r_axis = float(r_axis)
        self.z_axis = float(z_axis)
        self.a_minor = float(a_minor)
        self.kappa = float(kappa)
        self.triangularity = float(np.clip(triangularity, -0.95, 0.95))
        self.nfp = int(nfp)
        self.modes = list(modes or [])

    @classmethod
    def from_axisymmetric_lcfs(
        cls,
        lcfs_points: np.ndarray,
        *,
        r_axis: float,
        z_axis: float,
        nfp: int = 1,
        modes: Iterable[FourierMode3D] | None = None,
    ) -> "VMECStyleEquilibrium3D":
        """Infer baseline shaping from traced axisymmetric LCFS points."""
        points = np.asarray(lcfs_points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("lcfs_points must have shape (N, 2).")
        if points.shape[0] < 8:
            raise ValueError("lcfs_points must contain at least 8 points.")

        r_vals = points[:, 0]
        z_vals = points[:, 1]
        a_minor = max(0.5 * (float(np.max(r_vals)) - float(np.min(r_vals))), 1e-6)
        semi_z = max(0.5 * (float(np.max(z_vals)) - float(np.min(z_vals))), 1e-6)
        kappa = max(semi_z / a_minor, 0.2)

        i_top = int(np.argmax(z_vals))
        i_bottom = int(np.argmin(z_vals))
        r_top = float(r_vals[i_top])
        r_bottom = float(r_vals[i_bottom])
        delta = (float(r_axis) - 0.5 * (r_top + r_bottom)) / a_minor

        return cls(
            r_axis=float(r_axis),
            z_axis=float(z_axis),
            a_minor=a_minor,
            kappa=kappa,
            triangularity=float(np.clip(delta, -0.9, 0.9)),
            nfp=nfp,
            modes=modes,
        )

    def to_vmec_like_dict(self) -> dict[str, object]:
        """Export a deterministic VMEC-like boundary payload."""
        return {
            "format": "vmec_like_v1",
            "r_axis": float(self.r_axis),
            "z_axis": float(self.z_axis),
            "a_minor": float(self.a_minor),
            "kappa": float(self.kappa),
            "triangularity": float(self.triangularity),
            "nfp": int(self.nfp),
            "modes": [
                {
                    "m": int(mode.m),
                    "n": int(mode.n),
                    "r_cos": float(mode.r_cos),
                    "r_sin": float(mode.r_sin),
                    "z_cos": float(mode.z_cos),
                    "z_sin": float(mode.z_sin),
                }
                for mode in self.modes
            ],
        }

    @classmethod
    def from_vmec_like_dict(cls, payload: dict[str, object]) -> "VMECStyleEquilibrium3D":
        """Rebuild reduced equilibrium from VMEC-like boundary payload."""
        modes_payload = payload.get("modes", [])
        modes: list[FourierMode3D] = []
        if isinstance(modes_payload, list):
            for row in modes_payload:
                if not isinstance(row, dict):
                    raise ValueError("VMEC mode entries must be dictionaries.")
                modes.append(
                    FourierMode3D(
                        m=int(row.get("m", 0)),
                        n=int(row.get("n", 0)),
                        r_cos=float(row.get("r_cos", 0.0)),
                        r_sin=float(row.get("r_sin", 0.0)),
                        z_cos=float(row.get("z_cos", 0.0)),
                        z_sin=float(row.get("z_sin", 0.0)),
                    )
                )

        return cls(
            r_axis=float(payload["r_axis"]),
            z_axis=float(payload["z_axis"]),
            a_minor=float(payload["a_minor"]),
            kappa=float(payload.get("kappa", 1.0)),
            triangularity=float(payload.get("triangularity", 0.0)),
            nfp=int(payload.get("nfp", 1)),
            modes=modes,
        )

    @staticmethod
    def _broadcast(
        rho: float | np.ndarray,
        theta: float | np.ndarray,
        phi: float | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rho_arr = np.asarray(rho, dtype=float)
        theta_arr = np.asarray(theta, dtype=float)
        phi_arr = np.asarray(phi, dtype=float)
        rho_b, theta_b, phi_b = np.broadcast_arrays(rho_arr, theta_arr, phi_arr)
        return rho_b, theta_b, phi_b

    def flux_to_cylindrical(
        self,
        rho: float | np.ndarray,
        theta: float | np.ndarray,
        phi: float | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map native flux coordinates ``(rho, theta, phi)`` to ``(R, Z, phi)``."""
        rho_b, theta_b, phi_b = self._broadcast(rho, theta, phi)
        rho_b = np.clip(rho_b, 0.0, 1.25)

        theta_geo = theta_b + self.triangularity * np.sin(theta_b)
        minor = self.a_minor * rho_b

        r_val = self.r_axis + minor * np.cos(theta_geo)
        z_val = self.z_axis + self.kappa * minor * np.sin(theta_b)

        if self.modes:
            for mode in self.modes:
                phase = float(mode.m) * theta_b - float(mode.n * self.nfp) * phi_b
                r_val = r_val + minor * (
                    float(mode.r_cos) * np.cos(phase) + float(mode.r_sin) * np.sin(phase)
                )
                z_val = z_val + minor * (
                    float(mode.z_cos) * np.cos(phase) + float(mode.z_sin) * np.sin(phase)
                )

        return r_val, z_val, phi_b

    def flux_to_cartesian(
        self,
        rho: float | np.ndarray,
        theta: float | np.ndarray,
        phi: float | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map native flux coordinates ``(rho, theta, phi)`` to ``(x, y, z)``."""
        r_val, z_val, phi_val = self.flux_to_cylindrical(rho, theta, phi)
        x_val = r_val * np.cos(phi_val)
        y_val = r_val * np.sin(phi_val)
        return x_val, y_val, z_val

    def sample_surface(
        self,
        *,
        rho: float = 1.0,
        resolution_toroidal: int = 60,
        resolution_poloidal: int = 60,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample a triangulated flux surface from native 3D coordinates."""
        if resolution_toroidal < 3:
            raise ValueError("resolution_toroidal must be >= 3.")
        if resolution_poloidal < 8:
            raise ValueError("resolution_poloidal must be >= 8.")

        n_tor = int(resolution_toroidal)
        n_pol = int(resolution_poloidal)

        phi_vals = np.linspace(0.0, 2.0 * np.pi, n_tor, endpoint=False)
        theta_vals = np.linspace(0.0, 2.0 * np.pi, n_pol, endpoint=False)
        phi_grid, theta_grid = np.meshgrid(phi_vals, theta_vals, indexing="ij")
        rho_grid = np.full_like(phi_grid, float(rho), dtype=float)

        x_grid, y_grid, z_grid = self.flux_to_cartesian(rho_grid, theta_grid, phi_grid)
        vertices = np.stack(
            [x_grid.reshape(-1), y_grid.reshape(-1), z_grid.reshape(-1)],
            axis=1,
        )

        faces: list[list[int]] = []
        for i in range(n_tor):
            i_next = (i + 1) % n_tor
            for j in range(n_pol):
                j_next = (j + 1) % n_pol

                a = i * n_pol + j
                b = i * n_pol + j_next
                c = i_next * n_pol + j
                d = i_next * n_pol + j_next

                faces.append([a, b, c])
                faces.append([b, d, c])

        return vertices.astype(float), np.asarray(faces, dtype=np.int64)


# ──────────────────────────────────────────────────────────────────────
# 3D Force-Balance Solver (Reduced-Order Spectral Variational)
# ──────────────────────────────────────────────────────────────────────


@dataclass
class ForceBalanceResult:
    """Result of reduced-order 3D force-balance solve."""

    converged: bool
    iterations: int
    residual_norm: float
    initial_residual: float
    modes: list[FourierMode3D]
    force_residual_history: list[float]
    armijo_reject_count: int = 0
    non_decreasing_steps: int = 0


class ForceBalance3D:
    r"""Reduced-order 3D force-balance solver via spectral variational method.

    Takes a 2D Grad-Shafranov base equilibrium (Psi, pressure, current profiles)
    and iterates 3D Fourier mode coefficients to minimize the MHD force residual:

        ||J x B - nabla p||^2

    evaluated on a (rho, theta, phi) sampling grid. This is NOT a full 3D VMEC
    solver but provides a genuine force-balance closure by coupling the Fourier
    boundary parameterization to the MHD equilibrium condition.

    The method is a spectral Galerkin projection: compute the force residual on a
    3D sampling grid, project onto each Fourier mode via inner product, and update
    the mode amplitudes by gradient descent with Armijo line search.
    """

    def __init__(
        self,
        eq: VMECStyleEquilibrium3D,
        *,
        b0_tesla: float = 5.3,
        r0_major: float = 6.2,
        p0_pa: float = 5e5,
        j0_ma_m2: float = 1.0,
        pressure_exp: float = 2.0,
        current_exp: float = 1.5,
    ) -> None:
        self.eq = eq
        self.b0 = float(b0_tesla)
        self.r0 = float(r0_major)
        self.p0 = float(p0_pa)
        self.j0 = float(j0_ma_m2) * 1e6  # MA/m^2 -> A/m^2
        self.pressure_exp = float(pressure_exp)
        self.current_exp = float(current_exp)

    def _pressure_profile(self, rho: np.ndarray) -> np.ndarray:
        """p(rho) = p0 * (1 - rho^2)^alpha, clamped to [0, 1]."""
        rho_c = np.clip(rho, 0.0, 1.0)
        return self.p0 * (1.0 - rho_c**2) ** self.pressure_exp

    def _current_profile(self, rho: np.ndarray) -> np.ndarray:
        """J_phi(rho) = j0 * (1 - rho^2)^beta."""
        rho_c = np.clip(rho, 0.0, 1.0)
        return self.j0 * (1.0 - rho_c**2) ** self.current_exp

    def _magnetic_field(
        self, R: np.ndarray, Z: np.ndarray, rho: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Approximate (B_R, B_Z, B_phi) at given points.

        Toroidal field: B_phi = B0 * R0 / R.
        Poloidal field: derived from current profile via Ampere's law (cylindrical
        approximation): B_theta ~ mu0 * J_phi * a * rho / 2.
        """
        mu0 = 4.0 * np.pi * 1e-7
        B_phi = self.b0 * self.r0 / np.maximum(R, 0.1)
        J_phi = self._current_profile(rho)
        B_theta = mu0 * J_phi * self.eq.a_minor * np.clip(rho, 0.0, 1.0) / 2.0
        # Project B_theta into (R, Z) components via poloidal angle
        theta_approx = np.arctan2(Z - self.eq.z_axis, R - self.eq.r_axis)
        B_R = -B_theta * np.sin(theta_approx)
        B_Z = B_theta * np.cos(theta_approx)
        return B_R, B_Z, B_phi

    def compute_force_residual(
        self,
        n_rho: int = 12,
        n_theta: int = 24,
        n_phi: int = 16,
    ) -> float:
        r"""Compute ||J x B - nabla p||^2 on a (rho, theta, phi) grid.

        Returns the volume-averaged L2 norm of the force residual.
        """
        rho_pts = np.linspace(0.05, 0.95, n_rho)
        theta_pts = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
        phi_pts = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
        rho_g, theta_g, phi_g = np.meshgrid(rho_pts, theta_pts, phi_pts, indexing="ij")

        R, Z, _ = self.eq.flux_to_cylindrical(rho_g, theta_g, phi_g)
        rho_flat = rho_g.ravel()
        R_flat = R.ravel()
        Z_flat = Z.ravel()

        # Pressure gradient (radial in flux coordinates)
        drho = 0.01
        p_plus = self._pressure_profile(rho_flat + drho)
        p_minus = self._pressure_profile(rho_flat - drho)
        dp_drho = (p_plus - p_minus) / (2.0 * drho)

        # Map dp/drho to (R, Z) via chain rule: dp/dR ~ dp/drho * drho/dR
        # drho/dR ~ cos(theta) / a_minor, drho/dZ ~ sin(theta) / (kappa * a_minor)
        theta_flat = theta_g.ravel()
        grad_p_R = dp_drho * np.cos(theta_flat) / max(self.eq.a_minor, 1e-6)
        grad_p_Z = dp_drho * np.sin(theta_flat) / max(
            self.eq.kappa * self.eq.a_minor, 1e-6
        )

        # Current density and magnetic field
        B_R, B_Z, B_phi = self._magnetic_field(R_flat, Z_flat, rho_flat)
        J_phi = self._current_profile(rho_flat)

        # J x B (only phi-component of J, so J x B has R and Z components)
        # J = (0, 0, J_phi) in cylindrical
        # J x B = (J_phi * B_Z, -J_phi * B_R, 0) -- but we want radial force balance
        JxB_R = J_phi * B_Z
        JxB_Z = -J_phi * B_R

        # Force residual: F = J x B - grad(p)
        F_R = JxB_R - grad_p_R
        F_Z = JxB_Z - grad_p_Z

        residual = float(np.sqrt(np.mean(F_R**2 + F_Z**2)))
        return residual

    def _mode_gradient(
        self,
        mode_idx: int,
        component: str,
        epsilon: float = 1e-5,
        n_rho: int = 12,
        n_theta: int = 24,
        n_phi: int = 16,
    ) -> float:
        """Finite-difference gradient of residual w.r.t. a mode coefficient."""
        mode = self.eq.modes[mode_idx]
        orig_val = getattr(mode, component)

        # Perturb +epsilon
        kwargs = {
            "m": mode.m, "n": mode.n,
            "r_cos": mode.r_cos, "r_sin": mode.r_sin,
            "z_cos": mode.z_cos, "z_sin": mode.z_sin,
        }
        kwargs[component] = orig_val + epsilon
        self.eq.modes[mode_idx] = FourierMode3D(**kwargs)
        res_plus = self.compute_force_residual(n_rho, n_theta, n_phi)

        # Perturb -epsilon
        kwargs[component] = orig_val - epsilon
        self.eq.modes[mode_idx] = FourierMode3D(**kwargs)
        res_minus = self.compute_force_residual(n_rho, n_theta, n_phi)

        # Restore
        kwargs[component] = orig_val
        self.eq.modes[mode_idx] = FourierMode3D(**kwargs)

        return (res_plus - res_minus) / (2.0 * epsilon)

    def solve(
        self,
        *,
        max_iterations: int = 200,
        tolerance: float = 1e-4,
        learning_rate: float = 0.01,
        armijo_c: float = 1e-4,
        n_rho: int = 12,
        n_theta: int = 24,
        n_phi: int = 16,
    ) -> ForceBalanceResult:
        """Iterate Fourier coefficients to minimise force residual.

        Uses gradient descent with Armijo back-tracking line search. If the
        equilibrium has no modes, a default set (m=0..2, n=0..1) is seeded.
        """
        if not self.eq.modes:
            seed_modes = []
            for m in range(3):
                for n in range(2):
                    seed_modes.append(FourierMode3D(m=m, n=n))
            self.eq.modes = seed_modes

        initial_residual = self.compute_force_residual(n_rho, n_theta, n_phi)
        history = [initial_residual]
        armijo_reject_count = 0
        non_decreasing_steps = 0

        components = ["r_cos", "r_sin", "z_cos", "z_sin"]

        for iteration in range(max_iterations):
            current_residual = history[-1]
            if current_residual < tolerance:
                break

            # Compute gradient for all mode coefficients
            gradients: list[tuple[int, str, float]] = []
            for idx in range(len(self.eq.modes)):
                for comp in components:
                    g = self._mode_gradient(idx, comp, n_rho=n_rho, n_theta=n_theta, n_phi=n_phi)
                    gradients.append((idx, comp, g))

            # Gradient norm for Armijo
            grad_norm_sq = sum(g * g for _, _, g in gradients)
            if grad_norm_sq < 1e-20:
                break

            # Armijo line search
            step = learning_rate
            for _ in range(8):
                # Apply trial step
                saved = list(self.eq.modes)
                for idx, comp, g in gradients:
                    mode = self.eq.modes[idx]
                    kwargs = {
                        "m": mode.m, "n": mode.n,
                        "r_cos": mode.r_cos, "r_sin": mode.r_sin,
                        "z_cos": mode.z_cos, "z_sin": mode.z_sin,
                    }
                    kwargs[comp] = kwargs[comp] - step * g
                    self.eq.modes[idx] = FourierMode3D(**kwargs)

                trial_residual = self.compute_force_residual(n_rho, n_theta, n_phi)
                if trial_residual <= current_residual - armijo_c * step * grad_norm_sq:
                    break
                # Revert and halve step
                self.eq.modes = saved
                step *= 0.5
            else:
                armijo_reject_count += 1
                logger.debug(
                    "ForceBalance3D Armijo line-search rejected all trial steps "
                    "(iteration=%d, residual=%.6e, grad_norm_sq=%.6e)",
                    iteration,
                    current_residual,
                    grad_norm_sq,
                )

            new_residual = self.compute_force_residual(n_rho, n_theta, n_phi)
            if new_residual >= current_residual:
                non_decreasing_steps += 1
            history.append(new_residual)

        final_residual = history[-1]
        return ForceBalanceResult(
            converged=final_residual < tolerance,
            iterations=len(history) - 1,
            residual_norm=final_residual,
            initial_residual=initial_residual,
            modes=list(self.eq.modes),
            force_residual_history=history,
            armijo_reject_count=armijo_reject_count,
            non_decreasing_steps=non_decreasing_steps,
        )
