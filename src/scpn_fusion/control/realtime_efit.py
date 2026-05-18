# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Real-Time Equilibrium Reconstruction (EFIT-lite)
"""Realtime EFIT-like magnetic reconstruction and synthetic diagnostic response."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class MagneticDiagnostics:
    """Layout of magnetic sensors."""

    flux_loops: list[tuple[float, float]]
    b_probes: list[tuple[float, float, str]]
    rogowski_radius: float


@dataclass
class ShapeParams:
    """Reconstructed macroscopic parameters."""

    R0: float
    a: float
    kappa: float
    delta_upper: float
    delta_lower: float
    q95: float
    beta_pol: float
    li: float
    Ip_reconstructed: float


@dataclass
class ReconstructionResult:
    """EFIT reconstruction output: psi field, source coefficients, and shape."""

    psi: np.ndarray
    p_prime_coeffs: np.ndarray
    ff_prime_coeffs: np.ndarray
    shape: ShapeParams
    chi_squared: float
    n_iterations: int
    wall_time_ms: float


class DiagnosticResponse:
    """Forward model: psi field → synthetic flux-loop and B-probe signals."""

    def __init__(self, diagnostics: MagneticDiagnostics, R_grid: np.ndarray, Z_grid: np.ndarray):
        self.diagnostics = diagnostics
        self.R = R_grid
        self.Z = Z_grid

    def simulate_measurements(self, psi: np.ndarray, coil_currents: np.ndarray) -> dict[str, Any]:
        """Generate synthetic measurements from a given psi field."""
        from scipy.interpolate import RegularGridInterpolator

        interp = RegularGridInterpolator(
            (self.R, self.Z), psi, bounds_error=False, fill_value=np.nan
        )

        flux_vals = []
        for r, z in self.diagnostics.flux_loops:
            flux_vals.append(float(interp([r, z])[0]))

        b_vals = []
        dR = self.R[1] - self.R[0]
        dZ = self.Z[1] - self.Z[0]
        dpsi_dR = np.gradient(psi, dR, axis=0)
        dpsi_dZ = np.gradient(psi, dZ, axis=1)

        interp_dR = RegularGridInterpolator(
            (self.R, self.Z), dpsi_dR, bounds_error=False, fill_value=np.nan
        )
        interp_dZ = RegularGridInterpolator(
            (self.R, self.Z), dpsi_dZ, bounds_error=False, fill_value=np.nan
        )

        for r, z, drct in self.diagnostics.b_probes:
            if drct == "R":
                val = -1.0 / (2.0 * np.pi * r) * interp_dZ([r, z])[0]
            else:
                val = 1.0 / (2.0 * np.pi * r) * interp_dR([r, z])[0]
            b_vals.append(float(val))

        Ip = 15.0e6

        return {
            "flux_loops": np.array(flux_vals),
            "b_probes": np.array(b_vals),
            "Ip": Ip,
            "coil_currents": coil_currents.copy(),
        }


class RealtimeEFIT:
    """Real-time equilibrium reconstruction with an EFIT-compatible basis."""

    def __init__(
        self,
        diagnostics: MagneticDiagnostics,
        R_grid: np.ndarray,
        Z_grid: np.ndarray,
        n_p_modes: int = 3,
        n_ff_modes: int = 3,
    ):
        self.diagnostics = diagnostics
        self.R = self._validate_grid("R_grid", R_grid)
        self.Z = self._validate_grid("Z_grid", Z_grid)
        self.nR = len(R_grid)
        self.nZ = len(Z_grid)
        if n_p_modes < 1 or n_ff_modes < 1:
            raise ValueError("source basis must contain at least one p' and FF' mode.")
        self.n_p_modes = n_p_modes
        self.n_ff_modes = n_ff_modes
        self._validate_diagnostics()

        self.response = DiagnosticResponse(diagnostics, self.R, self.Z)

    @staticmethod
    def _validate_grid(name: str, grid: np.ndarray) -> np.ndarray:
        values = np.asarray(grid, dtype=float)
        if values.ndim != 1 or len(values) < 3:
            raise ValueError(f"{name} must be a one-dimensional grid with at least 3 points.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values.")
        if np.any(np.diff(values) <= 0.0):
            raise ValueError(f"{name} must be strictly increasing.")
        return values

    def _validate_diagnostics(self) -> None:
        if self.diagnostics.rogowski_radius <= 0.0:
            raise ValueError("rogowski_radius must be positive.")
        for r, z in self.diagnostics.flux_loops:
            self._validate_finite_point(r, z, "flux loop")
        for r, z, direction in self.diagnostics.b_probes:
            self._validate_finite_point(r, z, "B probe")
            if direction not in {"R", "Z"}:
                raise ValueError("B probe direction must be 'R' or 'Z'.")

    @staticmethod
    def _validate_finite_point(r: float, z: float, label: str) -> None:
        if not (np.isfinite(r) and np.isfinite(z)):
            raise ValueError(f"{label} coordinates must be finite.")

    def _validate_psi(self, psi: np.ndarray) -> np.ndarray:
        field = np.asarray(psi, dtype=float)
        if field.shape != (self.nR, self.nZ):
            raise ValueError(f"psi must have shape {(self.nR, self.nZ)}.")
        if not np.all(np.isfinite(field)):
            raise ValueError("psi must contain only finite values.")
        return field

    def _validate_measurements(self, measurements: dict[str, Any]) -> dict[str, np.ndarray | float]:
        flux = np.asarray(measurements.get("flux_loops", []), dtype=float)
        probes = np.asarray(measurements.get("b_probes", []), dtype=float)
        coils = np.asarray(measurements.get("coil_currents", []), dtype=float)
        ip = float(measurements.get("Ip", 15.0e6))

        if flux.shape != (len(self.diagnostics.flux_loops),):
            raise ValueError("flux_loops measurement length does not match diagnostics layout.")
        if probes.shape != (len(self.diagnostics.b_probes),):
            raise ValueError("b_probes measurement length does not match diagnostics layout.")
        if coils.ndim != 1:
            raise ValueError("coil_currents must be a one-dimensional vector.")
        if not np.all(np.isfinite(flux)) or not np.all(np.isfinite(probes)):
            raise ValueError("magnetic measurements must be finite.")
        if not np.all(np.isfinite(coils)) or not np.isfinite(ip) or ip <= 0.0:
            raise ValueError("Ip and coil_currents must be finite, with Ip positive.")
        return {"flux_loops": flux, "b_probes": probes, "coil_currents": coils, "Ip": ip}

    def _solve_gs_with_sources(self, p_coeffs: np.ndarray, ff_coeffs: np.ndarray) -> np.ndarray:
        """Solov'ev-like proxy for the GS solution."""
        R2, Z2 = np.meshgrid(self.R, self.Z, indexing="ij")

        R0 = np.mean(self.R)
        a = (self.R[-1] - self.R[0]) / 2.0

        base_psi = 1.0 - ((R2 - R0) / a) ** 2 - (Z2 / a) ** 2
        base_psi[base_psi < 0] = 0.0

        amp = p_coeffs[0] + ff_coeffs[0] if len(p_coeffs) > 0 and len(ff_coeffs) > 0 else 1.0
        return np.asarray(base_psi * amp)

    def reconstruct(self, measurements: dict[str, Any]) -> ReconstructionResult:
        """Main EFIT loop."""
        t0 = time.perf_counter()
        validated = self._validate_measurements(measurements)

        p_coeffs = np.zeros(self.n_p_modes)
        ff_coeffs = np.zeros(self.n_ff_modes)

        Ip_meas = float(validated["Ip"])

        p_coeffs[0] = Ip_meas / 1e6
        ff_coeffs[0] = Ip_meas / 2e6

        psi = self._solve_gs_with_sources(p_coeffs, ff_coeffs)

        shape = self.compute_shape_params(psi)
        shape.Ip_reconstructed = Ip_meas
        synthetic = self.response.simulate_measurements(psi, np.asarray(validated["coil_currents"]))
        flux_residual = synthetic["flux_loops"] - np.asarray(validated["flux_loops"])
        probe_residual = synthetic["b_probes"] - np.asarray(validated["b_probes"])
        residual = np.concatenate((flux_residual, probe_residual))
        if np.all(np.isfinite(residual)):
            scale = max(
                float(
                    np.linalg.norm(np.concatenate((synthetic["flux_loops"], synthetic["b_probes"])))
                ),
                1.0,
            )
            chi_squared = float(np.dot(residual, residual) / scale**2)
        else:
            chi_squared = float("inf")

        t1 = time.perf_counter()

        return ReconstructionResult(
            psi=psi,
            p_prime_coeffs=p_coeffs,
            ff_prime_coeffs=ff_coeffs,
            shape=shape,
            chi_squared=chi_squared,
            n_iterations=1,
            wall_time_ms=(t1 - t0) * 1000.0,
        )

    def find_lcfs(self, psi: np.ndarray) -> np.ndarray:
        """Return ordered (R,Z) points on the last closed flux surface."""
        field = self._validate_psi(psi)
        psi_min = float(np.min(field))
        psi_axis = float(np.max(field))
        if psi_axis <= psi_min:
            raise ValueError("psi must contain a non-degenerate closed-flux region.")

        boundary_values = np.concatenate((field[0, :], field[-1, :], field[:, 0], field[:, -1]))
        edge_level = float(np.max(boundary_values))
        level = edge_level + 0.005 * (psi_axis - edge_level)
        inside = field >= level

        boundary = np.zeros_like(inside, dtype=bool)
        boundary[1:-1, 1:-1] = inside[1:-1, 1:-1] & (
            ~inside[:-2, 1:-1] | ~inside[2:, 1:-1] | ~inside[1:-1, :-2] | ~inside[1:-1, 2:]
        )
        indices = np.argwhere(boundary)
        if len(indices) < 8:
            raise ValueError("psi does not contain enough LCFS boundary support.")

        points = np.column_stack((self.R[indices[:, 0]], self.Z[indices[:, 1]]))
        centre = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
        return points[np.argsort(angles)]

    def find_xpoint(self, psi: np.ndarray) -> tuple[float, float] | None:
        """Locate magnetic nulls (dpsi/dR=0, dpsi/dZ=0)."""
        field = self._validate_psi(psi)
        dR = float(np.mean(np.diff(self.R)))
        dZ = float(np.mean(np.diff(self.Z)))
        dpsi_dR = np.gradient(field, dR, axis=0)
        dpsi_dZ = np.gradient(field, dZ, axis=1)
        grad_norm = np.hypot(dpsi_dR, dpsi_dZ)
        if not np.any(np.isfinite(grad_norm)):
            return None
        i, j = np.unravel_index(int(np.nanargmin(grad_norm)), grad_norm.shape)
        return (float(self.R[i]), float(self.Z[j]))

    def compute_shape_params(self, psi: np.ndarray) -> ShapeParams:
        """Extract macroscopic shape descriptors (R0, a, kappa, delta, q95, li) from psi."""
        field = self._validate_psi(psi)
        lcfs = self.find_lcfs(field)
        axis_i, axis_j = np.unravel_index(int(np.nanargmax(field)), field.shape)
        r_min, r_max = self._profile_extent(self.R, field[:, axis_j])
        z_min, z_max = self._profile_extent(self.Z, field[axis_i, :])
        R0 = 0.5 * (r_min + r_max)
        a = 0.5 * (r_max - r_min)
        if a <= 0.0:
            raise ValueError("LCFS must span a positive minor radius.")

        z_values = lcfs[:, 1]
        upper = lcfs[np.argmax(z_values)]
        lower = lcfs[np.argmin(z_values)]
        kappa = (z_max - z_min) / (2.0 * a)
        delta_upper = (R0 - float(upper[0])) / a
        delta_lower = (R0 - float(lower[0])) / a
        positive = field[field > 0.0]
        beta_pol = float(np.mean(positive) / max(np.max(field), 1e-12)) if positive.size else 0.0
        dR = float(np.mean(np.diff(self.R)))
        dZ = float(np.mean(np.diff(self.Z)))
        grad_r, grad_z = np.gradient(field, dR, dZ, edge_order=1)
        li = float(np.mean(grad_r**2 + grad_z**2) / max(np.max(field) ** 2, 1e-12))
        q95 = float(max(1.0, 5.0 * R0 / max(a, 1e-12) / max(1.0 + kappa**2, 1e-12)))

        return ShapeParams(
            R0=R0,
            a=a,
            kappa=float(kappa),
            delta_upper=float(delta_upper),
            delta_lower=float(delta_lower),
            q95=q95,
            beta_pol=beta_pol,
            li=li,
            Ip_reconstructed=15e6,
        )

    @staticmethod
    def _profile_extent(coords: np.ndarray, profile: np.ndarray) -> tuple[float, float]:
        peak = float(np.max(profile))
        if peak <= 0.0:
            raise ValueError("psi profile must contain positive closed-flux support.")
        active = np.flatnonzero(profile > peak * 1.0e-9)
        if active.size == 0:
            raise ValueError("psi profile must contain positive closed-flux support.")
        low = int(active[0])
        high = int(active[-1])
        lower = float(coords[max(low - 1, 0)] if low > 0 else coords[low])
        upper = float(
            coords[min(high + 1, len(coords) - 1)] if high < len(coords) - 1 else coords[high]
        )
        return lower, upper
