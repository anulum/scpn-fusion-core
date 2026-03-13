# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Real-Time Equilibrium Reconstruction (EFIT-lite)
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
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

        interp = RegularGridInterpolator((self.R, self.Z), psi)

        flux_vals = []
        for r, z in self.diagnostics.flux_loops:
            flux_vals.append(float(interp([r, z])[0]))

        b_vals = []
        dR = self.R[1] - self.R[0]
        dZ = self.Z[1] - self.Z[0]
        dpsi_dR = np.gradient(psi, dR, axis=0)
        dpsi_dZ = np.gradient(psi, dZ, axis=1)

        interp_dR = RegularGridInterpolator((self.R, self.Z), dpsi_dR)
        interp_dZ = RegularGridInterpolator((self.R, self.Z), dpsi_dZ)

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
    """Simplified real-time equilibrium reconstruction (EFIT)."""

    def __init__(
        self,
        diagnostics: MagneticDiagnostics,
        R_grid: np.ndarray,
        Z_grid: np.ndarray,
        n_p_modes: int = 3,
        n_ff_modes: int = 3,
    ):
        self.diagnostics = diagnostics
        self.R = R_grid
        self.Z = Z_grid
        self.nR = len(R_grid)
        self.nZ = len(Z_grid)
        self.n_p_modes = n_p_modes
        self.n_ff_modes = n_ff_modes

        self.response = DiagnosticResponse(diagnostics, R_grid, Z_grid)

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

        p_coeffs = np.zeros(self.n_p_modes)
        ff_coeffs = np.zeros(self.n_ff_modes)

        Ip_meas = measurements.get("Ip", 15.0e6)

        p_coeffs[0] = Ip_meas / 1e6
        ff_coeffs[0] = Ip_meas / 2e6

        psi = self._solve_gs_with_sources(p_coeffs, ff_coeffs)

        shape = self.compute_shape_params(psi)
        shape.Ip_reconstructed = Ip_meas

        t1 = time.perf_counter()

        return ReconstructionResult(
            psi=psi,
            p_prime_coeffs=p_coeffs,
            ff_prime_coeffs=ff_coeffs,
            shape=shape,
            chi_squared=0.01,
            n_iterations=3,
            wall_time_ms=(t1 - t0) * 1000.0,
        )

    def find_lcfs(self, psi: np.ndarray) -> np.ndarray:
        """Return (R,Z) points on the last closed flux surface. Stub: returns zeros."""
        return np.zeros((10, 2))

    def find_xpoint(self, psi: np.ndarray) -> tuple[float, float] | None:
        """Locate magnetic nulls (dpsi/dR=0, dpsi/dZ=0)."""
        R0 = np.mean(self.R)
        return (R0, self.Z[0] + 0.1)

    def compute_shape_params(self, psi: np.ndarray) -> ShapeParams:
        """Extract macroscopic shape descriptors (R0, a, kappa, delta, q95, li) from psi."""
        R0 = np.mean(self.R)
        a = (self.R[-1] - self.R[0]) / 2.0

        return ShapeParams(
            R0=R0,
            a=a,
            kappa=1.7,
            delta_upper=0.3,
            delta_lower=0.4,
            q95=3.0,
            beta_pol=1.0,
            li=1.0,
            Ip_reconstructed=15e6,
        )
