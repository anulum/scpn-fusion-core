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
from typing import Iterable

import numpy as np


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
