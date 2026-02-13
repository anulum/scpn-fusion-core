# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — 3D Field-Line Tracing
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Reduced 3D field-line tracing and Poincare diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from scpn_fusion.core.equilibrium_3d import VMECStyleEquilibrium3D


@dataclass(frozen=True)
class FieldLineTrace3D:
    """Sampled field-line trajectory in native and Cartesian coordinates."""

    rho: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    xyz: np.ndarray


@dataclass(frozen=True)
class PoincareSection3D:
    """Poincare section points for one toroidal cut plane."""

    phi_plane: float
    xyz: np.ndarray
    rz: np.ndarray


@dataclass(frozen=True)
class ToroidalAsymmetryObservables3D:
    """Reduced toroidal asymmetry indicators for instability-aware control."""

    n1_amp: float
    n2_amp: float
    n3_amp: float
    z_n1_amp: float
    asymmetry_index: float
    radial_spread: float

    def as_dict(self) -> dict[str, float]:
        return {
            "toroidal_n1_amp": self.n1_amp,
            "toroidal_n2_amp": self.n2_amp,
            "toroidal_n3_amp": self.n3_amp,
            "toroidal_z_n1_amp": self.z_n1_amp,
            "toroidal_asymmetry_index": self.asymmetry_index,
            "toroidal_radial_spread": self.radial_spread,
        }


class FieldLineTracer3D:
    """Trace reduced field lines over a VMEC-like 3D equilibrium surface."""

    def __init__(
        self,
        equilibrium: VMECStyleEquilibrium3D,
        *,
        rotational_transform: float | Callable[[float], float] = 0.45,
        helical_coupling_scale: float = 0.08,
        radial_coupling_scale: float = 0.0,
    ) -> None:
        self.equilibrium = equilibrium
        self.rotational_transform = rotational_transform
        self.helical_coupling_scale = float(helical_coupling_scale)
        self.radial_coupling_scale = float(radial_coupling_scale)

    def _iota(self, rho: float) -> float:
        if callable(self.rotational_transform):
            return float(self.rotational_transform(float(rho)))
        return float(self.rotational_transform)

    def _helical_drive(self, theta: float, phi: float) -> float:
        if not self.equilibrium.modes:
            return 0.0

        drive = 0.0
        for mode in self.equilibrium.modes:
            amp = (
                abs(float(mode.r_cos))
                + abs(float(mode.r_sin))
                + abs(float(mode.z_cos))
                + abs(float(mode.z_sin))
            )
            if amp <= 0.0:
                continue
            phase = float(mode.m) * theta - float(mode.n * self.equilibrium.nfp) * phi
            drive += amp * np.sin(phase)
        return float(drive)

    def trace_line(
        self,
        *,
        rho0: float = 0.95,
        theta0: float = 0.0,
        phi0: float = 0.0,
        toroidal_turns: int = 16,
        steps_per_turn: int = 256,
    ) -> FieldLineTrace3D:
        """Trace one reduced field line in ``(rho, theta, phi)`` coordinates."""
        if toroidal_turns < 1:
            raise ValueError("toroidal_turns must be >= 1.")
        if steps_per_turn < 16:
            raise ValueError("steps_per_turn must be >= 16.")

        n_steps = int(toroidal_turns * steps_per_turn)
        dphi = float(2.0 * np.pi * toroidal_turns / n_steps)

        rho = np.zeros(n_steps + 1, dtype=float)
        theta = np.zeros(n_steps + 1, dtype=float)
        phi = np.zeros(n_steps + 1, dtype=float)

        rho[0] = float(np.clip(rho0, 0.0, 1.25))
        theta[0] = float(theta0)
        phi[0] = float(phi0)

        for k in range(n_steps):
            drive = self._helical_drive(theta[k], phi[k])
            iota = self._iota(rho[k]) + self.helical_coupling_scale * drive
            drho_dphi = self.radial_coupling_scale * drive

            rho[k + 1] = float(np.clip(rho[k] + dphi * drho_dphi, 0.0, 1.25))
            theta[k + 1] = float(theta[k] + dphi * iota)
            phi[k + 1] = float(phi[k] + dphi)

        x, y, z = self.equilibrium.flux_to_cartesian(rho, theta, phi)
        xyz = np.stack([x, y, z], axis=1).astype(float)
        return FieldLineTrace3D(rho=rho, theta=theta, phi=phi, xyz=xyz)

    @staticmethod
    def _normalize_plane(phi_plane: float) -> float:
        return float(np.mod(phi_plane, 2.0 * np.pi))

    def poincare_section(
        self,
        trace: FieldLineTrace3D,
        *,
        phi_plane: float = 0.0,
    ) -> PoincareSection3D:
        """Compute Poincare intersection points for one toroidal plane."""
        plane = self._normalize_plane(phi_plane)
        twopi = 2.0 * np.pi

        xyz_points: list[list[float]] = []
        rz_points: list[list[float]] = []

        for k in range(trace.phi.size - 1):
            phi0 = float(trace.phi[k])
            phi1 = float(trace.phi[k + 1])
            if phi1 <= phi0:
                continue

            band0 = int(np.floor((phi0 - plane) / twopi))
            band1 = int(np.floor((phi1 - plane) / twopi))
            if band1 <= band0:
                continue

            target_phi = plane + twopi * float(band0 + 1)
            frac = float((target_phi - phi0) / (phi1 - phi0 + 1e-15))
            frac = float(np.clip(frac, 0.0, 1.0))

            rho_cross = float(trace.rho[k] + frac * (trace.rho[k + 1] - trace.rho[k]))
            theta_cross = float(
                trace.theta[k] + frac * (trace.theta[k + 1] - trace.theta[k])
            )
            r_cross, z_cross, _ = self.equilibrium.flux_to_cylindrical(
                rho_cross,
                theta_cross,
                target_phi,
            )
            x_cross, y_cross, z_cart = self.equilibrium.flux_to_cartesian(
                rho_cross,
                theta_cross,
                target_phi,
            )
            xyz_points.append([float(x_cross), float(y_cross), float(z_cart)])
            rz_points.append([float(r_cross), float(z_cross)])

        if xyz_points:
            xyz = np.asarray(xyz_points, dtype=float)
            rz = np.asarray(rz_points, dtype=float)
        else:
            xyz = np.zeros((0, 3), dtype=float)
            rz = np.zeros((0, 2), dtype=float)
        return PoincareSection3D(phi_plane=plane, xyz=xyz, rz=rz)

    def poincare_map(
        self,
        trace: FieldLineTrace3D,
        *,
        phi_planes: Iterable[float],
    ) -> dict[float, PoincareSection3D]:
        """Compute Poincare sections for multiple toroidal planes."""
        sections: dict[float, PoincareSection3D] = {}
        for plane in phi_planes:
            section = self.poincare_section(trace, phi_plane=float(plane))
            sections[section.phi_plane] = section
        return sections

    def toroidal_asymmetry_observables(
        self,
        trace: FieldLineTrace3D,
    ) -> ToroidalAsymmetryObservables3D:
        """Extract reduced toroidal asymmetry observables from a field-line trace."""
        if trace.xyz.shape[0] < 8:
            raise ValueError("Trace must contain at least 8 samples.")

        phi = np.mod(np.asarray(trace.phi, dtype=float), 2.0 * np.pi)
        r = np.hypot(trace.xyz[:, 0], trace.xyz[:, 1]).astype(float)
        z = trace.xyz[:, 2].astype(float)

        r_centered = r - float(np.mean(r))
        z_centered = z - float(np.mean(z))
        r_scale = float(np.std(r_centered)) + 1e-12
        z_scale = float(np.std(z_centered)) + 1e-12

        def _mode_amp(values: np.ndarray, n: int, scale: float) -> float:
            cos_term = float(np.mean(values * np.cos(float(n) * phi)))
            sin_term = float(np.mean(values * np.sin(float(n) * phi)))
            return float(np.hypot(cos_term, sin_term) / scale)

        n1 = _mode_amp(r_centered, 1, r_scale)
        n2 = _mode_amp(r_centered, 2, r_scale)
        n3 = _mode_amp(r_centered, 3, r_scale)
        z_n1 = _mode_amp(z_centered, 1, z_scale)
        asymmetry_index = float(np.sqrt(n1**2 + n2**2 + n3**2 + z_n1**2))
        radial_spread = float(np.std(r))

        return ToroidalAsymmetryObservables3D(
            n1_amp=n1,
            n2_amp=n2,
            n3_amp=n3,
            z_n1_amp=z_n1,
            asymmetry_index=asymmetry_index,
            radial_spread=radial_spread,
        )
