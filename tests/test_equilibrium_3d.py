# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — 3D Equilibrium Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for VMEC-like reduced 3D equilibrium interface."""

from __future__ import annotations

import numpy as np

from scpn_fusion.core.equilibrium_3d import FourierMode3D, VMECStyleEquilibrium3D
from scpn_fusion.core.geometry_3d import Reactor3DBuilder


class _DummyKernel:
    def __init__(self) -> None:
        self.NR = 81
        self.NZ = 81
        self.R = np.linspace(1.0, 3.0, self.NR)
        self.Z = np.linspace(-1.0, 1.0, self.NZ)
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)
        radius2 = (self.RR - 2.0) ** 2 + self.ZZ**2
        self.Psi = 1.0 - radius2
        self.cfg = {"coils": []}

    def solve_equilibrium(self) -> None:
        return None

    def find_x_point(self, psi: np.ndarray) -> tuple[tuple[float, float], float]:
        _ = psi
        return (2.894, 0.0), 0.2


def test_vmec_like_axisymmetric_is_phi_periodic() -> None:
    eq = VMECStyleEquilibrium3D(
        r_axis=2.0,
        z_axis=0.0,
        a_minor=0.5,
        kappa=1.6,
        triangularity=0.2,
    )
    theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    rho = np.ones_like(theta)
    phi0 = np.zeros_like(theta)
    phi2 = np.full_like(theta, 2.0 * np.pi)

    r0, z0, _ = eq.flux_to_cylindrical(rho, theta, phi0)
    r2, z2, _ = eq.flux_to_cylindrical(rho, theta, phi2)

    assert np.allclose(r0, r2, atol=1e-10)
    assert np.allclose(z0, z2, atol=1e-10)


def test_vmec_like_n1_mode_introduces_toroidal_variation() -> None:
    eq = VMECStyleEquilibrium3D(
        r_axis=2.0,
        z_axis=0.0,
        a_minor=0.5,
        kappa=1.4,
        modes=[FourierMode3D(m=1, n=1, r_cos=0.08, z_sin=0.05)],
    )
    theta = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    rho = np.ones_like(theta)

    r_a, z_a, _ = eq.flux_to_cylindrical(rho, theta, np.zeros_like(theta))
    r_b, z_b, _ = eq.flux_to_cylindrical(
        rho, theta, np.full_like(theta, 0.5 * np.pi)
    )

    assert float(np.max(np.abs(r_a - r_b))) > 1e-3
    assert float(np.max(np.abs(z_a - z_b))) > 1e-3


def test_vmec_like_surface_sampling_shapes() -> None:
    eq = VMECStyleEquilibrium3D(
        r_axis=2.0,
        z_axis=0.0,
        a_minor=0.45,
        kappa=1.7,
        modes=[FourierMode3D(m=2, n=1, r_cos=0.03, z_sin=0.02)],
    )
    vertices, faces = eq.sample_surface(
        resolution_toroidal=14,
        resolution_poloidal=18,
    )

    assert vertices.shape == (14 * 18, 3)
    assert faces.shape == (2 * 14 * 18, 3)
    assert np.isfinite(vertices).all()
    assert np.issubdtype(faces.dtype, np.integer)


def test_builder_supports_native_3d_equilibrium_without_kernel() -> None:
    eq = VMECStyleEquilibrium3D(
        r_axis=2.0,
        z_axis=0.0,
        a_minor=0.45,
        kappa=1.3,
    )
    builder = Reactor3DBuilder(equilibrium_3d=eq, solve_equilibrium=False)
    vertices, faces = builder.generate_plasma_surface(
        resolution_toroidal=10,
        resolution_poloidal=12,
    )

    assert vertices.shape == (10 * 12, 3)
    assert faces.shape == (2 * 10 * 12, 3)
    assert builder.generate_coil_meshes() == []


def test_builder_vmec_like_from_axisymmetric_lcfs() -> None:
    kernel = _DummyKernel()
    builder = Reactor3DBuilder(kernel=kernel, solve_equilibrium=False)
    eq3d = builder.build_vmec_like_equilibrium(
        toroidal_modes=[FourierMode3D(m=1, n=1, r_cos=0.05)],
        lcfs_resolution=48,
        radial_steps=256,
    )

    assert isinstance(eq3d, VMECStyleEquilibrium3D)
    assert eq3d.a_minor > 0.0
    assert eq3d.kappa > 0.0

    builder.equilibrium_3d = eq3d
    vertices, faces = builder.generate_plasma_surface(
        resolution_toroidal=12,
        resolution_poloidal=16,
    )
    assert vertices.shape == (12 * 16, 3)
    assert faces.shape == (2 * 12 * 16, 3)
