# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Geometry 3D Tests
"""Unit tests for 3D flux-surface mesh generation."""

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from scpn_fusion.core.equilibrium_3d import VMECStyleEquilibrium3D
from scpn_fusion.core.geometry_3d import Reactor3DBuilder


class _DummyKernel:
    """Synthetic kernel with a smooth flux map for mesh generation."""

    def __init__(self) -> None:
        self.NR = 81
        self.NZ = 81
        self.R = np.linspace(1.0, 3.0, self.NR)
        self.Z = np.linspace(-1.0, 1.0, self.NZ)
        self.dR = float(self.R[1] - self.R[0])
        self.dZ = float(self.Z[1] - self.Z[0])
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)

        # Axis at (R=2.0, Z=0.0), smooth radial decay.
        radius2 = (self.RR - 2.0) ** 2 + self.ZZ**2
        self.Psi = 1.0 - radius2

        self.cfg = {
            "coils": [
                {"name": "PF1", "r": 2.5, "z": 0.3},
                {"name": "PF2", "r": 1.5, "z": -0.3},
            ]
        }
        self.solve_calls = 0

    def solve_equilibrium(self) -> None:
        self.solve_calls += 1

    def find_x_point(self, psi: np.ndarray[Any, Any]) -> tuple[tuple[float, float], float]:
        _ = psi
        # Boundary level intersects all rays in this synthetic setup.
        return (2.894, 0.0), 0.2


class _SparseBoundaryKernel(_DummyKernel):
    """Kernel whose boundary flux yields sparse contour crossings."""

    def find_x_point(self, psi: np.ndarray[Any, Any]) -> tuple[tuple[float, float], float]:
        _ = psi
        # Force a sparse/non-crossing contour for some angles to exercise fallback.
        return (2.894, 0.0), -0.5


class _EdgeAxisKernel(_DummyKernel):
    """Kernel with the magnetic axis near the inboard edge."""

    def __init__(self) -> None:
        super().__init__()
        # Move magnetic axis near the inboard boundary so many rays immediately
        # leave the domain and trigger low-point fallback behavior.
        radius2 = (self.RR - 1.08) ** 2 + self.ZZ**2
        self.Psi = 1.0 - radius2

    def find_x_point(self, psi: np.ndarray[Any, Any]) -> tuple[tuple[float, float], float]:
        _ = psi
        return (1.45, 0.0), 0.2


def test_geometry_mesh_generation_shapes() -> None:
    """Mesh generation returns vertex and face arrays of the expected shape."""
    kernel = _DummyKernel()
    builder = Reactor3DBuilder(kernel=cast(Any, kernel), solve_equilibrium=False)

    vertices, faces = builder.generate_plasma_surface(
        resolution_toroidal=12,
        resolution_poloidal=18,
        radial_steps=256,
    )

    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3
    assert vertices.shape[0] > 0
    assert faces.shape[0] > 0
    assert np.isfinite(vertices).all()
    assert np.issubdtype(faces.dtype, np.integer)

    n_tor = 12
    assert vertices.shape[0] % n_tor == 0
    n_pol = vertices.shape[0] // n_tor
    assert n_pol >= 8
    assert faces.shape[0] == 2 * n_tor * n_pol
    assert int(faces.max()) < vertices.shape[0]
    assert int(faces.min()) >= 0


def test_geometry_export_obj(tmp_path: Path) -> None:
    """OBJ export writes a mesh file to disk."""
    kernel = _DummyKernel()
    builder = Reactor3DBuilder(kernel=cast(Any, kernel), solve_equilibrium=False)
    vertices, faces = builder.generate_plasma_surface(
        resolution_toroidal=10,
        resolution_poloidal=16,
        radial_steps=192,
    )

    out_path = tmp_path / "plasma_test.obj"
    written = builder.export_obj(vertices, faces, out_path)
    text = written.read_text(encoding="utf-8").splitlines()

    assert written.exists()
    assert text[0].startswith("# SCPN 3D Plasma Export")
    assert sum(1 for line in text if line.startswith("v ")) == len(vertices)
    assert sum(1 for line in text if line.startswith("f ")) == len(faces)


def test_geometry_export_preview_png(tmp_path: Path) -> None:
    """PNG preview export writes a preview image."""
    kernel = _DummyKernel()
    builder = Reactor3DBuilder(kernel=cast(Any, kernel), solve_equilibrium=False)
    vertices, faces = builder.generate_plasma_surface(
        resolution_toroidal=10,
        resolution_poloidal=16,
        radial_steps=192,
    )
    png_path = tmp_path / "plasma_preview.png"
    written = builder.export_preview_png(vertices, faces, png_path)
    assert written.exists()
    assert written.stat().st_size > 0


def test_geometry_constructor_solve_flag() -> None:
    """The solve_equilibrium flag controls whether the kernel is solved."""
    kernel = _DummyKernel()
    Reactor3DBuilder(kernel=cast(Any, kernel), solve_equilibrium=False)
    assert kernel.solve_calls == 0

    Reactor3DBuilder(kernel=cast(Any, kernel), solve_equilibrium=True)
    assert kernel.solve_calls == 1


def test_geometry_lcfs_sparse_crossings_fallback() -> None:
    """Sparse contour crossings fall back to an ellipse."""
    kernel = _SparseBoundaryKernel()
    builder = Reactor3DBuilder(kernel=cast(Any, kernel), solve_equilibrium=False)

    n_tor = 12
    n_pol = 12
    vertices, faces = builder.generate_plasma_surface(
        resolution_toroidal=n_tor,
        resolution_poloidal=n_pol,
        radial_steps=128,
    )

    assert vertices.shape[0] == n_tor * n_pol
    assert faces.shape[0] == 2 * n_tor * n_pol
    assert np.isfinite(vertices).all()


def test_generate_coil_mesh_placeholders() -> None:
    """Coil-mesh generation returns placeholder coil geometries."""
    kernel = _DummyKernel()
    builder = Reactor3DBuilder(kernel=cast(Any, kernel), solve_equilibrium=False)
    coils = builder.generate_coil_meshes(n_toroidal_samples=36)
    assert len(coils) == 2
    assert coils[0]["name"] == "PF1"
    assert "R" in coils[0] and "Z" in coils[0]
    assert coils[0]["coil_type"] == "toroidal_loop"
    verts = coils[0]["vertices_xyz"]
    assert isinstance(verts, np.ndarray)
    assert verts.shape == (36, 3)
    assert cast(float, coils[0]["centerline_length_m"]) > 0.0


def test_generate_coil_mesh_rejects_too_few_samples() -> None:
    """Too few toroidal samples are rejected."""
    kernel = _DummyKernel()
    builder = Reactor3DBuilder(kernel=cast(Any, kernel), solve_equilibrium=False)
    with pytest.raises(ValueError, match="n_toroidal_samples"):
        builder.generate_coil_meshes(n_toroidal_samples=8)


def test_geometry_lcfs_low_point_edge_case_fallback() -> None:
    """An edge-axis kernel triggers the low-point fallback."""
    kernel = _EdgeAxisKernel()
    builder = Reactor3DBuilder(kernel=cast(Any, kernel), solve_equilibrium=False)

    n_pol = 24
    lcfs = builder._trace_lcfs(resolution_poloidal=n_pol, radial_steps=64)

    # Fallback path should still return a full closed contour.
    assert lcfs.shape == (n_pol, 2)
    assert np.isfinite(lcfs).all()
    assert np.all((lcfs[:, 0] >= kernel.R[0]) & (lcfs[:, 0] <= kernel.R[-1]))
    assert np.all((lcfs[:, 1] >= kernel.Z[0]) & (lcfs[:, 1] <= kernel.Z[-1]))


def test_build_stellarator_w7x_like_equilibrium_nonaxisymmetric() -> None:
    """The W7-X-like build yields a non-axisymmetric equilibrium."""
    base_eq = VMECStyleEquilibrium3D(
        r_axis=2.0,
        z_axis=0.0,
        a_minor=0.5,
        kappa=1.6,
        triangularity=0.2,
        nfp=1,
    )
    builder = Reactor3DBuilder(equilibrium_3d=base_eq, solve_equilibrium=False)
    eq = builder.build_stellarator_w7x_like_equilibrium(
        nfp=5,
        edge_ripple=0.10,
        vertical_ripple=0.06,
    )

    assert eq.nfp == 5
    assert len(eq.modes) >= 3

    rho = np.array([1.0, 1.0], dtype=float)
    theta = np.array([0.45 * np.pi, 0.45 * np.pi], dtype=float)
    phi = np.array([0.0, 0.6], dtype=float)
    r_val, _, _ = eq.flux_to_cylindrical(rho, theta, phi)
    assert abs(float(r_val[0] - r_val[1])) > 1e-4


def test_geometry_main_cli_exports_mesh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI entry point builds and exports an OBJ mesh and PNG preview."""
    import matplotlib.pyplot as plt

    import scpn_fusion.core.geometry_3d as geometry_3d

    monkeypatch.setattr(geometry_3d, "FusionKernel", lambda _config: _DummyKernel())
    monkeypatch.setattr(plt, "savefig", lambda *a, **k: None)
    monkeypatch.setattr(plt, "close", lambda *a, **k: None)

    obj_path = tmp_path / "mesh.obj"
    png_path = tmp_path / "preview.png"
    code = geometry_3d.main(
        [
            "--config",
            "ignored.json",
            "--output",
            str(obj_path),
            "--toroidal",
            "10",
            "--poloidal",
            "14",
            "--radial-steps",
            "64",
            "--preview-png",
            str(png_path),
        ]
    )
    assert code == 0
    assert obj_path.exists()


def test_generate_plasma_surface_rejects_low_toroidal_resolution() -> None:
    """A toroidal resolution below three is rejected."""
    builder = Reactor3DBuilder(kernel=cast(Any, _DummyKernel()), solve_equilibrium=False)
    with pytest.raises(ValueError, match="resolution_toroidal"):
        builder.generate_plasma_surface(resolution_toroidal=2)


def test_build_vmec_like_rejects_low_resolutions() -> None:
    """Sub-minimum LCFS or radial resolutions are rejected by the VMEC build."""
    builder = Reactor3DBuilder(kernel=cast(Any, _DummyKernel()), solve_equilibrium=False)
    with pytest.raises(ValueError, match="lcfs_resolution"):
        builder.build_vmec_like_equilibrium(lcfs_resolution=4)
    with pytest.raises(ValueError, match="radial_steps"):
        builder.build_vmec_like_equilibrium(lcfs_resolution=16, radial_steps=8)


def test_generate_coil_meshes_rejects_too_few_samples() -> None:
    """Too few toroidal coil samples are rejected."""
    builder = Reactor3DBuilder(kernel=cast(Any, _DummyKernel()), solve_equilibrium=False)
    with pytest.raises(ValueError, match="n_toroidal_samples"):
        builder.generate_coil_meshes(n_toroidal_samples=4)


def test_generate_coil_meshes_rejects_invalid_coil_radius() -> None:
    """A coil with a non-positive major radius is rejected."""
    kernel = _DummyKernel()
    kernel.cfg = {"coils": [{"name": "BAD", "r": -1.0, "z": 0.0}]}
    builder = Reactor3DBuilder(kernel=cast(Any, kernel), solve_equilibrium=False)
    with pytest.raises(ValueError, match="invalid major radius"):
        builder.generate_coil_meshes()


def test_sample_psi_outside_domain_rejected() -> None:
    """Sampling the flux map outside the kernel domain is rejected."""
    builder = Reactor3DBuilder(kernel=cast(Any, _DummyKernel()), solve_equilibrium=False)
    with pytest.raises(ValueError, match="outside kernel domain"):
        builder._sample_psi_bilinear(100.0, 100.0)


def test_kernel_required_operation_without_kernel() -> None:
    """A kernel-backed operation on an equilibrium-only builder raises."""
    eq = VMECStyleEquilibrium3D(
        r_axis=2.0, z_axis=0.0, a_minor=0.5, kappa=1.6, triangularity=0.2, nfp=1
    )
    builder = Reactor3DBuilder(equilibrium_3d=eq, solve_equilibrium=False)
    with pytest.raises(RuntimeError, match="requires a 2D kernel-backed builder"):
        builder._sample_psi_bilinear(2.0, 0.0)


def test_generate_coil_meshes_rejects_invalid_vertical_position() -> None:
    """A coil with a non-finite vertical position is rejected."""
    kernel = _DummyKernel()
    kernel.cfg = {"coils": [{"name": "BAD_Z", "r": 2.0, "z": float("nan")}]}
    builder = Reactor3DBuilder(kernel=cast(Any, kernel), solve_equilibrium=False)
    with pytest.raises(ValueError, match="invalid vertical position"):
        builder.generate_coil_meshes()
