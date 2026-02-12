# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Geometry 3D Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Unit tests for 3D flux-surface mesh generation."""

import numpy as np

from scpn_fusion.core.geometry_3d import Reactor3DBuilder


class _DummyKernel:
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

    def find_x_point(self, psi: np.ndarray) -> tuple[tuple[float, float], float]:
        _ = psi
        # Boundary level intersects all rays in this synthetic setup.
        return (2.894, 0.0), 0.2


def test_geometry_mesh_generation_shapes() -> None:
    kernel = _DummyKernel()
    builder = Reactor3DBuilder(kernel=kernel, solve_equilibrium=False)

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


def test_geometry_export_obj(tmp_path) -> None:
    kernel = _DummyKernel()
    builder = Reactor3DBuilder(kernel=kernel, solve_equilibrium=False)
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


def test_geometry_export_preview_png(tmp_path) -> None:
    kernel = _DummyKernel()
    builder = Reactor3DBuilder(kernel=kernel, solve_equilibrium=False)
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
    kernel = _DummyKernel()
    Reactor3DBuilder(kernel=kernel, solve_equilibrium=False)
    assert kernel.solve_calls == 0

    Reactor3DBuilder(kernel=kernel, solve_equilibrium=True)
    assert kernel.solve_calls == 1


def test_generate_coil_mesh_placeholders() -> None:
    kernel = _DummyKernel()
    builder = Reactor3DBuilder(kernel=kernel, solve_equilibrium=False)
    coils = builder.generate_coil_meshes()
    assert len(coils) == 2
    assert coils[0]["name"] == "PF1"
    assert "R" in coils[0] and "Z" in coils[0]
