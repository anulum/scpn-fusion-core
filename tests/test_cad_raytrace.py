# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — CAD Raytrace Tests

from __future__ import annotations

import struct
import sys
from pathlib import Path
import types
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_fusion.engineering.cad_raytrace import (
    _segment_intersects_triangle,
    estimate_surface_loading,
    load_cad_mesh,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _tetra_mesh() -> tuple[FloatArray, IntArray]:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def _sources() -> tuple[FloatArray, FloatArray]:
    points = np.asarray([[2.0, 2.0, 2.0], [1.5, 2.0, 1.2]], dtype=np.float64)
    strength = np.asarray([4.0e6, 2.0e6], dtype=np.float64)
    return points, strength


def test_estimate_surface_loading_rejects_negative_face_indices() -> None:
    vertices, faces = _tetra_mesh()
    source_points, source_strength = _sources()
    faces[0, 0] = -1
    with pytest.raises(ValueError, match="non-negative"):
        estimate_surface_loading(vertices, faces, source_points, source_strength)


def test_estimate_surface_loading_rejects_out_of_bounds_face_indices() -> None:
    vertices, faces = _tetra_mesh()
    source_points, source_strength = _sources()
    faces[0, 0] = vertices.shape[0]
    with pytest.raises(ValueError, match="out-of-bounds"):
        estimate_surface_loading(vertices, faces, source_points, source_strength)


def test_estimate_surface_loading_rejects_non_finite_vertices() -> None:
    vertices, faces = _tetra_mesh()
    source_points, source_strength = _sources()
    vertices[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        estimate_surface_loading(vertices, faces, source_points, source_strength)


def test_estimate_surface_loading_rejects_non_finite_source_points() -> None:
    vertices, faces = _tetra_mesh()
    source_points, source_strength = _sources()
    source_points[0, 1] = np.inf
    with pytest.raises(ValueError, match="source_points_xyz"):
        estimate_surface_loading(vertices, faces, source_points, source_strength)


def test_estimate_surface_loading_rejects_negative_source_strength() -> None:
    vertices, faces = _tetra_mesh()
    source_points, source_strength = _sources()
    source_strength[0] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        estimate_surface_loading(vertices, faces, source_points, source_strength)


def test_estimate_surface_loading_rejects_degenerate_triangles() -> None:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    source_points = np.asarray([[1.0, 1.0, 1.0]], dtype=np.float64)
    source_strength = np.asarray([1.0], dtype=np.float64)
    with pytest.raises(ValueError, match="non-degenerate"):
        estimate_surface_loading(vertices, faces, source_points, source_strength)


def test_estimate_surface_loading_rejects_invalid_occlusion_epsilon() -> None:
    vertices, faces = _tetra_mesh()
    source_points, source_strength = _sources()
    with pytest.raises(ValueError, match="occlusion_epsilon"):
        estimate_surface_loading(
            vertices,
            faces,
            source_points,
            source_strength,
            occlusion_cull=True,
            occlusion_epsilon=0.0,
        )


def test_estimate_surface_loading_occlusion_culls_shadowed_faces() -> None:
    vertices = np.asarray(
        [
            [0.0, -1.0, -1.0],
            [0.0, 1.0, -1.0],
            [0.0, -1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [1.0, -1.0, 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    source_points = np.asarray([[-2.0, 0.0, 0.0]], dtype=np.float64)
    source_strength = np.asarray([1.0e6], dtype=np.float64)

    report_no_cull = estimate_surface_loading(
        vertices,
        faces,
        source_points,
        source_strength,
        occlusion_cull=False,
    )
    report_cull = estimate_surface_loading(
        vertices,
        faces,
        source_points,
        source_strength,
        occlusion_cull=True,
    )

    assert report_no_cull.face_loading_w_m2[0] > 0.0
    assert report_no_cull.face_loading_w_m2[1] > 0.0
    assert report_cull.face_loading_w_m2[0] > 0.0
    assert report_cull.face_loading_w_m2[1] == pytest.approx(0.0, abs=1e-12)


def test_estimate_surface_loading_broadphase_matches_legacy_occlusion_path() -> None:
    vertices, faces = _tetra_mesh()
    source_points, source_strength = _sources()
    report_legacy = estimate_surface_loading(
        vertices,
        faces,
        source_points,
        source_strength,
        occlusion_cull=True,
        occlusion_broadphase=False,
    )
    report_broadphase = estimate_surface_loading(
        vertices,
        faces,
        source_points,
        source_strength,
        occlusion_cull=True,
        occlusion_broadphase=True,
    )
    np.testing.assert_allclose(
        report_legacy.face_loading_w_m2,
        report_broadphase.face_loading_w_m2,
        rtol=0.0,
        atol=0.0,
    )
    assert report_legacy.peak_loading_w_m2 == pytest.approx(
        report_broadphase.peak_loading_w_m2, rel=0.0, abs=0.0
    )
    assert report_legacy.mean_loading_w_m2 == pytest.approx(
        report_broadphase.mean_loading_w_m2, rel=0.0, abs=0.0
    )


def test_load_cad_mesh_rejects_invalid_extension(tmp_path: Path) -> None:
    bad = tmp_path / "mesh.obj"
    bad.write_text("dummy")
    with pytest.raises(ValueError, match="\\.stl, \\.step, or \\.stp"):
        load_cad_mesh(bad)


def test_load_cad_mesh_ascii_stl_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "trimesh", None)
    stl = tmp_path / "mesh.stl"
    stl.write_text(
        "\n".join(
            [
                "solid demo",
                "facet normal 0 0 1",
                "outer loop",
                "vertex 0 0 0",
                "vertex 1 0 0",
                "vertex 0 1 0",
                "endloop",
                "endfacet",
                "endsolid demo",
            ]
        ),
        encoding="utf-8",
    )
    vertices, faces = load_cad_mesh(stl)
    assert vertices.shape == (3, 3)
    assert faces.shape == (1, 3)


def test_load_cad_mesh_binary_stl_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "trimesh", None)
    stl = tmp_path / "mesh_binary.stl"

    header = b"binary-stl-test".ljust(80, b" ")
    tri_count = 1
    tri = struct.pack(
        "<12fH",
        0.0,
        0.0,
        1.0,  # normal
        0.0,
        0.0,
        0.0,  # v1
        1.0,
        0.0,
        0.0,  # v2
        0.0,
        1.0,
        0.0,  # v3
        0,  # attr byte count
    )
    stl.write_bytes(header + struct.pack("<I", tri_count) + tri)

    vertices, faces = load_cad_mesh(stl)
    assert vertices.shape == (3, 3)
    assert faces.shape == (1, 3)


def test_load_cad_mesh_binary_stl_truncated_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "trimesh", None)
    stl = tmp_path / "mesh_binary_bad.stl"
    # Header + tri_count=1 but no triangle payload.
    stl.write_bytes(b"x" * 80 + struct.pack("<I", 1))
    with pytest.raises(ValueError, match="Binary STL truncated"):
        load_cad_mesh(stl)


def test_load_cad_mesh_ascii_stl_rejects_non_finite_vertices(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "trimesh", None)
    stl = tmp_path / "mesh_nan.stl"
    stl.write_text(
        "\n".join(
            [
                "solid demo",
                "facet normal 0 0 1",
                "outer loop",
                "vertex 0 0 0",
                "vertex nan 0 0",
                "vertex 0 1 0",
                "endloop",
                "endfacet",
                "endsolid demo",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="finite"):
        load_cad_mesh(stl)


class _FakeTrimesh(types.ModuleType):
    """Trimesh module double that returns a deterministic triangular mesh."""

    def load(self, _path: Path, *, force: str) -> Any:
        """Return a mesh object compatible with the production trimesh path."""
        assert force == "mesh"
        return types.SimpleNamespace(
            vertices=np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float64,
            ),
            faces=np.asarray([[0, 1, 2]], dtype=np.int64),
        )


def test_load_cad_mesh_uses_trimesh_when_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The public loader accepts a trimesh-backed mesh when the backend exists."""
    monkeypatch.setitem(sys.modules, "trimesh", _FakeTrimesh("trimesh"))
    stl = tmp_path / "mesh.stl"
    stl.write_text("solid delegated\nendsolid delegated\n", encoding="utf-8")

    vertices, faces = load_cad_mesh(stl)

    assert vertices.shape == (3, 3)
    assert faces.tolist() == [[0, 1, 2]]


def test_load_cad_mesh_step_requires_trimesh(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """STEP and STP inputs fail closed when no trimesh backend is importable."""
    monkeypatch.setitem(sys.modules, "trimesh", None)
    step_file = tmp_path / "blanket.step"
    step_file.write_text("ISO-10303-21;\nEND-ISO-10303-21;\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="requires trimesh"):
        load_cad_mesh(step_file)


def test_load_cad_mesh_short_binary_stl_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A too-short binary STL payload is rejected before triangle parsing."""
    monkeypatch.setitem(sys.modules, "trimesh", None)
    stl = tmp_path / "short_binary.stl"
    stl.write_bytes(b"short")

    with pytest.raises(ValueError, match="Binary STL too short"):
        load_cad_mesh(stl)


def test_load_cad_mesh_empty_binary_stl_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A zero-triangle binary STL payload is rejected as an empty mesh."""
    monkeypatch.setitem(sys.modules, "trimesh", None)
    stl = tmp_path / "empty_binary.stl"
    stl.write_bytes(b"empty".ljust(80, b" ") + struct.pack("<I", 0))

    with pytest.raises(ValueError, match="No triangles parsed"):
        load_cad_mesh(stl)


def test_load_cad_mesh_ignores_malformed_ascii_vertex_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Malformed ASCII vertex rows are ignored before binary fallback rejects."""
    monkeypatch.setitem(sys.modules, "trimesh", None)
    stl = tmp_path / "malformed_ascii.stl"
    stl.write_text("vertex 0 0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Binary STL too short"):
        load_cad_mesh(stl)


def test_estimate_surface_loading_rejects_invalid_mesh_shapes() -> None:
    """The public loading estimator rejects invalid vertex and face shapes."""
    source_points, source_strength = _sources()
    with pytest.raises(ValueError, match="vertices"):
        estimate_surface_loading(
            np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
            np.asarray([[0, 1, 2]], dtype=np.int64),
            source_points,
            source_strength,
        )
    with pytest.raises(ValueError, match="faces"):
        estimate_surface_loading(
            np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
            np.asarray([0, 1, 2], dtype=np.int64),
            source_points,
            source_strength,
        )


def test_estimate_surface_loading_rejects_empty_mesh_parts() -> None:
    """The public loading estimator rejects empty vertex and face containers."""
    source_points, source_strength = _sources()
    with pytest.raises(ValueError, match="vertices"):
        estimate_surface_loading(
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.int64),
            source_points,
            source_strength,
        )
    with pytest.raises(ValueError, match="faces"):
        estimate_surface_loading(
            np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
            np.empty((0, 3), dtype=np.int64),
            source_points,
            source_strength,
        )


def test_estimate_surface_loading_rejects_non_finite_triangle_area() -> None:
    """Overflowed triangle area calculations fail closed."""
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0e308, 0.0, 0.0], [0.0, 1.0e308, 0.0]],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    source_points = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64)
    source_strength = np.asarray([1.0], dtype=np.float64)

    with pytest.raises(ValueError, match="triangle areas"):
        estimate_surface_loading(vertices, faces, source_points, source_strength)


def test_estimate_surface_loading_rejects_source_shape_and_strength_size() -> None:
    """Source geometry and strength arrays must agree in row count."""
    vertices, faces = _tetra_mesh()
    with pytest.raises(ValueError, match="source_points_xyz"):
        estimate_surface_loading(
            vertices,
            faces,
            np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
            np.asarray([1.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="source_strength_w length"):
        estimate_surface_loading(
            vertices,
            faces,
            np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64),
            np.asarray([1.0, 2.0], dtype=np.float64),
        )


def test_estimate_surface_loading_rejects_non_finite_source_strength() -> None:
    """Source strength values must be finite before loading is integrated."""
    vertices, faces = _tetra_mesh()
    source_points = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64)
    source_strength = np.asarray([np.nan], dtype=np.float64)

    with pytest.raises(ValueError, match="source_strength_w"):
        estimate_surface_loading(vertices, faces, source_points, source_strength)


def test_segment_intersection_rejects_parallel_segment() -> None:
    """The segment-triangle primitive rejects a segment parallel to the triangle."""
    triangle = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    p0 = np.asarray([0.1, 0.1, 1.0], dtype=np.float64)
    p1 = np.asarray([0.9, 0.1, 1.0], dtype=np.float64)

    assert not _segment_intersects_triangle(
        p0,
        p1,
        triangle,
        epsilon=1.0e-9,
    )
