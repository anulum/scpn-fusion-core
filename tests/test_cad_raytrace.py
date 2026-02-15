# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — CAD Raytrace Tests
# ──────────────────────────────────────────────────────────────────────

from __future__ import annotations

import sys

import numpy as np
import pytest

from scpn_fusion.engineering.cad_raytrace import estimate_surface_loading, load_cad_mesh


def _tetra_mesh() -> tuple[np.ndarray, np.ndarray]:
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


def _sources() -> tuple[np.ndarray, np.ndarray]:
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


def test_load_cad_mesh_rejects_invalid_extension(tmp_path) -> None:
    bad = tmp_path / "mesh.obj"
    bad.write_text("dummy")
    with pytest.raises(ValueError, match="\\.stl, \\.step, or \\.stp"):
        load_cad_mesh(bad)


def test_load_cad_mesh_ascii_stl_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
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


def test_load_cad_mesh_ascii_stl_rejects_non_finite_vertices(
    monkeypatch: pytest.MonkeyPatch, tmp_path
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
