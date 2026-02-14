# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — CAD Raytrace
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Reduced CAD mesh ray-tracing utilities (STEP/STL integration lane)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class CADLoadReport:
    face_loading_w_m2: FloatArray
    peak_loading_w_m2: float
    mean_loading_w_m2: float


def _parse_ascii_stl(path: Path) -> tuple[FloatArray, IntArray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    current: list[int] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line.lower().startswith("vertex"):
                continue
            parts = line.split()
            if len(parts) != 4:
                continue
            xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
            vertices.append(xyz)
            current.append(len(vertices) - 1)
            if len(current) == 3:
                faces.append(current)
                current = []
    if not faces:
        raise ValueError(f"No triangles parsed from STL: {path}")
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def load_cad_mesh(path: str | Path) -> tuple[FloatArray, IntArray]:
    """
    Load CAD mesh from STL/STEP using trimesh when available.
    Falls back to ASCII STL parser when trimesh is not available.
    """
    mesh_path = Path(path)
    suffix = mesh_path.suffix.lower()
    if suffix not in {".stl", ".step", ".stp"}:
        raise ValueError("CAD mesh path must end with .stl, .step, or .stp")

    try:
        import trimesh  # type: ignore[import-not-found]
    except Exception:
        trimesh = None

    if trimesh is not None:
        mesh = trimesh.load(mesh_path, force="mesh")
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if vertices.size == 0 or faces.size == 0:
            raise ValueError(f"Mesh load failed or empty geometry: {mesh_path}")
        return vertices, faces

    if suffix == ".stl":
        return _parse_ascii_stl(mesh_path)
    raise RuntimeError("STEP/STP loading requires trimesh backend.")


def _triangle_normals_and_areas(
    vertices: FloatArray,
    faces: IntArray,
) -> tuple[FloatArray, FloatArray]:
    tri = vertices[faces]
    e1 = tri[:, 1, :] - tri[:, 0, :]
    e2 = tri[:, 2, :] - tri[:, 0, :]
    cross = np.cross(e1, e2)
    area2 = np.linalg.norm(cross, axis=1)
    areas = 0.5 * area2
    normals = cross / np.maximum(area2[:, None], 1e-12)
    return normals, areas


def estimate_surface_loading(
    vertices: FloatArray,
    faces: IntArray,
    source_points_xyz: FloatArray,
    source_strength_w: FloatArray,
) -> CADLoadReport:
    """
    Compute reduced line-of-sight heat loading on CAD triangles.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    src = np.asarray(source_points_xyz, dtype=np.float64)
    strength = np.asarray(source_strength_w, dtype=np.float64).reshape(-1)
    if src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("source_points_xyz must have shape (N, 3)")
    if strength.size != src.shape[0]:
        raise ValueError("source_strength_w length must match source_points_xyz rows")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (M, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (K, 3)")

    tri = vertices[faces]
    centroids = np.mean(tri, axis=1)
    normals, _areas = _triangle_normals_and_areas(vertices, faces)

    loading = np.zeros(faces.shape[0], dtype=np.float64)
    for p, power in zip(src, strength):
        ray = centroids - p[None, :]
        dist2 = np.sum(ray * ray, axis=1)
        dist = np.sqrt(np.maximum(dist2, 1e-12))
        dirs = ray / dist[:, None]
        cos_incidence = np.clip(np.sum(normals * dirs, axis=1), 0.0, 1.0)
        loading += float(max(power, 0.0)) * cos_incidence / (4.0 * np.pi * np.maximum(dist2, 1e-12))

    return CADLoadReport(
        face_loading_w_m2=loading,
        peak_loading_w_m2=float(np.max(loading)) if loading.size else 0.0,
        mean_loading_w_m2=float(np.mean(loading)) if loading.size else 0.0,
    )
