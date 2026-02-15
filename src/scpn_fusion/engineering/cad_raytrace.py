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
        _validate_mesh(vertices, faces)
        return vertices, faces

    if suffix == ".stl":
        vertices, faces = _parse_ascii_stl(mesh_path)
        _validate_mesh(vertices, faces)
        return vertices, faces
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


def _segment_intersects_triangle(
    p0: FloatArray,
    p1: FloatArray,
    tri: FloatArray,
    *,
    epsilon: float,
) -> bool:
    """
    Segment-triangle intersection using Moller-Trumbore parameterization.
    Returns True only when the intersection lies strictly inside (p0, p1).
    """
    d = p1 - p0
    edge1 = tri[1] - tri[0]
    edge2 = tri[2] - tri[0]
    h = np.cross(d, edge2)
    a = float(np.dot(edge1, h))
    if abs(a) <= epsilon:
        return False
    f = 1.0 / a
    s = p0 - tri[0]
    u = f * float(np.dot(s, h))
    if u < 0.0 or u > 1.0:
        return False
    q = np.cross(s, edge1)
    v = f * float(np.dot(d, q))
    if v < 0.0 or (u + v) > 1.0:
        return False
    t = f * float(np.dot(edge2, q))
    return epsilon < t < (1.0 - epsilon)


def _validate_mesh(vertices: FloatArray, faces: IntArray) -> None:
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (M, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (K, 3)")
    if vertices.shape[0] == 0:
        raise ValueError("vertices must contain at least one point.")
    if faces.shape[0] == 0:
        raise ValueError("faces must contain at least one triangle.")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("vertices must contain only finite values.")
    if np.any(faces < 0):
        raise ValueError("faces must use non-negative vertex indices.")
    if np.any(faces >= vertices.shape[0]):
        raise ValueError("faces contain out-of-bounds vertex indices.")

    _, areas = _triangle_normals_and_areas(vertices, faces)
    if not np.all(np.isfinite(areas)):
        raise ValueError("triangle areas must be finite.")
    if np.any(areas <= 0.0):
        raise ValueError("faces must define non-degenerate triangles.")


def estimate_surface_loading(
    vertices: FloatArray,
    faces: IntArray,
    source_points_xyz: FloatArray,
    source_strength_w: FloatArray,
    *,
    occlusion_cull: bool = False,
    occlusion_epsilon: float = 1e-9,
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
    if src.size and not np.all(np.isfinite(src)):
        raise ValueError("source_points_xyz must contain only finite values.")
    if not np.all(np.isfinite(strength)):
        raise ValueError("source_strength_w must contain only finite values.")
    if np.any(strength < 0.0):
        raise ValueError("source_strength_w must be non-negative.")
    if not np.isfinite(occlusion_epsilon) or occlusion_epsilon <= 0.0:
        raise ValueError("occlusion_epsilon must be finite and > 0.")

    _validate_mesh(vertices, faces)

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
        if occlusion_cull:
            visible = cos_incidence > 0.0
            for i in range(faces.shape[0]):
                if not visible[i]:
                    continue
                c = centroids[i]
                for j in range(faces.shape[0]):
                    if i == j:
                        continue
                    if _segment_intersects_triangle(
                        p,
                        c,
                        tri[j],
                        epsilon=occlusion_epsilon,
                    ):
                        visible[i] = False
                        break
            cos_incidence = np.where(visible, cos_incidence, 0.0)
        loading += float(power) * cos_incidence / (4.0 * np.pi * np.maximum(dist2, 1e-12))

    return CADLoadReport(
        face_loading_w_m2=loading,
        peak_loading_w_m2=float(np.max(loading)) if loading.size else 0.0,
        mean_loading_w_m2=float(np.mean(loading)) if loading.size else 0.0,
    )
