# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Closed-Contour Geometry Contract
"""Closed poloidal contours and point classification in physical ``(R, Z)``.

One contract serves every consumer of limiter, LCFS and separatrix contours:

* A contour is a simple closed polygon in metres. A repeated closing vertex
  is accepted and dropped. Repeated consecutive vertices, fewer than three
  distinct vertices, zero enclosed area, spikes that fold an edge back onto
  its neighbour, and self-intersections are rejected with ``ValueError``.
* Vertex order may be clockwise or counter-clockwise; it is reported, never
  required.
* A point within ``tolerance_m`` of any edge is ``ON_BOUNDARY``. Otherwise the
  even-odd rule with signed edge crossings decides ``INSIDE`` or ``OUTSIDE``.
  Concave contours are handled by the same rule.

The tolerance is a physical distance, so it does not change when a contour is
translated; scaling a contour and its points scales the decision only through
the caller's choice of tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.core.fusion_kernel_numerics import FloatArray

INSIDE = 1
ON_BOUNDARY = 0
OUTSIDE = -1

DEFAULT_CONTOUR_TOLERANCE_M = 1.0e-9
"""Default boundary tolerance in metres.

It sits far above binary64 rounding at reactor radii (about 2e-15 m at
R = 10 m) and far below any engineering contour precision. Callers comparing
grid-resolved contours must pass a tolerance that matches their resolution.
"""

ContourKind = Literal["limiter", "lcfs", "separatrix"]
CONTOUR_KINDS: tuple[ContourKind, ...] = ("limiter", "lcfs", "separatrix")


@dataclass(frozen=True)
class ClosedContour:
    """Validated simple closed contour in metres.

    Attributes
    ----------
    vertices : FloatArray
        Distinct vertices, shape ``(n, 2)`` with ``n >= 3``, without a repeated
        closing vertex.
    signed_area_m2 : float
        Shoelace area; positive for counter-clockwise order.
    tolerance_m : float
        Distance used for validation and boundary classification.
    """

    vertices: FloatArray
    signed_area_m2: float
    tolerance_m: float

    @property
    def orientation(self) -> Literal["counterclockwise", "clockwise"]:
        """Return the vertex order of the contour."""
        return "counterclockwise" if self.signed_area_m2 > 0.0 else "clockwise"


def _check_tolerance(tolerance_m: float) -> float:
    tol = float(tolerance_m)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tolerance_m must be finite and positive.")
    return tol


def _finite_points(value: Any, *, name: str) -> FloatArray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 1:
        raise ValueError(f"{name} must have shape (n_points, 2) with n_points > 0.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values only.")
    return arr


def _point_segment_distance(points: FloatArray, start: FloatArray, end: FloatArray) -> FloatArray:
    """Return distances from ``points`` (n, 2) to every segment (m, 2) -> (n, m)."""
    edge = end - start
    length_sq = np.einsum("ij,ij->i", edge, edge)
    rel = points[:, None, :] - start[None, :, :]
    t = np.einsum("nmk,mk->nm", rel, edge) / length_sq[None, :]
    t = np.clip(t, 0.0, 1.0)
    nearest = start[None, :, :] + t[:, :, None] * edge[None, :, :]
    return np.asarray(np.linalg.norm(points[:, None, :] - nearest, axis=2), dtype=np.float64)


def _cross(o: FloatArray, a: FloatArray, b: FloatArray) -> FloatArray:
    return np.asarray(
        (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1])
        - (a[..., 1] - o[..., 1]) * (b[..., 0] - o[..., 0]),
        dtype=np.float64,
    )


def _reject_self_intersections(vertices: FloatArray, tol: float, name: str) -> None:
    n = vertices.shape[0]
    start = vertices
    end = np.roll(vertices, -1, axis=0)
    for i in range(n):
        # Adjacent edges share a vertex; a spike folds the far end of one
        # edge back onto the other.
        nxt = (i + 1) % n
        far_on_this = _point_segment_distance(end[nxt : nxt + 1], start[i : i + 1], end[i : i + 1])
        this_on_far = _point_segment_distance(
            start[i : i + 1], start[nxt : nxt + 1], end[nxt : nxt + 1]
        )
        if float(far_on_this[0, 0]) <= tol or float(this_on_far[0, 0]) <= tol:
            raise ValueError(f"{name} folds back on itself at vertex {nxt}.")
        others = np.array(
            [j for j in range(i + 2, n) if not (i == 0 and j == n - 1)], dtype=np.int64
        )
        if others.size == 0:
            continue
        a0 = np.broadcast_to(start[i], (others.size, 2))
        a1 = np.broadcast_to(end[i], (others.size, 2))
        b0 = start[others]
        b1 = end[others]
        d1 = _cross(a0, a1, b0)
        d2 = _cross(a0, a1, b1)
        d3 = _cross(b0, b1, a0)
        d4 = _cross(b0, b1, a1)
        proper = (np.sign(d1) * np.sign(d2) < 0) & (np.sign(d3) * np.sign(d4) < 0)
        # Two segments that do not cross are closest at an endpoint of one.
        near = np.minimum.reduce(
            [
                _point_segment_distance(b0, start[i : i + 1], end[i : i + 1])[:, 0],
                _point_segment_distance(b1, start[i : i + 1], end[i : i + 1])[:, 0],
                _point_segment_distance(start[i : i + 1], b0, b1)[0],
                _point_segment_distance(end[i : i + 1], b0, b1)[0],
            ]
        )
        hit = proper | (near <= tol)
        if np.any(hit):
            j = int(others[np.argmax(hit)])
            raise ValueError(f"{name} is self-intersecting: edges {i} and {j} meet.")


def validate_closed_contour(
    points: Any,
    *,
    name: str = "contour",
    tolerance_m: float = DEFAULT_CONTOUR_TOLERANCE_M,
) -> ClosedContour:
    """Validate a closed ``(R, Z)`` contour and return its canonical form.

    Parameters
    ----------
    points : array_like
        Contour vertices in metres, shape ``(n, 2)``. A final vertex equal to
        the first (within ``tolerance_m``) is treated as the closing point.
    name : str
        Name used in error messages.
    tolerance_m : float
        Physical distance below which two vertices coincide or two edges meet.

    Returns
    -------
    ClosedContour
        Distinct vertices, signed area and tolerance.

    Raises
    ------
    ValueError
        For non-finite or malformed input, repeated consecutive vertices,
        fewer than three distinct vertices, zero area, spikes or
        self-intersections.
    """
    tol = _check_tolerance(tolerance_m)
    arr = _finite_points(points, name=name).copy()
    if arr.shape[0] >= 2 and float(np.linalg.norm(arr[0] - arr[-1])) <= tol:
        arr = arr[:-1]
    if arr.shape[0] < 3:
        raise ValueError(f"{name} must contain at least three distinct vertices.")
    step = np.linalg.norm(np.roll(arr, -1, axis=0) - arr, axis=1)
    repeated = np.flatnonzero(step <= tol)
    if repeated.size:
        raise ValueError(f"{name} repeats vertex {int(repeated[0])} consecutively.")
    x = arr[:, 0]
    y = arr[:, 1]
    signed_area = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    perimeter = float(np.sum(step))
    if abs(signed_area) <= tol * perimeter:
        raise ValueError(f"{name} encloses zero area.")
    _reject_self_intersections(arr, tol, name)
    return ClosedContour(vertices=arr, signed_area_m2=signed_area, tolerance_m=tol)


def classify_points(points: Any, contour: ClosedContour) -> NDArray[np.int8]:
    """Classify points as ``INSIDE``, ``ON_BOUNDARY`` or ``OUTSIDE`` a contour.

    Parameters
    ----------
    points : array_like
        Finite points in metres, shape ``(n, 2)``.
    contour : ClosedContour
        Contour from :func:`validate_closed_contour`; its tolerance is used.

    Returns
    -------
    NDArray[np.int8]
        One code per point: ``1`` inside, ``0`` on the boundary, ``-1`` outside.
    """
    pts = _finite_points(points, name="points")
    start = contour.vertices
    end = np.roll(start, -1, axis=0)
    on_edge = np.min(_point_segment_distance(pts, start, end), axis=1) <= contour.tolerance_m

    x = pts[:, 0][:, None]
    y = pts[:, 1][:, None]
    xa = start[:, 0][None, :]
    ya = start[:, 1][None, :]
    xb = end[:, 0][None, :]
    yb = end[:, 1][None, :]
    straddles = (ya > y) != (yb > y)
    dy = np.where(straddles, yb - ya, 1.0)
    x_cross = xa + (y - ya) * (xb - xa) / dy
    crossings = np.sum(straddles & (x < x_cross), axis=1)
    inside = (crossings % 2) == 1

    codes = np.where(inside, INSIDE, OUTSIDE).astype(np.int8)
    codes[on_edge] = ON_BOUNDARY
    return codes


__all__ = [
    "CONTOUR_KINDS",
    "DEFAULT_CONTOUR_TOLERANCE_M",
    "INSIDE",
    "ON_BOUNDARY",
    "OUTSIDE",
    "ClosedContour",
    "ContourKind",
    "classify_points",
    "validate_closed_contour",
]
