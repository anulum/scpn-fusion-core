# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Closed-Contour Geometry Contract Tests
"""Limiter/LCFS contour contract checked against independent geometry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_fusion.core.contour_geometry import (
    DEFAULT_CONTOUR_TOLERANCE_M,
    INSIDE,
    ON_BOUNDARY,
    OUTSIDE,
    classify_points,
    validate_closed_contour,
)
from scpn_fusion.core.fusion_kernel import CoilSet
from scpn_fusion.core.fusion_kernel_free_boundary import reconstruct_boundary_flux_from_coils

ROOT = Path(__file__).resolve().parents[1]
SPARC_DEVICE = ROOT / "validation" / "reference_data" / "sparc" / "device_description.json"

AUDIT_TRIANGLE = np.array([[4.0, -1.0], [6.0, -1.0], [5.0, 1.0]])
# Concave "C" contour opening towards +R; the notch spans R in (1, 3), |Z| < 1.
CONCAVE_C = np.array(
    [
        [0.0, -2.0],
        [3.0, -2.0],
        [3.0, -1.0],
        [1.0, -1.0],
        [1.0, 1.0],
        [3.0, 1.0],
        [3.0, 2.0],
        [0.0, 2.0],
    ]
)


def _winding_number(point: NDArray[np.float64], vertices: NDArray[np.float64]) -> int:
    """Independent oracle: total signed angle swept around ``point``."""
    rel = vertices - point
    angles = np.arctan2(rel[:, 1], rel[:, 0])
    turns = np.diff(np.append(angles, angles[0]))
    turns = (turns + np.pi) % (2.0 * np.pi) - np.pi
    return int(round(float(np.sum(turns)) / (2.0 * np.pi)))


def _sparc_limiter() -> NDArray[np.float64]:
    data = json.loads(SPARC_DEVICE.read_text(encoding="utf-8"))
    outline = data["wall"]["description_2d"][0]["limiter"]["unit"][0]["outline"]
    return np.column_stack([outline["r"], outline["z"]]).astype(np.float64)


@pytest.mark.parametrize("order", [1, -1])
def test_audit_triangle_contains_its_interior_point_in_both_orders(order: int) -> None:
    contour = validate_closed_contour(AUDIT_TRIANGLE[::order])
    # Analytic cross-section at Z = 0 is R in [4.5, 5.5].
    codes = classify_points([[5.0, 0.0], [4.6, 0.0], [5.4, 0.0], [4.4, 0.0], [5.6, 0.0]], contour)
    assert codes.tolist() == [INSIDE, INSIDE, INSIDE, OUTSIDE, OUTSIDE]
    assert contour.orientation == ("counterclockwise" if order == 1 else "clockwise")


def test_public_reconstruction_reports_audit_point_inside_triangle_limiter() -> None:
    coils = CoilSet(positions=[(5.0, 3.0)], currents=np.array([1.0e3]), turns=[1])
    for limiter in (AUDIT_TRIANGLE, AUDIT_TRIANGLE[::-1]):
        result = reconstruct_boundary_flux_from_coils(
            None, coils, boundary_points=np.array([[5.0, 0.0]]), limiter_points=limiter
        )
        assert result["boundary_containment_fraction"] == 1.0
        assert result["boundary_containment_pass"] is True
        assert result["boundary_points_outside_limiter"] == 0


def test_edges_and_vertices_are_on_boundary_within_tolerance() -> None:
    contour = validate_closed_contour(AUDIT_TRIANGLE)
    tol = DEFAULT_CONTOUR_TOLERANCE_M
    points = [
        [4.0, -1.0],  # vertex
        [5.0, 1.0],  # apex vertex
        [5.0, -1.0],  # horizontal edge
        [4.5, 0.0],  # sloped edge
        [5.0, -1.0 - 0.5 * tol],  # just outside, within tolerance
        [5.0, -1.0 - 10.0 * tol],  # outside beyond tolerance
        [5.0, -1.0 + 10.0 * tol],  # inside beyond tolerance
    ]
    assert classify_points(points, contour).tolist() == [
        ON_BOUNDARY,
        ON_BOUNDARY,
        ON_BOUNDARY,
        ON_BOUNDARY,
        ON_BOUNDARY,
        OUTSIDE,
        INSIDE,
    ]


def test_concave_contour_matches_independent_winding_oracle() -> None:
    contour = validate_closed_contour(CONCAVE_C)
    rng = np.random.default_rng(20261101)
    points = rng.uniform([-0.5, -2.5], [3.5, 2.5], size=(400, 2))
    codes = classify_points(points, contour)
    for point, code in zip(points, codes):
        if code == ON_BOUNDARY:
            continue
        assert (code == INSIDE) == (_winding_number(point, CONCAVE_C) != 0)
    # The notch is outside although it lies between two arms of the contour.
    assert classify_points([[2.0, 0.0], [0.5, 0.0]], contour).tolist() == [OUTSIDE, INSIDE]


@pytest.mark.parametrize(
    ("shift", "scale"),
    [((0.0, 0.0), 1.0), ((7.25, -3.5), 1.0), ((0.0, 0.0), 0.01), ((2.0, 1.0), 40.0)],
)
def test_translation_and_scaling_preserve_classification(
    shift: tuple[float, float], scale: float
) -> None:
    base_points = np.array([[2.0, 0.0], [0.5, 0.0], [0.5, 1.5], [3.5, 0.0], [1.0, 0.0]])
    expected = [OUTSIDE, INSIDE, INSIDE, OUTSIDE, ON_BOUNDARY]
    offset = np.array(shift)
    contour = validate_closed_contour(CONCAVE_C * scale + offset)
    assert classify_points(base_points * scale + offset, contour).tolist() == expected


def test_repeated_closing_vertex_is_dropped() -> None:
    closed = np.vstack([AUDIT_TRIANGLE, AUDIT_TRIANGLE[:1]])
    contour = validate_closed_contour(closed)
    assert contour.vertices.shape == (3, 2)


@pytest.mark.parametrize(
    ("points", "message"),
    [
        ([[0.0, 0.0], [1.0, 0.0]], "at least three distinct vertices"),
        ([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], "repeats vertex"),
        ([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], "zero area"),
        ([[0.0, 0.0], [2.0, 0.0], [1.0, 0.0], [1.0, 1.0]], "folds back"),
        ([[0.0, 0.0], [3.0, 0.0], [0.0, 1.0], [2.0, 1.5]], "self-intersecting"),
        ([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [1.0, 0.0], [0.0, 2.0]], "self-intersecting"),
        ([[0.0, 0.0], [1.0, np.nan], [0.0, 1.0]], "finite values"),
        ([0.0, 1.0, 2.0], "shape"),
    ],
)
def test_malformed_contours_are_rejected(points: list[object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_closed_contour(points)


@pytest.mark.parametrize("tolerance", [0.0, -1.0, float("nan"), float("inf")])
def test_tolerance_must_be_finite_and_positive(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolerance_m"):
        validate_closed_contour(AUDIT_TRIANGLE, tolerance_m=tolerance)


def test_sparc_limiter_contour_matches_independent_oracle() -> None:
    limiter = _sparc_limiter()
    contour = validate_closed_contour(limiter, name="SPARC limiter")
    rng = np.random.default_rng(7)
    points = rng.uniform([1.0, -1.8], [2.6, 1.8], size=(600, 2))
    for order in (1, -1):
        reordered = validate_closed_contour(limiter[::order], name="SPARC limiter")
        codes = classify_points(points, reordered)
        for point, code in zip(points, codes):
            if code == ON_BOUNDARY:
                continue
            assert (code == INSIDE) == (_winding_number(point, contour.vertices) != 0)
    assert classify_points([[1.85, 0.0], [3.0, 0.0], [0.5, 0.0]], contour).tolist() == [
        INSIDE,
        OUTSIDE,
        OUTSIDE,
    ]


def test_limited_plasma_touching_limiter_counts_as_contained() -> None:
    coils = CoilSet(positions=[(5.0, 3.0)], currents=np.array([1.0e3]), turns=[1])
    limiter = np.array([[4.0, -1.0], [6.0, -1.0], [6.0, 1.0], [4.0, 1.0]])
    touching = np.array([[4.0, 0.0], [5.0, 0.5], [5.5, 0.0], [5.0, -0.5]])
    result = reconstruct_boundary_flux_from_coils(
        None, coils, boundary_points=touching, limiter_points=limiter, boundary_kind="lcfs"
    )
    assert result["boundary_containment_pass"] is True
    assert result["boundary_points_on_limiter"] == 1
    assert result["limiter_orientation"] == "counterclockwise"


def test_computational_box_wall_is_not_tested_against_the_limiter() -> None:
    coils = CoilSet(positions=[(5.0, 3.0)], currents=np.array([1.0e3]), turns=[1])
    limiter = np.array([[4.0, -1.0], [6.0, -1.0], [6.0, 1.0], [4.0, 1.0]])
    box = np.array([[3.0, -2.0], [7.0, -2.0], [7.0, 2.0], [3.0, 2.0]])
    result = reconstruct_boundary_flux_from_coils(
        None,
        coils,
        boundary_points=box,
        limiter_points=limiter,
        boundary_kind="computational_box_wall",
    )
    assert result["limiter_containment_applicable"] is False
    assert result["boundary_containment_fraction"] is None
    assert result["boundary_containment_pass"] is None


def test_unknown_boundary_kind_and_malformed_limiter_are_rejected() -> None:
    coils = CoilSet(positions=[(5.0, 3.0)], currents=np.array([1.0e3]), turns=[1])
    unknown_kind: Any = "wall"
    with pytest.raises(ValueError, match="boundary_kind"):
        reconstruct_boundary_flux_from_coils(
            None,
            coils,
            boundary_points=np.array([[5.0, 0.0]]),
            boundary_kind=unknown_kind,
        )
    with pytest.raises(ValueError, match="self-intersecting"):
        reconstruct_boundary_flux_from_coils(
            None,
            coils,
            boundary_points=np.array([[5.0, 0.0]]),
            limiter_points=np.array([[4.0, -1.0], [7.0, 1.0], [4.0, 1.5], [6.0, -1.0]]),
        )
