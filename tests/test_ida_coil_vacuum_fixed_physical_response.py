# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — Fixed-Physical Coil-Vacuum Response Tests
"""Numerical tests for the fixed-physical forcing and response partition."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from validation import ida_coil_vacuum_fixed_physical_response as fixed
from validation import ida_coil_vacuum_grid_contract as grid_contract
from validation import ida_coil_vacuum_grid_convergence as convergence
from validation import ida_coil_vacuum_grid_fields as fields


def _masks(resolution: int) -> tuple[np.ndarray[Any, Any], ...]:
    """Return nested interior, fixed-source-free, and plasma masks."""
    r_grid = np.linspace(*grid_contract.R_BOUNDS_M, resolution, dtype=np.float64)
    z_grid = np.linspace(*grid_contract.Z_BOUNDS_M, resolution, dtype=np.float64)
    r_mesh, z_mesh = np.meshgrid(r_grid, z_grid)
    interior = np.ones((resolution, resolution), dtype=np.bool_)
    interior[[0, -1], :] = False
    interior[:, [0, -1]] = False
    fixed_source = (r_mesh - 1.45) ** 2 + z_mesh**2 <= 0.225**2
    fixed_source_free = np.asarray(~fixed_source & interior, dtype=np.bool_)
    plasma = np.asarray(
        (r_mesh >= 0.8) & (r_mesh <= 2.1) & (z_mesh >= -0.9) & (z_mesh <= 0.9),
        dtype=np.bool_,
    )
    plasma &= interior
    return interior, fixed_source_free, plasma


def _grid(resolution: int = 33) -> convergence.GridResult:
    """Build one grid with a linear reference response for partition tests."""
    r_grid = np.linspace(*grid_contract.R_BOUNDS_M, resolution, dtype=np.float64)
    z_grid = np.linspace(*grid_contract.Z_BOUNDS_M, resolution, dtype=np.float64)
    r_mesh, z_mesh = np.meshgrid(r_grid, z_grid)
    forcing = np.asarray(
        np.sin(np.pi * (r_mesh - r_grid[0]) / np.ptp(r_grid))
        * np.sin(np.pi * (z_mesh - z_grid[0]) / np.ptp(z_grid)),
        dtype=np.float64,
    )
    forcing[[0, -1], :] = 0.0
    forcing[:, [0, -1]] = 0.0
    zero = np.zeros_like(forcing)
    interior, fixed_source_free, plasma = _masks(resolution)
    spacing_r = float(r_grid[1] - r_grid[0])
    spacing_z = float(z_grid[1] - z_grid[0])
    return convergence.GridResult(
        resolution=resolution,
        report={
            "current_recovery": {"weighted_fixed_physical_error": 0.01},
            "grid": {"cell_area_m2": spacing_r * spacing_z},
        },
        total_forcing_zr=forcing,
        source_forcing_zr=zero,
        source_free_forcing_zr=forcing,
        total_response_zr=0.5 * forcing,
        source_response_zr=zero,
        source_free_response_zr=0.5 * forcing,
        interior_mask=interior,
        primary_source_mask=np.asarray(~fixed_source_free & interior, dtype=np.bool_),
        fixed_source_free_mask=fixed_source_free,
        plasma_support_mask=plasma,
    )


def test_build_fixed_physical_grid_partitions_before_inverse_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public builder must pass one closed fixed partition to the inverse surface."""
    grid = _grid()

    def linear_inverse(
        total: np.ndarray[Any, Any],
        source: np.ndarray[Any, Any],
        source_free: np.ndarray[Any, Any],
        *,
        r_grid: np.ndarray[Any, Any],
        z_grid: np.ndarray[Any, Any],
    ) -> tuple[dict[str, Any], dict[str, np.ndarray[Any, Any]], dict[str, float]]:
        assert np.array_equal(total, source + source_free)
        d_area = float((r_grid[1] - r_grid[0]) * (z_grid[1] - z_grid[0]))
        responses = {
            "source": 0.5 * source,
            "source_free": 0.5 * source_free,
            "total": 0.5 * total,
        }
        report = {
            "closure_max_abs_wb": 0.0,
            **{
                name: fields.field_metric(value, d_area_m2=d_area)
                for name, value in responses.items()
            },
        }
        return (
            report,
            responses,
            {
                "inverse_source": 1.0,
                "inverse_source_free": 1.0,
                "inverse_total": 1.0,
            },
        )

    monkeypatch.setattr(fixed, "partition_inverse_response", linear_inverse)
    measured = fixed.build_fixed_physical_grid(grid)

    assert measured.report["forcing_closure_max_abs"] == 0.0
    assert measured.report["response_partition"]["closure_max_abs_wb"] <= 1.0e-12
    assert measured.report["total_response_reproduction_max_abs_wb"] <= 1.0e-12
    assert 0.0 < measured.report["fixed_source_l2_fraction"] < 1.0
    assert measured.report["source_free_response_fraction_of_total"] > 0.0
    assert np.array_equal(
        measured.source_forcing_zr + measured.source_free_forcing_zr,
        grid.total_forcing_zr,
    )


def _convergence_result(resolution: int) -> fixed.FixedPhysicalGridResult:
    """Return a nested second-order zero-limit response field."""
    r_grid = np.linspace(*grid_contract.R_BOUNDS_M, resolution, dtype=np.float64)
    z_grid = np.linspace(*grid_contract.Z_BOUNDS_M, resolution, dtype=np.float64)
    r_mesh, z_mesh = np.meshgrid(r_grid, z_grid)
    spacing = 1.0 / (resolution - 1)
    error = np.asarray(spacing**2 * (1.0 + r_mesh + z_mesh), dtype=np.float64)
    error[[0, -1], :] = 0.0
    error[:, [0, -1]] = 0.0
    interior, fixed_source_free, plasma = _masks(resolution)
    return fixed.FixedPhysicalGridResult(
        resolution=resolution,
        report={},
        source_forcing_zr=np.zeros_like(error),
        source_free_forcing_zr=error,
        source_response_zr=np.zeros_like(error),
        source_free_response_zr=0.25 * error,
        total_response_zr=0.25 * error,
        interior_mask=interior,
        fixed_source_free_mask=fixed_source_free,
        plasma_support_mask=plasma,
    )


def test_fixed_physical_convergence_uses_exact_restriction_and_zero_limit_order() -> None:
    """Both forcing and response orders must recover second order without interpolation."""
    measured = fixed.build_fixed_physical_convergence(
        [_convergence_result(size) for size in grid_contract.GRID_RESOLUTIONS]
    )

    assert set(measured["pairwise"]) == {"33_65", "65_129", "129_257"}
    assert set(measured["pairwise"]["33_65"]) == set(fixed.PAIRWISE_REGIONS)
    for surface in ("source_free_forcing_order", "source_free_response_order"):
        for row in measured[surface].values():
            assert row["observed_order"] == pytest.approx(2.0, abs=1.0e-11)


@pytest.mark.parametrize("resolutions", [[33, 65, 129], [65, 33, 129, 257]])
def test_fixed_physical_convergence_rejects_incomplete_or_reordered_ladders(
    resolutions: list[int],
) -> None:
    """No partial or reordered ladder may become CVGC2 evidence."""
    with pytest.raises(ValueError, match="exact ordered four-grid ladder"):
        fixed.build_fixed_physical_convergence([_convergence_result(size) for size in resolutions])


def test_fixed_physical_convergence_rejects_shared_mask_drift() -> None:
    """A physical mask change on a shared node must fail before comparison."""
    results = [_convergence_result(size) for size in grid_contract.GRID_RESOLUTIONS]
    results[1].fixed_source_free_mask[4, 4] = ~results[1].fixed_source_free_mask[4, 4]
    with pytest.raises(ValueError, match="fixed physical source-free mask drifted"):
        fixed.build_fixed_physical_convergence(results)


def test_fixed_physical_grid_rejects_mask_drift_and_zero_response_denominator() -> None:
    """Malformed masks and a zero response denominator must fail closed."""
    grid = _grid()
    grid.fixed_source_free_mask[0, 0] = True
    with pytest.raises(ValueError, match="inside the interior"):
        fixed.build_fixed_physical_grid(grid)

    with pytest.raises(ValueError, match="total response L2 must be finite and positive"):
        fixed._response_fraction(1.0, 0.0)
    with pytest.raises(ValueError, match="source-free response L2 must be finite and non-negative"):
        fixed._response_fraction(-1.0, 1.0)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("resolution", "frozen four-grid ladder"),
        ("forcing_shape", "forcing shape must match"),
        ("mask_shape", "mask shape must match"),
        ("plasma_outside", "plasma support mask must be inside"),
    ],
)
def test_fixed_physical_grid_rejects_declared_shape_and_mask_inconsistency(
    case: str,
    message: str,
) -> None:
    """Resolution, field shape, and both physical masks are independently bound."""
    grid = _grid()
    if case == "resolution":
        object.__setattr__(grid, "resolution", 17)
    elif case == "forcing_shape":
        object.__setattr__(grid, "total_forcing_zr", np.zeros((3, 3), dtype=np.float64))
    elif case == "mask_shape":
        object.__setattr__(grid, "plasma_support_mask", np.zeros((3, 3), dtype=np.bool_))
    elif case == "plasma_outside":
        grid.plasma_support_mask[0, 0] = True
    else:
        raise AssertionError(f"unhandled case {case}")
    with pytest.raises(ValueError, match=message):
        fixed.build_fixed_physical_grid(grid)


def test_fixed_physical_grid_rejects_recomputed_total_response_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The unchanged total inverse must reproduce the CVGC1 response."""
    grid = _grid()

    def drifted_inverse(
        total: np.ndarray[Any, Any],
        source: np.ndarray[Any, Any],
        source_free: np.ndarray[Any, Any],
        *,
        r_grid: np.ndarray[Any, Any],
        z_grid: np.ndarray[Any, Any],
    ) -> tuple[dict[str, Any], dict[str, np.ndarray[Any, Any]], dict[str, float]]:
        del source, source_free
        d_area = float((r_grid[1] - r_grid[0]) * (z_grid[1] - z_grid[0]))
        drifted = total + 1.0
        metric = fields.field_metric(drifted, d_area_m2=d_area)
        return (
            {"closure_max_abs_wb": 0.0, "source": metric, "source_free": metric, "total": metric},
            {"source": drifted, "source_free": drifted, "total": drifted},
            {"inverse_source": 1.0, "inverse_source_free": 1.0, "inverse_total": 1.0},
        )

    monkeypatch.setattr(fixed, "partition_inverse_response", drifted_inverse)
    with pytest.raises(ValueError, match="total response drifted"):
        fixed.build_fixed_physical_grid(grid)
