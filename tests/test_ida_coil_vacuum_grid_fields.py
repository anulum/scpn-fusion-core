# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — Coil-Vacuum Grid Field Tests
"""Behaviour tests for coil-vacuum manifest and numerical field operations."""

from __future__ import annotations

import importlib
import math
import warnings
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from validation import ida_coil_vacuum_grid_fields as fields


def _filament(*, r_m: float = 1.5, z_m: float = 0.0) -> fields.FilamentRecord:
    """Return one explicit filament for manufactured-grid tests."""
    return fields.FilamentRecord(
        parent_index=0,
        parent_name="FC1",
        filament_index=0,
        r_m=r_m,
        z_m=z_m,
        weight=1.0,
        effective_current_a_turns=-1.0e6,
    )


def _manufactured_plane(size: int) -> fields.FloatArray:
    """Return a smooth field with a controlled second-order grid error."""
    axis = np.linspace(0.0, 1.0, size, dtype=np.float64)
    r_mesh, z_mesh = np.meshgrid(axis, axis)
    exact = np.sin(np.pi * r_mesh) * np.sin(np.pi * z_mesh)
    error_shape = 1.0 + r_mesh + 2.0 * z_mesh
    spacing = 1.0 / (size - 1)
    return np.asarray(exact + spacing * spacing * error_shape, dtype=np.float64)


def _parent(
    *,
    filaments: tuple[fields.FilamentRecord, ...] | None = None,
) -> fields.ParentCoilRecord:
    """Return one manufactured parent manifest row."""
    rows = (_filament(),) if filaments is None else filaments
    return fields.ParentCoilRecord(
        index=0,
        name="FC1",
        coil_type="ShapedCoil",
        current_a=-1.0e6,
        turns=1.0,
        effective_current_a_turns=-1.0e6,
        filaments=rows,
    )


def test_real_diiid_manifest_preserves_parent_and_filament_lineage() -> None:
    """The installed public FreeGS DIII-D machine must match the frozen manifest."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        freegs = cast(Any, importlib.import_module("freegs"))
    parents = fields.extract_coil_manifest(freegs.machine.DIIID())
    filaments = fields.validate_frozen_manifest(
        parents,
        r_bounds=(0.1, 2.8),
        z_bounds=(-1.8, 1.8),
    )
    payload = fields.manifest_payload(parents)

    assert [parent.name for parent in parents] == [f"FC{index}" for index in range(1, 19)]
    assert len(filaments) == 216
    assert payload["parent_count"] == 18
    assert payload["filament_count"] == 216
    assert (
        payload["manifest_sha256"]
        == "f53d440fa251e1d9fff1e9b969974721c1655c982efb9b8d9e08a20d78cec8e3"
    )
    assert fields.flatten_filaments(parents) == filaments
    r_values, z_values, currents = fields.filament_arrays(filaments)
    assert r_values.shape == z_values.shape == currents.shape == (216,)
    assert np.isclose(
        np.sum(currents, dtype=np.float64),
        sum(parent.effective_current_a_turns for parent in parents),
    )


def test_nearest_source_geometry_masks_and_phase_are_deterministic() -> None:
    """Nearest-source geometry must preserve nested masks and physical distance."""
    r_grid = np.linspace(1.0, 2.0, 5, dtype=np.float64)
    z_grid = np.linspace(-0.5, 0.5, 5, dtype=np.float64)
    geometry = fields.nearest_source_geometry(
        r_grid,
        z_grid,
        (_filament(r_m=1.625, z_m=0.125),),
    )
    masks = fields.source_masks(geometry, fixed_physical_radius_m=0.2)

    assert geometry.filament_phase_rz == ((0.5, 0.5),)
    assert geometry.nearest_index.shape == (5, 5)
    assert np.all(geometry.nearest_index == 0)
    assert np.all(masks.rho_h_le_1 <= masks.rho_h_le_2)
    assert np.all(masks.rho_h_le_2 <= masks.rho_h_le_4)
    assert np.array_equal(
        masks.fixed_physical,
        geometry.distance_m <= 0.2,
    )
    summary = fields.mask_summary(masks.fixed_physical, d_area_m2=0.0625)
    assert summary["point_count"] == int(np.count_nonzero(masks.fixed_physical))
    assert summary["area_m2"] == pytest.approx(summary["point_count"] * 0.0625)


def test_exact_restriction_recovers_manufactured_second_order() -> None:
    """Nested restriction must recover order two without interpolation."""
    coarse = _manufactured_plane(33)
    medium = _manufactured_plane(65)
    fine = _manufactured_plane(129)
    mask = np.ones((33, 33), dtype=np.bool_)

    restricted = fields.restrict_to_shape(fine, (33, 33))
    assert restricted.shape == coarse.shape
    order = fields.observed_order_from_three(
        coarse,
        medium,
        fine,
        mask=mask,
    )
    assert order["observed_order"] == pytest.approx(2.0, abs=1.0e-12)
    assert order["coarse_to_medium_rms"] > order["medium_to_fine_rms"]


def test_operator_orientation_current_sign_and_wall_integration() -> None:
    """Z-R orientation and ``-Delta*psi/(mu0 R)`` sign must remain explicit."""
    r_grid = np.linspace(1.0, 2.0, 7, dtype=np.float64)
    z_grid = np.linspace(-1.0, 1.0, 5, dtype=np.float64)
    r_mesh, z_mesh = np.meshgrid(r_grid, z_grid)
    psi_zr = np.asarray(r_mesh**4 + 0.5 * z_mesh**2, dtype=np.float64)
    lhs = fields.native_lhs_zr(psi_zr, r_grid=r_grid, z_grid=z_grid)
    current = fields.implied_current_density(
        lhs,
        r_grid=r_grid,
        mu0=2.0,
    )
    expected = np.asarray(-lhs / (2.0 * r_grid[np.newaxis, :]), dtype=np.float64)
    assert lhs.shape == (5, 7)
    assert np.array_equal(current, expected)

    unit_density = np.ones((5, 7), dtype=np.float64)
    full_mask = np.ones((5, 7), dtype=np.bool_)
    assert fields.integrate_current(
        unit_density,
        mask=full_mask,
        d_area_m2=0.5,
    ) == pytest.approx(7.5)
    zero_wall = fields.zero_identity_wall(psi_zr, field="manufactured flux")
    assert np.count_nonzero(zero_wall[[0, -1], :]) == 0
    assert np.count_nonzero(zero_wall[:, [0, -1]]) == 0


def test_field_comparison_localisation_and_recovery_metrics() -> None:
    """Metrics must expose localisation, alignment, and cancellation-safe recovery."""
    reference = np.arange(25, dtype=np.float64).reshape(5, 5)
    actual = np.asarray(reference * 1.5, dtype=np.float64)
    mask = np.zeros((5, 5), dtype=np.bool_)
    mask[1:4, 1:4] = True

    field = fields.field_metric(actual, d_area_m2=0.25)
    comparison = fields.comparison_metric(
        actual,
        reference,
        mask=mask,
        d_area_m2=0.25,
    )
    assert field["linf"] == 36.0
    assert comparison["relative_l2"] == pytest.approx(0.5)
    assert comparison["cosine"] == pytest.approx(1.0)
    assert comparison["projection"] == pytest.approx(1.5)
    assert fields.forcing_l2_fraction(actual, mask=mask) == pytest.approx(
        np.linalg.norm(actual[mask]) / np.linalg.norm(actual)
    )

    positive = fields.recovery_metric(9.0, 10.0)
    negative = fields.recovery_metric(-18.0, -20.0)
    assert positive["signed_error_a_turns"] == -1.0
    assert negative["relative_error"] == 0.1
    assert fields.weighted_recovery_error((positive, negative)) == pytest.approx(3.0 / 30.0)


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (
            lambda: fields.finite_plane(np.ones((2, 2)), field="small"),
            "finite non-trivial",
        ),
        (
            lambda: fields.nearest_source_geometry(
                np.asarray([1.0, 0.5, 2.0]),
                np.asarray([-1.0, 0.0, 1.0]),
                (_filament(),),
            ),
            "uniformly and strictly increasing",
        ),
        (
            lambda: fields.source_masks(
                fields.NearestSourceGeometry(
                    distance_m=np.ones((3, 3)),
                    rho_h=np.ones((3, 3)),
                    nearest_index=np.zeros((3, 3), dtype=np.int64),
                    filament_phase_rz=((0.0, 0.0),),
                ),
                fixed_physical_radius_m=0.0,
            ),
            "positive and finite",
        ),
        (
            lambda: fields.restrict_to_shape(np.ones((6, 6)), (4, 4)),
            "not exactly nested",
        ),
        (
            lambda: fields.comparison_metric(
                np.ones((3, 3)),
                np.ones((3, 3)),
                mask=np.zeros((3, 3), dtype=np.bool_),
                d_area_m2=1.0,
            ),
            "incompatible",
        ),
        (
            lambda: fields.weighted_recovery_error((fields.recovery_metric(0.0, 0.0),)),
            "non-zero",
        ),
    ],
)
def test_public_helpers_fail_closed_on_invalid_geometry(
    operation: Any,
    message: str,
) -> None:
    """Invalid planes, axes, masks, nesting, and recovery must fail closed."""
    with pytest.raises(ValueError, match=message):
        operation()


@pytest.mark.parametrize(
    ("coils", "message"),
    [
        (None, "non-empty top-level coil list"),
        ([], "non-empty top-level coil list"),
        ([("FC1",)], "coil entries"),
        (
            [
                ("FC1", SimpleNamespace(current=1.0, turns=1.0, _points=[(1.0, 0.0, 1.0)])),
                ("FC1", SimpleNamespace(current=1.0, turns=1.0, _points=[(1.0, 0.0, 1.0)])),
            ],
            "names must be non-empty and unique",
        ),
        (
            [("FC1", SimpleNamespace(current=1.0, turns=1.0, _points=[]))],
            "must expose shaped-filament points",
        ),
        (
            [
                (
                    "FC1",
                    SimpleNamespace(current=math.inf, turns=1.0, _points=[(1.0, 0.0, 1.0)]),
                )
            ],
            "current/turns are invalid",
        ),
        (
            [("FC1", SimpleNamespace(current=1.0, turns=0.0, _points=[(1.0, 0.0, 1.0)]))],
            "current/turns are invalid",
        ),
        (
            [("FC1", SimpleNamespace(current=1.0, turns=1.0, _points=[(1.0, 0.0)]))],
            "malformed filament point",
        ),
        (
            [
                (
                    "FC1",
                    SimpleNamespace(current=1.0, turns=1.0, _points=[(math.nan, 0.0, 1.0)]),
                )
            ],
            "non-finite filament",
        ),
        (
            [("FC1", SimpleNamespace(current=1.0, turns=1.0, _points=[(1.0, 0.0, 0.5)]))],
            "filament currents do not close",
        ),
    ],
)
def test_manifest_extraction_rejects_malformed_machine_data(
    coils: object,
    message: str,
) -> None:
    """Machine lineage extraction must reject malformed or non-closing coils."""
    with pytest.raises(ValueError, match=message):
        fields.extract_coil_manifest(SimpleNamespace(coils=coils))


def test_manifest_helpers_reject_incomplete_or_out_of_domain_rows() -> None:
    """Frozen manifest helpers must reject missing groups, counts, bounds, and positions."""
    with pytest.raises(ValueError, match="non-empty filament groups"):
        fields.flatten_filaments(())
    with pytest.raises(ValueError, match="non-empty filament groups"):
        fields.flatten_filaments((_parent(filaments=()),))
    with pytest.raises(ValueError, match="18 parents and 216 filaments"):
        fields.validate_frozen_manifest(
            (_parent(),),
            r_bounds=(0.1, 2.8),
            z_bounds=(-1.8, 1.8),
        )

    parents = tuple(
        fields.ParentCoilRecord(
            index=index,
            name=f"FC{index + 1}",
            coil_type="ShapedCoil",
            current_a=1.0,
            turns=1.0,
            effective_current_a_turns=12.0,
            filaments=tuple(
                fields.FilamentRecord(
                    parent_index=index,
                    parent_name=f"FC{index + 1}",
                    filament_index=filament_index,
                    r_m=1.5,
                    z_m=0.0,
                    weight=1.0 / 12.0,
                    effective_current_a_turns=1.0,
                )
                for filament_index in range(12)
            ),
        )
        for index in range(18)
    )
    with pytest.raises(ValueError, match="domain bounds are invalid"):
        fields.validate_frozen_manifest(
            parents,
            r_bounds=(2.8, 0.1),
            z_bounds=(-1.8, 1.8),
        )
    outside = list(parents)
    outside[0] = _parent(filaments=(_filament(r_m=3.0),) * 12)
    with pytest.raises(ValueError, match="lies outside"):
        fields.validate_frozen_manifest(
            tuple(outside),
            r_bounds=(0.1, 2.8),
            z_bounds=(-1.8, 1.8),
        )
    with pytest.raises(ValueError, match="must not be empty"):
        fields.filament_arrays(())


@pytest.mark.parametrize(
    ("operation", "message", "exception"),
    [
        (
            lambda: fields.nearest_source_geometry(
                np.ones((3, 3)),
                np.arange(3, dtype=np.float64),
                (_filament(),),
            ),
            "one-dimensional",
            ValueError,
        ),
        (
            lambda: fields.nearest_source_geometry(
                np.asarray([1.0, math.nan, 2.0]),
                np.arange(3, dtype=np.float64),
                (_filament(),),
            ),
            "axes must be finite",
            ValueError,
        ),
        (
            lambda: fields.nearest_source_geometry(
                np.arange(3, dtype=np.float64),
                np.arange(3, dtype=np.float64),
                (),
            ),
            "must not be empty",
            ValueError,
        ),
        (
            lambda: fields.nearest_source_geometry(
                np.arange(3, dtype=np.float64),
                np.arange(3, dtype=np.float64),
                (_filament(r_m=math.nan),),
            ),
            "assignment is incomplete",
            RuntimeError,
        ),
        (
            lambda: fields.mask_summary(np.ones(3, dtype=np.bool_), d_area_m2=1.0),
            "inputs are invalid",
            ValueError,
        ),
        (
            lambda: fields.mask_summary(np.ones((3, 3), dtype=np.bool_), d_area_m2=0.0),
            "inputs are invalid",
            ValueError,
        ),
        (
            lambda: fields.native_lhs_zr(
                np.ones((3, 4)),
                r_grid=np.arange(3, dtype=np.float64),
                z_grid=np.arange(3, dtype=np.float64),
            ),
            "shape must match",
            ValueError,
        ),
        (
            lambda: fields.implied_current_density(
                np.ones((3, 3)),
                r_grid=np.asarray([0.0, 1.0, 2.0]),
                mu0=1.0,
            ),
            "forcing/radius geometry",
            ValueError,
        ),
        (
            lambda: fields.implied_current_density(
                np.ones((3, 3)),
                r_grid=np.arange(3, dtype=np.float64) + 1.0,
                mu0=0.0,
            ),
            "mu0 must be positive",
            ValueError,
        ),
        (
            lambda: fields.integrate_current(
                np.ones((3, 3)),
                mask=np.ones((4, 4), dtype=np.bool_),
                d_area_m2=1.0,
            ),
            "mask/area is invalid",
            ValueError,
        ),
        (
            lambda: fields.integrate_current(
                np.ones((3, 3)),
                mask=np.ones((3, 3), dtype=np.bool_),
                d_area_m2=math.nan,
            ),
            "mask/area is invalid",
            ValueError,
        ),
        (
            lambda: fields.field_metric(np.ones((3, 3)), d_area_m2=-1.0),
            "cell area must be positive",
            ValueError,
        ),
        (
            lambda: fields.comparison_metric(
                np.ones((3, 3)),
                np.ones((4, 4)),
                mask=np.ones((3, 3), dtype=np.bool_),
                d_area_m2=1.0,
            ),
            "incompatible",
            ValueError,
        ),
        (
            lambda: fields.comparison_metric(
                np.ones((3, 3)),
                np.ones((3, 3)),
                mask=np.ones((3, 3), dtype=np.bool_),
                d_area_m2=0.0,
            ),
            "incompatible",
            ValueError,
        ),
        (
            lambda: fields.restrict_to_shape(np.ones((5, 5)), (2, 3)),
            "target must be non-trivial",
            ValueError,
        ),
        (
            lambda: fields.observed_order_from_three(
                np.ones((3, 3)),
                np.ones((5, 5)),
                np.ones((9, 9)),
                mask=np.zeros((3, 3), dtype=np.bool_),
            ),
            "mask must select",
            ValueError,
        ),
        (
            lambda: fields.observed_order_from_three(
                np.ones((3, 3)),
                np.ones((5, 5)),
                np.ones((9, 9)),
                mask=np.ones((3, 3), dtype=np.bool_),
            ),
            "differences must be non-zero",
            ValueError,
        ),
        (
            lambda: fields.forcing_l2_fraction(
                np.ones((3, 3)),
                mask=np.ones((4, 4), dtype=np.bool_),
            ),
            "mask shape is invalid",
            ValueError,
        ),
        (
            lambda: fields.forcing_l2_fraction(
                np.zeros((3, 3)),
                mask=np.ones((3, 3), dtype=np.bool_),
            ),
            "norm must be non-zero",
            ValueError,
        ),
        (
            lambda: fields.recovery_metric(math.nan, 1.0),
            "values must be finite",
            ValueError,
        ),
        (
            lambda: fields.weighted_recovery_error(()),
            "must not be empty",
            ValueError,
        ),
    ],
)
def test_numerical_helpers_reject_invalid_inputs(
    operation: Any,
    message: str,
    exception: type[Exception],
) -> None:
    """Numerical helpers must reject invalid grids, masks, norms, and scalars."""
    with pytest.raises(exception, match=message):
        operation()
