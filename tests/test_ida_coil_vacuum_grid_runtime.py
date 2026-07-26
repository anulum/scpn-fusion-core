# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — Coil-Vacuum Grid Runtime Tests
"""Runtime and nested-grid tests for the coil-vacuum diagnostic."""

from __future__ import annotations

import importlib
import warnings
from typing import Any, cast

import numpy as np
import pytest

from validation import ida_coil_vacuum_grid_contract as contract
from validation import ida_coil_vacuum_grid_convergence as convergence
from validation import ida_coil_vacuum_grid_fields as fields
from validation import ida_coil_vacuum_grid_runtime as runtime


def _manufactured_plane(size: int, *, scale: float = 1.0) -> fields.FloatArray:
    """Return a smooth plane with one controlled second-order error term."""
    axis = np.linspace(0.0, 1.0, size, dtype=np.float64)
    r_mesh, z_mesh = np.meshgrid(axis, axis)
    exact = np.sin(np.pi * r_mesh) * np.sin(np.pi * z_mesh)
    error_shape = 1.0 + r_mesh + z_mesh
    spacing = 1.0 / (size - 1)
    return np.asarray(scale * (exact + spacing * spacing * error_shape), dtype=np.float64)


def _result(resolution: int) -> convergence.GridResult:
    """Return one required-grid result with deterministic private arrays."""
    forcing = _manufactured_plane(resolution)
    response = _manufactured_plane(resolution, scale=0.25)
    interior = np.ones((resolution, resolution), dtype=np.bool_)
    interior[[0, -1], :] = False
    interior[:, [0, -1]] = False
    axis = np.linspace(0.0, 1.0, resolution, dtype=np.float64)
    r_mesh, z_mesh = np.meshgrid(axis, axis)
    plasma_support = np.asarray(
        (r_mesh >= 0.25) & (r_mesh <= 0.75) & (z_mesh >= 0.25) & (z_mesh <= 0.75),
        dtype=np.bool_,
    )
    spacing = 1.0 / (resolution - 1)
    return convergence.GridResult(
        resolution=resolution,
        report={
            "grid": {
                "cell_area_m2": spacing * spacing,
            }
        },
        total_forcing_zr=forcing,
        source_forcing_zr=forcing,
        source_free_forcing_zr=forcing,
        total_response_zr=response,
        source_response_zr=response,
        source_free_response_zr=response,
        interior_mask=interior.copy(),
        primary_source_mask=interior.copy(),
        fixed_source_free_mask=interior.copy(),
        plasma_support_mask=plasma_support,
    )


def test_build_convergence_uses_exact_nested_restriction() -> None:
    """The ordered four-grid result must expose all pairs and order-two triples."""
    measured = convergence.build_convergence(
        [_result(resolution) for resolution in contract.GRID_RESOLUTIONS]
    )

    assert set(measured["pairwise"]) == {"33_65", "65_129", "129_257"}
    assert set(measured["pairwise"]["33_65"]) == set(convergence.PAIRWISE_REGIONS)
    assert set(measured["source_free_forcing_order"]) == set(contract.GRID_TRIPLES)
    for row in measured["source_free_forcing_order"].values():
        assert row["observed_order"] == pytest.approx(2.0, abs=1.0e-12)
    assert measured["finest_source_free_response"]["relative_l2"] > 0.0


@pytest.mark.parametrize(
    "resolutions",
    [
        [33, 65, 129],
        [65, 33, 129, 257],
        [33, 65, 129, 129],
    ],
)
def test_build_convergence_rejects_incomplete_reordered_or_duplicate_ladders(
    resolutions: list[int],
) -> None:
    """No missing, reordered, or duplicate grid may enter convergence evidence."""
    with pytest.raises(ValueError, match="exact ordered four-grid ladder"):
        convergence.build_convergence([_result(resolution) for resolution in resolutions])


@pytest.mark.parametrize(
    ("mask_name", "message"),
    [
        ("fixed_source_free_mask", "fixed physical source-free mask drifted"),
        ("interior_mask", "full interior mask drifted"),
        ("plasma_support_mask", "plasma support mask drifted"),
    ],
)
def test_build_convergence_rejects_physical_mask_drift_on_shared_nodes(
    mask_name: str,
    message: str,
) -> None:
    """Physical comparison regions must be identical at every nested shared node."""
    results = [_result(resolution) for resolution in contract.GRID_RESOLUTIONS]
    fine_mask = cast(fields.BoolArray, getattr(results[1], mask_name))
    fine_mask[4, 4] = ~fine_mask[4, 4]
    with pytest.raises(ValueError, match=message):
        convergence.build_convergence(results)


def test_run_grid_rejects_unregistered_resolution_before_execution() -> None:
    """Only the four frozen resolutions may reach expensive field execution."""
    with pytest.raises(ValueError, match="four required grids"):
        runtime.run_grid(
            resolution=17,
            tokamak=object(),
            parents=(),
            r_bounds=(0.1, 2.8),
            z_bounds=(-1.8, 1.8),
            fixed_physical_radius_m=0.2,
            reference_129_zr=None,
            reference_129_forcing_zr=None,
            plasma_support_mask=np.ones((17, 17), dtype=np.bool_),
        )


def test_reference_field_binding_accepts_roundoff_and_rejects_drift() -> None:
    """Reference binding must tolerate only sub-threshold numerical roundoff."""
    r_grid = np.linspace(1.0, 2.0, 3, dtype=np.float64)
    z_grid = np.linspace(-1.0, 1.0, 3, dtype=np.float64)
    tokamak = type(
        "ManufacturedTokamak",
        (),
        {"psi": staticmethod(lambda r, z: np.zeros_like(r + z))},
    )()
    filament = fields.FilamentRecord(
        parent_index=0,
        parent_name="FC1",
        filament_index=0,
        r_m=1.5,
        z_m=0.0,
        weight=1.0,
        effective_current_a_turns=1.0,
    )

    with pytest.raises(ValueError, match="shape drifted"):
        runtime._vacuum_fields(
            tokamak=tokamak,
            filaments=(filament,),
            r_grid=r_grid,
            z_grid=z_grid,
            reference_129_zr=np.zeros((4, 4), dtype=np.float64),
        )
    with pytest.raises(ValueError, match="disagrees with equilibrium"):
        runtime._vacuum_fields(
            tokamak=tokamak,
            filaments=(filament,),
            r_grid=r_grid,
            z_grid=z_grid,
            reference_129_zr=np.full((3, 3), 2.0e-14, dtype=np.float64),
        )

    reference = np.full((3, 3), 1.0e-15, dtype=np.float64)
    freegs_vacuum, native_vacuum, timings = runtime._vacuum_fields(
        tokamak=tokamak,
        filaments=(filament,),
        r_grid=r_grid,
        z_grid=z_grid,
        reference_129_zr=reference,
    )
    assert freegs_vacuum is reference
    assert native_vacuum.shape == (3, 3)
    assert set(timings) == {"freegs_field", "native_field"}
    exact_reference = np.zeros((3, 3), dtype=np.float64)
    exact_freegs, _, _ = runtime._vacuum_fields(
        tokamak=tokamak,
        filaments=(filament,),
        r_grid=r_grid,
        z_grid=z_grid,
        reference_129_zr=exact_reference,
    )
    assert exact_freegs is exact_reference


def test_reference_forcing_binding_checks_reconstruction_before_exact_anchor() -> None:
    """The 129 forcing must agree numerically before exact anchor bytes are used."""
    evaluated = np.zeros((3, 3), dtype=np.float64)
    reference = np.full((3, 3), 1.0e-13, dtype=np.float64)
    assert runtime._bind_reference_forcing(evaluated, None) is evaluated
    assert runtime._bind_reference_forcing(evaluated, reference) is reference

    with pytest.raises(ValueError, match="shape drifted"):
        runtime._bind_reference_forcing(
            evaluated,
            np.zeros((4, 4), dtype=np.float64),
        )
    with pytest.raises(ValueError, match="disagrees with the bound anchor"):
        runtime._bind_reference_forcing(
            evaluated,
            np.full((3, 3), 2.0e-12, dtype=np.float64),
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("shape", "forcing shapes must agree"),
        ("grid_shape", "grids must match forcing shape"),
        ("grid_order", "grid coordinates must increase"),
        ("nonfinite", "source inverse forcing must be finite"),
        ("boundary", "source inverse forcing must have zero boundary rows"),
        ("closure", "forcing partition does not close"),
    ],
)
def test_partition_inverse_response_rejects_invalid_inputs_before_solver(
    case: str,
    message: str,
) -> None:
    """Malformed partitions must fail before preconditioner or inverse construction."""
    total = np.zeros((3, 3), dtype=np.float64)
    source = np.zeros_like(total)
    source_free = np.zeros_like(total)
    r_grid = np.linspace(1.0, 2.0, 3, dtype=np.float64)
    z_grid = np.linspace(-1.0, 1.0, 3, dtype=np.float64)
    if case == "shape":
        source = np.zeros((4, 4), dtype=np.float64)
    elif case == "grid_shape":
        r_grid = np.linspace(1.0, 2.0, 4, dtype=np.float64)
    elif case == "grid_order":
        r_grid = r_grid[::-1]
    elif case == "nonfinite":
        source[1, 1] = np.nan
    elif case == "boundary":
        source[0, 1] = 1.0
    elif case == "closure":
        source[1, 1] = 1.0
    else:
        raise AssertionError(f"unhandled case {case}")
    with pytest.raises(ValueError, match=message):
        runtime.partition_inverse_response(
            total,
            source,
            source_free,
            r_grid=r_grid,
            z_grid=z_grid,
        )


def test_runtime_internal_contracts_reject_invalid_masks_and_missing_parent() -> None:
    """Parity and recovery internals must fail closed before emitting evidence."""
    plane = np.ones((3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="fields/mask are incompatible"):
        runtime._source_free_parity(
            plane,
            np.ones((4, 4), dtype=np.float64),
            mask=np.ones((3, 3), dtype=np.bool_),
        )
    with pytest.raises(ValueError, match="must select source-free points"):
        runtime._source_free_parity(
            plane,
            plane,
            mask=np.zeros((3, 3), dtype=np.bool_),
        )

    parent = fields.ParentCoilRecord(
        index=0,
        name="FC1",
        coil_type="ShapedCoil",
        current_a=1.0,
        turns=1.0,
        effective_current_a_turns=1.0,
        filaments=(
            fields.FilamentRecord(
                parent_index=0,
                parent_name="FC1",
                filament_index=0,
                r_m=1.5,
                z_m=0.0,
                weight=1.0,
                effective_current_a_turns=1.0,
            ),
        ),
    )
    masks = fields.SourceMasks(
        rho_h_le_1=np.ones((3, 3), dtype=np.bool_),
        rho_h_le_2=np.ones((3, 3), dtype=np.bool_),
        rho_h_le_4=np.ones((3, 3), dtype=np.bool_),
        fixed_physical=np.ones((3, 3), dtype=np.bool_),
    )
    with pytest.raises(ValueError, match="missing parent coil FC1"):
        runtime._current_recovery(
            tokamak=type("MissingCoilTokamak", (), {"coils": []})(),
            parents=(parent,),
            r_grid=np.linspace(1.0, 2.0, 3, dtype=np.float64),
            z_grid=np.linspace(-1.0, 1.0, 3, dtype=np.float64),
            fixed_physical_radius_m=0.2,
            aggregate_density=plane,
            aggregate_masks=masks,
        )


@pytest.mark.experimental
@pytest.mark.external_reference
@pytest.mark.dedicated_hardware
def test_real_diiid_33_grid_executes_production_field_and_inverse_chain() -> None:
    """The real public DIII-D machine must traverse the unchanged runtime chain."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        freegs = cast(Any, importlib.import_module("freegs"))
    tokamak = freegs.machine.DIIID()
    tokamak.setControlCurrents([((-1.0) ** index) * (index + 1) * 1.0e4 for index in range(18)])
    parents = fields.extract_coil_manifest(tokamak)
    result = runtime.run_grid(
        resolution=33,
        tokamak=tokamak,
        parents=parents,
        r_bounds=(0.1, 2.8),
        z_bounds=(-1.8, 1.8),
        fixed_physical_radius_m=0.225,
        reference_129_zr=None,
        reference_129_forcing_zr=None,
        plasma_support_mask=np.ones((33, 33), dtype=np.bool_),
    )

    assert result.resolution == 33
    assert len(result.report["current_recovery"]["parents"]) == 18
    assert result.report["forcing_partition"]["closure_max_abs"] <= (
        contract.PARTITION_CLOSURE_MAX_ABS
    )
    assert result.report["response_partition"]["closure_max_abs_wb"] <= (
        contract.PARTITION_CLOSURE_MAX_ABS
    )
    assert result.report["vacuum_fields"]["source_free_parity"]["passes"] is True
    assert result.report["forcing_partition"]["total"]["field_sha256"]
    assert result.report["response_partition"]["total"]["field_sha256"]
