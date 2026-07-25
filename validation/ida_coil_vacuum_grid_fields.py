# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN-FUSION-CORE — Coil-Vacuum Grid Field Operations
"""Numerical helpers for the IDA coil-vacuum grid-convergence diagnostic."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

_same_case = cast(Any, importlib.import_module("validation.benchmark_ida_same_case"))
_operator = cast(
    Any,
    importlib.import_module("validation.diagnose_ida_fixed_reference_operator"),
)

FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]
IntArray: TypeAlias = NDArray[np.int64]


@dataclass(frozen=True)
class FilamentRecord:
    """One shaped-coil filament with preserved parent lineage."""

    parent_index: int
    parent_name: str
    filament_index: int
    r_m: float
    z_m: float
    weight: float
    effective_current_a_turns: float

    @property
    def identifier(self) -> str:
        """Return the stable parent/filament identifier."""
        return f"{self.parent_name}:{self.filament_index:03d}"

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-ready record."""
        return {
            "effective_current_a_turns": self.effective_current_a_turns,
            "filament_id": self.identifier,
            "filament_index": self.filament_index,
            "parent_index": self.parent_index,
            "parent_name": self.parent_name,
            "r_m": self.r_m,
            "weight": self.weight,
            "z_m": self.z_m,
        }


@dataclass(frozen=True)
class ParentCoilRecord:
    """One top-level machine coil and its exact shaped-filament rows."""

    index: int
    name: str
    coil_type: str
    current_a: float
    turns: float
    effective_current_a_turns: float
    filaments: tuple[FilamentRecord, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-ready parent record."""
        return {
            "coil_type": self.coil_type,
            "current_a": self.current_a,
            "effective_current_a_turns": self.effective_current_a_turns,
            "filament_count": len(self.filaments),
            "filaments": [row.as_dict() for row in self.filaments],
            "name": self.name,
            "parent_index": self.index,
            "turns": self.turns,
        }


@dataclass(frozen=True)
class NearestSourceGeometry:
    """Nearest-filament distance fields and per-filament grid phases."""

    distance_m: FloatArray
    rho_h: FloatArray
    nearest_index: IntArray
    filament_phase_rz: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class SourceMasks:
    """Predeclared grid-relative and fixed-physical source masks."""

    rho_h_le_1: BoolArray
    rho_h_le_2: BoolArray
    rho_h_le_4: BoolArray
    fixed_physical: BoolArray


def finite_plane(value: object, *, field: str) -> FloatArray:
    """Return one finite non-trivial float64 plane or fail closed."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"{field} must be a finite non-trivial 2D array")
    return array


def canonical_sha256(value: object) -> str:
    """Hash one JSON-compatible value with the repository canonical encoding."""
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def extract_coil_manifest(tokamak: object) -> tuple[ParentCoilRecord, ...]:
    """Extract exact top-level shaped-coil lineage from the frozen FreeGS machine."""
    machine = cast(Any, tokamak)
    raw_coils = getattr(machine, "coils", None)
    if not isinstance(raw_coils, list) or not raw_coils:
        raise ValueError("tokamak must expose a non-empty top-level coil list")
    parents: list[ParentCoilRecord] = []
    names: set[str] = set()
    for parent_index, raw_entry in enumerate(raw_coils):
        if not isinstance(raw_entry, tuple) or len(raw_entry) != 2:
            raise ValueError("tokamak coil entries must be (name, coil) pairs")
        raw_name, raw_coil = raw_entry
        name = str(raw_name)
        if not name or name in names:
            raise ValueError("parent coil names must be non-empty and unique")
        names.add(name)
        coil = cast(Any, raw_coil)
        current = float(coil.current)
        turns = float(coil.turns if hasattr(coil, "turns") else 1.0)
        points = getattr(coil, "_points", None)
        if not isinstance(points, list) or not points:
            raise ValueError(f"parent coil {name} must expose shaped-filament points")
        if not all(math.isfinite(value) for value in (current, turns)) or turns <= 0.0:
            raise ValueError(f"parent coil {name} current/turns are invalid")
        effective_current = current * turns
        filaments: list[FilamentRecord] = []
        for filament_index, raw_point in enumerate(points):
            if not isinstance(raw_point, tuple) or len(raw_point) != 3:
                raise ValueError(f"parent coil {name} has a malformed filament point")
            r_m, z_m, weight = (float(item) for item in raw_point)
            if not all(math.isfinite(item) for item in (r_m, z_m, weight)):
                raise ValueError(f"parent coil {name} has a non-finite filament")
            row = FilamentRecord(
                parent_index=parent_index,
                parent_name=name,
                filament_index=filament_index,
                r_m=r_m,
                z_m=z_m,
                weight=weight,
                effective_current_a_turns=effective_current * weight,
            )
            filaments.append(row)
        child_sum = math.fsum(row.effective_current_a_turns for row in filaments)
        if not math.isclose(child_sum, effective_current, rel_tol=1.0e-12, abs_tol=1.0e-9):
            raise ValueError(f"parent coil {name} filament currents do not close")
        parents.append(
            ParentCoilRecord(
                index=parent_index,
                name=name,
                coil_type=type(coil).__name__,
                current_a=current,
                turns=turns,
                effective_current_a_turns=effective_current,
                filaments=tuple(filaments),
            )
        )
    return tuple(parents)


def flatten_filaments(parents: tuple[ParentCoilRecord, ...]) -> tuple[FilamentRecord, ...]:
    """Flatten a non-empty parent manifest without losing row identity."""
    if not parents or any(not parent.filaments for parent in parents):
        raise ValueError("parent manifest must contain non-empty filament groups")
    return tuple(row for parent in parents for row in parent.filaments)


def validate_frozen_manifest(
    parents: tuple[ParentCoilRecord, ...],
    *,
    r_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
) -> tuple[FilamentRecord, ...]:
    """Validate the frozen 18-parent/216-filament DIII-D manifest contract."""
    filaments = flatten_filaments(parents)
    if len(parents) != 18 or len(filaments) != 216:
        raise ValueError("frozen DIII-D manifest must contain 18 parents and 216 filaments")
    r_min, r_max = r_bounds
    z_min, z_max = z_bounds
    if not r_min < r_max or not z_min < z_max:
        raise ValueError("manifest domain bounds are invalid")
    for row in filaments:
        if not (r_min < row.r_m < r_max and z_min < row.z_m < z_max):
            raise ValueError(f"filament {row.identifier} lies outside the fixed domain")
    return filaments


def manifest_payload(parents: tuple[ParentCoilRecord, ...]) -> dict[str, Any]:
    """Return the canonical manifest payload and its deterministic digest."""
    rows = [parent.as_dict() for parent in parents]
    payload: dict[str, Any] = {
        "filament_count": sum(len(parent.filaments) for parent in parents),
        "parent_count": len(parents),
        "parents": rows,
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    return payload


def filament_arrays(
    filaments: tuple[FilamentRecord, ...],
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return R, Z, and effective-current vectors for a filament manifest."""
    if not filaments:
        raise ValueError("filament manifest must not be empty")
    return (
        np.asarray([row.r_m for row in filaments], dtype=np.float64),
        np.asarray([row.z_m for row in filaments], dtype=np.float64),
        np.asarray([row.effective_current_a_turns for row in filaments], dtype=np.float64),
    )


def nearest_source_geometry(
    r_grid: FloatArray,
    z_grid: FloatArray,
    filaments: tuple[FilamentRecord, ...],
) -> NearestSourceGeometry:
    """Measure nearest-filament physical/cell distance without large 3-D buffers."""
    if r_grid.ndim != 1 or z_grid.ndim != 1 or min(r_grid.size, z_grid.size) < 3:
        raise ValueError("grid axes must be non-trivial one-dimensional arrays")
    if not filaments:
        raise ValueError("filament manifest must not be empty")
    if not np.all(np.isfinite(r_grid)) or not np.all(np.isfinite(z_grid)):
        raise ValueError("grid axes must be finite")
    r_steps = np.diff(r_grid)
    z_steps = np.diff(z_grid)
    if (
        not np.all(r_steps > 0.0)
        or not np.all(z_steps > 0.0)
        or not np.allclose(r_steps, r_steps[0], rtol=1.0e-12, atol=0.0)
        or not np.allclose(z_steps, z_steps[0], rtol=1.0e-12, atol=0.0)
    ):
        raise ValueError("grid axes must be uniformly and strictly increasing")
    d_r = float(r_grid[1] - r_grid[0])
    d_z = float(z_grid[1] - z_grid[0])
    r_mesh, z_mesh = np.meshgrid(r_grid, z_grid)
    min_distance_sq = np.full(r_mesh.shape, np.inf, dtype=np.float64)
    min_rho_sq = np.full(r_mesh.shape, np.inf, dtype=np.float64)
    nearest = np.full(r_mesh.shape, -1, dtype=np.int64)
    phases: list[tuple[float, float]] = []
    for index, row in enumerate(filaments):
        delta_r = r_mesh - row.r_m
        delta_z = z_mesh - row.z_m
        distance_sq = delta_r * delta_r + delta_z * delta_z
        rho_sq = (delta_r / d_r) ** 2 + (delta_z / d_z) ** 2
        update = distance_sq < min_distance_sq
        min_distance_sq[update] = distance_sq[update]
        nearest[update] = index
        min_rho_sq = np.minimum(min_rho_sq, rho_sq)
        phase_r = float(((row.r_m - float(r_grid[0])) / d_r) % 1.0)
        phase_z = float(((row.z_m - float(z_grid[0])) / d_z) % 1.0)
        phases.append((phase_r, phase_z))
    if np.any(nearest < 0):
        raise RuntimeError("nearest-filament assignment is incomplete")
    return NearestSourceGeometry(
        distance_m=np.sqrt(min_distance_sq),
        rho_h=np.sqrt(min_rho_sq),
        nearest_index=nearest,
        filament_phase_rz=tuple(phases),
    )


def source_masks(
    geometry: NearestSourceGeometry,
    *,
    fixed_physical_radius_m: float,
) -> SourceMasks:
    """Build the three sensitivity masks and fixed-physical exclusion mask."""
    if not math.isfinite(fixed_physical_radius_m) or fixed_physical_radius_m <= 0.0:
        raise ValueError("fixed physical radius must be positive and finite")
    return SourceMasks(
        rho_h_le_1=np.asarray(geometry.rho_h <= 1.0, dtype=np.bool_),
        rho_h_le_2=np.asarray(geometry.rho_h <= 2.0, dtype=np.bool_),
        rho_h_le_4=np.asarray(geometry.rho_h <= 4.0, dtype=np.bool_),
        fixed_physical=np.asarray(
            geometry.distance_m <= fixed_physical_radius_m,
            dtype=np.bool_,
        ),
    )


def mask_summary(mask: BoolArray, *, d_area_m2: float) -> dict[str, Any]:
    """Summarise one deterministic mask by point count and physical area."""
    if mask.ndim != 2 or not math.isfinite(d_area_m2) or d_area_m2 <= 0.0:
        raise ValueError("mask summary inputs are invalid")
    count = int(np.count_nonzero(mask))
    return {
        "area_m2": count * d_area_m2,
        "point_count": count,
        "point_fraction": count / mask.size,
    }


def zero_identity_wall(value: object, *, field: str) -> FloatArray:
    """Copy one forcing plane and enforce zero identity-wall rows."""
    plane = finite_plane(value, field=field).copy()
    plane[0, :] = 0.0
    plane[-1, :] = 0.0
    plane[:, 0] = 0.0
    plane[:, -1] = 0.0
    return plane


def native_lhs_zr(
    psi_zr: object,
    *,
    r_grid: FloatArray,
    z_grid: FloatArray,
) -> FloatArray:
    """Apply the unchanged native second-order ``Delta*`` operator."""
    plane = finite_plane(psi_zr, field="vacuum flux")
    if plane.shape != (z_grid.size, r_grid.size):
        raise ValueError("vacuum flux shape must match the Z-R grid")
    lhs_rz = _operator._native_lhs(
        np.asarray(plane.T, dtype=np.float64),
        r_grid=r_grid,
        z_grid=z_grid,
    )
    return np.asarray(lhs_rz.T, dtype=np.float64)


def implied_current_density(
    forcing_zr: object,
    *,
    r_grid: FloatArray,
    mu0: float,
) -> FloatArray:
    """Recover implied toroidal current density from ``-Delta*psi/(mu0 R)``."""
    forcing = finite_plane(forcing_zr, field="coil-vacuum forcing")
    if (
        r_grid.ndim != 1
        or forcing.shape[1] != r_grid.size
        or not np.all(np.isfinite(r_grid))
        or np.any(r_grid <= 0.0)
    ):
        raise ValueError("forcing/radius geometry is invalid")
    if not math.isfinite(mu0) or mu0 <= 0.0:
        raise ValueError("mu0 must be positive and finite")
    return np.asarray(
        -forcing / (mu0 * r_grid[np.newaxis, :]),
        dtype=np.float64,
    )


def integrate_current(
    current_density_zr: object,
    *,
    mask: BoolArray,
    d_area_m2: float,
) -> float:
    """Integrate one current-density plane over a validated mask."""
    current = finite_plane(current_density_zr, field="current density")
    if mask.shape != current.shape or not math.isfinite(d_area_m2) or d_area_m2 <= 0.0:
        raise ValueError("current integration mask/area is invalid")
    interior = mask.copy()
    interior[0, :] = False
    interior[-1, :] = False
    interior[:, 0] = False
    interior[:, -1] = False
    return float(np.sum(current[interior], dtype=np.float64) * d_area_m2)


def field_metric(value: object, *, d_area_m2: float) -> dict[str, Any]:
    """Return deterministic area-weighted norms for one finite plane."""
    plane = finite_plane(value, field="metric field")
    if not math.isfinite(d_area_m2) or d_area_m2 <= 0.0:
        raise ValueError("metric cell area must be positive and finite")
    return {
        "area_weighted_l2": float(np.linalg.norm(plane) * math.sqrt(d_area_m2)),
        "area_weighted_rms": float(math.sqrt(np.mean(plane * plane))),
        "field_sha256": _same_case._array_sha256(plane),
        "l2": float(np.linalg.norm(plane)),
        "linf": float(np.max(np.abs(plane))),
    }


def comparison_metric(
    actual: object,
    reference: object,
    *,
    mask: BoolArray,
    d_area_m2: float,
) -> dict[str, Any]:
    """Compare two matching planes on a non-empty mask."""
    left = finite_plane(actual, field="comparison actual")
    right = finite_plane(reference, field="comparison reference")
    if (
        left.shape != right.shape
        or mask.shape != left.shape
        or not np.any(mask)
        or not math.isfinite(d_area_m2)
        or d_area_m2 <= 0.0
    ):
        raise ValueError("comparison planes/mask are incompatible")
    delta = np.asarray(left[mask] - right[mask], dtype=np.float64)
    left_values = np.asarray(left[mask], dtype=np.float64)
    right_values = np.asarray(right[mask], dtype=np.float64)
    denominator = max(float(np.linalg.norm(right_values)), 1.0e-30)
    cosine_denominator = max(
        float(np.linalg.norm(left_values) * np.linalg.norm(right_values)),
        1.0e-30,
    )
    return {
        "area_weighted_l2": float(np.linalg.norm(delta) * math.sqrt(d_area_m2)),
        "area_weighted_rms": float(math.sqrt(np.mean(delta * delta))),
        "cosine": float(np.dot(left_values, right_values) / cosine_denominator),
        "linf": float(np.max(np.abs(delta))),
        "projection": float(np.dot(left_values, right_values) / max(denominator**2, 1.0e-30)),
        "relative_l2": float(np.linalg.norm(delta) / denominator),
    }


def restrict_to_shape(value: object, shape: tuple[int, int]) -> FloatArray:
    """Restrict a nested odd grid exactly, without interpolation."""
    plane = finite_plane(value, field="fine-grid field")
    target_z, target_r = shape
    if target_z < 3 or target_r < 3:
        raise ValueError("restriction target must be non-trivial")
    stride_z_num = plane.shape[0] - 1
    stride_r_num = plane.shape[1] - 1
    stride_z_den = target_z - 1
    stride_r_den = target_r - 1
    if stride_z_num % stride_z_den or stride_r_num % stride_r_den:
        raise ValueError("fine and target grids are not exactly nested")
    stride_z = stride_z_num // stride_z_den
    stride_r = stride_r_num // stride_r_den
    return np.asarray(plane[::stride_z, ::stride_r], dtype=np.float64)


def observed_order_from_three(
    coarse: object,
    medium: object,
    fine: object,
    *,
    mask: BoolArray,
) -> dict[str, float]:
    """Estimate nested-grid order from consecutive restricted differences."""
    coarse_plane = finite_plane(coarse, field="coarse field")
    coarse_shape = (int(coarse_plane.shape[0]), int(coarse_plane.shape[1]))
    medium_on_coarse = restrict_to_shape(medium, coarse_shape)
    fine_on_coarse = restrict_to_shape(fine, coarse_shape)
    if mask.shape != coarse_plane.shape or not np.any(mask):
        raise ValueError("observed-order mask must select coarse-grid points")
    coarse_delta = np.asarray((coarse_plane - medium_on_coarse)[mask], dtype=np.float64)
    fine_delta = np.asarray((medium_on_coarse - fine_on_coarse)[mask], dtype=np.float64)
    coarse_rms = float(math.sqrt(np.mean(coarse_delta * coarse_delta)))
    fine_rms = float(math.sqrt(np.mean(fine_delta * fine_delta)))
    if coarse_rms <= 0.0 or fine_rms <= 0.0:
        raise ValueError("observed-order differences must be non-zero")
    return {
        "coarse_to_medium_rms": coarse_rms,
        "medium_to_fine_rms": fine_rms,
        "observed_order": float(math.log2(coarse_rms / fine_rms)),
    }


def forcing_l2_fraction(forcing: object, *, mask: BoolArray) -> float:
    """Return the fraction of forcing L2 contained in a selected footprint."""
    plane = finite_plane(forcing, field="forcing")
    if mask.shape != plane.shape:
        raise ValueError("forcing mask shape is invalid")
    total = float(np.linalg.norm(plane))
    if total <= 0.0:
        raise ValueError("forcing norm must be non-zero")
    return float(np.linalg.norm(plane[mask]) / total)


def recovery_metric(recovered: float, expected: float) -> dict[str, float]:
    """Summarise one signed parent-current recovery result."""
    if not math.isfinite(recovered) or not math.isfinite(expected):
        raise ValueError("current recovery values must be finite")
    absolute_error = abs(recovered - expected)
    return {
        "absolute_error_a_turns": absolute_error,
        "expected_a_turns": expected,
        "recovered_a_turns": recovered,
        "relative_error": absolute_error / max(abs(expected), 1.0e-30),
        "signed_error_a_turns": recovered - expected,
    }


def weighted_recovery_error(rows: tuple[dict[str, float], ...]) -> float:
    """Return cancellation-safe absolute-current-weighted aggregate error."""
    if not rows:
        raise ValueError("current recovery rows must not be empty")
    numerator = math.fsum(row["absolute_error_a_turns"] for row in rows)
    denominator = math.fsum(abs(row["expected_a_turns"]) for row in rows)
    if denominator <= 0.0:
        raise ValueError("absolute expected-current sum must be non-zero")
    return numerator / denominator
