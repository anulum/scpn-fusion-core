# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Impurity Transport Data Contracts
"""Impurity-species, ADAS, and Aurora/STRAHL data contracts for impurity transport.

This module holds the validated data contracts shared across the impurity
transport package: the strict-axis validation primitive, the ``FloatArray``
alias, and the four dataclasses (:class:`ImpuritySpecies`,
:class:`AdasChargeStateCoefficients`, :class:`AuroraStrahlArtifact`,
:class:`AuroraParityCase`). It has no dependency on the charge-state math,
cooling, parity-solver, diagnostics, or transport-solver clusters, forming the
bottom layer of the impurity-transport module DAG.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _strict_axis(name: str, values: FloatArray, *, min_length: int = 2) -> FloatArray:
    axis = np.asarray(values, dtype=np.float64)
    if axis.ndim != 1 or axis.size < min_length:
        raise ValueError(f"{name} must be a 1D axis with at least {min_length} points")
    if not np.all(np.isfinite(axis)) or not np.all(np.diff(axis) > 0.0):
        raise ValueError(f"{name} must be finite and strictly increasing")
    return axis


@dataclass
class ImpuritySpecies:
    """Impurity species metadata and edge source parameters."""

    element: str
    Z_nucleus: int
    mass_amu: float
    source_rate: float = 0.0
    source_decay_width_rho: float = 0.05

    def __post_init__(self) -> None:
        """Validate impurity-species fields after dataclass construction."""
        if self.Z_nucleus < 1:
            raise ValueError("Z_nucleus must be positive")
        if not np.isfinite(self.mass_amu) or self.mass_amu <= 0.0:
            raise ValueError("mass_amu must be finite and positive")
        if not np.isfinite(self.source_rate) or self.source_rate < 0.0:
            raise ValueError("source_rate must be finite and non-negative")
        if not np.isfinite(self.source_decay_width_rho) or self.source_decay_width_rho <= 0.0:
            raise ValueError("source_decay_width_rho must be finite and positive")


@dataclass(frozen=True)
class AdasChargeStateCoefficients:
    """ADAS-style charge-state coefficient tables for native CR contracts."""

    charge_states: FloatArray
    ionisation_m3_s: FloatArray
    recombination_m3_s: FloatArray
    line_radiation_w_m3: FloatArray

    def __post_init__(self) -> None:
        """Validate ADAS charge-state coefficient tables after construction."""
        charge = np.asarray(self.charge_states, dtype=np.float64)
        ion = np.asarray(self.ionisation_m3_s, dtype=np.float64)
        rec = np.asarray(self.recombination_m3_s, dtype=np.float64)
        rad = np.asarray(self.line_radiation_w_m3, dtype=np.float64)
        if charge.ndim != 1 or charge.size < 2:
            raise ValueError("charge_states must be a 1D axis with at least two states")
        if not np.all(np.isfinite(charge)) or not np.all(np.diff(charge) > 0.0):
            raise ValueError("charge_states must be finite and strictly increasing")
        if not np.all(np.equal(charge, np.floor(charge))):
            raise ValueError("charge_states must contain integer charge values")
        for name, values in {
            "ionisation_m3_s": ion,
            "recombination_m3_s": rec,
            "line_radiation_w_m3": rad,
        }.items():
            if values.shape != charge.shape:
                raise ValueError(f"{name} must match charge_states shape")
            if not np.all(np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class AuroraStrahlArtifact:
    """JSON-compatible Aurora/STRAHL-style charge-state artifact."""

    coordinates: dict[str, list[float]]
    coordinate_units: dict[str, str]
    observable_axes: dict[str, list[str]]
    observable_units: dict[str, str]
    observables: dict[str, Any]
    conservation: dict[str, float]
    provenance: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible artifact payload."""
        return {
            "schema": "aurora-strahl-charge-state-artifact.v1",
            "coordinates": self.coordinates,
            "coordinate_units": self.coordinate_units,
            "observable_axes": self.observable_axes,
            "observable_units": self.observable_units,
            "observables": self.observables,
            "conservation": self.conservation,
            "provenance": self.provenance,
        }

    def validate_contract(self) -> dict[str, Any]:
        """Validate Aurora/STRAHL-style axes, shapes, conservation, and finiteness."""
        required_axis_units = {
            "time_s": "s",
            "radius_m": "m",
            "charge_state": "integer_charge",
        }
        required_observable_axes = {
            "charge_state_density_r_t": ["time_s", "radius_m", "charge_state"],
            "total_impurity_density_r_t": ["time_s", "radius_m"],
            "line_radiation_power_t": ["time_s"],
            "line_radiation_power_t_r_z": ["time_s", "radius_m", "charge_state"],
            "source_sink_matrix_t_r_z_z": [
                "time_s",
                "radius_m",
                "charge_state",
                "charge_state",
            ],
            "total_impurity_inventory_t": ["time_s"],
            "ionisation_source_matrix": ["radius_m", "charge_state"],
            "recombination_sink_matrix": ["radius_m", "charge_state"],
        }
        required_observable_units = {
            "charge_state_density_r_t": "m^-3",
            "total_impurity_density_r_t": "m^-3",
            "line_radiation_power_t": "W",
            "line_radiation_power_t_r_z": "W",
            "source_sink_matrix_t_r_z_z": "m^-3_s^-1",
            "total_impurity_inventory_t": "particles",
            "ionisation_source_matrix": "m^-3_s^-1",
            "recombination_sink_matrix": "m^-3_s^-1",
        }
        required_axes_present = all(
            axis in self.coordinates and self.coordinate_units.get(axis) == unit
            for axis, unit in required_axis_units.items()
        )
        required_observables_present = all(
            name in self.observables
            and self.observable_axes.get(name) == axes
            and self.observable_units.get(name) == required_observable_units[name]
            for name, axes in required_observable_axes.items()
        )
        coordinate_lengths: dict[str, int] = {}
        coordinates_finite = True
        coordinates_strict = True
        for axis in required_axis_units:
            values = np.asarray(self.coordinates.get(axis, []), dtype=np.float64)
            coordinate_lengths[axis] = int(values.size)
            coordinates_finite = coordinates_finite and bool(np.all(np.isfinite(values)))
            coordinates_strict = coordinates_strict and bool(
                values.ndim == 1 and values.size >= 2 and np.all(np.diff(values) > 0.0)
            )

        observable_shapes: dict[str, list[int]] = {}
        finite_observables = True
        nonnegative_density_and_power = True
        shape_contract = True
        for name, axes in required_observable_axes.items():
            arr = np.asarray(self.observables.get(name, []), dtype=np.float64)
            observable_shapes[name] = [int(v) for v in arr.shape]
            expected_shape = tuple(coordinate_lengths[axis] for axis in axes)
            shape_contract = shape_contract and bool(arr.shape == expected_shape)
            finite_observables = finite_observables and bool(np.all(np.isfinite(arr)))
            if name != "source_sink_matrix_t_r_z_z":
                nonnegative_density_and_power = nonnegative_density_and_power and bool(
                    np.all(arr >= 0.0)
                )

        charge_density = np.asarray(
            self.observables.get("charge_state_density_r_t", []), dtype=np.float64
        )
        total_density = np.asarray(
            self.observables.get("total_impurity_density_r_t", []), dtype=np.float64
        )
        source_sink = np.asarray(
            self.observables.get("source_sink_matrix_t_r_z_z", []), dtype=np.float64
        )
        density_closure = bool(
            charge_density.ndim == 3
            and total_density.shape == charge_density.shape[:2]
            and np.allclose(total_density, np.sum(charge_density, axis=2), rtol=1.0e-13)
        )
        source_sink_scale = max(
            float(np.max(np.abs(source_sink))) if source_sink.size else 0.0, 1.0
        )
        source_sink_tolerance = 1.0e-12 * source_sink_scale + 1.0e-6
        source_sink_conservative = bool(
            source_sink.ndim == 4
            and np.all(np.abs(np.sum(source_sink, axis=3)) <= source_sink_tolerance)
        )
        inventory_error = float(self.conservation.get("relative_inventory_error", np.inf))
        inventory_conserved = bool(np.isfinite(inventory_error) and inventory_error <= 1.0e-12)
        passed = bool(
            required_axes_present
            and required_observables_present
            and coordinates_finite
            and coordinates_strict
            and shape_contract
            and finite_observables
            and nonnegative_density_and_power
            and density_closure
            and source_sink_conservative
            and inventory_conserved
        )
        return {
            "coordinate_lengths": coordinate_lengths,
            "coordinates_finite": coordinates_finite,
            "coordinates_strictly_increasing": coordinates_strict,
            "density_closure": density_closure,
            "finite_observables": finite_observables,
            "inventory_conserved": inventory_conserved,
            "nonnegative_density_and_power": nonnegative_density_and_power,
            "observable_shapes": observable_shapes,
            "passed": passed,
            "required_axes_present": required_axes_present,
            "required_observables_present": required_observables_present,
            "shape_contract": shape_contract,
            "source_sink_conservative": source_sink_conservative,
        }


@dataclass(frozen=True)
class AuroraParityCase:
    """Native Aurora-compatible impurity-transport parity input contract."""

    element: str
    charge_states: FloatArray
    radius_m: FloatArray
    time_s: FloatArray
    ne_t_r: FloatArray
    Te_t_r: FloatArray
    initial_charge_state_density_rz: FloatArray
    diffusion_m2_s_r_z: FloatArray
    convection_m_s_r_z: FloatArray
    major_radius_m: float
    ionisation_m3_s_t_r_z: FloatArray | None = None
    recombination_m3_s_t_r_z: FloatArray | None = None
    line_radiation_w_m3_t_r_z: FloatArray | None = None
    effective_source_m3_s_t_r_z: FloatArray | None = None

    def __post_init__(self) -> None:
        """Validate the charge-state artifact axes and fields after construction."""
        charge = _strict_axis("charge_state", np.asarray(self.charge_states, dtype=np.float64))
        radius = _strict_axis("radius_m", np.asarray(self.radius_m, dtype=np.float64))
        time = _strict_axis("time_s", np.asarray(self.time_s, dtype=np.float64))
        if radius[0] < 0.0:
            raise ValueError("radius_m must be non-negative")
        if not np.isfinite(self.major_radius_m) or self.major_radius_m <= 0.0:
            raise ValueError("major_radius_m must be finite and positive")
        shape_t_r = (time.size, radius.size)
        shape_r_z = (radius.size, charge.size)
        shape_t_r_z = (time.size, radius.size, charge.size)
        arrays = {
            "ne_t_r": (np.asarray(self.ne_t_r, dtype=np.float64), shape_t_r, True),
            "Te_t_r": (np.asarray(self.Te_t_r, dtype=np.float64), shape_t_r, True),
            "initial_charge_state_density_rz": (
                np.asarray(self.initial_charge_state_density_rz, dtype=np.float64),
                shape_r_z,
                False,
            ),
            "diffusion_m2_s_r_z": (
                np.asarray(self.diffusion_m2_s_r_z, dtype=np.float64),
                shape_r_z,
                False,
            ),
            "convection_m_s_r_z": (
                np.asarray(self.convection_m_s_r_z, dtype=np.float64),
                shape_r_z,
                False,
            ),
        }
        for name, (array, expected, positive) in arrays.items():
            if array.shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be finite")
            if positive and np.any(array <= 0.0):
                raise ValueError(f"{name} must be positive")
            if not positive and name != "convection_m_s_r_z" and np.any(array < 0.0):
                raise ValueError(f"{name} must be non-negative")
        for name, optional in {
            "ionisation_m3_s_t_r_z": self.ionisation_m3_s_t_r_z,
            "recombination_m3_s_t_r_z": self.recombination_m3_s_t_r_z,
            "line_radiation_w_m3_t_r_z": self.line_radiation_w_m3_t_r_z,
        }.items():
            if optional is None:
                continue
            array = np.asarray(optional, dtype=np.float64)
            if array.shape != shape_t_r_z:
                raise ValueError(f"{name} must have shape {shape_t_r_z}")
            if not np.all(np.isfinite(array)) or np.any(array < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.effective_source_m3_s_t_r_z is not None:
            effective_source = np.asarray(self.effective_source_m3_s_t_r_z, dtype=np.float64)
            if effective_source.shape != shape_t_r_z:
                raise ValueError(f"effective_source_m3_s_t_r_z must have shape {shape_t_r_z}")
            if not np.all(np.isfinite(effective_source)):
                raise ValueError("effective_source_m3_s_t_r_z must be finite")
