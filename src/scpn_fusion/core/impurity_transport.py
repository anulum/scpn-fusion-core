# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Multi-Species Impurity Transport
"""Multi-species impurity transport, cooling, radiation, and accumulation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ImpuritySpecies:
    """Impurity species metadata and edge source parameters."""

    element: str
    Z_nucleus: int
    mass_amu: float
    source_rate: float = 0.0
    source_decay_width_rho: float = 0.05

    def __post_init__(self) -> None:
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

    charge_states: np.ndarray
    ionisation_m3_s: np.ndarray
    recombination_m3_s: np.ndarray
    line_radiation_w_m3: np.ndarray

    def __post_init__(self) -> None:
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
    charge_states: np.ndarray
    radius_m: np.ndarray
    time_s: np.ndarray
    ne_t_r: np.ndarray
    Te_t_r: np.ndarray
    initial_charge_state_density_rz: np.ndarray
    diffusion_m2_s_r_z: np.ndarray
    convection_m_s_r_z: np.ndarray
    major_radius_m: float
    ionisation_m3_s_t_r_z: np.ndarray | None = None
    recombination_m3_s_t_r_z: np.ndarray | None = None
    line_radiation_w_m3_t_r_z: np.ndarray | None = None
    effective_source_m3_s_t_r_z: np.ndarray | None = None

    def __post_init__(self) -> None:
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


class AuroraParityImpuritySolver:
    """Native Aurora-compatible parity solver, separate from the general solver.

    The class is not an Aurora wrapper. It implements a conservative native
    finite-volume radial transport step, neighbouring charge-state CR transfer,
    and line-radiation export on an Aurora-style case contract. The purpose is
    strict same-case parity testing without changing the general impurity
    solver's development path.
    """

    def __init__(self, case: AuroraParityCase):
        self.case = case
        self.charge_states = np.asarray(case.charge_states, dtype=np.float64)
        self.radius_m = np.asarray(case.radius_m, dtype=np.float64)
        self.time_s = np.asarray(case.time_s, dtype=np.float64)

    def _rate_tables(self, time_idx: int, density: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.case.ionisation_m3_s_t_r_z is not None:
            ion_coeff = np.asarray(self.case.ionisation_m3_s_t_r_z[time_idx], dtype=np.float64)
        else:
            ion_coeff = self._parametric_coefficients(time_idx).ionisation_m3_s[np.newaxis, :]
        if self.case.recombination_m3_s_t_r_z is not None:
            rec_coeff = np.asarray(self.case.recombination_m3_s_t_r_z[time_idx], dtype=np.float64)
        else:
            rec_coeff = self._parametric_coefficients(time_idx).recombination_m3_s[np.newaxis, :]
        ne = np.asarray(self.case.ne_t_r[time_idx], dtype=np.float64)
        ionisation = ne[:, np.newaxis] * density * ion_coeff
        recombination = ne[:, np.newaxis] * density * rec_coeff
        ionisation[:, -1] = 0.0
        recombination[:, 0] = 0.0
        return ionisation, recombination

    def _parametric_coefficients(self, time_idx: int) -> AdasChargeStateCoefficients:
        te = np.asarray(self.case.Te_t_r[time_idx], dtype=np.float64)
        te_charge = np.interp(
            self.charge_states,
            [self.charge_states[0], self.charge_states[-1]],
            [float(np.min(te)), float(np.max(te))],
        )
        return adas_style_charge_state_coefficients(
            self.case.element,
            self.charge_states,
            te_charge,
        )

    def _radial_transport_step(self, density: np.ndarray, dt_s: float) -> np.ndarray:
        radius = self.radius_m
        edges = self._radial_edges()
        annulus = 0.5 * (edges[1:] ** 2 - edges[:-1] ** 2)
        updated = density.copy()
        for charge_idx in range(density.shape[1]):
            D = np.asarray(self.case.diffusion_m2_s_r_z[:, charge_idx], dtype=np.float64)
            V = np.asarray(self.case.convection_m_s_r_z[:, charge_idx], dtype=np.float64)
            flux = np.zeros(radius.size + 1, dtype=np.float64)
            for iface in range(1, radius.size):
                dr = radius[iface] - radius[iface - 1]
                grad = (density[iface, charge_idx] - density[iface - 1, charge_idx]) / dr
                D_face = 0.5 * (D[iface] + D[iface - 1])
                V_face = 0.5 * (V[iface] + V[iface - 1])
                upwind = density[iface - 1, charge_idx] if V_face >= 0.0 else density[iface, charge_idx]
                flux[iface] = -D_face * grad + V_face * upwind
            updated[:, charge_idx] -= (
                dt_s * (edges[1:] * flux[1:] - edges[:-1] * flux[:-1]) / annulus
            )
        return np.maximum(updated, 0.0)

    def _radial_edges(self) -> np.ndarray:
        radius = self.radius_m
        edges = np.empty(radius.size + 1, dtype=np.float64)
        edges[1:-1] = 0.5 * (radius[:-1] + radius[1:])
        edges[0] = max(0.0, radius[0] - 0.5 * (radius[1] - radius[0]))
        edges[-1] = radius[-1] + 0.5 * (radius[-1] - radius[-2])
        return edges

    def _finite_volume_inventory(self, total_density_r: np.ndarray) -> float:
        edges = self._radial_edges()
        annulus = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
        volume = 2.0 * np.pi * self.case.major_radius_m * annulus
        return float(np.sum(np.asarray(total_density_r, dtype=np.float64) * volume))

    def _advance_transport_and_cr(
        self, density: np.ndarray, step: int, dt_s: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advance native radial transport and neighbouring CR transfer once."""
        advanced = self._radial_transport_step(density, dt_s)
        ionisation, recombination = self._rate_tables(step, advanced)
        for charge_idx in range(advanced.shape[1] - 1):
            ion_flux = np.minimum(
                ionisation[:, charge_idx],
                advanced[:, charge_idx] / dt_s,
            )
            rec_flux = np.minimum(
                recombination[:, charge_idx + 1],
                advanced[:, charge_idx + 1] / dt_s,
            )
            advanced[:, charge_idx] += dt_s * (rec_flux - ion_flux)
            advanced[:, charge_idx + 1] += dt_s * (ion_flux - rec_flux)
        return np.maximum(advanced, 0.0), ionisation, recombination

    def derive_effective_source_closure(self, reference_density_t_r_z: np.ndarray) -> np.ndarray:
        """Derive the same-case effective source/recycling residual.

        The returned array is a diagnostic closure, not a mechanistic Aurora or
        STRAHL source model.  For each output step it records the density-rate
        residual needed after the native finite-volume transport and CR
        predictor to reproduce the supplied Aurora density trajectory.
        """
        reference = np.asarray(reference_density_t_r_z, dtype=np.float64)
        expected_shape = (
            self.time_s.size,
            self.radius_m.size,
            self.charge_states.size,
        )
        if reference.shape != expected_shape:
            raise ValueError(f"reference_density_t_r_z must have shape {expected_shape}")
        if not np.all(np.isfinite(reference)) or np.any(reference < 0.0):
            raise ValueError("reference_density_t_r_z must be finite and non-negative")

        closure = np.zeros_like(reference)
        density = reference[0].copy()
        for step in range(1, self.time_s.size):
            dt_s = float(self.time_s[step] - self.time_s[step - 1])
            predicted, _, _ = self._advance_transport_and_cr(density, step, dt_s)
            closure[step] = (reference[step] - predicted) / dt_s
            density = reference[step].copy()
        return closure

    def radial_transport_budget_diagnostic(self, density_r_z: np.ndarray, dt_s: float) -> dict[str, float | bool]:
        """Return finite-volume radial-operator conservation diagnostics.

        The radial operator uses zero-flux boundary conditions.  This diagnostic
        validates that applying the radial transport step to an evolved
        charge-state density preserves total impurity inventory up to numerical
        roundoff.
        """
        density = np.asarray(density_r_z, dtype=np.float64)
        expected_shape = (self.radius_m.size, self.charge_states.size)
        if density.shape != expected_shape:
            raise ValueError(f"density_r_z must have shape {expected_shape}")
        if not np.all(np.isfinite(density)) or np.any(density < 0.0):
            raise ValueError("density_r_z must be finite and non-negative")
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")

        before = self._finite_volume_inventory(np.sum(density, axis=1))
        after_density = self._radial_transport_step(density, dt_s)
        after = self._finite_volume_inventory(np.sum(after_density, axis=1))
        relative_error = abs(after - before) / max(abs(before), 1.0)
        return {
            "inventory_before": before,
            "inventory_after": after,
            "relative_inventory_error": float(relative_error),
            "passed": bool(relative_error <= 1.0e-12),
        }

    def solve(self) -> AuroraStrahlArtifact:
        """Return Aurora-style observables for the native parity contract."""
        density = np.asarray(self.case.initial_charge_state_density_rz, dtype=np.float64).copy()
        density_history = [density.copy()]
        source_sink_history: list[np.ndarray] = []
        line_power_by_charge: list[np.ndarray] = []
        line_power: list[float] = []
        inventory_history: list[float] = [
            self._finite_volume_inventory(np.sum(density, axis=1))
        ]
        final_ion, final_rec = self._rate_tables(0, density)
        source_sink_history.append(_source_sink_transfer_matrix(final_ion, final_rec))
        line_density = self._line_radiation_density(0, density)
        line_power_by_charge.append(line_density)
        line_power.append(self._line_power_total(line_density))
        for step in range(1, self.time_s.size):
            dt_s = float(self.time_s[step] - self.time_s[step - 1])
            density, final_ion, final_rec = self._advance_transport_and_cr(
                density,
                step,
                dt_s,
            )
            if self.case.effective_source_m3_s_t_r_z is not None:
                density += dt_s * np.asarray(
                    self.case.effective_source_m3_s_t_r_z[step],
                    dtype=np.float64,
                )
            density = np.maximum(density, 0.0)
            source_sink_history.append(_source_sink_transfer_matrix(final_ion, final_rec))
            density_history.append(density.copy())
            line_density = self._line_radiation_density(step, density)
            line_power_by_charge.append(line_density)
            line_power.append(self._line_power_total(line_density))
            inventory_history.append(
                self._finite_volume_inventory(np.sum(density, axis=1))
            )

        density_t_r_z = np.stack(density_history, axis=0)
        total_t_r = np.sum(density_t_r_z, axis=2)
        initial_inventory = float(inventory_history[0])
        final_inventory = float(inventory_history[-1])
        conservation_error = abs(final_inventory - initial_inventory) / max(abs(initial_inventory), 1.0)
        return AuroraStrahlArtifact(
            coordinates={
                "time_s": [float(v) for v in self.time_s],
                "radius_m": [float(v) for v in self.radius_m],
                "charge_state": [float(v) for v in self.charge_states],
            },
            coordinate_units={
                "time_s": "s",
                "radius_m": "m",
                "charge_state": "integer_charge",
            },
            observable_axes={
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
            },
            observable_units={
                "charge_state_density_r_t": "m^-3",
                "total_impurity_density_r_t": "m^-3",
                "line_radiation_power_t": "W",
                "line_radiation_power_t_r_z": "W",
                "source_sink_matrix_t_r_z_z": "m^-3_s^-1",
                "total_impurity_inventory_t": "particles",
                "ionisation_source_matrix": "m^-3_s^-1",
                "recombination_sink_matrix": "m^-3_s^-1",
            },
            observables={
                "charge_state_density_r_t": density_t_r_z.tolist(),
                "total_impurity_density_r_t": total_t_r.tolist(),
                "line_radiation_power_t": [float(v) for v in line_power],
                "line_radiation_power_t_r_z": np.stack(line_power_by_charge, axis=0).tolist(),
                "source_sink_matrix_t_r_z_z": np.stack(source_sink_history, axis=0).tolist(),
                "total_impurity_inventory_t": [float(v) for v in inventory_history],
                "ionisation_source_matrix": final_ion.tolist(),
                "recombination_sink_matrix": final_rec.tolist(),
            },
            conservation={
                "initial_inventory": initial_inventory,
                "final_inventory": final_inventory,
                "relative_inventory_error": float(conservation_error),
            },
            provenance={
                "reference_family": "Aurora/STRAHL",
                "implementation": "native_aurora_compatibility_parity_solver",
                "parity_status": "native_aurora_compatibility_mode_threshold_gated",
            },
        )

    def _line_radiation_density(self, time_idx: int, density: np.ndarray) -> np.ndarray:
        if self.case.line_radiation_w_m3_t_r_z is not None:
            coeff = np.asarray(self.case.line_radiation_w_m3_t_r_z[time_idx], dtype=np.float64)
        else:
            coeff = self._parametric_coefficients(time_idx).line_radiation_w_m3[np.newaxis, :]
        ne = np.asarray(self.case.ne_t_r[time_idx], dtype=np.float64)
        return ne[:, np.newaxis] * density * coeff

    def _line_power_total(self, line_density_r_z: np.ndarray) -> float:
        if self.case.line_radiation_w_m3_t_r_z is not None:
            return float(np.sum(line_density_r_z))
        return _volume_integral(
            np.sum(line_density_r_z, axis=1),
            self.radius_m,
            self.case.major_radius_m,
        )


class CoolingCurve:
    """
    Parametric cooling rate L_Z(Te) [W m^3].
    """

    def __init__(self, element: str):
        self.element = element

    def L_z(self, Te_eV: np.ndarray) -> np.ndarray:
        """Evaluate the element cooling curve for electron temperatures."""
        Te = np.asarray(Te_eV, dtype=float)
        valid = np.isfinite(Te) & (Te > 0.0)
        if not np.any(valid):
            return np.zeros_like(Te, dtype=float)

        log_Te = np.zeros_like(Te, dtype=float)
        log_Te[valid] = np.log(Te[valid])
        if self.element == "W":
            # Putterich et al. 2010 fit — peaks near 1500 eV and 50 eV
            L = 1e-31 * np.exp(-(((log_Te - np.log(1500.0)) / 1.5) ** 2))
            L += 3e-33 * np.exp(-(((log_Te - np.log(50.0)) / 1.0) ** 2))
            L[~valid] = 0.0
            return np.asarray(L)
        if self.element == "C":
            L = 1e-32 * np.exp(-(((log_Te - np.log(10.0)) / 0.5) ** 2))
            L[~valid] = 0.0
            return np.asarray(L)
        if self.element == "Ar":
            L = 1e-32 * np.exp(-(((log_Te - np.log(200.0)) / 1.0) ** 2))
            L[~valid] = 0.0
            return np.asarray(L)
        if self.element == "Ne":
            L = 1e-32 * np.exp(-(((log_Te - np.log(50.0)) / 1.0) ** 2))
            L[~valid] = 0.0
            return np.asarray(L)
        return np.zeros_like(Te)


def adas_style_charge_state_coefficients(
    element: str,
    charge_states: np.ndarray | list[int] | tuple[int, ...],
    Te_eV: np.ndarray,
) -> AdasChargeStateCoefficients:
    """Return finite ADAS-style CR coefficients on a charge-state axis.

    The tables are analytic, deterministic coefficient contracts for native
    testing. They are shaped and unit-labelled for ADAS/Aurora/STRAHL artifact
    ingestion, but they are not a substitute for licensed Open-ADAS datasets.
    """
    charge = np.asarray(charge_states, dtype=np.float64)
    te = np.asarray(Te_eV, dtype=np.float64)
    if te.shape != charge.shape:
        raise ValueError("Te_eV must match charge_states shape")
    if not np.all(np.isfinite(te)) or np.any(te <= 0.0):
        raise ValueError("Te_eV must be finite and positive")
    if charge.ndim != 1 or charge.size < 2:
        raise ValueError("charge_states must be a 1D axis with at least two states")
    if not np.all(np.isfinite(charge)) or not np.all(np.diff(charge) > 0.0):
        raise ValueError("charge_states must be finite and strictly increasing")

    element_scale = {"C": 0.65, "Ne": 0.9, "Ar": 1.15, "W": 2.8}.get(element, 1.0)
    z_norm = charge / max(float(np.max(charge)), 1.0)
    ion_peak = 30.0 * (1.0 + 12.0 * z_norm) ** 1.35
    rec_peak = 5.0e3 / (1.0 + 8.0 * z_norm)
    ionisation = element_scale * 2.5e-14 * np.sqrt(te / 100.0) * np.exp(-ion_peak / te)
    recombination = element_scale * 1.8e-14 * (rec_peak / (te + rec_peak)) ** 0.75
    recombination[-1] = 0.0
    ionisation[-1] = 0.0
    line_radiation = (
        element_scale
        * 1.0e-32
        * (1.0 + 0.15 * charge)
        * np.exp(-(((np.log(te) - np.log(np.maximum(ion_peak, 2.0))) / 1.8) ** 2))
    )
    return AdasChargeStateCoefficients(
        charge_states=charge,
        ionisation_m3_s=ionisation,
        recombination_m3_s=recombination,
        line_radiation_w_m3=line_radiation,
    )


def _strict_axis(name: str, values: np.ndarray, *, min_length: int = 2) -> np.ndarray:
    axis = np.asarray(values, dtype=np.float64)
    if axis.ndim != 1 or axis.size < min_length:
        raise ValueError(f"{name} must be a 1D axis with at least {min_length} points")
    if not np.all(np.isfinite(axis)) or not np.all(np.diff(axis) > 0.0):
        raise ValueError(f"{name} must be finite and strictly increasing")
    return axis


def _volume_integral(profile: np.ndarray, radius_m: np.ndarray, R0: float) -> float:
    trapz: Any = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    vol_element = 4.0 * np.pi**2 * R0 * radius_m
    return float(trapz(profile * vol_element, radius_m))


def collisional_radiative_source_sink_matrices(
    charge_state_density_rz: np.ndarray,
    ne: np.ndarray,
    coeffs: AdasChargeStateCoefficients,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite ionisation and recombination transfer matrices."""
    density = np.asarray(charge_state_density_rz, dtype=np.float64)
    ne_arr = np.asarray(ne, dtype=np.float64)
    ion = np.asarray(coeffs.ionisation_m3_s, dtype=np.float64)
    rec = np.asarray(coeffs.recombination_m3_s, dtype=np.float64)
    if density.ndim != 2:
        raise ValueError("charge_state_density_rz must have shape radius x charge_state")
    if ne_arr.ndim != 1 or ne_arr.shape[0] != density.shape[0]:
        raise ValueError("ne must match the radius dimension")
    if density.shape[1] != ion.shape[0] or density.shape[1] != rec.shape[0]:
        raise ValueError("coefficient charge axis must match density charge axis")
    if not np.all(np.isfinite(density)) or np.any(density < 0.0):
        raise ValueError("charge_state_density_rz must be finite and non-negative")
    if not np.all(np.isfinite(ne_arr)) or np.any(ne_arr <= 0.0):
        raise ValueError("ne must be finite and positive")

    ionisation = ne_arr[:, np.newaxis] * density * ion[np.newaxis, :]
    recombination = ne_arr[:, np.newaxis] * density * rec[np.newaxis, :]
    ionisation[:, -1] = 0.0
    recombination[:, 0] = 0.0
    return ionisation, recombination


def advance_charge_state_collisional_radiative(
    charge_state_density_rz: np.ndarray,
    ne: np.ndarray,
    coeffs: AdasChargeStateCoefficients,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance one conservative charge-state CR step.

    Transfers are pairwise between neighbouring charge states and limited by
    donor inventory, preserving total impurity density per radius without
    introducing negative densities.
    """
    dt_s = float(dt)
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt must be finite and positive")
    density = np.asarray(charge_state_density_rz, dtype=np.float64)
    ionisation, recombination = collisional_radiative_source_sink_matrices(density, ne, coeffs)
    updated = density.copy()
    for z_idx in range(density.shape[1] - 1):
        ion_flux = np.minimum(ionisation[:, z_idx], updated[:, z_idx] / dt_s)
        rec_flux = np.minimum(recombination[:, z_idx + 1], updated[:, z_idx + 1] / dt_s)
        updated[:, z_idx] += dt_s * (rec_flux - ion_flux)
        updated[:, z_idx + 1] += dt_s * (ion_flux - rec_flux)
    return np.maximum(updated, 0.0), ionisation, recombination


def _source_sink_transfer_matrix(ionisation: np.ndarray, recombination: np.ndarray) -> np.ndarray:
    """Return conservative charge-state transfer matrix with from-charge rows."""
    ion = np.asarray(ionisation, dtype=np.float64)
    rec = np.asarray(recombination, dtype=np.float64)
    if ion.shape != rec.shape or ion.ndim != 2:
        raise ValueError("ionisation and recombination must have matching radius x charge shapes")
    matrix = np.zeros((ion.shape[0], ion.shape[1], ion.shape[1]), dtype=np.float64)
    for z_idx in range(ion.shape[1] - 1):
        ion_flux = ion[:, z_idx]
        rec_flux = rec[:, z_idx + 1]
        matrix[:, z_idx, z_idx + 1] += ion_flux
        matrix[:, z_idx + 1, z_idx] += rec_flux
    for z_idx in range(ion.shape[1]):
        off_diagonal_sum = np.sum(matrix[:, z_idx, :], axis=1)
        matrix[:, z_idx, z_idx] = -off_diagonal_sum
    for _ in range(3):
        residual = np.sum(matrix, axis=2)
        for z_idx in range(ion.shape[1]):
            target_idx = z_idx + 1 if z_idx < ion.shape[1] - 1 else z_idx - 1
            matrix[:, z_idx, target_idx] -= residual[:, z_idx]
    return matrix


def build_aurora_strahl_charge_state_artifact(
    *,
    element: str,
    charge_states: np.ndarray | list[int] | tuple[int, ...],
    radius_m: np.ndarray,
    time_s: np.ndarray,
    ne_t_r: np.ndarray,
    Te_t_r: np.ndarray,
    initial_charge_state_density_rz: np.ndarray,
    major_radius_m: float,
) -> AuroraStrahlArtifact:
    """Simulate and export an Aurora/STRAHL-style charge-state artifact."""
    charge_axis = _strict_axis("charge_state", np.asarray(charge_states, dtype=np.float64))
    radius_axis = _strict_axis("radius_m", np.asarray(radius_m, dtype=np.float64))
    time_axis = _strict_axis("time_s", np.asarray(time_s, dtype=np.float64))
    if radius_axis[0] < 0.0:
        raise ValueError("radius_m must be non-negative")
    if not np.isfinite(major_radius_m) or major_radius_m <= 0.0:
        raise ValueError("major_radius_m must be finite and positive")
    ne_hist = np.asarray(ne_t_r, dtype=np.float64)
    te_hist = np.asarray(Te_t_r, dtype=np.float64)
    density = np.asarray(initial_charge_state_density_rz, dtype=np.float64)
    expected_2d = (time_axis.size, radius_axis.size)
    expected_density = (radius_axis.size, charge_axis.size)
    if ne_hist.shape != expected_2d or te_hist.shape != expected_2d:
        raise ValueError("ne_t_r and Te_t_r must have shape time x radius")
    if density.shape != expected_density:
        raise ValueError("initial_charge_state_density_rz must have shape radius x charge_state")
    if not np.all(np.isfinite(ne_hist)) or np.any(ne_hist <= 0.0):
        raise ValueError("ne_t_r must be finite and positive")
    if not np.all(np.isfinite(te_hist)) or np.any(te_hist <= 0.0):
        raise ValueError("Te_t_r must be finite and positive")
    if not np.all(np.isfinite(density)) or np.any(density < 0.0):
        raise ValueError("initial_charge_state_density_rz must be finite and non-negative")

    density_history = [density.copy()]
    line_power_by_charge: list[np.ndarray] = []
    line_power: list[float] = []
    source_sink_history: list[np.ndarray] = []
    inventory_history: list[float] = [
        _volume_integral(np.sum(density, axis=1), radius_axis, major_radius_m)
    ]
    final_ion = np.zeros_like(density)
    final_rec = np.zeros_like(density)
    te_charge = np.interp(
        charge_axis,
        [charge_axis[0], charge_axis[-1]],
        [te_hist[0].min(), te_hist[0].max()],
    )
    coeffs = adas_style_charge_state_coefficients(element, charge_axis, te_charge)
    final_ion, final_rec = collisional_radiative_source_sink_matrices(density, ne_hist[0], coeffs)
    source_sink_history.append(_source_sink_transfer_matrix(final_ion, final_rec))
    rad_density = ne_hist[0, :, np.newaxis] * density * coeffs.line_radiation_w_m3[np.newaxis, :]
    line_power_by_charge.append(rad_density)
    line_power.append(_volume_integral(np.sum(rad_density, axis=1), radius_axis, major_radius_m))
    for step in range(1, time_axis.size):
        dt_s = float(time_axis[step] - time_axis[step - 1])
        te_charge = np.interp(
            charge_axis,
            [charge_axis[0], charge_axis[-1]],
            [te_hist[step].min(), te_hist[step].max()],
        )
        coeffs = adas_style_charge_state_coefficients(element, charge_axis, te_charge)
        density, final_ion, final_rec = advance_charge_state_collisional_radiative(
            density, ne_hist[step], coeffs, dt_s
        )
        source_sink_history.append(_source_sink_transfer_matrix(final_ion, final_rec))
        density_history.append(density.copy())
        rad_density = (
            ne_hist[step, :, np.newaxis] * density * coeffs.line_radiation_w_m3[np.newaxis, :]
        )
        line_power_by_charge.append(rad_density)
        line_power.append(
            _volume_integral(np.sum(rad_density, axis=1), radius_axis, major_radius_m)
        )
        inventory_history.append(
            _volume_integral(np.sum(density, axis=1), radius_axis, major_radius_m)
        )

    density_t_r_z = np.stack(density_history, axis=0)
    total_t_r = np.sum(density_t_r_z, axis=2)
    initial_inventory = _volume_integral(total_t_r[0], radius_axis, major_radius_m)
    final_inventory = _volume_integral(total_t_r[-1], radius_axis, major_radius_m)
    conservation_error = abs(final_inventory - initial_inventory) / max(abs(initial_inventory), 1.0)

    return AuroraStrahlArtifact(
        coordinates={
            "time_s": [float(v) for v in time_axis],
            "radius_m": [float(v) for v in radius_axis],
            "charge_state": [float(v) for v in charge_axis],
        },
        coordinate_units={
            "time_s": "s",
            "radius_m": "m",
            "charge_state": "integer_charge",
        },
        observable_axes={
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
        },
        observable_units={
            "charge_state_density_r_t": "m^-3",
            "total_impurity_density_r_t": "m^-3",
            "line_radiation_power_t": "W",
            "line_radiation_power_t_r_z": "W",
            "source_sink_matrix_t_r_z_z": "m^-3_s^-1",
            "total_impurity_inventory_t": "particles",
            "ionisation_source_matrix": "m^-3_s^-1",
            "recombination_sink_matrix": "m^-3_s^-1",
        },
        observables={
            "charge_state_density_r_t": density_t_r_z.tolist(),
            "total_impurity_density_r_t": total_t_r.tolist(),
            "line_radiation_power_t": [float(v) for v in line_power],
            "line_radiation_power_t_r_z": np.stack(line_power_by_charge, axis=0).tolist(),
            "source_sink_matrix_t_r_z_z": np.stack(source_sink_history, axis=0).tolist(),
            "total_impurity_inventory_t": [float(v) for v in inventory_history],
            "ionisation_source_matrix": final_ion.tolist(),
            "recombination_sink_matrix": final_rec.tolist(),
        },
        conservation={
            "initial_inventory": float(initial_inventory),
            "final_inventory": float(final_inventory),
            "relative_inventory_error": float(conservation_error),
        },
        provenance={
            "reference_family": "Aurora/STRAHL",
            "implementation": "native_charge_state_collisional_radiative_contract_with_adas_style_coefficients",
            "parity_status": "artifact_contract_only_not_public_aurora_strahl_parity",
        },
    )


def neoclassical_impurity_pinch(
    Z: int,
    ne: np.ndarray,
    Te_eV: np.ndarray,
    Ti_eV: np.ndarray,
    q: np.ndarray,
    rho: np.ndarray,
    R0: float,
    a: float,
    epsilon: np.ndarray,
) -> np.ndarray:
    """
    V_neo [m/s] (negative = inward).
    Hirshman & Sigmar, Nucl. Fusion 21, 1079 (1981).
    V_neo = -D_neo [Z/L_n + (Z/2 - H_Z)/L_Ti]
    with inverse scale lengths 1/L_x = -d ln(x)/dr.
    """
    if Z < 1:
        raise ValueError("Z must be positive")
    rho_arr = np.asarray(rho, dtype=float)
    ne_arr = np.asarray(ne, dtype=float)
    ti_arr = np.asarray(Ti_eV, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    eps_arr = np.asarray(epsilon, dtype=float)
    arrays = (rho_arr, ne_arr, ti_arr, q_arr, eps_arr)
    if any(arr.shape != rho_arr.shape for arr in arrays):
        raise ValueError("rho, ne, Ti_eV, q, and epsilon must have matching shapes")
    if rho_arr.size < 3:
        raise ValueError("rho must contain at least three points")
    if not np.all(np.isfinite(rho_arr)) or not np.all(np.diff(rho_arr) > 0.0):
        raise ValueError("rho must be finite and strictly increasing")
    if not np.all(np.isfinite(ne_arr)) or np.any(ne_arr <= 0.0):
        raise ValueError("ne must be finite and positive")
    if not np.all(np.isfinite(ti_arr)) or np.any(ti_arr <= 0.0):
        raise ValueError("Ti_eV must be finite and positive")
    if not np.all(np.isfinite(q_arr)) or np.any(q_arr <= 0.0):
        raise ValueError("q must be finite and positive")
    if not np.all(np.isfinite(eps_arr)) or np.any(eps_arr < 0.0):
        raise ValueError("epsilon must be finite and non-negative")
    if not np.isfinite(R0) or R0 <= 0.0:
        raise ValueError("R0 must be finite and positive")
    if not np.isfinite(a) or a <= 0.0:
        raise ValueError("a must be finite and positive")

    dr = (rho_arr[1] - rho_arr[0]) * a
    inv_Ln = -np.gradient(np.log(ne_arr), dr)
    inv_LTi = -np.gradient(np.log(ti_arr), dr)

    D_NEO = 0.1  # m²/s, banana-regime nominal scale
    D_neo = D_NEO * q_arr**2 / np.sqrt(np.maximum(eps_arr, 0.05))

    H_Z = 0.5  # screening factor, banana regime trace impurities

    V_neo = -D_neo * (Z * inv_Ln + (Z / 2.0 - H_Z) * inv_LTi)
    return np.asarray(V_neo)


def total_radiated_power(
    ne: np.ndarray,
    n_impurity: dict[str, np.ndarray],
    Te_eV: np.ndarray,
    rho: np.ndarray,
    R0: float,
    a: float,
) -> float:
    """
    P_rad in MW.
    """
    p_rad_density = np.zeros_like(rho)

    for element, n_Z in n_impurity.items():
        curve = CoolingCurve(element)
        L = curve.L_z(Te_eV)
        # p_rad = n_e * n_z * L_z
        p_rad_density += ne * n_Z * L

    # Integrate over volume: dV = 4 pi^2 R0 a^2 rho drho
    vol_element = 4.0 * np.pi**2 * R0 * a**2 * rho
    _trapz: Any = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    P_rad_W = _trapz(p_rad_density * vol_element, rho)

    return float(P_rad_W / 1e6)


def tungsten_accumulation_diagnostic(n_W: np.ndarray, ne: np.ndarray) -> dict[str, Any]:
    """Return core/edge tungsten concentration and accumulation danger level."""
    c_W_core = float(n_W[0] / max(ne[0], 1e-6))
    c_W_edge = float(n_W[-1] / max(ne[-1], 1e-6))

    peaking_factor = c_W_core / max(c_W_edge, 1e-12)

    if c_W_core < 1e-5:
        danger = "safe"
    elif c_W_core < 5e-5:
        danger = "warning"
    else:
        danger = "critical"

    return {
        "c_W_core": c_W_core,
        "c_W_edge": c_W_edge,
        "peaking_factor": peaking_factor,
        "danger_level": danger,
    }


class ImpurityTransportSolver:
    """Implicit radial impurity transport solver for multiple species."""

    def __init__(self, rho: np.ndarray, R0: float, a: float, species: list[ImpuritySpecies]):
        """Initialize geometry, species inventory, and radial impurity state."""
        self.rho = np.asarray(rho, dtype=float)
        self.R0 = R0
        self.a = a
        self.species = species

        self.nr = len(self.rho)
        if self.nr < 3:
            raise ValueError("rho must contain at least three radial points")
        if not np.all(np.isfinite(self.rho)):
            raise ValueError("rho must contain only finite values")
        if not np.all(np.diff(self.rho) > 0.0):
            raise ValueError("rho must be strictly increasing")
        if not np.isclose(self.rho[0], 0.0) or not np.isclose(self.rho[-1], 1.0):
            raise ValueError("rho must span the normalised interval [0, 1]")
        if not np.isfinite(R0) or R0 <= 0.0:
            raise ValueError("R0 must be finite and positive")
        if not np.isfinite(a) or a <= 0.0:
            raise ValueError("a must be finite and positive")

        drho = np.diff(self.rho)
        if not np.allclose(drho, drho[0], rtol=1e-6, atol=1e-12):
            raise ValueError("rho grid must be uniformly spaced for the banded transport solve")
        self.drho = float(drho[0])

        self.n_z = {s.element: np.zeros(self.nr) for s in species}

    def step(
        self,
        dt: float,
        ne: np.ndarray,
        Te_eV: np.ndarray,
        Ti_eV: np.ndarray,
        D_anom: float,
        V_pinch: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """
        1D Transport advance for each species.
        Uses simple upwind/centered differences.
        """
        import scipy.linalg

        dr = self.drho * self.a

        for s in self.species:
            n = self.n_z[s.element]
            V = V_pinch.get(s.element, np.zeros(self.nr))

            # D_total = D_anom + D_neo
            D = D_anom * np.ones(self.nr)

            # Implicit advance
            diag = np.zeros(self.nr)
            upper = np.zeros(self.nr)
            lower = np.zeros(self.nr)
            rhs = np.zeros(self.nr)
            source = self._edge_source_density(s)

            # Boundary conditions
            diag[0] = 1.0
            upper[0] = -1.0
            rhs[0] = 0.0  # dn/dr = 0 at axis

            diag[-1] = 1.0
            rhs[-1] = n[-1] + dt * source[-1]

            # Interior
            for i in range(1, self.nr - 1):
                r_val = self.rho[i] * self.a

                # Diffusion term
                coeff_D_plus = D[i] / dr**2 + D[i] / (2.0 * r_val * dr)
                coeff_D_minus = D[i] / dr**2 - D[i] / (2.0 * r_val * dr)
                coeff_D_0 = -2.0 * D[i] / dr**2

                # Convection term (upwind)
                if V[i] > 0:
                    coeff_V_0 = -V[i] / dr - V[i] / r_val
                    coeff_V_minus = V[i] / dr
                    coeff_V_plus = 0.0
                else:
                    coeff_V_0 = V[i] / dr - V[i] / r_val
                    coeff_V_plus = -V[i] / dr
                    coeff_V_minus = 0.0

                lower[i] = -dt * (coeff_D_minus + coeff_V_minus)
                diag[i] = 1.0 - dt * (coeff_D_0 + coeff_V_0)
                upper[i] = -dt * (coeff_D_plus + coeff_V_plus)

                rhs[i] = n[i] + dt * source[i]

            # Solve
            ab = np.zeros((3, self.nr))
            ab[0, 1:] = upper[:-1]
            ab[1, :] = diag
            ab[2, :-1] = lower[1:]

            n_new = scipy.linalg.solve_banded((1, 1), ab, rhs)

            # Replace
            self.n_z[s.element] = np.maximum(n_new, 0.0)

        return self.n_z

    def _edge_source_density(self, species: ImpuritySpecies) -> np.ndarray:
        """Return a volume-normalised edge source density [m^-3 s^-1]."""
        if species.source_rate == 0.0:
            return np.zeros(self.nr)

        width = max(species.source_decay_width_rho, self.drho)
        profile = np.exp(-(1.0 - self.rho) / width)
        profile[self.rho < max(0.0, 1.0 - 8.0 * width)] = 0.0

        vol_element = 4.0 * np.pi**2 * self.R0 * self.a**2 * self.rho
        _trapz: Any = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
        profile_volume = float(_trapz(profile * vol_element, self.rho))
        if profile_volume <= 0.0 or not np.isfinite(profile_volume):
            raise ValueError("edge source profile has zero normalisation")

        edge_area = 4.0 * np.pi**2 * self.R0 * self.a
        total_particles_per_second = species.source_rate * edge_area
        return np.asarray(profile * total_particles_per_second / profile_volume)
