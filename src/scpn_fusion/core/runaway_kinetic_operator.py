# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Conservative Runaway Kinetic Operator
"""Conservative finite-volume evolution on radius, momentum and pitch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from scpn_fusion.core.runaway_kinetic_coefficients import (
    RunawayKineticCoefficients,
)
from scpn_fusion.core.runaway_kinetic_grid import FloatArray, RunawayKineticGrid


Index: TypeAlias = slice | int


def _frozen(values: FloatArray) -> FloatArray:
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class RunawayKineticGeometry:
    """Cell measures and oriented face measures for conservative divergence."""

    cell_measure: FloatArray
    density_cell_measure: FloatArray
    radial_face_measure: FloatArray
    momentum_face_measure: FloatArray
    pitch_face_measure: FloatArray

    @classmethod
    def cylindrical(cls, grid: RunawayKineticGrid) -> RunawayKineticGeometry:
        """Construct exact tensor-product cylindrical phase-space measures."""

        radial_shell = grid.radial_shell_measure_m2
        pitch_width = np.diff(grid.pitch_faces)
        momentum_shell = grid.momentum_shell_measure
        two_pi = 2.0 * np.pi

        cell = (
            radial_shell[:, None, None]
            * pitch_width[None, :, None]
            * (two_pi * momentum_shell)[None, None, :]
        )
        density_cell = (
            np.ones((grid.nr, 1, 1), dtype=np.float64)
            * pitch_width[None, :, None]
            * (two_pi * momentum_shell)[None, None, :]
        )
        radial_face = (
            grid.radius_faces_m[:, None, None]
            * pitch_width[None, :, None]
            * (two_pi * momentum_shell)[None, None, :]
        )
        momentum_face = (
            radial_shell[:, None, None]
            * pitch_width[None, :, None]
            * (two_pi * grid.momentum_faces_mc**2)[None, None, :]
        )
        pitch_face = (
            radial_shell[:, None, None]
            * np.ones((1, grid.nxi + 1, 1), dtype=np.float64)
            * (two_pi * momentum_shell)[None, None, :]
        )
        return cls.checked(
            grid,
            cell_measure=cell,
            density_cell_measure=density_cell,
            radial_face_measure=radial_face,
            momentum_face_measure=momentum_face,
            pitch_face_measure=pitch_face,
        )

    @classmethod
    def checked(
        cls,
        grid: RunawayKineticGrid,
        *,
        cell_measure: FloatArray,
        density_cell_measure: FloatArray,
        radial_face_measure: FloatArray,
        momentum_face_measure: FloatArray,
        pitch_face_measure: FloatArray,
    ) -> RunawayKineticGeometry:
        """Construct geometry imported from an independent kinetic code."""

        expected = {
            "cell_measure": grid.shape,
            "density_cell_measure": grid.shape,
            "radial_face_measure": (grid.nr + 1, grid.nxi, grid.np),
            "momentum_face_measure": (grid.nr, grid.nxi, grid.np + 1),
            "pitch_face_measure": (grid.nr, grid.nxi + 1, grid.np),
        }
        arrays: dict[str, FloatArray] = {}
        supplied = {
            "cell_measure": cell_measure,
            "density_cell_measure": density_cell_measure,
            "radial_face_measure": radial_face_measure,
            "momentum_face_measure": momentum_face_measure,
            "pitch_face_measure": pitch_face_measure,
        }
        for name, shape in expected.items():
            values = np.asarray(supplied[name], dtype=np.float64)
            if values.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {values.shape}")
            if not np.all(np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError(f"{name} must contain finite non-negative values")
            arrays[name] = _frozen(values)
        if np.any(arrays["cell_measure"] <= 0.0):
            raise ValueError("cell_measure must be strictly positive")
        return cls(**arrays)


@dataclass(frozen=True)
class RunawayKineticTendencies:
    """Every independently auditable contribution to ``df/dt``."""

    radial_transport: FloatArray
    electric_acceleration: FloatArray
    collisional_drag_diffusion: FloatArray
    pitch_scattering: FloatArray
    cross_diffusion: FloatArray
    synchrotron_loss: FloatArray
    bremsstrahlung_loss: FloatArray
    avalanche_generation: FloatArray
    external_source: FloatArray
    runaway_density_radial_transport_m3_s: FloatArray
    runaway_density_avalanche_generation_m3_s: FloatArray
    runaway_density_external_source_m3_s: FloatArray
    runaway_density_tendency_m3_s: FloatArray

    @property
    def total(self) -> FloatArray:
        """Complete operator tendency with no hidden or omitted component."""

        return (
            self.radial_transport
            + self.electric_acceleration
            + self.collisional_drag_diffusion
            + self.pitch_scattering
            + self.cross_diffusion
            + self.synchrotron_loss
            + self.bremsstrahlung_loss
            + self.avalanche_generation
            + self.external_source
        )


class RunawayKineticOperator:
    """Evaluate one complete conservative radius-momentum-pitch operator."""

    def __init__(
        self,
        grid: RunawayKineticGrid,
        coefficients: RunawayKineticCoefficients,
        *,
        geometry: RunawayKineticGeometry | None = None,
    ) -> None:
        self.grid = grid
        self.coefficients = coefficients
        self.geometry = geometry or RunawayKineticGeometry.cylindrical(grid)

    @staticmethod
    def _upwind_face(
        state: FloatArray,
        advection: FloatArray,
        axis: int,
        *,
        zero_low: bool = False,
        zero_high: bool = False,
    ) -> FloatArray:
        shape = list(state.shape)
        shape[axis] += 1
        face: FloatArray = np.zeros(shape, dtype=np.float64)

        inner: list[Index] = [slice(None)] * state.ndim
        inner[axis] = slice(1, -1)
        left: list[Index] = [slice(None)] * state.ndim
        left[axis] = slice(None, -1)
        right: list[Index] = [slice(None)] * state.ndim
        right[axis] = slice(1, None)
        a_inner = advection[tuple(inner)]
        face[tuple(inner)] = np.where(
            a_inner >= 0.0,
            state[tuple(left)],
            state[tuple(right)],
        )

        low: list[Index] = [slice(None)] * state.ndim
        low[axis] = 0
        high: list[Index] = [slice(None)] * state.ndim
        high[axis] = -1
        first: list[Index] = [slice(None)] * state.ndim
        first[axis] = 0
        last: list[Index] = [slice(None)] * state.ndim
        last[axis] = -1
        if not zero_low:
            face[tuple(low)] = np.where(advection[tuple(low)] < 0.0, state[tuple(first)], 0.0)
        if not zero_high:
            face[tuple(high)] = np.where(advection[tuple(high)] > 0.0, state[tuple(last)], 0.0)
        return face

    @staticmethod
    def _face_gradient(
        state: FloatArray,
        centres: FloatArray,
        faces: FloatArray,
        *,
        axis: int,
        zero_high: bool,
        high_boundary_distance: float | None = None,
    ) -> FloatArray:
        shape = list(state.shape)
        shape[axis] += 1
        gradient: FloatArray = np.zeros(shape, dtype=np.float64)
        inner: list[Index] = [slice(None)] * state.ndim
        inner[axis] = slice(1, -1)
        left: list[Index] = [slice(None)] * state.ndim
        left[axis] = slice(None, -1)
        right: list[Index] = [slice(None)] * state.ndim
        right[axis] = slice(1, None)
        delta_shape = [1] * state.ndim
        delta_shape[axis] = centres.size - 1
        gradient[tuple(inner)] = (state[tuple(right)] - state[tuple(left)]) / np.diff(
            centres
        ).reshape(delta_shape)

        high: list[Index] = [slice(None)] * state.ndim
        high[axis] = -1
        last: list[Index] = [slice(None)] * state.ndim
        last[axis] = -1
        if not zero_high:
            distance = (
                faces[-1] - centres[-1]
                if high_boundary_distance is None
                else high_boundary_distance
            )
            gradient[tuple(high)] = -state[tuple(last)] / (distance)
        return gradient

    def _divergence(
        self,
        flux: FloatArray,
        face_measure: FloatArray,
        *,
        axis: int,
    ) -> FloatArray:
        weighted = flux * face_measure
        upper: list[Index] = [slice(None)] * weighted.ndim
        upper[axis] = slice(1, None)
        lower: list[Index] = [slice(None)] * weighted.ndim
        lower[axis] = slice(None, -1)
        return -(weighted[tuple(upper)] - weighted[tuple(lower)]) / (self.geometry.cell_measure)

    def _advection_tendency(
        self,
        state: FloatArray,
        advection: FloatArray,
        face_measure: FloatArray,
        *,
        axis: int,
        upwind_advection: FloatArray | None = None,
        zero_low: bool = False,
        zero_high: bool = False,
    ) -> FloatArray:
        interpolation_advection = advection if upwind_advection is None else upwind_advection
        return self._divergence(
            advection
            * self._upwind_face(
                state,
                interpolation_advection,
                axis,
                zero_low=zero_low,
                zero_high=zero_high,
            ),
            face_measure,
            axis=axis,
        )

    def _diffusion_tendency(
        self,
        state: FloatArray,
        diffusion: FloatArray,
        face_measure: FloatArray,
        centres: FloatArray,
        faces: FloatArray,
        *,
        axis: int,
        zero_high: bool,
        high_boundary_distance: float | None = None,
    ) -> FloatArray:
        gradient = self._face_gradient(
            state,
            centres,
            faces,
            axis=axis,
            zero_high=zero_high,
            high_boundary_distance=high_boundary_distance,
        )
        return self._divergence(
            -diffusion * gradient,
            face_measure,
            axis=axis,
        )

    def _cross_gradient(self, state: FloatArray, *, face_axis: int, grad_axis: int) -> FloatArray:
        centres = (
            self.grid.radius_m,
            self.grid.pitch,
            self.grid.momentum_mc,
        )[grad_axis]
        if centres.size == 1:
            cell_gradient = np.zeros_like(state)
        else:
            cell_gradient = np.asarray(
                np.gradient(state, centres, axis=grad_axis, edge_order=1),
                dtype=np.float64,
            )
        shape = list(state.shape)
        shape[face_axis] += 1
        result: FloatArray = np.zeros(shape, dtype=np.float64)
        inner: list[Index] = [slice(None)] * state.ndim
        inner[face_axis] = slice(1, -1)
        left: list[Index] = [slice(None)] * state.ndim
        left[face_axis] = slice(None, -1)
        right: list[Index] = [slice(None)] * state.ndim
        right[face_axis] = slice(1, None)
        result[tuple(inner)] = 0.5 * (cell_gradient[tuple(left)] + cell_gradient[tuple(right)])
        first: list[Index] = [slice(None)] * state.ndim
        first[face_axis] = 0
        last: list[Index] = [slice(None)] * state.ndim
        last[face_axis] = -1
        low: list[Index] = [slice(None)] * state.ndim
        low[face_axis] = 0
        high: list[Index] = [slice(None)] * state.ndim
        high[face_axis] = -1
        result[tuple(low)] = cell_gradient[tuple(first)]
        result[tuple(high)] = cell_gradient[tuple(last)]
        return result

    def evaluate(
        self,
        distribution: FloatArray,
        runaway_density_m3: FloatArray | None = None,
    ) -> RunawayKineticTendencies:
        """Evaluate every operator contribution for one finite distribution."""

        state = self.grid.require_state("distribution", distribution)
        c = self.coefficients
        g = self.geometry

        radial = self._advection_tendency(
            state, c.radial_advection, g.radial_face_measure, axis=0
        ) + self._diffusion_tendency(
            state,
            c.radial_diffusion,
            g.radial_face_measure,
            self.grid.radius_m,
            self.grid.radius_faces_m,
            axis=0,
            zero_high=False,
            high_boundary_distance=float(
                self.grid.radius_m[-1] - self.grid.radius_m[-2]
                if self.grid.nr > 1
                else np.diff(self.grid.radius_faces_m)[0]
            ),
        )
        electric = self._advection_tendency(
            state,
            c.momentum_electric_advection,
            g.momentum_face_measure,
            axis=2,
            upwind_advection=c.momentum_advection,
            zero_low=True,
        ) + self._advection_tendency(
            state,
            c.pitch_electric_advection,
            g.pitch_face_measure,
            axis=1,
            upwind_advection=c.pitch_advection,
            zero_low=True,
            zero_high=True,
        )
        collision = self._advection_tendency(
            state,
            c.momentum_collision_advection,
            g.momentum_face_measure,
            axis=2,
            upwind_advection=c.momentum_advection,
            zero_low=True,
        ) + self._diffusion_tendency(
            state,
            c.momentum_diffusion,
            g.momentum_face_measure,
            self.grid.momentum_mc,
            self.grid.momentum_faces_mc,
            axis=2,
            zero_high=False,
            high_boundary_distance=float(
                self.grid.momentum_mc[-1] - self.grid.momentum_mc[-2]
                if self.grid.np > 1
                else np.diff(self.grid.momentum_faces_mc)[0]
            ),
        )
        pitch_scattering = self._diffusion_tendency(
            state,
            c.pitch_diffusion,
            g.pitch_face_measure,
            self.grid.pitch,
            self.grid.pitch_faces,
            axis=1,
            zero_high=True,
        )
        momentum_cross_flux = -c.momentum_pitch_diffusion * self._cross_gradient(
            state, face_axis=2, grad_axis=1
        )
        pitch_cross_flux = -c.pitch_momentum_diffusion * self._cross_gradient(
            state, face_axis=1, grad_axis=2
        )
        cross = self._divergence(
            momentum_cross_flux, g.momentum_face_measure, axis=2
        ) + self._divergence(pitch_cross_flux, g.pitch_face_measure, axis=1)
        synchrotron = self._advection_tendency(
            state,
            c.momentum_synchrotron_advection,
            g.momentum_face_measure,
            axis=2,
            upwind_advection=c.momentum_advection,
            zero_low=True,
        ) + self._advection_tendency(
            state,
            c.pitch_synchrotron_advection,
            g.pitch_face_measure,
            axis=1,
            upwind_advection=c.pitch_advection,
            zero_low=True,
            zero_high=True,
        )
        bremsstrahlung = self._advection_tendency(
            state,
            c.momentum_bremsstrahlung_advection,
            g.momentum_face_measure,
            axis=2,
            upwind_advection=c.momentum_advection,
            zero_low=True,
        )
        if runaway_density_m3 is None:
            runaway_density = np.sum(state * self.geometry.density_cell_measure, axis=(1, 2))
        else:
            runaway_density = np.asarray(runaway_density_m3, dtype=np.float64)
            if runaway_density.shape != (self.grid.nr,):
                raise ValueError(
                    "runaway_density_m3 must have shape "
                    f"({self.grid.nr},), got {runaway_density.shape}"
                )
            if not np.all(np.isfinite(runaway_density)) or np.any(runaway_density < 0.0):
                raise ValueError("runaway_density_m3 must be finite and non-negative")
        avalanche = (
            c.avalanche_source_kernel
            * c.total_electron_density_m3[:, None, None]
            * runaway_density[:, None, None]
        )
        density_radial_transport = np.sum(
            radial * self.geometry.density_cell_measure,
            axis=(1, 2),
        )
        density_avalanche = c.total_density_avalanche_rate_s_inv * runaway_density
        density_external_source = c.total_density_external_source_m3_s
        density_tendency = density_radial_transport + density_avalanche + density_external_source

        return RunawayKineticTendencies(
            radial_transport=_frozen(radial),
            electric_acceleration=_frozen(electric),
            collisional_drag_diffusion=_frozen(collision),
            pitch_scattering=_frozen(pitch_scattering),
            cross_diffusion=_frozen(cross),
            synchrotron_loss=_frozen(synchrotron),
            bremsstrahlung_loss=_frozen(bremsstrahlung),
            avalanche_generation=_frozen(avalanche),
            external_source=c.external_source,
            runaway_density_radial_transport_m3_s=_frozen(density_radial_transport),
            runaway_density_avalanche_generation_m3_s=_frozen(density_avalanche),
            runaway_density_external_source_m3_s=c.total_density_external_source_m3_s,
            runaway_density_tendency_m3_s=_frozen(density_tendency),
        )


__all__ = [
    "RunawayKineticGeometry",
    "RunawayKineticOperator",
    "RunawayKineticTendencies",
]
