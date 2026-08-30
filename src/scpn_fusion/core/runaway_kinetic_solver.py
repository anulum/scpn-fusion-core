# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Full Runaway Kinetic Solver
"""Public time integrator for the full radius-momentum-pitch operator."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np

from scpn_fusion.core.runaway_kinetic_grid import FloatArray
from scpn_fusion.core.runaway_kinetic_operator import (
    RunawayKineticOperator,
    RunawayKineticTendencies,
)


ELECTRON_CHARGE_C = 1.602176634e-19
SPEED_OF_LIGHT_M_PER_S = 299792458.0
ELECTRON_REST_ENERGY_J = 8.1871057769e-14


def _readonly(values: FloatArray) -> FloatArray:
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class RunawayKineticMoments:
    """Radially resolved moments of a kinetic trajectory."""

    density_m3: FloatArray
    current_density_a_m2: FloatArray
    kinetic_energy_density_j_m3: FloatArray


@dataclass(frozen=True)
class RunawayKineticTrajectory:
    """Full state history, operator budgets and resolved physical moments."""

    times_s: FloatArray
    distribution: FloatArray
    radial_transport: FloatArray
    electric_acceleration: FloatArray
    collisional_drag_diffusion: FloatArray
    pitch_scattering: FloatArray
    cross_diffusion: FloatArray
    synchrotron_loss: FloatArray
    bremsstrahlung_loss: FloatArray
    avalanche_generation: FloatArray
    external_source: FloatArray
    total_tendency: FloatArray
    runaway_density_m3: FloatArray
    runaway_density_radial_transport_m3_s: FloatArray
    runaway_density_avalanche_generation_m3_s: FloatArray
    runaway_density_external_source_m3_s: FloatArray
    runaway_density_tendency_m3_s: FloatArray
    moments: RunawayKineticMoments
    internal_steps: int
    minimum_distribution: float


class RunawayKineticSolver:
    """Deterministic SSPRK3 evolution of every physical kinetic axis.

    ``maximum_step_s`` is part of the numerical contract rather than a hidden
    adaptive heuristic.  Convergence studies must refine it together with the
    radius, momentum, pitch and requested-output time grids.
    """

    def __init__(
        self,
        operator: RunawayKineticOperator,
        *,
        maximum_step_s: float,
        negativity_tolerance: float = 1.0e-12,
    ) -> None:
        if not np.isfinite(maximum_step_s) or maximum_step_s <= 0.0:
            raise ValueError("maximum_step_s must be finite and positive")
        if not np.isfinite(negativity_tolerance) or negativity_tolerance < 0.0:
            raise ValueError("negativity_tolerance must be finite and non-negative")
        self.operator = operator
        self.maximum_step_s = float(maximum_step_s)
        self.negativity_tolerance = float(negativity_tolerance)

    @staticmethod
    def _rhs(
        operator: RunawayKineticOperator,
        state: FloatArray,
        runaway_density_m3: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        tendency = operator.evaluate(state, runaway_density_m3)
        return (
            np.asarray(tendency.total, dtype=np.float64),
            np.asarray(tendency.runaway_density_tendency_m3_s, dtype=np.float64),
        )

    def _step(
        self,
        state: FloatArray,
        density: FloatArray,
        dt: float,
    ) -> tuple[FloatArray, FloatArray]:
        rhs0, density_rhs0 = self._rhs(self.operator, state, density)
        first = state + dt * rhs0
        density_first = density + dt * density_rhs0
        rhs1, density_rhs1 = self._rhs(self.operator, first, density_first)
        second = 0.75 * state + 0.25 * (first + dt * rhs1)
        density_second = 0.75 * density + 0.25 * (density_first + dt * density_rhs1)
        rhs2, density_rhs2 = self._rhs(self.operator, second, density_second)
        state_result = cast(
            FloatArray,
            (1.0 / 3.0) * state + (2.0 / 3.0) * (second + dt * rhs2),
        )
        density_result = cast(
            FloatArray,
            (1.0 / 3.0) * density + (2.0 / 3.0) * (density_second + dt * density_rhs2),
        )
        return state_result, density_result

    def _moments(self, history: FloatArray) -> RunawayKineticMoments:
        grid = self.operator.grid
        weight = self.operator.geometry.density_cell_measure
        density = np.sum(history * weight[None, :, :, :], axis=(2, 3))

        p = grid.momentum_mc[None, :]
        xi = grid.pitch[:, None]
        gamma = np.sqrt(1.0 + p * p)
        parallel_speed = SPEED_OF_LIGHT_M_PER_S * p * xi / gamma
        current_weight = weight * parallel_speed[None, :, :]
        current = ELECTRON_CHARGE_C * np.sum(history * current_weight[None, :, :, :], axis=(2, 3))
        energy_weight = weight * (gamma - 1.0)[None, :, :] * ELECTRON_REST_ENERGY_J
        energy = np.sum(history * energy_weight[None, :, :, :], axis=(2, 3))
        return RunawayKineticMoments(
            density_m3=_readonly(density),
            current_density_a_m2=_readonly(current),
            kinetic_energy_density_j_m3=_readonly(energy),
        )

    @staticmethod
    def _stack_tendencies(
        tendencies: list[RunawayKineticTendencies],
        name: str,
    ) -> FloatArray:
        stacked = np.stack([getattr(item, name) for item in tendencies])
        return _readonly(cast(FloatArray, stacked))

    def solve(
        self,
        initial_distribution: FloatArray,
        times_s: FloatArray,
        *,
        initial_runaway_density_m3: FloatArray | None = None,
        backend: Literal["numpy", "rust"] = "numpy",
    ) -> RunawayKineticTrajectory:
        """Evolve and return the unprojected distribution at every given time."""

        if backend == "rust":
            return self._solve_rust(
                initial_distribution,
                times_s,
                initial_runaway_density_m3=initial_runaway_density_m3,
            )
        if backend != "numpy":
            raise ValueError("backend must be exactly 'numpy' or 'rust'")

        times = np.asarray(times_s, dtype=np.float64)
        if times.ndim != 1 or times.size < 2:
            raise ValueError("times_s must contain at least two one-dimensional entries")
        if not np.all(np.isfinite(times)) or times[0] != 0.0:
            raise ValueError("times_s must be finite and start exactly at zero")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("times_s must be strictly increasing")

        state = self.operator.grid.require_state("initial_distribution", initial_distribution)
        if initial_runaway_density_m3 is None:
            density = np.sum(
                state * self.operator.geometry.density_cell_measure,
                axis=(1, 2),
            )
        else:
            density = np.asarray(initial_runaway_density_m3, dtype=np.float64)
            if density.shape != (self.operator.grid.nr,):
                raise ValueError(
                    "initial_runaway_density_m3 must have shape "
                    f"({self.operator.grid.nr},), got {density.shape}"
                )
            if not np.all(np.isfinite(density)) or np.any(density < 0.0):
                raise ValueError("initial_runaway_density_m3 must be finite and non-negative")
            density = np.array(density, copy=True)
        scale = max(float(np.max(np.abs(state))), 1.0)
        history = [np.array(state, copy=True)]
        density_history = [np.array(density, copy=True)]
        internal_steps = 0

        for start, stop in zip(times[:-1], times[1:], strict=True):
            interval = float(stop - start)
            count = max(1, int(np.ceil(interval / self.maximum_step_s)))
            dt = interval / count
            for _ in range(count):
                state, density = self._step(state, density, dt)
                internal_steps += 1
                minimum = float(np.min(state))
                if not np.all(np.isfinite(state)):
                    raise FloatingPointError("kinetic evolution produced a non-finite state")
                if minimum < -self.negativity_tolerance * scale:
                    raise FloatingPointError(
                        f"kinetic evolution violated the declared negativity tolerance: {minimum}"
                    )
                if not np.all(np.isfinite(density)) or np.any(density < 0.0):
                    raise FloatingPointError("runaway-density evolution produced an invalid state")
            history.append(np.array(state, copy=True))
            density_history.append(np.array(density, copy=True))

        distribution = np.stack(history)
        density_trajectory = np.stack(density_history)
        tendency_history = [
            self.operator.evaluate(frame, frame_density)
            for frame, frame_density in zip(distribution, density_trajectory, strict=True)
        ]
        total = _readonly(np.stack([item.total for item in tendency_history]))
        return RunawayKineticTrajectory(
            times_s=_readonly(times),
            distribution=_readonly(distribution),
            radial_transport=self._stack_tendencies(tendency_history, "radial_transport"),
            electric_acceleration=self._stack_tendencies(tendency_history, "electric_acceleration"),
            collisional_drag_diffusion=self._stack_tendencies(
                tendency_history, "collisional_drag_diffusion"
            ),
            pitch_scattering=self._stack_tendencies(tendency_history, "pitch_scattering"),
            cross_diffusion=self._stack_tendencies(tendency_history, "cross_diffusion"),
            synchrotron_loss=self._stack_tendencies(tendency_history, "synchrotron_loss"),
            bremsstrahlung_loss=self._stack_tendencies(tendency_history, "bremsstrahlung_loss"),
            avalanche_generation=self._stack_tendencies(tendency_history, "avalanche_generation"),
            external_source=self._stack_tendencies(tendency_history, "external_source"),
            total_tendency=total,
            runaway_density_m3=_readonly(density_trajectory),
            runaway_density_radial_transport_m3_s=self._stack_tendencies(
                tendency_history, "runaway_density_radial_transport_m3_s"
            ),
            runaway_density_avalanche_generation_m3_s=self._stack_tendencies(
                tendency_history,
                "runaway_density_avalanche_generation_m3_s",
            ),
            runaway_density_external_source_m3_s=self._stack_tendencies(
                tendency_history, "runaway_density_external_source_m3_s"
            ),
            runaway_density_tendency_m3_s=_readonly(
                np.stack([item.runaway_density_tendency_m3_s for item in tendency_history])
            ),
            moments=self._moments(distribution),
            internal_steps=internal_steps,
            minimum_distribution=float(np.min(distribution)),
        )

    def _solve_rust(
        self,
        initial_distribution: FloatArray,
        times_s: FloatArray,
        *,
        initial_runaway_density_m3: FloatArray | None,
    ) -> RunawayKineticTrajectory:
        """Run the compiled full-fidelity backend without silent fallback."""

        try:
            extension = importlib.import_module("scpn_fusion_rs")
        except ImportError as exc:
            raise RuntimeError(
                "backend='rust' requires the compiled scpn_fusion_rs extension"
            ) from exc
        solve_rust = cast(Any, getattr(extension, "runaway_kinetic_solve_rust", None))
        if solve_rust is None:
            raise RuntimeError(
                "the installed scpn_fusion_rs extension lacks "
                "runaway_kinetic_solve_rust; rebuild the workspace extension"
            )
        c = self.operator.coefficients
        g = self.operator.geometry
        coefficients = {
            "radial_advection": c.radial_advection,
            "momentum_electric_advection": c.momentum_electric_advection,
            "momentum_collision_advection": c.momentum_collision_advection,
            "momentum_synchrotron_advection": c.momentum_synchrotron_advection,
            "momentum_bremsstrahlung_advection": c.momentum_bremsstrahlung_advection,
            "pitch_electric_advection": c.pitch_electric_advection,
            "pitch_synchrotron_advection": c.pitch_synchrotron_advection,
            "radial_diffusion": c.radial_diffusion,
            "momentum_diffusion": c.momentum_diffusion,
            "pitch_diffusion": c.pitch_diffusion,
            "momentum_pitch_diffusion": c.momentum_pitch_diffusion,
            "pitch_momentum_diffusion": c.pitch_momentum_diffusion,
            "avalanche_source_kernel": c.avalanche_source_kernel,
            "total_electron_density_m3": c.total_electron_density_m3,
            "total_density_avalanche_rate_s_inv": c.total_density_avalanche_rate_s_inv,
            "total_density_external_source_m3_s": c.total_density_external_source_m3_s,
            "external_source": c.external_source,
        }
        geometry = {
            "cell_measure": g.cell_measure,
            "density_cell_measure": g.density_cell_measure,
            "radial_face_measure": g.radial_face_measure,
            "momentum_face_measure": g.momentum_face_measure,
            "pitch_face_measure": g.pitch_face_measure,
        }
        grid = self.operator.grid
        raw = cast(
            dict[str, Any],
            solve_rust(
                grid.radius_faces_m,
                grid.pitch_faces,
                grid.momentum_faces_mc,
                coefficients,
                geometry,
                np.asarray(initial_distribution, dtype=np.float64),
                np.asarray(times_s, dtype=np.float64),
                None
                if initial_runaway_density_m3 is None
                else np.asarray(initial_runaway_density_m3, dtype=np.float64),
                self.maximum_step_s,
                self.negativity_tolerance,
            ),
        )

        def array(name: str) -> FloatArray:
            return _readonly(np.asarray(raw[name], dtype=np.float64))

        return RunawayKineticTrajectory(
            times_s=array("times_s"),
            distribution=array("distribution"),
            radial_transport=array("radial_transport"),
            electric_acceleration=array("electric_acceleration"),
            collisional_drag_diffusion=array("collisional_drag_diffusion"),
            pitch_scattering=array("pitch_scattering"),
            cross_diffusion=array("cross_diffusion"),
            synchrotron_loss=array("synchrotron_loss"),
            bremsstrahlung_loss=array("bremsstrahlung_loss"),
            avalanche_generation=array("avalanche_generation"),
            external_source=array("external_source"),
            total_tendency=array("total_tendency"),
            runaway_density_m3=array("runaway_density_m3"),
            runaway_density_radial_transport_m3_s=array("runaway_density_radial_transport_m3_s"),
            runaway_density_avalanche_generation_m3_s=array(
                "runaway_density_avalanche_generation_m3_s"
            ),
            runaway_density_external_source_m3_s=array("runaway_density_external_source_m3_s"),
            runaway_density_tendency_m3_s=array("runaway_density_tendency_m3_s"),
            moments=RunawayKineticMoments(
                density_m3=array("density_m3"),
                current_density_a_m2=array("current_density_a_m2"),
                kinetic_energy_density_j_m3=array("kinetic_energy_density_j_m3"),
            ),
            internal_steps=int(raw["internal_steps"]),
            minimum_distribution=float(raw["minimum_distribution"]),
        )


__all__ = [
    "RunawayKineticMoments",
    "RunawayKineticSolver",
    "RunawayKineticTrajectory",
]
