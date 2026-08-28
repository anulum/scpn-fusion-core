# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Differentiable Coupled Transport Runtime
"""End-to-end autodiff and bounded optimisation for coupled transport.

The functions here reproduce the public prescribed-coefficient Ti/Te/ne/flux
path with JAX-native tridiagonal solves. They are attached to the production
``TransportSolver`` class and are checked against stateful production replay;
they do not imply differentiability of transport models outside this declared
coupled model intersection.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from scpn_fusion.core._integrated_transport_solver_base import TransportSolverState
from scpn_fusion.core._integrated_transport_solver_differentiable_numerics import (
    JaxState,
    objective,
    one_step,
)
from scpn_fusion.core.current_diffusion import CurrentDiffusionSolver
from scpn_fusion.core.integrated_transport_solver_coupled_contracts import (
    CoupledTransportInputs,
    FloatArray,
)
from scpn_fusion.core.integrated_transport_solver_differentiable_contracts import (
    CoupledTransportControls,
    CoupledTransportOptimisationResult,
    CoupledTransportTarget,
    DifferentiableCoupledTransportResult,
)

JaxEvaluation = tuple[tuple[jax.Array, JaxState], jax.Array]


class CoupledTransportDifferentiableMixin(TransportSolverState):
    """Production-bound public autodiff surface for the coupled runtime."""

    _coupled_current_solver: CurrentDiffusionSolver | None
    _coupled_current_geometry: tuple[float, float, float] | None

    def _differentiable_initial_state(
        self, inputs: CoupledTransportInputs
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        geometry = (inputs.major_radius_m, inputs.minor_radius_m, inputs.magnetic_field_t)
        current_solver = getattr(self, "_coupled_current_solver", None)
        if (
            current_solver is not None
            and getattr(self, "_coupled_current_geometry", None) == geometry
        ):
            psi = current_solver.psi.copy()
        else:
            psi = CurrentDiffusionSolver(
                np.asarray(self.rho, dtype=np.float64),
                R0=inputs.major_radius_m,
                a=inputs.minor_radius_m,
                B0=inputs.magnetic_field_t,
            ).psi
        return (
            np.asarray(self.Ti, dtype=np.float64).copy(),
            np.asarray(self.Te, dtype=np.float64).copy(),
            np.asarray(self.ne, dtype=np.float64).copy(),
            np.asarray(psi, dtype=np.float64).copy(),
        )

    def _coupled_evaluator(
        self,
        inputs_history: Sequence[CoupledTransportInputs],
        target: CoupledTransportTarget,
    ) -> Callable[[jax.Array], JaxEvaluation]:
        if not inputs_history:
            raise ValueError("inputs_history must contain at least one step")
        first = inputs_history[0]
        geometry = (first.major_radius_m, first.minor_radius_m, first.magnetic_field_t)
        if any(
            (item.major_radius_m, item.minor_radius_m, item.magnetic_field_t) != geometry
            for item in inputs_history
        ):
            raise ValueError("all differentiable steps must share one geometry")
        radial_points = int(np.asarray(self.rho).size)
        checked_target = target.validated(radial_points)
        initial = self._differentiable_initial_state(first)
        rho = jnp.asarray(self.rho, dtype=jnp.float64)
        initial_jax = (
            jnp.asarray(initial[0], dtype=jnp.float64),
            jnp.asarray(initial[1], dtype=jnp.float64),
            jnp.asarray(initial[2], dtype=jnp.float64),
            jnp.asarray(initial[3], dtype=jnp.float64),
        )
        target_jax = (
            jnp.asarray(checked_target.ion_temperature_kev, dtype=jnp.float64),
            jnp.asarray(checked_target.electron_temperature_kev, dtype=jnp.float64),
            jnp.asarray(checked_target.electron_density_1e19_m3, dtype=jnp.float64),
            jnp.asarray(checked_target.poloidal_flux_wb_per_rad, dtype=jnp.float64),
        )

        def evaluated(control_values: jax.Array) -> tuple[jax.Array, Any]:
            state = initial_jax
            for item in inputs_history:
                state = one_step(state, control_values, rho, item)
            return objective(state, target_jax, checked_target.state_weights), state

        return cast(
            "Callable[[jax.Array], JaxEvaluation]",
            jax.jit(jax.value_and_grad(evaluated, has_aux=True)),
        )

    @staticmethod
    def _result_from_evaluation(
        controls: CoupledTransportControls,
        evaluation: JaxEvaluation,
    ) -> DifferentiableCoupledTransportResult:
        (objective, final_state), gradient = evaluation
        final_numpy = [np.asarray(values, dtype=np.float64) for values in final_state]
        return DifferentiableCoupledTransportResult(
            controls=controls,
            objective=float(objective),
            gradient=np.asarray(gradient, dtype=np.float64),
            ion_temperature_kev=final_numpy[0],
            electron_temperature_kev=final_numpy[1],
            electron_density_1e19_m3=final_numpy[2],
            poloidal_flux_wb_per_rad=final_numpy[3],
        )

    def differentiate_coupled_transport(
        self,
        inputs_history: Sequence[CoupledTransportInputs],
        controls: CoupledTransportControls,
        target: CoupledTransportTarget,
    ) -> DifferentiableCoupledTransportResult:
        """Evaluate a full-trajectory objective and its three control gradients."""
        evaluator = self._coupled_evaluator(inputs_history, target)
        evaluation = evaluator(jnp.asarray(controls.as_array(), dtype=jnp.float64))
        return self._result_from_evaluation(controls, evaluation)

    def optimise_coupled_transport(
        self,
        inputs_history: Sequence[CoupledTransportInputs],
        initial_controls: CoupledTransportControls,
        target: CoupledTransportTarget,
        *,
        iterations: int = 24,
        learning_rate: float = 0.08,
        lower_bound: float = 0.5,
        upper_bound: float = 1.5,
    ) -> CoupledTransportOptimisationResult:
        """Run deterministic bounded Adam over heat, particle, and current scales."""
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and > 0")
        if not 0.0 < lower_bound < upper_bound:
            raise ValueError("control bounds must satisfy 0 < lower < upper")
        controls = initial_controls.as_array()
        evaluator = self._coupled_evaluator(inputs_history, target)
        first = self._result_from_evaluation(
            initial_controls,
            evaluator(jnp.asarray(controls, dtype=jnp.float64)),
        )
        objectives = [first.objective]
        history = [controls.copy()]
        moment = np.zeros(3, dtype=np.float64)
        velocity = np.zeros(3, dtype=np.float64)
        gradient = first.gradient
        for iteration in range(1, iterations + 1):
            moment = 0.9 * moment + 0.1 * gradient
            velocity = 0.999 * velocity + 0.001 * gradient**2
            corrected_moment = moment / (1.0 - 0.9**iteration)
            corrected_velocity = velocity / (1.0 - 0.999**iteration)
            controls = np.clip(
                controls - learning_rate * corrected_moment / (np.sqrt(corrected_velocity) + 1e-8),
                lower_bound,
                upper_bound,
            )
            next_controls = CoupledTransportControls.from_array(controls)
            evaluation = self._result_from_evaluation(
                next_controls,
                evaluator(jnp.asarray(controls, dtype=jnp.float64)),
            )
            objectives.append(evaluation.objective)
            history.append(controls.copy())
            gradient = evaluation.gradient
        return CoupledTransportOptimisationResult(
            initial_controls=initial_controls,
            final_controls=CoupledTransportControls.from_array(controls),
            initial_objective=objectives[0],
            final_objective=objectives[-1],
            objective_history=np.asarray(objectives, dtype=np.float64),
            control_history=np.asarray(history, dtype=np.float64),
            final_gradient=np.asarray(gradient, dtype=np.float64),
            iterations=iterations,
        )


def scaled_inputs(
    inputs: CoupledTransportInputs, controls: CoupledTransportControls
) -> CoupledTransportInputs:
    """Apply controls to a production-runtime input for perturbation replay."""
    return replace(
        inputs,
        heat_power_w=inputs.heat_power_w * controls.heat_power_scale,
        particle_rate_s=inputs.particle_rate_s * controls.particle_rate_scale,
        driven_current_a=inputs.driven_current_a * controls.driven_current_scale,
    )


__all__ = [
    "CoupledTransportControls",
    "CoupledTransportDifferentiableMixin",
    "CoupledTransportOptimisationResult",
    "CoupledTransportTarget",
    "DifferentiableCoupledTransportResult",
    "scaled_inputs",
]
