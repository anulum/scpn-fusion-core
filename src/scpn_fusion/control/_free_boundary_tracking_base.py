# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Free-Boundary Tracking Shared State Interface
"""Shared typed state interface for the free-boundary tracking controller mixins.

``FreeBoundaryTrackingController`` is composed from the config, control, runtime and
shot mixins, which freely reference one another's attributes (coil set, response matrix,
objective targets, observer state) and methods (free-boundary solve, objective
evaluation, actuator snapshot/restore, safe-state recovery). Declaring that shared
surface once here lets every mixin be strict-type-checked in isolation; the concrete
attributes are assigned by the controller's ``__init__`` and the config mixin, and the
concrete methods are implemented across the control/runtime/shot mixins.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from scpn_fusion.control._free_boundary_tracking_types import (
    FloatArray,
    _ActuatorSnapshot,
    _ObjectiveBlock,
    _ObservationSnapshot,
)
from scpn_fusion.control.state_estimator import ExtendedKalmanFilter
from scpn_fusion.control.tokamak_flight_sim import FirstOrderActuator


class _FreeBoundaryTrackingState:
    """Typed declaration of the composed-controller state shared across the mixins."""

    # Kernel, estimator and scalar identification/solve settings.
    kernel: Any
    state_estimator: ExtendedKalmanFilter | None
    verbose: bool
    identification_perturbation: float
    correction_limit: float
    response_regularization: float
    response_refresh_steps: int
    solve_max_outer_iter: int
    solve_tol: float

    # Coil set and actuator configuration.
    coils: Any
    n_coils: int
    coil_current_limits: FloatArray
    control_dt_s: float
    coil_actuator_tau_s: float
    coil_slew_limits: FloatArray
    hold_steps_after_reject: int
    supervisor_limits: dict[str, float]
    observer_gain: float
    observer_forgetting: float
    observer_max_abs: float
    fallback_currents: FloatArray | None
    _hold_steps_remaining: int
    _coil_actuators: list[FirstOrderActuator]

    # Objective targets and measurement-noise vectors.
    objective_tolerances: dict[str, float]
    target_vector: FloatArray
    objective_blocks: tuple[_ObjectiveBlock, ...]
    measurement_bias_vector: FloatArray
    measurement_drift_per_step: FloatArray
    measurement_correction_bias: FloatArray
    measurement_correction_drift_per_step: FloatArray
    measurement_latency_steps: int
    latency_compensation_gain: float
    latency_rate_max_abs: float
    control_objective_weights: FloatArray
    last_coil_penalties: FloatArray

    # Response-matrix identification diagnostics.
    response_matrix: FloatArray
    response_degenerate: bool
    response_rank: int
    response_condition_number: float
    response_max_singular_value: float

    # Runtime observer and latency state.
    objective_bias_estimate: FloatArray
    objective_rate_estimate: FloatArray
    history: dict[str, list[float]]
    measurement_drift_state: FloatArray
    measurement_correction_drift_state: FloatArray
    _measurement_latency_buffer: deque[FloatArray]
    _last_delayed_measurement: FloatArray | None

    def _solve_free_boundary_state(self) -> dict[str, Any]:
        """Solve the free-boundary Grad-Shafranov equilibrium for the current coils."""
        raise NotImplementedError

    def evaluate_objectives(self, observation: np.ndarray[Any, Any]) -> dict[str, Any]:
        """Evaluate the shape/flux control objectives for an observation vector."""
        raise NotImplementedError

    def _observe_snapshot(self, *, apply_latency: bool = True) -> _ObservationSnapshot:
        """Observe the current objective snapshot, optionally applying measurement latency."""
        raise NotImplementedError

    def _restore_actuator_states(self, snapshots: tuple[_ActuatorSnapshot, ...]) -> None:
        """Restore coil actuators to previously captured snapshots."""
        raise NotImplementedError

    def _recover_to_safe_state(
        self,
        *,
        actuator_snapshot: tuple[_ActuatorSnapshot, ...],
        baseline_currents: FloatArray,
        metrics_before: dict[str, Any],
        true_metrics_before: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], float, bool]:
        """Roll the controller back to a safe baseline after a rejected step."""
        raise NotImplementedError

    def _snapshot_actuator_states(self) -> tuple[_ActuatorSnapshot, ...]:
        """Capture the current coil actuator states for later restoration."""
        raise NotImplementedError

    def _observe_objectives(self) -> FloatArray:
        """Observe the latency-shaped objective vector for the current step."""
        raise NotImplementedError

    def _observe_true_objectives(self) -> FloatArray:
        """Observe the noise-free objective vector for diagnostics."""
        raise NotImplementedError

    def _apply_commanded_currents(self, commanded: FloatArray) -> FloatArray:
        """Apply commanded coil currents through the actuator model."""
        raise NotImplementedError

    def _command_currents(self, delta_currents: FloatArray, gain: float) -> FloatArray:
        """Command an incremental coil-current update scaled by gain."""
        raise NotImplementedError

    def _apply_correction(self, delta_currents: FloatArray, gain: float) -> FloatArray:
        """Apply a gain-scaled coil-current correction to the actuators."""
        raise NotImplementedError

    def _map_ekf_to_observation(
        self, observation: FloatArray, x_ekf: np.ndarray[Any, Any]
    ) -> FloatArray:
        """Map an EKF state estimate back into the observation space."""
        raise NotImplementedError

    def _map_observation_to_ekf(self, observation: FloatArray) -> FloatArray | None:
        """Map an observation vector into the EKF state space."""
        raise NotImplementedError

    def compute_correction(
        self, observation: FloatArray, *, metrics: dict[str, Any] | None = None
    ) -> FloatArray:
        """Compute the coil-current correction for an objective observation."""
        raise NotImplementedError

    def _update_objective_observer(self, observation: FloatArray) -> None:
        """Update the objective bias/rate observer with a new observation."""
        raise NotImplementedError

    def identify_response_matrix(self, perturbation: float | None = None) -> FloatArray:
        """Identify the coil-to-objective response matrix by perturbation."""
        raise NotImplementedError

    def evaluate_supervisor(
        self,
        metrics: dict[str, Any],
        *,
        max_abs_coil_current: float,
        max_abs_actuator_lag: float,
    ) -> dict[str, Any]:
        """Evaluate supervisor safety limits for the current step metrics."""
        raise NotImplementedError

    def _sync_actuators_from_coils(self) -> None:
        """Synchronise actuator setpoints from the current coil currents."""
        raise NotImplementedError

    def _reset_latency_estimator(self) -> None:
        """Reset the measurement-latency compensation estimator."""
        raise NotImplementedError

    def _reset_measurement_offsets(self) -> None:
        """Reset the accumulated measurement bias/drift offsets."""
        raise NotImplementedError

    def _reset_objective_observer(self) -> None:
        """Reset the objective bias/rate observer state."""
        raise NotImplementedError

    def _advance_measurement_offsets(self) -> None:
        """Advance the measurement drift offsets by one control step."""
        raise NotImplementedError

    def _log(self, message: str) -> None:
        """Emit a controller log line when verbose."""
        raise NotImplementedError

    def _current_measurement_offset(self) -> FloatArray:
        """Return the current accumulated measurement offset vector."""
        raise NotImplementedError

    @staticmethod
    def _detect_tolerance_regressions(
        metrics_before: dict[str, Any], metrics_after: dict[str, Any]
    ) -> dict[str, bool]:
        """Flag per-objective tolerance regressions between two metric sets."""
        raise NotImplementedError


__all__ = ["_FreeBoundaryTrackingState"]
