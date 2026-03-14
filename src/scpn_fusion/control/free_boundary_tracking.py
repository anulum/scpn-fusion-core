# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Free-Boundary Tracking Control
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: AGPL-3.0-or-later
# ──────────────────────────────────────────────────────────────────────
"""Closed-loop free-boundary tracking without a surrogate plant.

The controller re-identifies the local coil-to-objective response directly
from repeated :class:`scpn_fusion.core.fusion_kernel.FusionKernel` solves and
applies bounded least-squares coil corrections. When configured, supervisor
rejection can ramp the coil set toward explicit safe fallback currents instead
of freezing at the previous command, and an objective-space disturbance
observer can accumulate persistent residuals without introducing a reduced-order
plant model. Deterministic objective-space sensor bias and drift can also be
applied and compensated through configuration so hidden plant performance stays
visible during calibration-fault stress tests. Fixed-step measurement latency
can be injected in the same objective space, and an extrapolating current-state
estimator can compensate that latency without replacing the full kernel.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.control.tokamak_flight_sim import FirstOrderActuator
from scpn_fusion.core.fusion_kernel import CoilSet, FusionKernel
from scpn_fusion.control.state_estimator import ExtendedKalmanFilter

logger = logging.getLogger(__name__)
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class _ObjectiveBlock:
    name: str
    start: int
    stop: int


@dataclass(frozen=True)
class _ActuatorSnapshot:
    state: float
    delay_buffer: tuple[float, ...]


@dataclass(frozen=True)
class _ObservationSnapshot:
    true: FloatArray
    measured: FloatArray
    delayed: FloatArray
    effective: FloatArray


class FreeBoundaryTrackingController:
    """Direct free-boundary controller using local coil-response identification.

    This path keeps the full Grad-Shafranov kernel in the loop instead of
    replacing it with a reduced-order plant.

    Examples
    --------
    >>> from scpn_fusion.control.free_boundary_tracking import run_free_boundary_tracking
    >>> summary = run_free_boundary_tracking(
    ...     "iter_config.json",
    ...     shot_steps=3,
    ...     gain=0.8,
    ...     verbose=False,
    ... )
    >>> summary["boundary_variant"]
    'free_boundary'
    """

    def __init__(
        self,
        config_file: str,
        *,
        kernel_factory: Callable[[str], Any] = FusionKernel,
        verbose: bool = True,
        identification_perturbation: float = 0.25,
        correction_limit: float = 2.0,
        response_regularization: float = 1e-3,
        response_refresh_steps: int = 1,
        solve_max_outer_iter: int = 10,
        solve_tol: float = 1e-4,
        objective_tolerances: dict[str, float] | None = None,
        control_dt_s: float | None = None,
        coil_actuator_tau_s: float | None = None,
        coil_slew_limits: float | list[float] | None = None,
        supervisor_limits: dict[str, float] | None = None,
        hold_steps_after_reject: int | None = None,
        state_estimator: ExtendedKalmanFilter | None = None,
    ) -> None:
        self.kernel = kernel_factory(config_file)
        self.state_estimator = state_estimator
        self.verbose = bool(verbose)
        self.identification_perturbation = float(identification_perturbation)
        if (
            not np.isfinite(self.identification_perturbation)
            or self.identification_perturbation <= 0.0
        ):
            raise ValueError("identification_perturbation must be finite and > 0.")
        self.correction_limit = float(correction_limit)
        if not np.isfinite(self.correction_limit) or self.correction_limit <= 0.0:
            raise ValueError("correction_limit must be finite and > 0.")
        self.response_regularization = float(response_regularization)
        if not np.isfinite(self.response_regularization) or self.response_regularization < 0.0:
            raise ValueError("response_regularization must be finite and >= 0.")
        self.response_refresh_steps = int(response_refresh_steps)
        if self.response_refresh_steps < 1:
            raise ValueError("response_refresh_steps must be >= 1.")
        self.solve_max_outer_iter = int(solve_max_outer_iter)
        if self.solve_max_outer_iter < 1:
            raise ValueError("solve_max_outer_iter must be >= 1.")
        self.solve_tol = float(solve_tol)
        if not np.isfinite(self.solve_tol) or self.solve_tol <= 0.0:
            raise ValueError("solve_tol must be finite and > 0.")

        build_coilset = getattr(self.kernel, "build_coilset_from_config", None)
        if not callable(build_coilset):
            raise TypeError(
                "kernel must define build_coilset_from_config() for free-boundary tracking."
            )
        self.coils = build_coilset()
        self.n_coils = int(len(self.coils.positions))
        if self.n_coils < 1:
            raise ValueError("free-boundary tracking requires at least one external coil.")

        if self.coils.current_limits is None:
            self.coil_current_limits = np.full(self.n_coils, np.inf, dtype=np.float64)
        else:
            self.coil_current_limits = np.asarray(
                self.coils.current_limits, dtype=np.float64
            ).reshape(-1)
            if self.coil_current_limits.shape != (self.n_coils,):
                raise ValueError("CoilSet.current_limits must match the number of coils.")
            if np.any(~np.isfinite(self.coil_current_limits)) or np.any(
                self.coil_current_limits <= 0.0
            ):
                raise ValueError("CoilSet.current_limits must be finite and > 0.")

        tracking_cfg = self.kernel.cfg.get("free_boundary_tracking", {})
        self.control_dt_s = self._resolve_positive_float(
            tracking_cfg.get("control_dt_s"),
            control_dt_s,
            default=0.05,
            name="control_dt_s",
        )
        self.coil_actuator_tau_s = self._resolve_positive_float(
            tracking_cfg.get("coil_actuator_tau_s"),
            coil_actuator_tau_s,
            default=1e-6,
            name="coil_actuator_tau_s",
        )
        self.coil_slew_limits = self._resolve_coil_slew_limits(
            tracking_cfg.get("coil_slew_limits"),
            coil_slew_limits,
        )
        self.hold_steps_after_reject = self._resolve_nonnegative_int(
            tracking_cfg.get("hold_steps_after_reject"),
            hold_steps_after_reject,
            default=0,
            name="hold_steps_after_reject",
        )
        self.supervisor_limits = self._resolve_supervisor_limits(
            tracking_cfg.get("supervisor_limits"),
            supervisor_limits,
        )
        self.observer_gain = self._resolve_nonnegative_float(
            tracking_cfg.get("observer_gain"),
            default=0.0,
            name="free_boundary_tracking.observer_gain",
        )
        self.observer_forgetting = self._resolve_fraction(
            tracking_cfg.get("observer_forgetting"),
            default=0.0,
            name="free_boundary_tracking.observer_forgetting",
        )
        self.observer_max_abs = self._resolve_nonnegative_float(
            tracking_cfg.get("observer_max_abs"),
            default=np.inf,
            name="free_boundary_tracking.observer_max_abs",
        )
        self.fallback_currents = self._resolve_fallback_currents(
            tracking_cfg.get("fallback_currents"),
        )
        self._hold_steps_remaining = 0
        self._coil_actuators = self._build_coil_actuators()

        self.objective_tolerances = self._resolve_objective_tolerances(
            self.kernel.cfg.get("free_boundary", {}).get("objective_tolerances"),
            objective_tolerances,
        )
        self.target_vector, self.objective_blocks = self._build_target_vector()
        if self.target_vector.size < 1:
            raise ValueError("free-boundary tracking requires explicit target values.")
        self.measurement_bias_vector = self._resolve_measurement_vector(
            tracking_cfg.get("measurement_bias"),
            name="free_boundary_tracking.measurement_bias",
        )
        self.measurement_drift_per_step = self._resolve_measurement_vector(
            tracking_cfg.get("measurement_drift_per_step"),
            name="free_boundary_tracking.measurement_drift_per_step",
        )
        self.measurement_correction_bias = self._resolve_measurement_vector(
            tracking_cfg.get("measurement_correction_bias"),
            name="free_boundary_tracking.measurement_correction_bias",
        )
        self.measurement_correction_drift_per_step = self._resolve_measurement_vector(
            tracking_cfg.get("measurement_correction_drift_per_step"),
            name="free_boundary_tracking.measurement_correction_drift_per_step",
        )
        self.measurement_latency_steps = self._resolve_nonnegative_int(
            tracking_cfg.get("measurement_latency_steps"),
            None,
            default=0,
            name="free_boundary_tracking.measurement_latency_steps",
        )
        self.latency_compensation_gain = self._resolve_fraction(
            tracking_cfg.get("latency_compensation_gain"),
            default=0.0,
            name="free_boundary_tracking.latency_compensation_gain",
        )
        self.latency_rate_max_abs = self._resolve_nonnegative_float(
            tracking_cfg.get("latency_rate_max_abs"),
            default=np.inf,
            name="free_boundary_tracking.latency_rate_max_abs",
        )
        self.measurement_drift_state = np.zeros_like(self.target_vector, dtype=np.float64)
        self.measurement_correction_drift_state = np.zeros_like(
            self.target_vector, dtype=np.float64
        )
        self.control_objective_weights = self._build_control_objective_weights()
        self.objective_bias_estimate = np.zeros_like(self.target_vector, dtype=np.float64)
        self.objective_rate_estimate = np.zeros_like(self.target_vector, dtype=np.float64)
        self._measurement_latency_buffer: deque[FloatArray] = deque()
        self._last_delayed_measurement: FloatArray | None = None

        self.response_matrix = np.zeros((self.target_vector.size, self.n_coils), dtype=np.float64)
        self.response_rank = 0
        self.response_condition_number = float("inf")
        self.response_max_singular_value = 0.0
        self.response_degenerate = True
        self.last_coil_penalties = np.ones(self.n_coils, dtype=np.float64)
        self.history: dict[str, list[Any]] = {
            "t": [],
            "tracking_error_norm": [],
            "true_tracking_error_norm": [],
            "control_error_norm": [],
            "true_control_error_norm": [],
            "shape_rms": [],
            "true_shape_rms": [],
            "shape_max_abs": [],
            "true_shape_max_abs": [],
            "x_point_position_error": [],
            "true_x_point_position_error": [],
            "x_point_flux_error": [],
            "true_x_point_flux_error": [],
            "divertor_rms": [],
            "true_divertor_rms": [],
            "divertor_max_abs": [],
            "true_divertor_max_abs": [],
            "max_abs_delta_i": [],
            "max_abs_coil_current": [],
            "max_abs_actuator_lag": [],
            "max_abs_measurement_offset": [],
            "mean_abs_measurement_offset": [],
            "measurement_error_norm": [],
            "delayed_observation_error_norm": [],
            "estimated_observation_error_norm": [],
            "accepted_gain": [],
            "objective_converged": [],
            "max_abs_objective_bias_estimate": [],
            "mean_abs_objective_bias_estimate": [],
            "max_abs_objective_rate_estimate": [],
            "response_rank": [],
            "response_condition_number": [],
            "response_max_singular_value": [],
            "response_degenerate": [],
            "active_control_rows": [],
            "max_coil_penalty": [],
            "supervisor_intervened": [],
            "supervisor_safe": [],
            "supervisor_hold_steps_remaining": [],
            "fallback_active": [],
            "tolerance_regression_blocked": [],
        }

    def _log(self, message: str) -> None:
        if self.verbose:
            logger.info(message)

    @staticmethod
    def _resolve_objective_tolerances(
        cfg_tolerances: Any,
        override_tolerances: dict[str, float] | None,
    ) -> dict[str, float]:
        allowed = {
            "shape_rms",
            "shape_max_abs",
            "x_point_position",
            "x_point_flux",
            "divertor_rms",
            "divertor_max_abs",
        }
        merged: dict[str, float] = {}
        for raw, name in (
            (cfg_tolerances, "free_boundary.objective_tolerances"),
            (override_tolerances, "objective_tolerances"),
        ):
            if raw is None:
                continue
            if not isinstance(raw, dict):
                raise ValueError(
                    f"{name} must be a mapping of tolerance names to non-negative floats."
                )
            for key, value in raw.items():
                if key not in allowed:
                    allowed_keys = ", ".join(sorted(allowed))
                    raise ValueError(f"Unknown {name} key {key!r}. Allowed keys: {allowed_keys}.")
                tol_value = float(value)
                if not np.isfinite(tol_value) or tol_value < 0.0:
                    raise ValueError(f"{name}.{key} must be finite and >= 0.")
                merged[key] = tol_value
        return merged

    @staticmethod
    def _resolve_positive_float(
        cfg_value: Any,
        override_value: float | None,
        *,
        default: float,
        name: str,
    ) -> float:
        raw_value = (
            default
            if override_value is None and cfg_value is None
            else (cfg_value if override_value is None else override_value)
        )
        value = float(raw_value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and > 0.")
        return value

    @staticmethod
    def _resolve_nonnegative_int(
        cfg_value: Any,
        override_value: int | None,
        *,
        default: int,
        name: str,
    ) -> int:
        raw_value = (
            default
            if override_value is None and cfg_value is None
            else (cfg_value if override_value is None else override_value)
        )
        value = int(raw_value)
        if value < 0:
            raise ValueError(f"{name} must be >= 0.")
        return value

    @staticmethod
    def _resolve_nonnegative_float(
        cfg_value: Any,
        *,
        default: float,
        name: str,
    ) -> float:
        raw_value = default if cfg_value is None else cfg_value
        value = float(raw_value)
        if not np.isfinite(value) and not np.isinf(value):
            raise ValueError(f"{name} must be finite or infinity.")
        if value < 0.0:
            raise ValueError(f"{name} must be >= 0.")
        return value

    @staticmethod
    def _resolve_fraction(
        cfg_value: Any,
        *,
        default: float,
        name: str,
    ) -> float:
        raw_value = default if cfg_value is None else cfg_value
        value = float(raw_value)
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1].")
        return value

    def _resolve_coil_slew_limits(
        self,
        cfg_limits: Any,
        override_limits: float | list[float] | None,
    ) -> FloatArray:
        raw = cfg_limits if override_limits is None else override_limits
        if raw is None:
            return np.full(self.n_coils, np.inf, dtype=np.float64)
        if np.isscalar(raw):
            limits = np.full(self.n_coils, float(cast(Any, raw)), dtype=np.float64)
        else:
            limits = np.asarray(raw, dtype=np.float64).reshape(-1)
        if limits.shape != (self.n_coils,):
            raise ValueError("coil_slew_limits must be a scalar or match the number of coils.")
        if np.any(~np.isfinite(limits)) or np.any(limits <= 0.0):
            raise ValueError("coil_slew_limits must contain finite values > 0.")
        return cast(FloatArray, np.asarray(limits, dtype=np.float64))

    @staticmethod
    def _resolve_supervisor_limits(
        cfg_limits: Any,
        override_limits: dict[str, float] | None,
    ) -> dict[str, float]:
        allowed = {
            "tracking_error_norm",
            "shape_rms",
            "shape_max_abs",
            "x_point_position",
            "x_point_flux",
            "divertor_rms",
            "divertor_max_abs",
            "max_abs_coil_current",
            "max_abs_actuator_lag",
        }
        merged: dict[str, float] = {}
        for raw, source_name in (
            (cfg_limits, "free_boundary_tracking.supervisor_limits"),
            (override_limits, "supervisor_limits"),
        ):
            if raw is None:
                continue
            if not isinstance(raw, dict):
                raise ValueError(
                    f"{source_name} must be a mapping of limit names to non-negative floats."
                )
            for key, value in raw.items():
                if key not in allowed:
                    allowed_keys = ", ".join(sorted(allowed))
                    raise ValueError(
                        f"Unknown {source_name} key {key!r}. Allowed keys: {allowed_keys}."
                    )
                limit_value = float(value)
                if not np.isfinite(limit_value) or limit_value < 0.0:
                    raise ValueError(f"{source_name}.{key} must be finite and >= 0.")
                merged[key] = limit_value
        return merged

    def _resolve_fallback_currents(self, cfg_value: Any) -> FloatArray | None:
        if cfg_value is None:
            return None
        values = np.asarray(cfg_value, dtype=np.float64).reshape(-1)
        if values.shape != (self.n_coils,):
            raise ValueError(
                "free_boundary_tracking.fallback_currents must match the number of coils."
            )
        if np.any(~np.isfinite(values)):
            raise ValueError("free_boundary_tracking.fallback_currents must be finite.")
        if np.any(np.abs(values) - self.coil_current_limits > 1e-12):
            raise ValueError(
                "free_boundary_tracking.fallback_currents must respect CoilSet.current_limits."
            )
        return cast(FloatArray, values.copy())

    def _build_coil_actuators(self) -> list[FirstOrderActuator]:
        actuators: list[FirstOrderActuator] = []
        for idx in range(self.n_coils):
            limit = float(self.coil_current_limits[idx])
            actuator = FirstOrderActuator(
                tau_s=self.coil_actuator_tau_s,
                dt_s=self.control_dt_s,
                u_min=-limit,
                u_max=limit,
                rate_limit=float(self.coil_slew_limits[idx]),
            )
            actuator.state = float(self.coils.currents[idx])
            actuator._delay_buffer = [float(self.coils.currents[idx])] * max(
                actuator.delay_steps, 1
            )
            actuators.append(actuator)
        return actuators

    def _snapshot_actuator_states(self) -> tuple[_ActuatorSnapshot, ...]:
        return tuple(
            _ActuatorSnapshot(
                state=float(actuator.state),
                delay_buffer=tuple(float(v) for v in actuator._delay_buffer),
            )
            for actuator in self._coil_actuators
        )

    def _restore_actuator_states(self, snapshots: tuple[_ActuatorSnapshot, ...]) -> None:
        if len(snapshots) != len(self._coil_actuators):
            raise ValueError("actuator snapshot count must match the number of coils.")
        for actuator, snapshot in zip(self._coil_actuators, snapshots):
            actuator.state = float(snapshot.state)
            actuator._delay_buffer = [float(v) for v in snapshot.delay_buffer]

    def _sync_actuators_from_coils(self) -> None:
        for idx, actuator in enumerate(self._coil_actuators):
            current = float(self.coils.currents[idx])
            actuator.state = current
            actuator._delay_buffer = [current] * max(actuator.delay_steps, 1)

    def _build_target_vector(self) -> tuple[FloatArray, tuple[_ObjectiveBlock, ...]]:
        values: list[float] = []
        blocks: list[_ObjectiveBlock] = []
        start = 0

        if self.coils.target_flux_points is not None and self.coils.target_flux_values is not None:
            target_flux = np.asarray(self.coils.target_flux_values, dtype=np.float64).reshape(-1)
            values.extend(float(v) for v in target_flux)
            stop = start + target_flux.size
            blocks.append(_ObjectiveBlock("shape_flux", start, stop))
            start = stop

        if self.coils.x_point_target is not None:
            x_target = np.asarray(self.coils.x_point_target, dtype=np.float64).reshape(2)
            values.extend((float(x_target[0]), float(x_target[1])))
            stop = start + 2
            blocks.append(_ObjectiveBlock("x_point_position", start, stop))
            start = stop
            if self.coils.x_point_flux_target is not None:
                values.append(float(self.coils.x_point_flux_target))
                stop = start + 1
                blocks.append(_ObjectiveBlock("x_point_flux", start, stop))
                start = stop
        elif self.coils.x_point_flux_target is not None:
            raise ValueError(
                "x_point_flux_target requires x_point_target for free-boundary tracking."
            )

        if (
            self.coils.divertor_strike_points is not None
            and self.coils.divertor_flux_values is not None
        ):
            divertor_flux = np.asarray(self.coils.divertor_flux_values, dtype=np.float64).reshape(
                -1
            )
            values.extend(float(v) for v in divertor_flux)
            stop = start + divertor_flux.size
            blocks.append(_ObjectiveBlock("divertor_flux", start, stop))
            start = stop

        return np.asarray(values, dtype=np.float64), tuple(blocks)

    def _resolve_measurement_vector(self, raw_value: Any, *, name: str) -> FloatArray:
        vector = np.zeros_like(self.target_vector, dtype=np.float64)
        if raw_value is None:
            return cast(FloatArray, vector)
        if not isinstance(raw_value, dict):
            raise ValueError(
                f"{name} must be a mapping of objective block names to finite scalars or vectors."
            )

        block_map = {block.name: block for block in self.objective_blocks}
        allowed_keys = ", ".join(sorted(block_map))
        for key, raw_block_value in raw_value.items():
            block = block_map.get(key)
            if block is None:
                raise ValueError(f"Unknown {name} key {key!r}. Allowed keys: {allowed_keys}.")
            width = block.stop - block.start
            if np.isscalar(raw_block_value):
                block_values = np.full(width, float(cast(Any, raw_block_value)), dtype=np.float64)
            else:
                block_values = np.asarray(raw_block_value, dtype=np.float64).reshape(-1)
                if block_values.size == 1:
                    block_values = np.full(width, float(block_values[0]), dtype=np.float64)
            if block_values.shape != (width,):
                raise ValueError(
                    f"{name}.{key} must be a scalar or contain exactly {width} entries."
                )
            if np.any(~np.isfinite(block_values)):
                raise ValueError(f"{name}.{key} must contain only finite values.")
            vector[block.start : block.stop] = block_values
        return cast(FloatArray, np.asarray(vector, dtype=np.float64))

    def _weight_from_tolerances(self, *keys: str) -> float:
        weight = 1.0
        for key in keys:
            tol = self.objective_tolerances.get(key)
            if tol is None:
                continue
            weight = max(weight, 1.0 / max(float(tol), 1.0e-12))
        return float(weight)

    def _build_control_objective_weights(self) -> FloatArray:
        weights = np.ones(self.target_vector.shape, dtype=np.float64)
        for block in self.objective_blocks:
            if block.name == "shape_flux":
                block_weight = self._weight_from_tolerances("shape_rms", "shape_max_abs")
            elif block.name == "x_point_position":
                block_weight = self._weight_from_tolerances("x_point_position")
            elif block.name == "x_point_flux":
                block_weight = self._weight_from_tolerances("x_point_flux")
            elif block.name == "divertor_flux":
                block_weight = self._weight_from_tolerances("divertor_rms", "divertor_max_abs")
            else:
                raise ValueError(f"Unknown objective block {block.name!r}.")
            weights[block.start : block.stop] = block_weight
        return cast(FloatArray, np.asarray(weights, dtype=np.float64))

    def _sync_config_currents(self) -> None:
        coils_cfg = self.kernel.cfg.setdefault("coils", [])
        while len(coils_cfg) < self.n_coils:
            coils_cfg.append({})
        for idx in range(self.n_coils):
            coils_cfg[idx]["current"] = float(self.coils.currents[idx])

    def _reset_objective_observer(self) -> None:
        self.objective_bias_estimate = np.zeros_like(self.target_vector, dtype=np.float64)

    def _reset_measurement_offsets(self) -> None:
        self.measurement_drift_state = np.zeros_like(self.target_vector, dtype=np.float64)
        self.measurement_correction_drift_state = np.zeros_like(
            self.target_vector, dtype=np.float64
        )

    def _reset_latency_estimator(self) -> None:
        self.objective_rate_estimate = np.zeros_like(self.target_vector, dtype=np.float64)
        self._measurement_latency_buffer.clear()
        self._last_delayed_measurement = None

    def _advance_measurement_offsets(self) -> None:
        self.measurement_drift_state = np.asarray(
            self.measurement_drift_state + self.measurement_drift_per_step,
            dtype=np.float64,
        )
        self.measurement_correction_drift_state = np.asarray(
            self.measurement_correction_drift_state + self.measurement_correction_drift_per_step,
            dtype=np.float64,
        )

    def _current_measurement_offset(self) -> FloatArray:
        return cast(
            FloatArray,
            np.asarray(
                self.measurement_bias_vector
                + self.measurement_drift_state
                - self.measurement_correction_bias
                - self.measurement_correction_drift_state,
                dtype=np.float64,
            ),
        )

    def _apply_measurement_latency(self, measurement: FloatArray, *, record: bool) -> FloatArray:
        measured = np.asarray(measurement, dtype=np.float64).reshape(-1)
        if (not record) or self.measurement_latency_steps < 1:
            return cast(FloatArray, measured.copy())
        self._measurement_latency_buffer.append(measured.copy())
        max_buffer = self.measurement_latency_steps + 1
        while len(self._measurement_latency_buffer) > max_buffer:
            self._measurement_latency_buffer.popleft()
        return cast(
            FloatArray, np.asarray(self._measurement_latency_buffer[0].copy(), dtype=np.float64)
        )

    def _predict_current_objectives(
        self,
        delayed_observation: FloatArray,
        *,
        allow_compensation: bool,
        update_state: bool,
    ) -> FloatArray:
        delayed = np.asarray(delayed_observation, dtype=np.float64).reshape(-1)
        prediction_horizon = 1.0 if self.measurement_latency_steps > 0 else 0.0
        if self.measurement_latency_steps < 1:
            if update_state:
                self.objective_rate_estimate = np.zeros_like(self.target_vector, dtype=np.float64)
                self._last_delayed_measurement = delayed.copy()
            return cast(FloatArray, delayed.copy())
        if self._last_delayed_measurement is None:
            delta = np.zeros_like(delayed, dtype=np.float64)
        else:
            delta = np.asarray(delayed - self._last_delayed_measurement, dtype=np.float64)
        latency_projection_ready = bool(np.linalg.norm(delta) > 1.0e-12)
        if update_state:
            self._last_delayed_measurement = delayed.copy()
        if (not allow_compensation) or self.latency_compensation_gain <= 0.0:
            if update_state:
                self.objective_rate_estimate = np.zeros_like(self.target_vector, dtype=np.float64)
            return cast(FloatArray, delayed.copy())
        if not latency_projection_ready:
            if update_state:
                self.objective_rate_estimate = np.zeros_like(self.target_vector, dtype=np.float64)
            return cast(FloatArray, delayed.copy())
        if not update_state:
            predicted = delayed + prediction_horizon * delta
            return cast(FloatArray, np.asarray(predicted, dtype=np.float64))
        updated = (1.0 - self.latency_compensation_gain) * self.objective_rate_estimate
        updated = np.asarray(updated + self.latency_compensation_gain * delta, dtype=np.float64)
        if np.isfinite(self.latency_rate_max_abs):
            updated = np.clip(updated, -self.latency_rate_max_abs, self.latency_rate_max_abs)
        self.objective_rate_estimate = cast(FloatArray, np.asarray(updated, dtype=np.float64))
        predicted = delayed + prediction_horizon * self.objective_rate_estimate
        return cast(FloatArray, np.asarray(predicted, dtype=np.float64))

    def _observe_snapshot(self, *, apply_latency: bool = True) -> _ObservationSnapshot:
        true_observation = self._observe_true_objectives()
        measured_observation = cast(
            FloatArray,
            np.asarray(true_observation + self._current_measurement_offset(), dtype=np.float64),
        )

        # ── Optional EKF Refinement ──
        if self.state_estimator is not None:
            # Predict step (assuming control_dt_s as time step)
            self.state_estimator.predict(self.control_dt_s)

            # Map measured observation to EKF measurement vector [R, Z, Ip, Te]
            # Since the observation vector format can vary, we only update
            # if we can find R, Z from x_point_position objective.
            z_ekf = self._map_observation_to_ekf(measured_observation)
            if z_ekf is not None:
                self.state_estimator.update(z_ekf)
                # Refine the measured observation with EKF state estimate
                measured_observation = self._map_ekf_to_observation(
                    measured_observation, self.state_estimator.estimate()
                )

        delayed_observation = self._apply_measurement_latency(
            measured_observation, record=apply_latency
        )
        effective_observation = self._predict_current_objectives(
            delayed_observation,
            allow_compensation=apply_latency,
            update_state=apply_latency,
        )
        return _ObservationSnapshot(
            true=true_observation,
            measured=measured_observation,
            delayed=delayed_observation,
            effective=effective_observation,
        )

    def _update_objective_observer(self, observation: FloatArray) -> None:
        if self.observer_gain <= 0.0 or self.objective_bias_estimate.size < 1:
            return
        residual = self.target_vector - np.asarray(observation, dtype=np.float64).reshape(-1)
        updated = (
            1.0 - self.observer_forgetting
        ) * self.objective_bias_estimate + self.observer_gain * residual
        if np.isfinite(self.observer_max_abs):
            updated = np.clip(updated, -self.observer_max_abs, self.observer_max_abs)
        self.objective_bias_estimate = np.asarray(updated, dtype=np.float64)

    def _command_currents(self, delta_currents: FloatArray, gain: float) -> FloatArray:
        g = float(gain)
        if not np.isfinite(g) or g <= 0.0:
            raise ValueError("gain must be finite and > 0.")
        commanded = np.asarray(self.coils.currents.copy(), dtype=np.float64)
        for idx in range(self.n_coils):
            limit = float(self.coil_current_limits[idx])
            commanded[idx] = float(
                np.clip(commanded[idx] + g * float(delta_currents[idx]), -limit, limit)
            )
        return cast(FloatArray, np.asarray(commanded, dtype=np.float64))

    def _apply_commanded_currents(self, commanded: FloatArray) -> FloatArray:
        command = np.asarray(commanded, dtype=np.float64).reshape(-1)
        if command.shape != (self.n_coils,):
            raise ValueError("commanded currents must match the number of coils.")
        applied = np.zeros(self.n_coils, dtype=np.float64)
        for idx, actuator in enumerate(self._coil_actuators):
            applied[idx] = float(actuator.step(float(command[idx])))
        self.coils.currents = applied
        return cast(FloatArray, np.asarray(applied.copy(), dtype=np.float64))

    def _solve_free_boundary_state(self) -> dict[str, Any]:
        self._sync_config_currents()
        solve = getattr(self.kernel, "solve", None)
        if callable(solve):
            return cast(
                dict[str, Any],
                solve(
                    boundary_variant="free_boundary",
                    coils=self.coils,
                    max_outer_iter=self.solve_max_outer_iter,
                    tol=self.solve_tol,
                    optimize_shape=False,
                ),
            )

        solve_free_boundary = getattr(self.kernel, "solve_free_boundary", None)
        if callable(solve_free_boundary):
            return cast(
                dict[str, Any],
                solve_free_boundary(
                    self.coils,
                    max_outer_iter=self.solve_max_outer_iter,
                    tol=self.solve_tol,
                    optimize_shape=False,
                ),
            )

        raise AttributeError(
            "kernel must define solve() or solve_free_boundary() for free-boundary tracking."
        )

    def _observe_true_objectives(self) -> FloatArray:
        observed: list[float] = []
        for block in self.objective_blocks:
            if block.name == "shape_flux":
                if self.coils.target_flux_points is None:
                    raise ValueError("shape-flux observation requires target_flux_points.")
                flux = np.asarray(
                    self.kernel._sample_flux_at_points(self.coils.target_flux_points),
                    dtype=np.float64,
                ).reshape(-1)
                observed.extend(float(v) for v in flux)
            elif block.name == "x_point_position":
                x_pos, _ = self.kernel.find_x_point(self.kernel.Psi)
                observed.extend((float(x_pos[0]), float(x_pos[1])))
            elif block.name == "x_point_flux":
                if self.coils.x_point_target is None:
                    raise ValueError("x_point_flux observation requires x_point_target.")
                x_target = np.asarray(self.coils.x_point_target, dtype=np.float64).reshape(2)
                observed.append(
                    float(self.kernel._interp_psi(float(x_target[0]), float(x_target[1])))
                )
            elif block.name == "divertor_flux":
                if self.coils.divertor_strike_points is None:
                    raise ValueError("divertor-flux observation requires divertor_strike_points.")
                flux = np.asarray(
                    self.kernel._sample_flux_at_points(self.coils.divertor_strike_points),
                    dtype=np.float64,
                ).reshape(-1)
                observed.extend(float(v) for v in flux)
            else:
                raise ValueError(f"Unknown objective block {block.name!r}.")
        return np.asarray(observed, dtype=np.float64)

    def _observe_objectives(self) -> FloatArray:
        return self._observe_snapshot().effective

    def identify_response_matrix(self, perturbation: float | None = None) -> FloatArray:
        p = self.identification_perturbation if perturbation is None else float(perturbation)
        if not np.isfinite(p) or p <= 0.0:
            raise ValueError("perturbation must be finite and > 0.")

        self._solve_free_boundary_state()
        original_currents = self.coils.currents.copy()
        actuator_snapshot = self._snapshot_actuator_states()
        for idx in range(self.n_coils):
            hi = float(self.coil_current_limits[idx])
            plus = original_currents.copy()
            minus = original_currents.copy()
            plus[idx] = float(np.clip(original_currents[idx] + p, -hi, hi))
            minus[idx] = float(np.clip(original_currents[idx] - p, -hi, hi))
            denom = plus[idx] - minus[idx]
            if abs(denom) < 1e-12:
                self.response_matrix[:, idx] = 0.0
                continue

            self.coils.currents = plus
            self._solve_free_boundary_state()
            obs_plus = self._observe_snapshot(apply_latency=False).effective

            self.coils.currents = minus
            self._solve_free_boundary_state()
            obs_minus = self._observe_snapshot(apply_latency=False).effective

            self.response_matrix[:, idx] = (obs_plus - obs_minus) / denom

        self.coils.currents = original_currents
        self._restore_actuator_states(actuator_snapshot)
        self._solve_free_boundary_state()
        self._update_response_diagnostics()
        return self.response_matrix.copy()

    def _update_response_diagnostics(self) -> None:
        singular_values = np.asarray(
            np.linalg.svd(self.response_matrix, compute_uv=False), dtype=np.float64
        ).reshape(-1)
        if singular_values.size < 1:
            self.response_rank = 0
            self.response_condition_number = float("inf")
            self.response_max_singular_value = 0.0
            self.response_degenerate = True
            return

        sigma_max = float(np.max(singular_values))
        eps = float(np.finfo(np.float64).eps)
        cutoff = float(max(eps * max(1.0, sigma_max), 1.0e-12))
        nonzero = singular_values[np.asarray(singular_values > cutoff, dtype=np.bool_)]
        self.response_rank = int(nonzero.size)
        self.response_max_singular_value = sigma_max
        self.response_condition_number = (
            float(sigma_max / float(nonzero[-1])) if nonzero.size > 0 else float("inf")
        )
        self.response_degenerate = bool(
            (not np.isfinite(sigma_max)) or sigma_max <= cutoff or nonzero.size < 1
        )

    def _build_control_activation_mask(self, metrics: dict[str, Any]) -> FloatArray:
        objective_checks = cast(dict[str, bool], metrics.get("objective_checks", {}))
        mask = np.ones(self.target_vector.shape, dtype=np.float64)
        for block in self.objective_blocks:
            if block.name == "shape_flux":
                relevant = [
                    key
                    for key in ("shape_rms", "shape_max_abs")
                    if key in self.objective_tolerances
                ]
            elif block.name == "x_point_position":
                relevant = [
                    key for key in ("x_point_position",) if key in self.objective_tolerances
                ]
            elif block.name == "x_point_flux":
                relevant = [key for key in ("x_point_flux",) if key in self.objective_tolerances]
            elif block.name == "divertor_flux":
                relevant = [
                    key
                    for key in ("divertor_rms", "divertor_max_abs")
                    if key in self.objective_tolerances
                ]
            else:
                raise ValueError(f"Unknown objective block {block.name!r}.")
            if relevant and all(objective_checks.get(key, False) for key in relevant):
                mask[block.start : block.stop] = 0.0
        return cast(FloatArray, mask)

    def _build_coil_penalties(self, delta_hint: FloatArray) -> FloatArray:
        headrooms = np.ones(self.n_coils, dtype=np.float64)
        penalties = np.ones(self.n_coils, dtype=np.float64)
        for idx in range(self.n_coils):
            limit = float(self.coil_current_limits[idx])
            if not np.isfinite(limit) or limit <= 0.0:
                headrooms[idx] = np.inf
                continue
            current = float(self.coils.currents[idx])
            direction = float(delta_hint[idx]) if idx < delta_hint.size else 0.0
            if direction > 1.0e-12:
                headroom = limit - current
            elif direction < -1.0e-12:
                headroom = limit + current
            else:
                headroom = limit - abs(current)
            headrooms[idx] = max(headroom, 1.0e-9)
        finite_headrooms = headrooms[np.isfinite(headrooms)]
        reference_headroom = float(np.max(finite_headrooms)) if finite_headrooms.size > 0 else 1.0
        for idx in range(self.n_coils):
            if not np.isfinite(headrooms[idx]):
                penalties[idx] = 1.0
                continue
            penalties[idx] = float(np.sqrt(max(reference_headroom / float(headrooms[idx]), 1.0)))
        return cast(FloatArray, penalties)

    def compute_correction(
        self,
        observation: FloatArray,
        *,
        metrics: dict[str, Any] | None = None,
    ) -> FloatArray:
        obs = np.asarray(observation, dtype=np.float64).reshape(-1)
        if obs.shape != self.target_vector.shape:
            raise ValueError("observation must match the free-boundary target vector shape.")
        metrics_now = self.evaluate_objectives(obs) if metrics is None else metrics
        control_mask = self._build_control_activation_mask(metrics_now)
        error = self.target_vector + self.objective_bias_estimate - obs
        weight_vector = self.control_objective_weights * control_mask
        weighted_response = weight_vector[:, None] * self.response_matrix
        weighted_error = weight_vector * error
        base_reg = np.sqrt(self.response_regularization) * np.eye(self.n_coils, dtype=np.float64)
        base_aug_matrix = np.vstack([weighted_response, base_reg])
        base_aug_rhs = np.concatenate([weighted_error, np.zeros(self.n_coils, dtype=np.float64)])
        delta_hint, *_ = np.linalg.lstsq(base_aug_matrix, base_aug_rhs, rcond=None)
        delta_hint = np.asarray(delta_hint, dtype=np.float64)
        coil_penalties = self._build_coil_penalties(delta_hint)
        self.last_coil_penalties[:] = coil_penalties
        aug_matrix = np.vstack(
            [
                weighted_response,
                np.sqrt(self.response_regularization) * np.diag(coil_penalties),
            ]
        )
        aug_rhs = np.concatenate([weighted_error, np.zeros(self.n_coils, dtype=np.float64)])
        delta, *_ = np.linalg.lstsq(aug_matrix, aug_rhs, rcond=None)
        clipped = np.clip(
            np.asarray(delta, dtype=np.float64), -self.correction_limit, self.correction_limit
        )
        return cast(FloatArray, np.asarray(clipped, dtype=np.float64))

    def _apply_correction(self, delta_currents: FloatArray, gain: float) -> FloatArray:
        commanded = self._command_currents(delta_currents, gain=gain)
        return self._apply_commanded_currents(commanded)

    def _apply_fallback_currents(self) -> float:
        if self.fallback_currents is None:
            raise ValueError("fallback currents are not configured.")
        applied = self._apply_commanded_currents(self.fallback_currents)
        if applied.size < 1:
            return 0.0
        return float(np.max(np.abs(self.fallback_currents - applied)))

    def _recover_to_safe_state(
        self,
        *,
        actuator_snapshot: tuple[_ActuatorSnapshot, ...],
        baseline_currents: FloatArray,
        metrics_before: dict[str, Any],
        true_metrics_before: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], float, bool]:
        self._restore_actuator_states(actuator_snapshot)
        self.coils.currents = baseline_currents.copy()
        fallback_active = False
        max_abs_actuator_lag = 0.0
        if self.fallback_currents is not None:
            max_abs_actuator_lag = self._apply_fallback_currents()
            fallback_active = True
        self._solve_free_boundary_state()
        metrics_after = self.evaluate_objectives(self._observe_objectives())
        true_metrics_after = self.evaluate_objectives(self._observe_true_objectives())
        if not fallback_active:
            metrics_after = metrics_before
            true_metrics_after = true_metrics_before
        supervisor_after = self.evaluate_supervisor(
            metrics_after,
            max_abs_coil_current=float(np.max(np.abs(self.coils.currents)))
            if self.coils.currents.size > 0
            else 0.0,
            max_abs_actuator_lag=max_abs_actuator_lag,
        )
        return (
            metrics_after,
            true_metrics_after,
            supervisor_after,
            max_abs_actuator_lag,
            fallback_active,
        )

    def evaluate_objectives(self, observation: np.ndarray) -> dict[str, Any]:
        obs = np.asarray(observation, dtype=np.float64).reshape(-1)
        error = self.target_vector - obs
        metrics: dict[str, Any] = {
            "tracking_error_norm": float(np.linalg.norm(error)),
            "control_error_norm": 0.0,
            "shape_rms": None,
            "shape_max_abs": None,
            "x_point_position_error": None,
            "x_point_flux_error": None,
            "divertor_rms": None,
            "divertor_max_abs": None,
            "active_control_rows": 0,
        }

        for block in self.objective_blocks:
            block_error = error[block.start : block.stop]
            if block.name == "shape_flux":
                metrics["shape_rms"] = float(np.sqrt(np.mean(block_error**2)))
                metrics["shape_max_abs"] = float(np.max(np.abs(block_error)))
            elif block.name == "x_point_position":
                metrics["x_point_position_error"] = float(np.linalg.norm(block_error))
            elif block.name == "x_point_flux":
                metrics["x_point_flux_error"] = float(abs(block_error[0]))
            elif block.name == "divertor_flux":
                metrics["divertor_rms"] = float(np.sqrt(np.mean(block_error**2)))
                metrics["divertor_max_abs"] = float(np.max(np.abs(block_error)))

        checks: dict[str, bool] = {}
        if "shape_rms" in self.objective_tolerances and metrics["shape_rms"] is not None:
            checks["shape_rms"] = bool(
                metrics["shape_rms"] <= self.objective_tolerances["shape_rms"]
            )
        if "shape_max_abs" in self.objective_tolerances and metrics["shape_max_abs"] is not None:
            checks["shape_max_abs"] = bool(
                metrics["shape_max_abs"] <= self.objective_tolerances["shape_max_abs"]
            )
        if (
            "x_point_position" in self.objective_tolerances
            and metrics["x_point_position_error"] is not None
        ):
            checks["x_point_position"] = bool(
                metrics["x_point_position_error"] <= self.objective_tolerances["x_point_position"]
            )
        if (
            "x_point_flux" in self.objective_tolerances
            and metrics["x_point_flux_error"] is not None
        ):
            checks["x_point_flux"] = bool(
                metrics["x_point_flux_error"] <= self.objective_tolerances["x_point_flux"]
            )
        if "divertor_rms" in self.objective_tolerances and metrics["divertor_rms"] is not None:
            checks["divertor_rms"] = bool(
                metrics["divertor_rms"] <= self.objective_tolerances["divertor_rms"]
            )
        if (
            "divertor_max_abs" in self.objective_tolerances
            and metrics["divertor_max_abs"] is not None
        ):
            checks["divertor_max_abs"] = bool(
                metrics["divertor_max_abs"] <= self.objective_tolerances["divertor_max_abs"]
            )
        metrics["objective_checks"] = checks
        metrics["objective_convergence_active"] = bool(checks)
        metrics["objective_converged"] = all(checks.values()) if checks else True
        metrics["objective_tolerances"] = self.objective_tolerances.copy()
        control_mask = self._build_control_activation_mask(metrics)
        control_error = control_mask * self.control_objective_weights * error
        metrics["control_error_norm"] = float(np.linalg.norm(control_error))
        metrics["active_control_rows"] = int(np.count_nonzero(np.abs(control_error) > 1.0e-12))
        return metrics

    def evaluate_supervisor(
        self,
        metrics: dict[str, Any],
        *,
        max_abs_coil_current: float,
        max_abs_actuator_lag: float,
    ) -> dict[str, Any]:
        metric_map = {
            "tracking_error_norm": metrics.get("tracking_error_norm"),
            "shape_rms": metrics.get("shape_rms"),
            "shape_max_abs": metrics.get("shape_max_abs"),
            "x_point_position": metrics.get("x_point_position_error"),
            "x_point_flux": metrics.get("x_point_flux_error"),
            "divertor_rms": metrics.get("divertor_rms"),
            "divertor_max_abs": metrics.get("divertor_max_abs"),
            "max_abs_coil_current": max_abs_coil_current,
            "max_abs_actuator_lag": max_abs_actuator_lag,
        }
        checks: dict[str, bool] = {}
        for key, limit in self.supervisor_limits.items():
            value = metric_map.get(key)
            checks[key] = bool(
                value is not None and np.isfinite(float(value)) and float(abs(value)) <= limit
            )
        return {
            "supervisor_limits": self.supervisor_limits.copy(),
            "supervisor_checks": checks,
            "supervisor_active": bool(checks),
            "supervisor_safe": all(checks.values()) if checks else True,
        }

    @staticmethod
    def _detect_tolerance_regressions(
        metrics_before: dict[str, Any],
        metrics_after: dict[str, Any],
    ) -> dict[str, bool]:
        metric_names = {
            "shape_rms": "shape_rms",
            "shape_max_abs": "shape_max_abs",
            "x_point_position": "x_point_position_error",
            "x_point_flux": "x_point_flux_error",
            "divertor_rms": "divertor_rms",
            "divertor_max_abs": "divertor_max_abs",
        }
        tolerances = cast(dict[str, float], metrics_before.get("objective_tolerances", {}))
        regressions: dict[str, bool] = {}
        for key, metric_name in metric_names.items():
            tolerance = tolerances.get(key)
            if tolerance is None:
                continue
            before_value = metrics_before.get(metric_name)
            after_value = metrics_after.get(metric_name)
            if before_value is None or after_value is None:
                continue
            before_scalar = float(before_value)
            after_scalar = float(after_value)
            regressions[key] = bool(
                before_scalar <= tolerance + 1.0e-12 and after_scalar > tolerance + 1.0e-12
            )
        return regressions

    def run_tracking_shot(
        self,
        *,
        shot_steps: int = 10,
        gain: float = 1.0,
        disturbance_callback: Callable[[Any, CoilSet, int], None] | None = None,
        stop_on_convergence: bool = False,
    ) -> dict[str, Any]:
        steps = int(shot_steps)
        if steps < 1:
            raise ValueError("shot_steps must be >= 1.")
        gain_value = float(gain)
        if not np.isfinite(gain_value) or gain_value <= 0.0:
            raise ValueError("gain must be finite and > 0.")
        start_time = time.time()
        self.history = {key: [] for key in self.history}
        self._hold_steps_remaining = 0
        self._reset_objective_observer()
        self._reset_measurement_offsets()
        self._reset_latency_estimator()
        self._sync_actuators_from_coils()
        self._solve_free_boundary_state()
        initial_snapshot = self._observe_snapshot()
        last_metrics = self.evaluate_objectives(initial_snapshot.effective)
        last_true_metrics = self.evaluate_objectives(initial_snapshot.true)
        last_supervisor = self.evaluate_supervisor(
            last_metrics,
            max_abs_coil_current=float(np.max(np.abs(self.coils.currents)))
            if self.coils.currents.size > 0
            else 0.0,
            max_abs_actuator_lag=0.0,
        )

        for step in range(steps):
            if disturbance_callback is not None:
                disturbance_callback(self.kernel, self.coils, step)

            self._solve_free_boundary_state()
            if step % self.response_refresh_steps == 0:
                self.identify_response_matrix()

            observation_snapshot_before = self._observe_snapshot()
            observation_before = observation_snapshot_before.effective
            true_observation_before = observation_snapshot_before.true
            self._update_objective_observer(observation_before)
            metrics_before = self.evaluate_objectives(observation_before)
            true_metrics_before = self.evaluate_objectives(true_observation_before)
            delta_currents = self.compute_correction(observation_before, metrics=metrics_before)
            baseline_currents = self.coils.currents.copy()
            actuator_snapshot = self._snapshot_actuator_states()
            accepted_gain = 0.0
            last_metrics = metrics_before
            last_true_metrics = true_metrics_before
            supervisor_intervened = False
            max_abs_actuator_lag = 0.0
            fallback_active = False
            tolerance_regression_blocked = False

            if self.response_degenerate:
                delta_currents = np.zeros_like(delta_currents)
                (
                    last_metrics,
                    last_true_metrics,
                    last_supervisor,
                    max_abs_actuator_lag,
                    fallback_active,
                ) = self._recover_to_safe_state(
                    actuator_snapshot=actuator_snapshot,
                    baseline_currents=baseline_currents,
                    metrics_before=metrics_before,
                    true_metrics_before=true_metrics_before,
                )
                supervisor_intervened = True
                if self.hold_steps_after_reject > 0:
                    self._hold_steps_remaining = self.hold_steps_after_reject
            elif self._hold_steps_remaining > 0:
                self._hold_steps_remaining -= 1
                delta_currents = np.zeros_like(delta_currents)
                (
                    last_metrics,
                    last_true_metrics,
                    last_supervisor,
                    max_abs_actuator_lag,
                    fallback_active,
                ) = self._recover_to_safe_state(
                    actuator_snapshot=actuator_snapshot,
                    baseline_currents=baseline_currents,
                    metrics_before=metrics_before,
                    true_metrics_before=true_metrics_before,
                )
                supervisor_intervened = True
            else:
                trial_gain = gain_value
                for _ in range(6):
                    self._restore_actuator_states(actuator_snapshot)
                    self.coils.currents = baseline_currents.copy()
                    commanded_currents = self._command_currents(delta_currents, gain=trial_gain)
                    applied_currents = self._apply_correction(delta_currents, gain=trial_gain)
                    max_abs_actuator_lag = (
                        float(np.max(np.abs(commanded_currents - applied_currents)))
                        if applied_currents.size > 0
                        else 0.0
                    )
                    self._solve_free_boundary_state()
                    observation_snapshot_after = self._observe_snapshot(apply_latency=False)
                    observation_after = observation_snapshot_after.effective
                    true_observation_after = observation_snapshot_after.true
                    metrics_after = self.evaluate_objectives(observation_after)
                    true_metrics_after = self.evaluate_objectives(true_observation_after)
                    supervisor_after = self.evaluate_supervisor(
                        metrics_after,
                        max_abs_coil_current=(
                            float(np.max(np.abs(self.coils.currents)))
                            if self.coils.currents.size > 0
                            else 0.0
                        ),
                        max_abs_actuator_lag=max_abs_actuator_lag,
                    )
                    improved = bool(
                        metrics_after["control_error_norm"]
                        <= metrics_before["control_error_norm"] + 1e-12
                    )
                    tolerance_regressions = self._detect_tolerance_regressions(
                        metrics_before, metrics_after
                    )
                    newly_converged = (not metrics_before["objective_converged"]) and metrics_after[
                        "objective_converged"
                    ]
                    if (
                        (improved or newly_converged)
                        and supervisor_after["supervisor_safe"]
                        and not any(tolerance_regressions.values())
                    ):
                        accepted_gain = trial_gain
                        last_metrics = metrics_after
                        last_true_metrics = true_metrics_after
                        last_supervisor = supervisor_after
                        break
                    if any(tolerance_regressions.values()):
                        tolerance_regression_blocked = True
                    trial_gain *= 0.5

                if accepted_gain == 0.0:
                    (
                        last_metrics,
                        last_true_metrics,
                        last_supervisor,
                        max_abs_actuator_lag,
                        fallback_active,
                    ) = self._recover_to_safe_state(
                        actuator_snapshot=actuator_snapshot,
                        baseline_currents=baseline_currents,
                        metrics_before=metrics_before,
                        true_metrics_before=true_metrics_before,
                    )
                    supervisor_intervened = True
                    if self.hold_steps_after_reject > 0:
                        self._hold_steps_remaining = self.hold_steps_after_reject

            max_abs_delta = (
                float(np.max(np.abs(delta_currents))) if delta_currents.size > 0 else 0.0
            )
            max_abs_coil = (
                float(np.max(np.abs(self.coils.currents))) if self.coils.currents.size > 0 else 0.0
            )
            measurement_offset = self._current_measurement_offset()
            measurement_offset_abs = np.abs(measurement_offset)
            measurement_error_norm = float(
                np.linalg.norm(
                    observation_snapshot_before.measured - observation_snapshot_before.true
                )
            )
            delayed_observation_error_norm = float(
                np.linalg.norm(
                    observation_snapshot_before.delayed - observation_snapshot_before.true
                )
            )
            estimated_observation_error_norm = float(
                np.linalg.norm(
                    observation_snapshot_before.effective - observation_snapshot_before.true
                )
            )
            self.history["t"].append(int(step))
            self.history["tracking_error_norm"].append(last_metrics["tracking_error_norm"])
            self.history["true_tracking_error_norm"].append(
                last_true_metrics["tracking_error_norm"]
            )
            self.history["control_error_norm"].append(last_metrics["control_error_norm"])
            self.history["true_control_error_norm"].append(last_true_metrics["control_error_norm"])
            self.history["shape_rms"].append(last_metrics["shape_rms"])
            self.history["true_shape_rms"].append(last_true_metrics["shape_rms"])
            self.history["shape_max_abs"].append(last_metrics["shape_max_abs"])
            self.history["true_shape_max_abs"].append(last_true_metrics["shape_max_abs"])
            self.history["x_point_position_error"].append(last_metrics["x_point_position_error"])
            self.history["true_x_point_position_error"].append(
                last_true_metrics["x_point_position_error"]
            )
            self.history["x_point_flux_error"].append(last_metrics["x_point_flux_error"])
            self.history["true_x_point_flux_error"].append(last_true_metrics["x_point_flux_error"])
            self.history["divertor_rms"].append(last_metrics["divertor_rms"])
            self.history["true_divertor_rms"].append(last_true_metrics["divertor_rms"])
            self.history["divertor_max_abs"].append(last_metrics["divertor_max_abs"])
            self.history["true_divertor_max_abs"].append(last_true_metrics["divertor_max_abs"])
            self.history["max_abs_delta_i"].append(max_abs_delta)
            self.history["max_abs_coil_current"].append(max_abs_coil)
            self.history["max_abs_actuator_lag"].append(max_abs_actuator_lag)
            self.history["max_abs_measurement_offset"].append(
                float(np.max(measurement_offset_abs)) if measurement_offset_abs.size else 0.0
            )
            self.history["mean_abs_measurement_offset"].append(
                float(np.mean(measurement_offset_abs)) if measurement_offset_abs.size else 0.0
            )
            self.history["measurement_error_norm"].append(measurement_error_norm)
            self.history["delayed_observation_error_norm"].append(delayed_observation_error_norm)
            self.history["estimated_observation_error_norm"].append(
                estimated_observation_error_norm
            )
            self.history["accepted_gain"].append(float(accepted_gain))
            self.history["objective_converged"].append(bool(last_metrics["objective_converged"]))
            bias_abs = np.abs(self.objective_bias_estimate)
            self.history["max_abs_objective_bias_estimate"].append(
                float(np.max(bias_abs)) if bias_abs.size else 0.0
            )
            self.history["mean_abs_objective_bias_estimate"].append(
                float(np.mean(bias_abs)) if bias_abs.size else 0.0
            )
            rate_abs = np.abs(self.objective_rate_estimate)
            self.history["max_abs_objective_rate_estimate"].append(
                float(np.max(rate_abs)) if rate_abs.size else 0.0
            )
            self.history["response_rank"].append(int(self.response_rank))
            self.history["response_condition_number"].append(float(self.response_condition_number))
            self.history["response_max_singular_value"].append(
                float(self.response_max_singular_value)
            )
            self.history["response_degenerate"].append(bool(self.response_degenerate))
            self.history["active_control_rows"].append(int(last_metrics["active_control_rows"]))
            self.history["max_coil_penalty"].append(
                float(np.max(self.last_coil_penalties)) if self.last_coil_penalties.size else 1.0
            )
            self.history["supervisor_intervened"].append(bool(supervisor_intervened))
            self.history["supervisor_safe"].append(bool(last_supervisor["supervisor_safe"]))
            self.history["supervisor_hold_steps_remaining"].append(int(self._hold_steps_remaining))
            self.history["fallback_active"].append(bool(fallback_active))
            self.history["tolerance_regression_blocked"].append(bool(tolerance_regression_blocked))

            self._log(
                f"Step {step}: err={last_metrics['tracking_error_norm']:.4e} | "
                f"ctrl={last_metrics['control_error_norm']:.4e} | "
                f"active={int(last_metrics['active_control_rows'])} | "
                f"max dI={max_abs_delta:.3e} | gain={accepted_gain:.3e} | "
                f"coil_pen={self.history['max_coil_penalty'][-1]:.2f} | "
                f"resp_rank={self.response_rank} | resp_deg={self.response_degenerate} | "
                f"obs_bias={self.history['max_abs_objective_bias_estimate'][-1]:.3e} | "
                f"meas_off={self.history['max_abs_measurement_offset'][-1]:.3e} | "
                f"delay_obs={delayed_observation_error_norm:.3e} | "
                f"est_obs={estimated_observation_error_norm:.3e} | "
                f"lag={max_abs_actuator_lag:.3e} | fallback={fallback_active} | "
                f"safe={last_supervisor['supervisor_safe']} | "
                f"converged={last_metrics['objective_converged']}"
            )
            self._advance_measurement_offsets()
            if stop_on_convergence and last_metrics["objective_converged"]:
                break

        runtime_seconds = time.time() - start_time
        error_arr = np.asarray(self.history["tracking_error_norm"], dtype=np.float64)
        true_error_arr = np.asarray(self.history["true_tracking_error_norm"], dtype=np.float64)
        control_error_arr = np.asarray(self.history["control_error_norm"], dtype=np.float64)
        true_control_error_arr = np.asarray(
            self.history["true_control_error_norm"], dtype=np.float64
        )
        delta_arr = np.asarray(self.history["max_abs_delta_i"], dtype=np.float64)
        coil_arr = np.asarray(self.history["max_abs_coil_current"], dtype=np.float64)
        lag_arr = np.asarray(self.history["max_abs_actuator_lag"], dtype=np.float64)
        measurement_offset_max_arr = np.asarray(
            self.history["max_abs_measurement_offset"], dtype=np.float64
        )
        measurement_offset_mean_arr = np.asarray(
            self.history["mean_abs_measurement_offset"], dtype=np.float64
        )
        measurement_error_arr = np.asarray(self.history["measurement_error_norm"], dtype=np.float64)
        delayed_error_arr = np.asarray(
            self.history["delayed_observation_error_norm"], dtype=np.float64
        )
        estimated_error_arr = np.asarray(
            self.history["estimated_observation_error_norm"], dtype=np.float64
        )
        gain_arr = np.asarray(self.history["accepted_gain"], dtype=np.float64)
        bias_max_arr = np.asarray(self.history["max_abs_objective_bias_estimate"], dtype=np.float64)
        bias_mean_arr = np.asarray(
            self.history["mean_abs_objective_bias_estimate"], dtype=np.float64
        )
        rate_max_arr = np.asarray(self.history["max_abs_objective_rate_estimate"], dtype=np.float64)
        response_rank_arr = np.asarray(self.history["response_rank"], dtype=np.float64)
        response_cond_arr = np.asarray(self.history["response_condition_number"], dtype=np.float64)
        response_sigma_arr = np.asarray(
            self.history["response_max_singular_value"], dtype=np.float64
        )
        response_deg_arr = np.asarray(self.history["response_degenerate"], dtype=np.float64)
        active_control_arr = np.asarray(self.history["active_control_rows"], dtype=np.float64)
        coil_penalty_arr = np.asarray(self.history["max_coil_penalty"], dtype=np.float64)
        supervisor_arr = np.asarray(self.history["supervisor_intervened"], dtype=np.float64)
        fallback_arr = np.asarray(self.history["fallback_active"], dtype=np.float64)
        tolerance_block_arr = np.asarray(
            self.history["tolerance_regression_blocked"], dtype=np.float64
        )
        measurement_distortion_enabled = bool(
            np.any(np.abs(self.measurement_bias_vector) > 0.0)
            or np.any(np.abs(self.measurement_drift_per_step) > 0.0)
        )
        measurement_compensation_enabled = bool(
            np.any(np.abs(self.measurement_correction_bias) > 0.0)
            or np.any(np.abs(self.measurement_correction_drift_per_step) > 0.0)
        )
        measurement_latency_enabled = bool(self.measurement_latency_steps > 0)
        latency_compensation_enabled = bool(
            measurement_latency_enabled and self.latency_compensation_gain > 0.0
        )
        return {
            "steps": int(len(self.history["t"])),
            "runtime_seconds": float(runtime_seconds),
            "boundary_variant": "free_boundary",
            "final_tracking_error_norm": float(error_arr[-1]) if error_arr.size else 0.0,
            "mean_tracking_error_norm": float(np.mean(error_arr)) if error_arr.size else 0.0,
            "final_true_tracking_error_norm": float(true_error_arr[-1])
            if true_error_arr.size
            else 0.0,
            "mean_true_tracking_error_norm": float(np.mean(true_error_arr))
            if true_error_arr.size
            else 0.0,
            "final_control_error_norm": float(control_error_arr[-1])
            if control_error_arr.size
            else 0.0,
            "mean_control_error_norm": float(np.mean(control_error_arr))
            if control_error_arr.size
            else 0.0,
            "final_true_control_error_norm": float(true_control_error_arr[-1])
            if true_control_error_arr.size
            else 0.0,
            "mean_true_control_error_norm": float(np.mean(true_control_error_arr))
            if true_control_error_arr.size
            else 0.0,
            "max_abs_delta_i": float(np.max(delta_arr)) if delta_arr.size else 0.0,
            "max_abs_coil_current": float(np.max(coil_arr)) if coil_arr.size else 0.0,
            "max_abs_actuator_lag": float(np.max(lag_arr)) if lag_arr.size else 0.0,
            "mean_abs_actuator_lag": float(np.mean(lag_arr)) if lag_arr.size else 0.0,
            "max_abs_measurement_offset": (
                float(np.max(measurement_offset_max_arr))
                if measurement_offset_max_arr.size
                else 0.0
            ),
            "mean_abs_measurement_offset": (
                float(np.mean(measurement_offset_mean_arr))
                if measurement_offset_mean_arr.size
                else 0.0
            ),
            "max_measurement_error_norm": float(np.max(measurement_error_arr))
            if measurement_error_arr.size
            else 0.0,
            "mean_measurement_error_norm": float(np.mean(measurement_error_arr))
            if measurement_error_arr.size
            else 0.0,
            "max_delayed_observation_error_norm": float(np.max(delayed_error_arr))
            if delayed_error_arr.size
            else 0.0,
            "mean_delayed_observation_error_norm": float(np.mean(delayed_error_arr))
            if delayed_error_arr.size
            else 0.0,
            "max_estimated_observation_error_norm": (
                float(np.max(estimated_error_arr)) if estimated_error_arr.size else 0.0
            ),
            "mean_estimated_observation_error_norm": (
                float(np.mean(estimated_error_arr)) if estimated_error_arr.size else 0.0
            ),
            "mean_accepted_gain": float(np.mean(gain_arr)) if gain_arr.size else 0.0,
            "min_accepted_gain": float(np.min(gain_arr)) if gain_arr.size else 0.0,
            "observer_enabled": bool(self.observer_gain > 0.0),
            "observer_gain": float(self.observer_gain),
            "measurement_distortion_enabled": measurement_distortion_enabled,
            "measurement_compensation_enabled": measurement_compensation_enabled,
            "measurement_latency_enabled": measurement_latency_enabled,
            "measurement_latency_steps": int(self.measurement_latency_steps),
            "latency_compensation_enabled": latency_compensation_enabled,
            "latency_compensation_gain": float(self.latency_compensation_gain),
            "max_abs_objective_bias_estimate": float(np.max(bias_max_arr))
            if bias_max_arr.size
            else 0.0,
            "mean_abs_objective_bias_estimate": float(np.mean(bias_mean_arr))
            if bias_mean_arr.size
            else 0.0,
            "max_abs_objective_rate_estimate": float(np.max(rate_max_arr))
            if rate_max_arr.size
            else 0.0,
            "min_response_rank": int(np.min(response_rank_arr)) if response_rank_arr.size else 0,
            "max_response_condition_number": float(np.max(response_cond_arr))
            if response_cond_arr.size
            else 0.0,
            "max_response_singular_value": float(np.max(response_sigma_arr))
            if response_sigma_arr.size
            else 0.0,
            "response_degenerate_count": int(np.sum(response_deg_arr)),
            "final_active_control_rows": int(active_control_arr[-1])
            if active_control_arr.size
            else 0,
            "mean_active_control_rows": float(np.mean(active_control_arr))
            if active_control_arr.size
            else 0.0,
            "max_coil_penalty": float(np.max(coil_penalty_arr)) if coil_penalty_arr.size else 1.0,
            "objective_tolerances": last_metrics["objective_tolerances"],
            "objective_checks": last_metrics["objective_checks"],
            "objective_convergence_active": last_metrics["objective_convergence_active"],
            "objective_converged": last_metrics["objective_converged"],
            "supervisor_limits": last_supervisor["supervisor_limits"],
            "supervisor_checks": last_supervisor["supervisor_checks"],
            "supervisor_active": last_supervisor["supervisor_active"],
            "supervisor_safe": last_supervisor["supervisor_safe"],
            "supervisor_intervention_count": int(np.sum(supervisor_arr)),
            "fallback_configured": bool(self.fallback_currents is not None),
            "fallback_active_steps": int(np.sum(fallback_arr)),
            "tolerance_regression_blocked_count": int(np.sum(tolerance_block_arr)),
            "hold_steps_after_reject": int(self.hold_steps_after_reject),
            "shape_rms": last_metrics["shape_rms"],
            "true_shape_rms": last_true_metrics["shape_rms"],
            "shape_max_abs": last_metrics["shape_max_abs"],
            "true_shape_max_abs": last_true_metrics["shape_max_abs"],
            "x_point_position_error": last_metrics["x_point_position_error"],
            "true_x_point_position_error": last_true_metrics["x_point_position_error"],
            "x_point_flux_error": last_metrics["x_point_flux_error"],
            "true_x_point_flux_error": last_true_metrics["x_point_flux_error"],
            "divertor_rms": last_metrics["divertor_rms"],
            "true_divertor_rms": last_true_metrics["divertor_rms"],
            "divertor_max_abs": last_metrics["divertor_max_abs"],
            "true_divertor_max_abs": last_true_metrics["divertor_max_abs"],
        }

    def _map_observation_to_ekf(self, observation: FloatArray) -> FloatArray | None:
        """Extract [R, Z, Ip, Te] measurement from observation vector."""
        z = np.zeros(4)
        found_rz = False

        for block in self.objective_blocks:
            if block.name == "x_point_position":
                z[0] = observation[block.start]  # R
                z[1] = observation[block.start + 1]  # Z
                found_rz = True
            # Note: Ip and Te could be added if they were tracked objectives

        return z if found_rz else None

    def _map_ekf_to_observation(self, observation: FloatArray, x_ekf: np.ndarray) -> FloatArray:
        """Inject EKF estimate [R, Z] back into observation vector."""
        refined = observation.copy()
        for block in self.objective_blocks:
            if block.name == "x_point_position":
                refined[block.start] = x_ekf[0]  # R_est
                refined[block.start + 1] = x_ekf[1]  # Z_est
        return refined


def run_free_boundary_tracking(
    config_file: str | None = None,
    *,
    shot_steps: int = 10,
    gain: float = 1.0,
    verbose: bool = True,
    kernel_factory: Callable[[str], Any] = FusionKernel,
    objective_tolerances: dict[str, float] | None = None,
    control_dt_s: float | None = None,
    coil_actuator_tau_s: float | None = None,
    coil_slew_limits: float | list[float] | None = None,
    supervisor_limits: dict[str, float] | None = None,
    hold_steps_after_reject: int | None = None,
    disturbance_callback: Callable[[Any, CoilSet, int], None] | None = None,
    stop_on_convergence: bool = False,
) -> dict[str, Any]:
    """Run deterministic free-boundary tracking over the configured objectives.

    Examples
    --------
    >>> summary = run_free_boundary_tracking("iter_config.json", shot_steps=2, gain=0.7, verbose=False)
    >>> bool(summary["objective_convergence_active"])
    True
    """
    if config_file is None:
        repo_root = Path(__file__).resolve().parents[3]
        config_file = str(repo_root / "iter_config.json")

    controller = FreeBoundaryTrackingController(
        str(config_file),
        kernel_factory=kernel_factory,
        verbose=verbose,
        objective_tolerances=objective_tolerances,
        control_dt_s=control_dt_s,
        coil_actuator_tau_s=coil_actuator_tau_s,
        coil_slew_limits=coil_slew_limits,
        supervisor_limits=supervisor_limits,
        hold_steps_after_reject=hold_steps_after_reject,
    )
    summary = controller.run_tracking_shot(
        shot_steps=shot_steps,
        gain=gain,
        disturbance_callback=disturbance_callback,
        stop_on_convergence=stop_on_convergence,
    )
    summary["config_path"] = str(config_file)
    return summary


__all__ = ["FreeBoundaryTrackingController", "run_free_boundary_tracking"]
