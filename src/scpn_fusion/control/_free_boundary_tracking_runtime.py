# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Free-Boundary Tracking Control
"""Runtime observation, latency, actuator, and solver helpers for tracking control."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from scpn_fusion.control._free_boundary_tracking_types import (
    FloatArray,
    _ObservationSnapshot,
)


class _FreeBoundaryTrackingRuntimeMixin:
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
