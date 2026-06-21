# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Free-Boundary Tracking Control
"""Shot execution loop for free-boundary tracking controller evaluations."""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from scpn_fusion.control._free_boundary_tracking_base import _FreeBoundaryTrackingState
from scpn_fusion.control._free_boundary_tracking_types import FloatArray
from scpn_fusion.core.fusion_kernel import CoilSet


class _FreeBoundaryTrackingShotMixin(_FreeBoundaryTrackingState):
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
            max_abs_coil_current=(
                float(np.max(np.abs(self.coils.currents))) if self.coils.currents.size > 0 else 0.0
            ),
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
        fallback_mask = fallback_arr > 0.5
        fallback_lag_arr = (
            lag_arr[fallback_mask] if lag_arr.size == fallback_arr.size else lag_arr[:0]
        )
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
            "final_true_tracking_error_norm": (
                float(true_error_arr[-1]) if true_error_arr.size else 0.0
            ),
            "mean_true_tracking_error_norm": (
                float(np.mean(true_error_arr)) if true_error_arr.size else 0.0
            ),
            "final_control_error_norm": (
                float(control_error_arr[-1]) if control_error_arr.size else 0.0
            ),
            "mean_control_error_norm": (
                float(np.mean(control_error_arr)) if control_error_arr.size else 0.0
            ),
            "final_true_control_error_norm": (
                float(true_control_error_arr[-1]) if true_control_error_arr.size else 0.0
            ),
            "mean_true_control_error_norm": (
                float(np.mean(true_control_error_arr)) if true_control_error_arr.size else 0.0
            ),
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
            "max_measurement_error_norm": (
                float(np.max(measurement_error_arr)) if measurement_error_arr.size else 0.0
            ),
            "mean_measurement_error_norm": (
                float(np.mean(measurement_error_arr)) if measurement_error_arr.size else 0.0
            ),
            "max_delayed_observation_error_norm": (
                float(np.max(delayed_error_arr)) if delayed_error_arr.size else 0.0
            ),
            "mean_delayed_observation_error_norm": (
                float(np.mean(delayed_error_arr)) if delayed_error_arr.size else 0.0
            ),
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
            "max_abs_objective_bias_estimate": (
                float(np.max(bias_max_arr)) if bias_max_arr.size else 0.0
            ),
            "mean_abs_objective_bias_estimate": (
                float(np.mean(bias_mean_arr)) if bias_mean_arr.size else 0.0
            ),
            "max_abs_objective_rate_estimate": (
                float(np.max(rate_max_arr)) if rate_max_arr.size else 0.0
            ),
            "min_response_rank": int(np.min(response_rank_arr)) if response_rank_arr.size else 0,
            "max_response_condition_number": (
                float(np.max(response_cond_arr)) if response_cond_arr.size else 0.0
            ),
            "max_response_singular_value": (
                float(np.max(response_sigma_arr)) if response_sigma_arr.size else 0.0
            ),
            "response_degenerate_count": int(np.sum(response_deg_arr)),
            "final_active_control_rows": (
                int(active_control_arr[-1]) if active_control_arr.size else 0
            ),
            "mean_active_control_rows": (
                float(np.mean(active_control_arr)) if active_control_arr.size else 0.0
            ),
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
            "fallback_activation_rate": (
                float(np.mean(fallback_arr)) if fallback_arr.size else 0.0
            ),
            "max_abs_actuator_lag_during_fallback": (
                float(np.max(fallback_lag_arr)) if fallback_lag_arr.size else 0.0
            ),
            "mean_abs_actuator_lag_during_fallback": (
                float(np.mean(fallback_lag_arr)) if fallback_lag_arr.size else 0.0
            ),
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

    def _map_ekf_to_observation(
        self, observation: FloatArray, x_ekf: np.ndarray[Any, Any]
    ) -> FloatArray:
        """Inject EKF estimate [R, Z] back into observation vector."""
        refined = observation.copy()
        for block in self.objective_blocks:
            if block.name == "x_point_position":
                refined[block.start] = x_ekf[0]  # R_est
                refined[block.start + 1] = x_ekf[1]  # Z_est
        return refined
