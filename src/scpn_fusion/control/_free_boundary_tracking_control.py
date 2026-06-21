# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Free-Boundary Tracking Control
"""Response identification and constrained correction logic for tracking control."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from scpn_fusion.control._free_boundary_tracking_base import _FreeBoundaryTrackingState
from scpn_fusion.control._free_boundary_tracking_types import FloatArray, _ActuatorSnapshot


class _FreeBoundaryTrackingControlMixin(_FreeBoundaryTrackingState):
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
        return mask

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
        return penalties

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
            max_abs_coil_current=(
                float(np.max(np.abs(self.coils.currents))) if self.coils.currents.size > 0 else 0.0
            ),
            max_abs_actuator_lag=max_abs_actuator_lag,
        )
        return (
            metrics_after,
            true_metrics_after,
            supervisor_after,
            max_abs_actuator_lag,
            fallback_active,
        )

    def evaluate_objectives(self, observation: np.ndarray[Any, Any]) -> dict[str, Any]:
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
