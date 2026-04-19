# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Free-Boundary Supervisory Control
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.control.disruption_risk_runtime import predict_disruption_risk
from scpn_fusion.control.fusion_sota_mpc import NeuralSurrogate

try:
    from scpn_fusion.core._rust_compat import FusionKernel
except ImportError:
    from scpn_fusion.core.fusion_kernel import FusionKernel


logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

DEFAULT_TARGET_VECTOR = np.array([6.0, 0.0, 5.0, -3.5], dtype=np.float64)
SUPERVISORY_DISRUPTION_RISK_BIAS = 2.5
SUPERVISORY_ALERT_LEVEL_NAMES = ("nominal", "warning", "guarded", "fallback")


def _require_positive_int(name: str, value: int) -> int:
    out = int(value)
    if out < 1:
        raise ValueError(f"{name} must be >= 1.")
    return out


def _require_nonnegative_int(name: str, value: int) -> int:
    out = int(value)
    if out < 0:
        raise ValueError(f"{name} must be >= 0.")
    return out


def _require_positive_finite(name: str, value: float) -> float:
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be finite and > 0.")
    return out


def _normalize_bounds(bounds: tuple[float, float], name: str) -> tuple[float, float]:
    lo = float(bounds[0])
    hi = float(bounds[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        raise ValueError(f"{name} must be finite with lower < upper.")
    return lo, hi


def _normalize_vector(
    values: Optional[np.ndarray | tuple[float, ...] | list[float]],
    *,
    length: int,
    default: float = 0.0,
    name: str,
) -> FloatArray:
    if values is None:
        return np.full(length, float(default), dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size != length:
        raise ValueError(f"{name} must have length {length}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values.")
    return arr


def _normalize_mask(
    values: Optional[np.ndarray | tuple[float, ...] | list[float]],
    *,
    length: int,
    name: str,
) -> NDArray[np.bool_]:
    if values is None:
        return np.zeros(length, dtype=bool)
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size != length:
        raise ValueError(f"{name} must have length {length}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values.")
    return np.asarray(arr != 0.0, dtype=bool)


@dataclass(frozen=True)
class FreeBoundaryTarget:
    r_axis_m: float = 6.0
    z_axis_m: float = 0.0
    x_point_r_m: float = 5.0
    x_point_z_m: float = -3.5

    def as_vector(self) -> FloatArray:
        return np.array(
            [self.r_axis_m, self.z_axis_m, self.x_point_r_m, self.x_point_z_m],
            dtype=np.float64,
        )


@dataclass
class FreeBoundaryEstimate:
    state_hat: FloatArray
    bias_hat: FloatArray
    actuator_bias_hat: FloatArray
    innovation: FloatArray
    corrected_state: FloatArray
    uncertainty_norm: float


@dataclass
class FreeBoundarySafetyMargins:
    signal_scalar: float
    q95: float
    beta_n: float
    disruption_risk: float
    q95_margin: float
    beta_margin: float
    risk_margin: float


@dataclass
class SafetyFilterResult:
    action: FloatArray
    predicted_currents: FloatArray
    safe_target_ip_ma: float
    intervention_active: bool
    saturation_active: bool
    failsafe_active: bool
    fallback_mode_active: bool
    invariant_violation_active: bool
    physics_guard_active: bool
    q95_guard_active: bool
    beta_guard_active: bool
    risk_guard_active: bool
    q95: float
    beta_n: float
    disruption_risk: float
    q95_margin: float
    beta_margin: float
    risk_margin: float
    degraded_mode_active: bool
    diagnostic_dropout_active: bool
    actuator_dropout_active: bool
    alert_level: int
    requested_alert_level: int
    alert_mode: str
    alert_transition_active: bool
    recovery_transition_active: bool
    risk_score: float
    reasons: list[str]


def _build_margin_signal_scalar(
    *,
    q95: float,
    beta_n: float,
    axis_error: float,
    xpoint_error: float,
    bias_norm: float,
    coil_spread: float,
) -> float:
    return float(
        0.32
        + 0.24 * max(beta_n - 1.9, 0.0)
        + 0.30 * max(4.2 - q95, 0.0)
        + 0.34 * axis_error
        + 0.26 * xpoint_error
        + 0.11 * bias_norm
        + 0.08 * coil_spread
    )


def estimate_free_boundary_safety_margins(
    *,
    corrected_state: np.ndarray,
    target_state: np.ndarray,
    bias_hat: np.ndarray,
    coil_currents: np.ndarray,
    target_ip_ma: float,
    q95_floor: float,
    beta_n_ceiling: float,
    disruption_risk_ceiling: float,
    risk_signal_history: Optional[np.ndarray | list[float]] = None,
) -> FreeBoundarySafetyMargins:
    corrected = np.asarray(corrected_state, dtype=np.float64).reshape(-1)
    target = np.asarray(target_state, dtype=np.float64).reshape(-1)
    bias = np.asarray(bias_hat, dtype=np.float64).reshape(-1)
    currents = np.asarray(coil_currents, dtype=np.float64).reshape(-1)
    if corrected.size != 4 or target.size != 4:
        raise ValueError("corrected_state and target_state must be finite 4-vectors.")
    if bias.size != 4 or currents.size == 0:
        raise ValueError("bias_hat must be a finite 4-vector and coil_currents must be non-empty.")
    if not (
        np.all(np.isfinite(corrected))
        and np.all(np.isfinite(target))
        and np.all(np.isfinite(bias))
        and np.all(np.isfinite(currents))
    ):
        raise ValueError("safety-margin inputs must be finite.")

    axis_error = float(np.linalg.norm(target[:2] - corrected[:2]))
    xpoint_error = float(np.linalg.norm(target[2:] - corrected[2:]))
    bias_norm = float(np.linalg.norm(bias))
    coil_abs = float(np.max(np.abs(currents)))
    coil_spread = float(np.std(currents))
    target_sep = float(np.linalg.norm(target[:2] - target[2:]))
    state_sep = float(np.linalg.norm(corrected[:2] - corrected[2:]))
    shape_error = abs(state_sep - target_sep)
    current_drive = max(float(target_ip_ma) - 7.0, 0.0)

    q95 = float(
        np.clip(
            4.65
            - 0.22 * current_drive
            - 2.10 * axis_error
            - 1.60 * xpoint_error
            - 0.28 * coil_spread
            - 0.55 * shape_error,
            2.2,
            6.2,
        )
    )
    beta_n = float(
        np.clip(
            1.45
            + 0.16 * current_drive
            + 0.45 * coil_abs
            + 1.70 * bias_norm
            + 1.20 * xpoint_error
            + 0.70 * shape_error,
            0.6,
            4.2,
        )
    )
    signal_scalar = _build_margin_signal_scalar(
        q95=q95,
        beta_n=beta_n,
        axis_error=axis_error,
        xpoint_error=xpoint_error,
        bias_norm=bias_norm,
        coil_spread=coil_spread,
    )
    if risk_signal_history is None:
        signal_history = np.array([signal_scalar], dtype=np.float64)
    else:
        history = np.asarray(risk_signal_history, dtype=np.float64).reshape(-1)
        if history.size == 0 or not np.all(np.isfinite(history)):
            raise ValueError("risk_signal_history must contain finite values when provided.")
        signal_history = np.concatenate((history, np.array([signal_scalar], dtype=np.float64)))
    toroidal_obs = {
        "toroidal_n1_amp": float(0.04 + 0.35 * xpoint_error + 0.05 * coil_spread),
        "toroidal_n2_amp": float(0.03 + 0.25 * axis_error + 0.05 * bias_norm),
        "toroidal_n3_amp": float(0.02 + 0.10 * shape_error),
        "toroidal_asymmetry_index": float(0.06 + 0.45 * coil_spread + 0.15 * xpoint_error),
        "toroidal_radial_spread": float(0.02 + 0.25 * axis_error),
    }
    disruption_risk = float(
        predict_disruption_risk(
            signal_history,
            toroidal_obs,
            bias_delta=SUPERVISORY_DISRUPTION_RISK_BIAS,
        )
    )
    return FreeBoundarySafetyMargins(
        signal_scalar=signal_scalar,
        q95=q95,
        beta_n=beta_n,
        disruption_risk=disruption_risk,
        q95_margin=float(q95 - q95_floor),
        beta_margin=float(beta_n_ceiling - beta_n),
        risk_margin=float(disruption_risk_ceiling - disruption_risk),
    )


class FreeBoundaryStateEstimator:
    """Observer with persistent-bias tracking for free-boundary state."""

    def __init__(
        self,
        surrogate: NeuralSurrogate,
        *,
        measurement_gain: float = 0.55,
        bias_gain: float = 0.08,
        bias_decay: float = 0.985,
        actuator_bias_gain: float = 0.06,
        actuator_bias_decay: float = 0.992,
        max_actuator_bias: float = 0.35,
    ) -> None:
        self.surrogate = surrogate
        self.measurement_gain = _require_positive_finite("measurement_gain", measurement_gain)
        self.bias_gain = _require_positive_finite("bias_gain", bias_gain)
        self.bias_decay = float(bias_decay)
        if not np.isfinite(self.bias_decay) or not (0.0 < self.bias_decay <= 1.0):
            raise ValueError("bias_decay must be finite and in (0, 1].")
        self.actuator_bias_gain = _require_positive_finite("actuator_bias_gain", actuator_bias_gain)
        self.actuator_bias_decay = float(actuator_bias_decay)
        if not np.isfinite(self.actuator_bias_decay) or not (0.0 < self.actuator_bias_decay <= 1.0):
            raise ValueError("actuator_bias_decay must be finite and in (0, 1].")
        self.max_actuator_bias = _require_positive_finite("max_actuator_bias", max_actuator_bias)
        self.allocation = np.linalg.pinv(
            np.asarray(self.surrogate.B, dtype=np.float64),
            rcond=1e-6,
        )
        self.state_hat = np.zeros(4, dtype=np.float64)
        self.bias_hat = np.zeros(4, dtype=np.float64)
        self.actuator_bias_hat = np.zeros(self.surrogate.B.shape[1], dtype=np.float64)
        self.innovation = np.zeros(4, dtype=np.float64)
        self.corrected_state = np.zeros(4, dtype=np.float64)
        self._initialized = False

    def reset(self, initial_state: np.ndarray | list[float] | tuple[float, ...]) -> None:
        state = np.asarray(initial_state, dtype=np.float64).reshape(-1)
        if state.size != 4 or not np.all(np.isfinite(state)):
            raise ValueError("initial_state must be a finite 4-vector.")
        self.state_hat = state.copy()
        self.bias_hat.fill(0.0)
        self.actuator_bias_hat.fill(0.0)
        self.innovation.fill(0.0)
        self.corrected_state = state.copy()
        self._initialized = True

    def step(
        self,
        measurement: np.ndarray,
        applied_action: np.ndarray,
        measured_actuator_action: Optional[np.ndarray] = None,
    ) -> FreeBoundaryEstimate:
        meas = np.asarray(measurement, dtype=np.float64).reshape(-1)
        action = np.asarray(applied_action, dtype=np.float64).reshape(-1)
        measured_action = (
            np.asarray(measured_actuator_action, dtype=np.float64).reshape(-1)
            if measured_actuator_action is not None
            else None
        )
        if meas.size != 4 or not np.all(np.isfinite(meas)):
            raise ValueError("measurement must be a finite 4-vector.")
        if action.size != self.surrogate.B.shape[1] or not np.all(np.isfinite(action)):
            raise ValueError("applied_action must match the surrogate control width.")
        if measured_action is not None and (
            measured_action.size != action.size or not np.all(np.isfinite(measured_action))
        ):
            raise ValueError("measured_actuator_action must match the surrogate control width.")

        if not self._initialized:
            self.reset(meas)

        effective_action = action + self.actuator_bias_hat
        predicted = self.surrogate.predict(self.corrected_state, effective_action)
        innovation = meas - predicted
        self.state_hat = predicted + self.measurement_gain * innovation
        self.bias_hat = self.bias_decay * self.bias_hat + self.bias_gain * innovation
        if measured_action is not None:
            actuator_residual = measured_action - action
        else:
            actuator_residual = self.allocation @ (innovation - self.bias_hat)
        actuator_prior = self.actuator_bias_decay * self.actuator_bias_hat
        self.actuator_bias_hat = np.clip(
            actuator_prior + self.actuator_bias_gain * (actuator_residual - actuator_prior),
            -self.max_actuator_bias,
            self.max_actuator_bias,
        )
        self.innovation = innovation
        self.corrected_state = self.state_hat + self.bias_hat
        return FreeBoundaryEstimate(
            state_hat=self.state_hat.copy(),
            bias_hat=self.bias_hat.copy(),
            actuator_bias_hat=self.actuator_bias_hat.copy(),
            innovation=self.innovation.copy(),
            corrected_state=self.corrected_state.copy(),
            uncertainty_norm=float(
                np.linalg.norm(innovation) + 0.25 * np.linalg.norm(self.actuator_bias_hat)
            ),
        )


class FreeBoundarySafetySupervisor:
    """Constraint and safety layer for closed-loop free-boundary control."""

    def __init__(
        self,
        *,
        coil_current_limits: tuple[float, float] = (-40.0, 40.0),
        coil_delta_limit: float = 2.0,
        total_action_l1_limit: float = 4.0,
        current_margin_fraction: float = 0.92,
        ip_bounds_ma: tuple[float, float] = (5.0, 16.0),
        severe_axis_error_m: float = 0.10,
        severe_xpoint_error_m: float = 0.12,
        severe_bias_norm_m: float = 0.08,
        ip_backoff_ma: float = 0.25,
        axis_r_bounds_m: tuple[float, float] = (5.88, 6.12),
        axis_z_bounds_m: tuple[float, float] = (-0.12, 0.12),
        xpoint_r_bounds_m: tuple[float, float] = (4.92, 5.10),
        xpoint_z_bounds_m: tuple[float, float] = (-3.60, -3.36),
        q95_floor: float = 3.5,
        beta_n_ceiling: float = 2.6,
        disruption_risk_ceiling: float = 0.90,
        warning_risk_score_threshold: float = 0.88,
        guarded_risk_score_threshold: float = 1.05,
        alert_recovery_hold_steps: int = 6,
        fallback_hold_steps: int = 4,
        fallback_action_scale: float = 0.45,
    ) -> None:
        self.coil_current_limits = _normalize_bounds(coil_current_limits, "coil_current_limits")
        self.coil_delta_limit = _require_positive_finite("coil_delta_limit", coil_delta_limit)
        self.total_action_l1_limit = _require_positive_finite(
            "total_action_l1_limit", total_action_l1_limit
        )
        self.current_margin_fraction = float(current_margin_fraction)
        if not np.isfinite(self.current_margin_fraction) or not (
            0.0 < self.current_margin_fraction <= 1.0
        ):
            raise ValueError("current_margin_fraction must be finite and in (0, 1].")
        self.ip_bounds_ma = _normalize_bounds(ip_bounds_ma, "ip_bounds_ma")
        self.severe_axis_error_m = _require_positive_finite(
            "severe_axis_error_m", severe_axis_error_m
        )
        self.severe_xpoint_error_m = _require_positive_finite(
            "severe_xpoint_error_m", severe_xpoint_error_m
        )
        self.severe_bias_norm_m = _require_positive_finite("severe_bias_norm_m", severe_bias_norm_m)
        self.ip_backoff_ma = _require_positive_finite("ip_backoff_ma", ip_backoff_ma)
        self.axis_r_bounds_m = _normalize_bounds(axis_r_bounds_m, "axis_r_bounds_m")
        self.axis_z_bounds_m = _normalize_bounds(axis_z_bounds_m, "axis_z_bounds_m")
        self.xpoint_r_bounds_m = _normalize_bounds(xpoint_r_bounds_m, "xpoint_r_bounds_m")
        self.xpoint_z_bounds_m = _normalize_bounds(xpoint_z_bounds_m, "xpoint_z_bounds_m")
        self.q95_floor = _require_positive_finite("q95_floor", q95_floor)
        self.beta_n_ceiling = _require_positive_finite("beta_n_ceiling", beta_n_ceiling)
        self.disruption_risk_ceiling = _require_positive_finite(
            "disruption_risk_ceiling",
            disruption_risk_ceiling,
        )
        if self.disruption_risk_ceiling >= 1.0:
            raise ValueError("disruption_risk_ceiling must be < 1.0.")
        self.warning_risk_score_threshold = _require_positive_finite(
            "warning_risk_score_threshold",
            warning_risk_score_threshold,
        )
        self.guarded_risk_score_threshold = _require_positive_finite(
            "guarded_risk_score_threshold",
            guarded_risk_score_threshold,
        )
        if self.guarded_risk_score_threshold <= self.warning_risk_score_threshold:
            raise ValueError("guarded_risk_score_threshold must be > warning_risk_score_threshold.")
        self.alert_recovery_hold_steps = _require_positive_int(
            "alert_recovery_hold_steps",
            alert_recovery_hold_steps,
        )
        self.fallback_hold_steps = _require_positive_int("fallback_hold_steps", fallback_hold_steps)
        self.fallback_action_scale = _require_positive_finite(
            "fallback_action_scale", fallback_action_scale
        )
        self._fallback_steps_remaining = 0
        self._alert_level = 0
        self._alert_recovery_steps = 0

    def reset(self) -> None:
        self._fallback_steps_remaining = 0
        self._alert_level = 0
        self._alert_recovery_steps = 0

    def _classify_requested_alert_level(
        self,
        *,
        margins: FreeBoundarySafetyMargins,
        risk_score: float,
        q95_guard_active: bool,
        beta_guard_active: bool,
        risk_guard_active: bool,
        diagnostic_dropout_active: bool,
        actuator_dropout_active: bool,
        failsafe_active: bool,
        invariant_violation_active: bool,
    ) -> int:
        risk_fraction = float(margins.disruption_risk / max(self.disruption_risk_ceiling, 1e-9))
        requested = 0
        if diagnostic_dropout_active:
            requested = max(requested, 2)
        if actuator_dropout_active:
            requested = max(requested, 3)
        if (
            failsafe_active
            or invariant_violation_active
            or risk_guard_active
            or (q95_guard_active and beta_guard_active)
        ):
            requested = max(requested, 3)
        if (
            q95_guard_active
            or beta_guard_active
            or risk_fraction >= 0.96
            or margins.q95_margin < 0.08
            or margins.beta_margin < 0.08
            or risk_score >= self.guarded_risk_score_threshold
        ):
            requested = max(requested, 2)
        if (
            risk_fraction >= 0.80
            or margins.q95_margin < 0.24
            or margins.beta_margin < 0.24
            or risk_score >= self.warning_risk_score_threshold
        ):
            requested = max(requested, 1)
        return int(requested)

    def _advance_alert_level(
        self,
        *,
        requested_alert_level: int,
        margins: FreeBoundarySafetyMargins,
        risk_score: float,
        invariant_violation_active: bool,
        failsafe_active: bool,
    ) -> tuple[int, bool, bool]:
        previous_level = self._alert_level
        if requested_alert_level > self._alert_level:
            self._alert_level = requested_alert_level
            self._alert_recovery_steps = 0
        elif requested_alert_level < self._alert_level:
            recovery_safe = bool(
                not invariant_violation_active
                and not failsafe_active
                and margins.risk_margin >= 0.02
                and margins.q95_margin >= 0.12
                and margins.beta_margin >= 0.10
                and risk_score <= 0.82
            )
            if recovery_safe:
                self._alert_recovery_steps += 1
                if self._alert_recovery_steps >= self.alert_recovery_hold_steps:
                    self._alert_level = max(requested_alert_level, self._alert_level - 1)
                    self._alert_recovery_steps = 0
            else:
                self._alert_recovery_steps = 0
        else:
            self._alert_recovery_steps = 0

        alert_transition_active = self._alert_level != previous_level
        recovery_transition_active = self._alert_level < previous_level
        return self._alert_level, alert_transition_active, recovery_transition_active

    def filter_action(
        self,
        proposed_action: np.ndarray,
        *,
        corrected_state: np.ndarray,
        target_state: np.ndarray,
        bias_hat: np.ndarray,
        coil_currents: np.ndarray,
        target_ip_ma: float,
        predicted_next_state: Optional[np.ndarray] = None,
        safety_margins: Optional[FreeBoundarySafetyMargins] = None,
        diagnostic_dropout_active: bool = False,
        actuator_dropout_active: bool = False,
    ) -> SafetyFilterResult:
        action = np.asarray(proposed_action, dtype=np.float64).reshape(-1)
        corrected = np.asarray(corrected_state, dtype=np.float64).reshape(-1)
        target = np.asarray(target_state, dtype=np.float64).reshape(-1)
        bias = np.asarray(bias_hat, dtype=np.float64).reshape(-1)
        currents = np.asarray(coil_currents, dtype=np.float64).reshape(-1)

        if action.size != currents.size:
            raise ValueError("proposed_action and coil_currents must have identical length.")

        reasons: list[str] = []
        clipped_action = np.clip(action, -self.coil_delta_limit, self.coil_delta_limit)
        saturation_active = not np.allclose(clipped_action, action)
        if saturation_active:
            reasons.append("delta_limit")

        action_l1 = float(np.sum(np.abs(clipped_action)))
        if action_l1 > self.total_action_l1_limit:
            clipped_action = clipped_action * (self.total_action_l1_limit / max(action_l1, 1e-12))
            saturation_active = True
            reasons.append("total_action_l1_limit")

        margin_lo = self.current_margin_fraction * self.coil_current_limits[0]
        margin_hi = self.current_margin_fraction * self.coil_current_limits[1]
        predicted = currents + clipped_action
        if np.any(predicted < margin_lo) or np.any(predicted > margin_hi):
            clipped_predicted = np.clip(predicted, margin_lo, margin_hi)
            clipped_action = clipped_predicted - currents
            predicted = clipped_predicted
            saturation_active = True
            reasons.append("current_margin")
        else:
            predicted = np.clip(predicted, *self.coil_current_limits)

        axis_error = float(np.linalg.norm(target[:2] - corrected[:2]))
        xpoint_error = float(np.linalg.norm(target[2:] - corrected[2:]))
        bias_norm = float(np.linalg.norm(bias))
        current_ratio = float(
            np.max(np.abs(predicted))
            / max(abs(self.coil_current_limits[1]), abs(self.coil_current_limits[0]), 1e-9)
        )
        risk_score = float(
            max(
                axis_error / self.severe_axis_error_m,
                xpoint_error / self.severe_xpoint_error_m,
                bias_norm / self.severe_bias_norm_m,
                current_ratio / max(self.current_margin_fraction, 1e-6),
            )
        )
        safe_target_ip_ma = float(np.clip(target_ip_ma, *self.ip_bounds_ma))
        margins = safety_margins or estimate_free_boundary_safety_margins(
            corrected_state=corrected,
            target_state=target,
            bias_hat=bias,
            coil_currents=currents,
            target_ip_ma=target_ip_ma,
            q95_floor=self.q95_floor,
            beta_n_ceiling=self.beta_n_ceiling,
            disruption_risk_ceiling=self.disruption_risk_ceiling,
        )
        geometry = (
            np.asarray(predicted_next_state, dtype=np.float64).reshape(-1)
            if predicted_next_state is not None
            else corrected
        )
        if geometry.size != 4 or not np.all(np.isfinite(geometry)):
            raise ValueError("predicted_next_state must be a finite 4-vector when provided.")
        invariant_violation_active = bool(
            geometry[0] < self.axis_r_bounds_m[0]
            or geometry[0] > self.axis_r_bounds_m[1]
            or geometry[1] < self.axis_z_bounds_m[0]
            or geometry[1] > self.axis_z_bounds_m[1]
            or geometry[2] < self.xpoint_r_bounds_m[0]
            or geometry[2] > self.xpoint_r_bounds_m[1]
            or geometry[3] < self.xpoint_z_bounds_m[0]
            or geometry[3] > self.xpoint_z_bounds_m[1]
        )
        if invariant_violation_active:
            safe_target_ip_ma = max(
                self.ip_bounds_ma[0], safe_target_ip_ma - 2.0 * self.ip_backoff_ma
            )
            reasons.append("geometry_guard")
        q95_guard_active = margins.q95_margin < 0.0
        beta_guard_active = margins.beta_margin < 0.0
        risk_guard_active = margins.risk_margin < 0.0
        physics_guard_active = False
        if q95_guard_active:
            safe_target_ip_ma = max(
                self.ip_bounds_ma[0], safe_target_ip_ma - 1.5 * self.ip_backoff_ma
            )
            reasons.append("q95_guard")
            physics_guard_active = True
        if beta_guard_active:
            safe_target_ip_ma = max(
                self.ip_bounds_ma[0], safe_target_ip_ma - 1.5 * self.ip_backoff_ma
            )
            clipped_action = 0.85 * clipped_action
            predicted = np.clip(currents + clipped_action, *self.coil_current_limits)
            reasons.append("beta_guard")
            physics_guard_active = True
        if risk_guard_active:
            safe_target_ip_ma = max(
                self.ip_bounds_ma[0], safe_target_ip_ma - 2.0 * self.ip_backoff_ma
            )
            clipped_action = 0.75 * clipped_action
            predicted = np.clip(currents + clipped_action, *self.coil_current_limits)
            reasons.append("disruption_risk_guard")
            physics_guard_active = True
        degraded_mode_active = bool(diagnostic_dropout_active or actuator_dropout_active)
        if diagnostic_dropout_active:
            safe_target_ip_ma = max(
                self.ip_bounds_ma[0], safe_target_ip_ma - 0.75 * self.ip_backoff_ma
            )
            clipped_action = 0.82 * clipped_action
            predicted = np.clip(currents + clipped_action, *self.coil_current_limits)
            reasons.append("diagnostic_dropout_guard")
        if actuator_dropout_active:
            safe_target_ip_ma = max(
                self.ip_bounds_ma[0], safe_target_ip_ma - 1.25 * self.ip_backoff_ma
            )
            clipped_action = 0.68 * clipped_action
            predicted = np.clip(currents + clipped_action, *self.coil_current_limits)
            reasons.append("actuator_loss_guard")

        if axis_error > self.severe_axis_error_m:
            safe_target_ip_ma = max(self.ip_bounds_ma[0], safe_target_ip_ma - self.ip_backoff_ma)
            reasons.append("axis_backoff")
        if xpoint_error > self.severe_xpoint_error_m:
            safe_target_ip_ma = max(self.ip_bounds_ma[0], safe_target_ip_ma - self.ip_backoff_ma)
            reasons.append("xpoint_backoff")
        if bias_norm > self.severe_bias_norm_m:
            safe_target_ip_ma = max(
                self.ip_bounds_ma[0], safe_target_ip_ma - 0.5 * self.ip_backoff_ma
            )
            reasons.append("bias_backoff")

        failsafe_active = bool(risk_score > 1.35 and saturation_active)
        if failsafe_active:
            safe_target_ip_ma = max(
                self.ip_bounds_ma[0], safe_target_ip_ma - 4.0 * self.ip_backoff_ma
            )
            clipped_action = 0.6 * clipped_action
            predicted = np.clip(currents + clipped_action, *self.coil_current_limits)
            reasons.append("failsafe_trip")

        requested_alert_level = self._classify_requested_alert_level(
            margins=margins,
            risk_score=risk_score,
            q95_guard_active=bool(q95_guard_active),
            beta_guard_active=bool(beta_guard_active),
            risk_guard_active=bool(risk_guard_active),
            diagnostic_dropout_active=bool(diagnostic_dropout_active),
            actuator_dropout_active=bool(actuator_dropout_active),
            failsafe_active=failsafe_active,
            invariant_violation_active=invariant_violation_active,
        )
        alert_level, alert_transition_active, recovery_transition_active = (
            self._advance_alert_level(
                requested_alert_level=requested_alert_level,
                margins=margins,
                risk_score=risk_score,
                invariant_violation_active=invariant_violation_active,
                failsafe_active=failsafe_active,
            )
        )
        if alert_level == 1:
            safe_target_ip_ma = max(
                self.ip_bounds_ma[0], safe_target_ip_ma - 0.50 * self.ip_backoff_ma
            )
            clipped_action = 0.94 * clipped_action
            predicted = np.clip(currents + clipped_action, *self.coil_current_limits)
            reasons.append("policy_warning")
        elif alert_level == 2:
            safe_target_ip_ma = max(
                self.ip_bounds_ma[0], safe_target_ip_ma - 1.10 * self.ip_backoff_ma
            )
            clipped_action = 0.82 * clipped_action
            predicted = np.clip(currents + clipped_action, *self.coil_current_limits)
            reasons.append("policy_guarded")

        if failsafe_active or invariant_violation_active or requested_alert_level >= 3:
            self._fallback_steps_remaining = self.fallback_hold_steps
        fallback_mode_active = self._fallback_steps_remaining > 0
        if fallback_mode_active:
            safe_target_ip_ma = max(
                self.ip_bounds_ma[0], safe_target_ip_ma - 1.5 * self.ip_backoff_ma
            )
            clipped_action = self.fallback_action_scale * clipped_action
            predicted = np.clip(currents + clipped_action, *self.coil_current_limits)
            reasons.append("fallback_latched")
            self._fallback_steps_remaining = max(self._fallback_steps_remaining - 1, 0)

        return SafetyFilterResult(
            action=clipped_action.astype(np.float64, copy=True),
            predicted_currents=predicted.astype(np.float64, copy=True),
            safe_target_ip_ma=safe_target_ip_ma,
            intervention_active=bool(reasons),
            saturation_active=bool(saturation_active),
            failsafe_active=failsafe_active,
            fallback_mode_active=fallback_mode_active,
            invariant_violation_active=invariant_violation_active,
            physics_guard_active=physics_guard_active,
            q95_guard_active=bool(q95_guard_active),
            beta_guard_active=bool(beta_guard_active),
            risk_guard_active=bool(risk_guard_active),
            q95=float(margins.q95),
            beta_n=float(margins.beta_n),
            disruption_risk=float(margins.disruption_risk),
            q95_margin=float(margins.q95_margin),
            beta_margin=float(margins.beta_margin),
            risk_margin=float(margins.risk_margin),
            degraded_mode_active=bool(degraded_mode_active),
            diagnostic_dropout_active=bool(diagnostic_dropout_active),
            actuator_dropout_active=bool(actuator_dropout_active),
            alert_level=int(alert_level),
            requested_alert_level=int(requested_alert_level),
            alert_mode=SUPERVISORY_ALERT_LEVEL_NAMES[int(alert_level)],
            alert_transition_active=bool(alert_transition_active),
            recovery_transition_active=bool(recovery_transition_active),
            risk_score=risk_score,
            reasons=reasons,
        )


class FreeBoundarySupervisoryController:
    """Target-tracking controller for axis and X-point free-boundary geometry."""

    def __init__(
        self,
        surrogate: NeuralSurrogate,
        target: FreeBoundaryTarget,
        *,
        state_gains: np.ndarray | tuple[float, ...] = (1.8, 2.1, 1.4, 1.7),
        bias_rejection_gains: np.ndarray | tuple[float, ...] = (0.7, 0.8, 0.55, 0.65),
        innovation_damping: float = 0.18,
        actuator_compensation_gain: float = 0.85,
    ) -> None:
        self.surrogate = surrogate
        self.target = target
        self.state_gains = _normalize_vector(state_gains, length=4, name="state_gains")
        self.bias_rejection_gains = _normalize_vector(
            bias_rejection_gains, length=4, name="bias_rejection_gains"
        )
        self.innovation_damping = _require_positive_finite("innovation_damping", innovation_damping)
        self.actuator_compensation_gain = _require_positive_finite(
            "actuator_compensation_gain",
            actuator_compensation_gain,
        )
        self.allocation = np.linalg.pinv(
            np.asarray(self.surrogate.B, dtype=np.float64),
            rcond=1e-6,
        )

    def propose_action(self, estimate: FreeBoundaryEstimate) -> FloatArray:
        error = self.target.as_vector() - estimate.corrected_state
        desired_delta = (
            self.state_gains * error
            - self.bias_rejection_gains * estimate.bias_hat
            - self.innovation_damping * estimate.innovation
        )
        action = (
            self.allocation @ desired_delta
            - self.actuator_compensation_gain * estimate.actuator_bias_hat
        )
        return np.asarray(action, dtype=np.float64).reshape(-1)


def extract_free_boundary_state(kernel: Any) -> FloatArray:
    psi = np.asarray(kernel.Psi, dtype=np.float64)
    idx_max = int(np.argmax(psi))
    iz, ir = np.unravel_index(idx_max, psi.shape)
    r_axis = float(kernel.R[ir])
    z_axis = float(kernel.Z[iz])

    if hasattr(kernel, "dR") and 1 <= ir < len(kernel.R) - 1:
        a = float(psi[iz, ir - 1])
        b = float(psi[iz, ir])
        c = float(psi[iz, ir + 1])
        denom = 2.0 * (a - 2.0 * b + c)
        if abs(denom) > 1e-30:
            r_axis += float(np.clip(-(c - a) / denom, -0.5, 0.5)) * float(kernel.dR)
    if hasattr(kernel, "dZ") and 1 <= iz < len(kernel.Z) - 1:
        a = float(psi[iz - 1, ir])
        b = float(psi[iz, ir])
        c = float(psi[iz + 1, ir])
        denom = 2.0 * (a - 2.0 * b + c)
        if abs(denom) > 1e-30:
            z_axis += float(np.clip(-(c - a) / denom, -0.5, 0.5)) * float(kernel.dZ)

    x_point, _ = kernel.find_x_point(psi)
    return np.array([r_axis, z_axis, float(x_point[0]), float(x_point[1])], dtype=np.float64)


def _plot_free_boundary_control(
    *,
    time_axis: FloatArray,
    states: FloatArray,
    target: FreeBoundaryTarget,
    actions: FloatArray,
    output_path: str,
) -> tuple[bool, Optional[str]]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return False, f"matplotlib unavailable: {exc}"

    try:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        axes[0].set_title("Axis Tracking")
        axes[0].plot(time_axis, states[:, 0], label="R axis")
        axes[0].plot(time_axis, states[:, 1], label="Z axis")
        axes[0].axhline(target.r_axis_m, color="C0", linestyle="--", alpha=0.5)
        axes[0].axhline(target.z_axis_m, color="C1", linestyle="--", alpha=0.5)
        axes[0].grid(True)
        axes[0].legend()

        axes[1].set_title("X-Point Tracking")
        axes[1].plot(time_axis, states[:, 2], label="X-point R")
        axes[1].plot(time_axis, states[:, 3], label="X-point Z")
        axes[1].axhline(target.x_point_r_m, color="C0", linestyle="--", alpha=0.5)
        axes[1].axhline(target.x_point_z_m, color="C1", linestyle="--", alpha=0.5)
        axes[1].grid(True)
        axes[1].legend()

        axes[2].set_title("Coil Commands")
        for idx in range(actions.shape[1]):
            axes[2].plot(time_axis, actions[:, idx], label=f"coil_{idx}")
        axes[2].grid(True)
        axes[2].legend()

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
        return True, None
    except Exception as exc:
        return False, str(exc)


def run_free_boundary_supervisory_simulation(
    config_file: Optional[str] = None,
    *,
    shot_length: int = 72,
    target: Optional[FreeBoundaryTarget] = None,
    control_dt_s: float = 0.05,
    disturbance_start_step: int = 18,
    disturbance_stop_step: Optional[int] = None,
    disturbance_per_step_ma: float = 0.08,
    disturbance_recovery_step: Optional[int] = None,
    disturbance_recovery_per_step_ma: float = 0.0,
    diagnostic_dropout_step: Optional[int] = None,
    diagnostic_dropout_duration_steps: int = 0,
    diagnostic_dropout_mask: Optional[np.ndarray | tuple[float, ...] | list[float]] = None,
    coil_kick_step: int = 24,
    coil_kick_vector: Optional[np.ndarray | tuple[float, ...] | list[float]] = None,
    sensor_bias_step: int = 30,
    sensor_bias_vector: Optional[np.ndarray | tuple[float, ...] | list[float]] = None,
    sensor_bias_clear_step: Optional[int] = None,
    actuator_bias_step: Optional[int] = None,
    actuator_bias_vector: Optional[np.ndarray | tuple[float, ...] | list[float]] = None,
    actuator_bias_clear_step: Optional[int] = None,
    actuator_dropout_step: Optional[int] = None,
    actuator_dropout_duration_steps: int = 0,
    actuator_dropout_mask: Optional[np.ndarray | tuple[float, ...] | list[float]] = None,
    measurement_noise_std: float = 0.0015,
    current_target_bounds: tuple[float, float] = (7.0, 10.0),
    coil_current_limits: tuple[float, float] = (-1.5, 1.5),
    coil_delta_limit: float = 0.35,
    supervisor_total_action_l1_limit: float = 4.0,
    supervisor_axis_r_bounds_m: tuple[float, float] = (5.88, 6.12),
    supervisor_axis_z_bounds_m: tuple[float, float] = (-0.12, 0.12),
    supervisor_xpoint_r_bounds_m: tuple[float, float] = (4.92, 5.10),
    supervisor_xpoint_z_bounds_m: tuple[float, float] = (-3.60, -3.36),
    supervisor_q95_floor: float = 3.5,
    supervisor_beta_n_ceiling: float = 2.6,
    supervisor_disruption_risk_ceiling: float = 0.90,
    supervisor_warning_risk_score_threshold: float = 0.88,
    supervisor_guarded_risk_score_threshold: float = 1.05,
    supervisor_alert_recovery_hold_steps: int = 6,
    supervisor_fallback_hold_steps: int = 4,
    supervisor_fallback_action_scale: float = 0.45,
    estimator_measurement_gain: float = 0.55,
    estimator_bias_gain: float = 0.08,
    estimator_bias_decay: float = 0.985,
    estimator_actuator_bias_gain: float = 0.06,
    estimator_actuator_bias_decay: float = 0.992,
    estimator_max_actuator_bias: float = 0.35,
    return_trace: bool = False,
    save_plot: bool = False,
    output_path: str = "Free_Boundary_Supervisory_Control.png",
    verbose: bool = False,
    rng_seed: int = 42,
    kernel_factory: Callable[[str], Any] = FusionKernel,
) -> dict[str, Any]:
    steps = _require_positive_int("shot_length", shot_length)
    dt_s = _require_positive_finite("control_dt_s", control_dt_s)
    disturbance_start = _require_nonnegative_int("disturbance_start_step", disturbance_start_step)
    disturbance_stop = (
        _require_nonnegative_int("disturbance_stop_step", disturbance_stop_step)
        if disturbance_stop_step is not None
        else steps + 1
    )
    disturbance_recovery_at = (
        _require_nonnegative_int("disturbance_recovery_step", disturbance_recovery_step)
        if disturbance_recovery_step is not None
        else steps + 1
    )
    diagnostic_dropout_at = (
        _require_nonnegative_int("diagnostic_dropout_step", diagnostic_dropout_step)
        if diagnostic_dropout_step is not None
        else steps + 1
    )
    diagnostic_dropout_duration = _require_nonnegative_int(
        "diagnostic_dropout_duration_steps",
        diagnostic_dropout_duration_steps,
    )
    coil_kick_at = _require_nonnegative_int("coil_kick_step", coil_kick_step)
    bias_step = _require_nonnegative_int("sensor_bias_step", sensor_bias_step)
    sensor_bias_clear_at = (
        _require_nonnegative_int("sensor_bias_clear_step", sensor_bias_clear_step)
        if sensor_bias_clear_step is not None
        else steps + 1
    )
    actuator_bias_at = (
        _require_nonnegative_int("actuator_bias_step", actuator_bias_step)
        if actuator_bias_step is not None
        else steps + 1
    )
    actuator_bias_clear_at = (
        _require_nonnegative_int("actuator_bias_clear_step", actuator_bias_clear_step)
        if actuator_bias_clear_step is not None
        else steps + 1
    )
    actuator_dropout_at = (
        _require_nonnegative_int("actuator_dropout_step", actuator_dropout_step)
        if actuator_dropout_step is not None
        else steps + 1
    )
    actuator_dropout_duration = _require_nonnegative_int(
        "actuator_dropout_duration_steps",
        actuator_dropout_duration_steps,
    )
    measurement_noise_std = float(measurement_noise_std)
    if not np.isfinite(measurement_noise_std) or measurement_noise_std < 0.0:
        raise ValueError("measurement_noise_std must be finite and >= 0.")
    disturbance_recovery_rate = float(disturbance_recovery_per_step_ma)
    if not np.isfinite(disturbance_recovery_rate):
        raise ValueError("disturbance_recovery_per_step_ma must be finite.")
    lo_ip, hi_ip = _normalize_bounds(current_target_bounds, "current_target_bounds")
    _normalize_bounds(coil_current_limits, "coil_current_limits")
    _require_positive_finite("coil_delta_limit", coil_delta_limit)

    if config_file is None:
        repo_root = Path(__file__).resolve().parents[3]
        config_file = str(repo_root / "iter_config.json")

    rng = np.random.default_rng(int(rng_seed))
    kernel = kernel_factory(str(config_file))
    kernel.solve_equilibrium()

    target_obj = (
        target if target is not None else FreeBoundaryTarget(*DEFAULT_TARGET_VECTOR.tolist())
    )
    surrogate = NeuralSurrogate(
        n_coils=len(kernel.cfg["coils"]),
        n_state=4,
        verbose=verbose,
    )
    surrogate.train_on_kernel(kernel, perturbation=0.25)

    estimator = FreeBoundaryStateEstimator(
        surrogate,
        measurement_gain=estimator_measurement_gain,
        bias_gain=estimator_bias_gain,
        bias_decay=estimator_bias_decay,
        actuator_bias_gain=estimator_actuator_bias_gain,
        actuator_bias_decay=estimator_actuator_bias_decay,
        max_actuator_bias=estimator_max_actuator_bias,
    )
    controller = FreeBoundarySupervisoryController(surrogate, target_obj)
    supervisor = FreeBoundarySafetySupervisor(
        coil_current_limits=coil_current_limits,
        coil_delta_limit=coil_delta_limit,
        total_action_l1_limit=supervisor_total_action_l1_limit,
        ip_bounds_ma=current_target_bounds,
        axis_r_bounds_m=supervisor_axis_r_bounds_m,
        axis_z_bounds_m=supervisor_axis_z_bounds_m,
        xpoint_r_bounds_m=supervisor_xpoint_r_bounds_m,
        xpoint_z_bounds_m=supervisor_xpoint_z_bounds_m,
        q95_floor=supervisor_q95_floor,
        beta_n_ceiling=supervisor_beta_n_ceiling,
        disruption_risk_ceiling=supervisor_disruption_risk_ceiling,
        warning_risk_score_threshold=supervisor_warning_risk_score_threshold,
        guarded_risk_score_threshold=supervisor_guarded_risk_score_threshold,
        alert_recovery_hold_steps=supervisor_alert_recovery_hold_steps,
        fallback_hold_steps=supervisor_fallback_hold_steps,
        fallback_action_scale=supervisor_fallback_action_scale,
    )
    supervisor.reset()

    initial_state = extract_free_boundary_state(kernel)
    estimator.reset(initial_state)

    n_coils = len(kernel.cfg["coils"])
    kick = _normalize_vector(
        coil_kick_vector,
        length=n_coils,
        default=0.0,
        name="coil_kick_vector",
    )
    diagnostic_dropout_mask_arr = _normalize_mask(
        diagnostic_dropout_mask,
        length=4,
        name="diagnostic_dropout_mask",
    )
    sensor_bias = np.zeros(4, dtype=np.float64)
    bias_step_vector = _normalize_vector(
        sensor_bias_vector,
        length=4,
        default=0.0,
        name="sensor_bias_vector",
    )
    actuator_bias = np.zeros(n_coils, dtype=np.float64)
    actuator_bias_step_vector = _normalize_vector(
        actuator_bias_vector,
        length=n_coils,
        default=0.0,
        name="actuator_bias_vector",
    )
    actuator_dropout_mask_arr = _normalize_mask(
        actuator_dropout_mask,
        length=n_coils,
        name="actuator_dropout_mask",
    )

    physics_cfg = kernel.cfg.setdefault("physics", {})
    target_ip_ma = float(np.clip(physics_cfg.get("plasma_current_target", lo_ip), lo_ip, hi_ip))
    physics_cfg["plasma_current_target"] = target_ip_ma

    target_vec = target_obj.as_vector()
    last_action = np.zeros(n_coils, dtype=np.float64)
    last_applied_action = np.zeros(n_coils, dtype=np.float64)

    times: list[float] = []
    measured_states: list[FloatArray] = []
    estimated_states: list[FloatArray] = []
    true_states: list[FloatArray] = []
    bias_norm_hist: list[float] = []
    uncertainty_hist: list[float] = []
    axis_error_hist: list[float] = []
    xpoint_error_hist: list[float] = []
    action_hist: list[FloatArray] = []
    applied_action_hist: list[FloatArray] = []
    actuator_bias_hat_hist: list[FloatArray] = []
    actuator_bias_true_hist: list[FloatArray] = []
    intervention_hist: list[int] = []
    saturation_hist: list[int] = []
    failsafe_hist: list[int] = []
    degraded_hist: list[int] = []
    diagnostic_dropout_hist: list[int] = []
    actuator_dropout_hist: list[int] = []
    fallback_hist: list[int] = []
    invariant_hist: list[int] = []
    physics_guard_hist: list[int] = []
    q95_guard_hist: list[int] = []
    beta_guard_hist: list[int] = []
    risk_guard_hist: list[int] = []
    q95_hist: list[float] = []
    beta_n_hist: list[float] = []
    disruption_risk_hist: list[float] = []
    q95_margin_hist: list[float] = []
    beta_margin_hist: list[float] = []
    risk_margin_hist: list[float] = []
    alert_level_hist: list[int] = []
    requested_alert_level_hist: list[int] = []
    alert_transition_hist: list[int] = []
    recovery_transition_hist: list[int] = []
    risk_score_hist: list[float] = []
    risk_signal_hist: list[float] = []
    target_ip_hist: list[float] = []
    last_valid_measurement = initial_state.copy()

    t0 = time.perf_counter()
    for step in range(steps):
        if disturbance_start <= step < disturbance_stop:
            target_ip_ma = float(
                np.clip(target_ip_ma + float(disturbance_per_step_ma), lo_ip, hi_ip)
            )
        if step >= disturbance_recovery_at and disturbance_recovery_rate != 0.0:
            target_ip_ma = float(np.clip(target_ip_ma + disturbance_recovery_rate, lo_ip, hi_ip))
        diagnostic_dropout_active = bool(
            diagnostic_dropout_duration > 0
            and np.any(diagnostic_dropout_mask_arr)
            and diagnostic_dropout_at <= step < diagnostic_dropout_at + diagnostic_dropout_duration
        )
        actuator_dropout_active = bool(
            actuator_dropout_duration > 0
            and np.any(actuator_dropout_mask_arr)
            and actuator_dropout_at <= step < actuator_dropout_at + actuator_dropout_duration
        )
        if step == coil_kick_at and np.any(kick != 0.0):
            for idx, delta in enumerate(kick):
                kernel.cfg["coils"][idx]["current"] = float(
                    kernel.cfg["coils"][idx].get("current", 0.0)
                ) + float(delta)
        if step == bias_step:
            sensor_bias = sensor_bias + bias_step_vector
        if step == sensor_bias_clear_at:
            sensor_bias = np.zeros_like(sensor_bias)
        if step == actuator_bias_at:
            actuator_bias = actuator_bias + actuator_bias_step_vector
        if step == actuator_bias_clear_at:
            actuator_bias = np.zeros_like(actuator_bias)

        physics_cfg["plasma_current_target"] = target_ip_ma
        true_state = extract_free_boundary_state(kernel)
        measurement = true_state + sensor_bias
        if measurement_noise_std > 0.0:
            measurement = measurement + rng.normal(0.0, measurement_noise_std, size=4)
        if diagnostic_dropout_active:
            measurement = np.where(diagnostic_dropout_mask_arr, last_valid_measurement, measurement)
        else:
            last_valid_measurement = np.asarray(measurement, dtype=np.float64).copy()

        estimate = estimator.step(
            measurement,
            last_action,
            measured_actuator_action=last_applied_action,
        )
        coil_currents = np.asarray(
            [float(coil.get("current", 0.0)) for coil in kernel.cfg["coils"]],
            dtype=np.float64,
        )
        proposed_action = controller.propose_action(estimate)
        safety_margins = estimate_free_boundary_safety_margins(
            corrected_state=estimate.corrected_state,
            target_state=target_vec,
            bias_hat=estimate.bias_hat,
            coil_currents=coil_currents,
            target_ip_ma=target_ip_ma,
            q95_floor=supervisor.q95_floor,
            beta_n_ceiling=supervisor.beta_n_ceiling,
            disruption_risk_ceiling=supervisor.disruption_risk_ceiling,
            risk_signal_history=(
                np.asarray(risk_signal_hist[-31:], dtype=np.float64) if risk_signal_hist else None
            ),
        )
        risk_signal_hist.append(float(safety_margins.signal_scalar))
        predictive_action = proposed_action + estimate.actuator_bias_hat
        if actuator_dropout_active:
            predictive_action = np.where(actuator_dropout_mask_arr, 0.0, predictive_action)
        predicted_next_state = surrogate.predict(
            estimate.corrected_state,
            predictive_action,
        )
        filtered = supervisor.filter_action(
            proposed_action,
            corrected_state=estimate.corrected_state,
            target_state=target_vec,
            bias_hat=estimate.bias_hat,
            coil_currents=coil_currents,
            target_ip_ma=target_ip_ma,
            predicted_next_state=predicted_next_state,
            safety_margins=safety_margins,
            diagnostic_dropout_active=diagnostic_dropout_active,
            actuator_dropout_active=actuator_dropout_active,
        )

        effective_action = np.asarray(filtered.action, dtype=np.float64)
        if actuator_dropout_active:
            effective_action = np.where(actuator_dropout_mask_arr, 0.0, effective_action)
        applied_currents = np.clip(
            coil_currents + effective_action + actuator_bias, *coil_current_limits
        )
        applied_action = applied_currents - coil_currents
        for idx, current in enumerate(applied_currents):
            kernel.cfg["coils"][idx]["current"] = float(current)
        target_ip_ma = float(filtered.safe_target_ip_ma)
        physics_cfg["plasma_current_target"] = target_ip_ma
        kernel.solve_equilibrium()

        post_state = extract_free_boundary_state(kernel)
        axis_error = float(np.linalg.norm(post_state[:2] - target_vec[:2]))
        xpoint_error = float(np.linalg.norm(post_state[2:] - target_vec[2:]))

        times.append(float(step * dt_s))
        measured_states.append(np.asarray(measurement, dtype=np.float64))
        estimated_states.append(np.asarray(estimate.corrected_state, dtype=np.float64))
        true_states.append(post_state)
        bias_norm_hist.append(float(np.linalg.norm(estimate.bias_hat)))
        uncertainty_hist.append(float(estimate.uncertainty_norm))
        axis_error_hist.append(axis_error)
        xpoint_error_hist.append(xpoint_error)
        action_hist.append(np.asarray(filtered.action, dtype=np.float64))
        applied_action_hist.append(np.asarray(applied_action, dtype=np.float64))
        actuator_bias_hat_hist.append(np.asarray(estimate.actuator_bias_hat, dtype=np.float64))
        actuator_bias_true_hist.append(np.asarray(actuator_bias, dtype=np.float64))
        intervention_hist.append(int(filtered.intervention_active))
        saturation_hist.append(int(filtered.saturation_active))
        failsafe_hist.append(int(filtered.failsafe_active))
        degraded_hist.append(int(filtered.degraded_mode_active))
        diagnostic_dropout_hist.append(int(filtered.diagnostic_dropout_active))
        actuator_dropout_hist.append(int(filtered.actuator_dropout_active))
        fallback_hist.append(int(filtered.fallback_mode_active))
        invariant_hist.append(int(filtered.invariant_violation_active))
        physics_guard_hist.append(int(filtered.physics_guard_active))
        q95_guard_hist.append(int(filtered.q95_guard_active))
        beta_guard_hist.append(int(filtered.beta_guard_active))
        risk_guard_hist.append(int(filtered.risk_guard_active))
        q95_hist.append(float(filtered.q95))
        beta_n_hist.append(float(filtered.beta_n))
        disruption_risk_hist.append(float(filtered.disruption_risk))
        q95_margin_hist.append(float(filtered.q95_margin))
        beta_margin_hist.append(float(filtered.beta_margin))
        risk_margin_hist.append(float(filtered.risk_margin))
        alert_level_hist.append(int(filtered.alert_level))
        requested_alert_level_hist.append(int(filtered.requested_alert_level))
        alert_transition_hist.append(int(filtered.alert_transition_active))
        recovery_transition_hist.append(int(filtered.recovery_transition_active))
        risk_score_hist.append(float(filtered.risk_score))
        target_ip_hist.append(float(target_ip_ma))
        last_action = np.asarray(filtered.action, dtype=np.float64)
        last_applied_action = np.asarray(applied_action, dtype=np.float64)

        if verbose and step % 10 == 0:
            logger.info(
                "Step %d: axis_err=%.4f xpoint_err=%.4f bias_norm=%.4f interventions=%d",
                step,
                axis_error,
                xpoint_error,
                bias_norm_hist[-1],
                intervention_hist[-1],
            )

    runtime_s = float(time.perf_counter() - t0)
    times_arr = np.asarray(times, dtype=np.float64)
    true_arr = np.asarray(true_states, dtype=np.float64)
    measured_arr = np.asarray(measured_states, dtype=np.float64)
    estimated_arr = np.asarray(estimated_states, dtype=np.float64)
    action_arr = np.asarray(action_hist, dtype=np.float64)
    applied_action_arr = np.asarray(applied_action_hist, dtype=np.float64)
    axis_err_arr = np.asarray(axis_error_hist, dtype=np.float64)
    xpoint_err_arr = np.asarray(xpoint_error_hist, dtype=np.float64)
    bias_arr = np.asarray(bias_norm_hist, dtype=np.float64)
    uncertainty_arr = np.asarray(uncertainty_hist, dtype=np.float64)
    actuator_bias_hat_arr = np.asarray(actuator_bias_hat_hist, dtype=np.float64)
    actuator_bias_true_arr = np.asarray(actuator_bias_true_hist, dtype=np.float64)
    intervention_arr = np.asarray(intervention_hist, dtype=np.int64)
    saturation_arr = np.asarray(saturation_hist, dtype=np.int64)
    failsafe_arr = np.asarray(failsafe_hist, dtype=np.int64)
    degraded_arr = np.asarray(degraded_hist, dtype=np.int64)
    diagnostic_dropout_arr = np.asarray(diagnostic_dropout_hist, dtype=np.int64)
    actuator_dropout_arr = np.asarray(actuator_dropout_hist, dtype=np.int64)
    fallback_arr = np.asarray(fallback_hist, dtype=np.int64)
    invariant_arr = np.asarray(invariant_hist, dtype=np.int64)
    physics_guard_arr = np.asarray(physics_guard_hist, dtype=np.int64)
    q95_guard_arr = np.asarray(q95_guard_hist, dtype=np.int64)
    beta_guard_arr = np.asarray(beta_guard_hist, dtype=np.int64)
    risk_guard_arr = np.asarray(risk_guard_hist, dtype=np.int64)
    q95_arr = np.asarray(q95_hist, dtype=np.float64)
    beta_n_arr = np.asarray(beta_n_hist, dtype=np.float64)
    disruption_risk_arr = np.asarray(disruption_risk_hist, dtype=np.float64)
    q95_margin_arr = np.asarray(q95_margin_hist, dtype=np.float64)
    beta_margin_arr = np.asarray(beta_margin_hist, dtype=np.float64)
    risk_margin_arr = np.asarray(risk_margin_hist, dtype=np.float64)
    alert_level_arr = np.asarray(alert_level_hist, dtype=np.int64)
    requested_alert_level_arr = np.asarray(requested_alert_level_hist, dtype=np.int64)
    alert_transition_arr = np.asarray(alert_transition_hist, dtype=np.int64)
    recovery_transition_arr = np.asarray(recovery_transition_hist, dtype=np.int64)
    risk_arr = np.asarray(risk_score_hist, dtype=np.float64)
    target_ip_arr = np.asarray(target_ip_hist, dtype=np.float64)
    warmup = max(steps // 3, 1)
    stabilized = (
        (axis_err_arr[warmup:] <= 0.06) & (xpoint_err_arr[warmup:] <= 0.08)
        if axis_err_arr.size > warmup
        else np.zeros(0, dtype=bool)
    )

    plot_saved = False
    plot_error: Optional[str] = None
    if save_plot:
        plot_saved, plot_error = _plot_free_boundary_control(
            time_axis=times_arr,
            states=true_arr,
            target=target_obj,
            actions=action_arr,
            output_path=output_path,
        )

    trace_hash = hashlib.sha256()
    for arr in (
        true_arr,
        measured_arr,
        estimated_arr,
        action_arr,
        applied_action_arr,
        axis_err_arr,
        xpoint_err_arr,
        bias_arr,
        uncertainty_arr,
        actuator_bias_hat_arr,
        actuator_bias_true_arr,
        intervention_arr,
        saturation_arr,
        failsafe_arr,
        degraded_arr,
        diagnostic_dropout_arr,
        actuator_dropout_arr,
        fallback_arr,
        invariant_arr,
        physics_guard_arr,
        q95_guard_arr,
        beta_guard_arr,
        risk_guard_arr,
        q95_arr,
        beta_n_arr,
        disruption_risk_arr,
        q95_margin_arr,
        beta_margin_arr,
        risk_margin_arr,
        alert_level_arr,
        requested_alert_level_arr,
        alert_transition_arr,
        recovery_transition_arr,
        risk_arr,
        target_ip_arr,
    ):
        trace_hash.update(np.asarray(arr).tobytes())

    summary: dict[str, Any] = {
        "config_path": str(config_file),
        "steps": int(steps),
        "runtime_seconds": runtime_s,
        "final_target_ip_ma": float(target_ip_ma),
        "final_r_axis": float(true_arr[-1, 0]) if true_arr.size else 0.0,
        "final_z_axis": float(true_arr[-1, 1]) if true_arr.size else 0.0,
        "final_xpoint_r": float(true_arr[-1, 2]) if true_arr.size else 0.0,
        "final_xpoint_z": float(true_arr[-1, 3]) if true_arr.size else 0.0,
        "mean_axis_error_m": float(np.mean(axis_err_arr)) if axis_err_arr.size else 0.0,
        "p95_axis_error_m": float(np.percentile(axis_err_arr, 95)) if axis_err_arr.size else 0.0,
        "mean_xpoint_error_m": float(np.mean(xpoint_err_arr)) if xpoint_err_arr.size else 0.0,
        "p95_xpoint_error_m": (
            float(np.percentile(xpoint_err_arr, 95)) if xpoint_err_arr.size else 0.0
        ),
        "stabilization_rate": float(np.mean(stabilized)) if stabilized.size else 0.0,
        "mean_measurement_error_m": (
            float(np.sqrt(np.mean((measured_arr - true_arr) ** 2))) if measured_arr.size else 0.0
        ),
        "mean_estimation_error_m": (
            float(np.sqrt(np.mean((estimated_arr - true_arr) ** 2))) if estimated_arr.size else 0.0
        ),
        "max_bias_norm_m": float(np.max(bias_arr)) if bias_arr.size else 0.0,
        "final_bias_norm_m": float(bias_arr[-1]) if bias_arr.size else 0.0,
        "max_uncertainty_norm": float(np.max(uncertainty_arr)) if uncertainty_arr.size else 0.0,
        "final_uncertainty_norm": float(uncertainty_arr[-1]) if uncertainty_arr.size else 0.0,
        "mean_actuator_bias_estimation_error": (
            float(np.sqrt(np.mean((actuator_bias_hat_arr - actuator_bias_true_arr) ** 2)))
            if actuator_bias_hat_arr.size
            else 0.0
        ),
        "final_actuator_bias_estimation_error": (
            float(np.linalg.norm(actuator_bias_hat_arr[-1] - actuator_bias_true_arr[-1]))
            if actuator_bias_hat_arr.size
            else 0.0
        ),
        "supervisor_intervention_count": int(np.sum(intervention_arr)),
        "saturation_event_count": int(np.sum(saturation_arr)),
        "failsafe_trip_count": int(np.sum(failsafe_arr)),
        "degraded_mode_count": int(np.sum(degraded_arr)),
        "diagnostic_dropout_count": int(np.sum(diagnostic_dropout_arr)),
        "actuator_dropout_count": int(np.sum(actuator_dropout_arr)),
        "fallback_mode_count": int(np.sum(fallback_arr)),
        "invariant_violation_count": int(np.sum(invariant_arr)),
        "physics_guard_count": int(np.sum(physics_guard_arr)),
        "q95_guard_count": int(np.sum(q95_guard_arr)),
        "beta_guard_count": int(np.sum(beta_guard_arr)),
        "risk_guard_count": int(np.sum(risk_guard_arr)),
        "min_q95": float(np.min(q95_arr)) if q95_arr.size else 0.0,
        "max_beta_n": float(np.max(beta_n_arr)) if beta_n_arr.size else 0.0,
        "max_disruption_risk": (
            float(np.max(disruption_risk_arr)) if disruption_risk_arr.size else 0.0
        ),
        "min_q95_margin": float(np.min(q95_margin_arr)) if q95_margin_arr.size else 0.0,
        "min_beta_margin": float(np.min(beta_margin_arr)) if beta_margin_arr.size else 0.0,
        "min_risk_margin": float(np.min(risk_margin_arr)) if risk_margin_arr.size else 0.0,
        "warning_mode_count": int(np.sum(alert_level_arr == 1)),
        "guarded_mode_count": int(np.sum(alert_level_arr == 2)),
        "peak_alert_level": int(np.max(alert_level_arr)) if alert_level_arr.size else 0,
        "final_alert_level": int(alert_level_arr[-1]) if alert_level_arr.size else 0,
        "alert_transition_count": int(np.sum(alert_transition_arr)),
        "recovery_transition_count": int(np.sum(recovery_transition_arr)),
        "max_risk_score": float(np.max(risk_arr)) if risk_arr.size else 0.0,
        "final_risk_score": float(risk_arr[-1]) if risk_arr.size else 0.0,
        "min_target_ip_ma": float(np.min(target_ip_arr)) if target_ip_arr.size else 0.0,
        "max_target_ip_ma": float(np.max(target_ip_arr)) if target_ip_arr.size else 0.0,
        "max_abs_action": float(np.max(np.abs(action_arr))) if action_arr.size else 0.0,
        "max_action_l1": (
            float(np.max(np.sum(np.abs(action_arr), axis=1))) if action_arr.size else 0.0
        ),
        "max_abs_coil_current": float(
            np.max(
                np.abs(
                    np.asarray(
                        [float(coil.get("current", 0.0)) for coil in kernel.cfg["coils"]],
                        dtype=np.float64,
                    )
                )
            )
        ),
        "replay_signature": trace_hash.hexdigest()[:16],
        "plot_saved": bool(plot_saved),
        "plot_error": plot_error,
    }
    if return_trace:
        summary["trace"] = {
            "time_s": times_arr.tolist(),
            "true_states": true_arr.tolist(),
            "measured_states": measured_arr.tolist(),
            "estimated_states": estimated_arr.tolist(),
            "actions": action_arr.tolist(),
            "applied_actions": applied_action_arr.tolist(),
            "axis_error_m": axis_err_arr.tolist(),
            "xpoint_error_m": xpoint_err_arr.tolist(),
            "bias_norm_m": bias_arr.tolist(),
            "uncertainty_norm": uncertainty_arr.tolist(),
            "actuator_bias_hat": actuator_bias_hat_arr.tolist(),
            "actuator_bias_true": actuator_bias_true_arr.tolist(),
            "supervisor_interventions": intervention_arr.astype(int).tolist(),
            "saturation_events": saturation_arr.astype(int).tolist(),
            "failsafe_trips": failsafe_arr.astype(int).tolist(),
            "degraded_mode": degraded_arr.astype(int).tolist(),
            "diagnostic_dropout": diagnostic_dropout_arr.astype(int).tolist(),
            "actuator_dropout": actuator_dropout_arr.astype(int).tolist(),
            "fallback_mode": fallback_arr.astype(int).tolist(),
            "invariant_violations": invariant_arr.astype(int).tolist(),
            "physics_guard": physics_guard_arr.astype(int).tolist(),
            "q95_guard": q95_guard_arr.astype(int).tolist(),
            "beta_guard": beta_guard_arr.astype(int).tolist(),
            "risk_guard": risk_guard_arr.astype(int).tolist(),
            "q95": q95_arr.tolist(),
            "beta_n": beta_n_arr.tolist(),
            "disruption_risk": disruption_risk_arr.tolist(),
            "q95_margin": q95_margin_arr.tolist(),
            "beta_margin": beta_margin_arr.tolist(),
            "risk_margin": risk_margin_arr.tolist(),
            "alert_level": alert_level_arr.astype(int).tolist(),
            "requested_alert_level": requested_alert_level_arr.astype(int).tolist(),
            "alert_transitions": alert_transition_arr.astype(int).tolist(),
            "recovery_transitions": recovery_transition_arr.astype(int).tolist(),
            "risk_score": risk_arr.tolist(),
            "target_ip_ma": target_ip_arr.tolist(),
        }
    return summary


if __name__ == "__main__":
    run_free_boundary_supervisory_simulation(verbose=True)
