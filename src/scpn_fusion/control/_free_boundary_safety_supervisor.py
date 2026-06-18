# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core - Free-Boundary Safety Supervisor
"""Safety filtering and alert-state management for free-boundary coil commands."""

from __future__ import annotations

from typing import Optional

import numpy as np

from scpn_fusion.control._free_boundary_supervisory_types import (
    FloatArray,
    FreeBoundarySafetyMargins,
    SafetyFilterResult,
    SUPERVISORY_ALERT_LEVEL_NAMES,
    _normalize_bounds,
    _require_positive_finite,
    _require_positive_int,
    estimate_free_boundary_safety_margins,
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
        """Clear fallback latches and alert-recovery state before a new shot."""
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
        proposed_action: FloatArray,
        *,
        corrected_state: FloatArray,
        target_state: FloatArray,
        bias_hat: FloatArray,
        coil_currents: FloatArray,
        target_ip_ma: float,
        predicted_next_state: Optional[FloatArray] = None,
        safety_margins: Optional[FreeBoundarySafetyMargins] = None,
        diagnostic_dropout_active: bool = False,
        actuator_dropout_active: bool = False,
    ) -> SafetyFilterResult:
        """Apply current, geometry, plasma-physics, and degraded-mode guards to an action."""
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
