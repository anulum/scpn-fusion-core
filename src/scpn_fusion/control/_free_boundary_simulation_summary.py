# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core - Free-Boundary Simulation Summary
"""Summary and trace-packet construction for free-boundary supervisory simulations."""

from __future__ import annotations

import hashlib
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.control._free_boundary_supervisory_types import BoolArray, FloatArray

IntArray = NDArray[np.int64]


def build_free_boundary_simulation_summary(
    *,
    config_file: str,
    steps: int,
    runtime_s: float,
    target_ip_ma: float,
    kernel: Any,
    true_arr: FloatArray,
    measured_arr: FloatArray,
    estimated_arr: FloatArray,
    action_arr: FloatArray,
    applied_action_arr: FloatArray,
    axis_err_arr: FloatArray,
    xpoint_err_arr: FloatArray,
    bias_arr: FloatArray,
    uncertainty_arr: FloatArray,
    actuator_bias_hat_arr: FloatArray,
    actuator_bias_true_arr: FloatArray,
    intervention_arr: IntArray,
    saturation_arr: IntArray,
    failsafe_arr: IntArray,
    degraded_arr: IntArray,
    diagnostic_dropout_arr: IntArray,
    actuator_dropout_arr: IntArray,
    fallback_arr: IntArray,
    invariant_arr: IntArray,
    physics_guard_arr: IntArray,
    q95_guard_arr: IntArray,
    beta_guard_arr: IntArray,
    risk_guard_arr: IntArray,
    q95_arr: FloatArray,
    beta_n_arr: FloatArray,
    disruption_risk_arr: FloatArray,
    q95_margin_arr: FloatArray,
    beta_margin_arr: FloatArray,
    risk_margin_arr: FloatArray,
    alert_level_arr: IntArray,
    requested_alert_level_arr: IntArray,
    alert_transition_arr: IntArray,
    recovery_transition_arr: IntArray,
    risk_arr: FloatArray,
    target_ip_arr: FloatArray,
    stabilized: BoolArray,
    times_arr: FloatArray,
    plot_saved: bool,
    plot_error: Optional[str],
    return_trace: bool,
) -> dict[str, Any]:
    """Build scalar metrics, replay signature, and optional trace data for one shot."""

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
