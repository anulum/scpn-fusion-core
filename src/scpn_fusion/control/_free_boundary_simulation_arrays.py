# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core - Free-Boundary Simulation Arrays
"""Array normalisation for free-boundary supervisory simulation histories."""

from __future__ import annotations

from typing import Any, Mapping, TypedDict

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


class FreeBoundaryHistoryArrays(TypedDict):
    """Typed, per-channel arrays for one free-boundary supervisory shot history."""

    times_arr: FloatArray
    true_arr: FloatArray
    measured_arr: FloatArray
    estimated_arr: FloatArray
    action_arr: FloatArray
    applied_action_arr: FloatArray
    axis_err_arr: FloatArray
    xpoint_err_arr: FloatArray
    bias_arr: FloatArray
    uncertainty_arr: FloatArray
    actuator_bias_hat_arr: FloatArray
    actuator_bias_true_arr: FloatArray
    intervention_arr: IntArray
    saturation_arr: IntArray
    failsafe_arr: IntArray
    degraded_arr: IntArray
    diagnostic_dropout_arr: IntArray
    actuator_dropout_arr: IntArray
    fallback_arr: IntArray
    invariant_arr: IntArray
    physics_guard_arr: IntArray
    q95_guard_arr: IntArray
    beta_guard_arr: IntArray
    risk_guard_arr: IntArray
    q95_arr: FloatArray
    beta_n_arr: FloatArray
    disruption_risk_arr: FloatArray
    q95_margin_arr: FloatArray
    beta_margin_arr: FloatArray
    risk_margin_arr: FloatArray
    alert_level_arr: IntArray
    requested_alert_level_arr: IntArray
    alert_transition_arr: IntArray
    recovery_transition_arr: IntArray
    risk_arr: FloatArray
    target_ip_arr: FloatArray
    stabilized: BoolArray


def build_free_boundary_history_arrays(
    histories: Mapping[str, Any],
    *,
    steps: int,
) -> FreeBoundaryHistoryArrays:
    """Convert shot history lists into typed arrays and derived stabilisation flags."""
    axis_err_arr: FloatArray = np.asarray(histories["axis_error_hist"], dtype=np.float64)
    xpoint_err_arr: FloatArray = np.asarray(histories["xpoint_error_hist"], dtype=np.float64)
    warmup = max(steps // 3, 1)
    if axis_err_arr.size > warmup:
        stabilized: BoolArray = (axis_err_arr[warmup:] <= 0.06) & (xpoint_err_arr[warmup:] <= 0.08)
    else:
        stabilized = np.zeros(0, dtype=np.bool_)

    arrays: FreeBoundaryHistoryArrays = {
        "times_arr": np.asarray(histories["times"], dtype=np.float64),
        "true_arr": np.asarray(histories["true_states"], dtype=np.float64),
        "measured_arr": np.asarray(histories["measured_states"], dtype=np.float64),
        "estimated_arr": np.asarray(histories["estimated_states"], dtype=np.float64),
        "action_arr": np.asarray(histories["action_hist"], dtype=np.float64),
        "applied_action_arr": np.asarray(histories["applied_action_hist"], dtype=np.float64),
        "axis_err_arr": axis_err_arr,
        "xpoint_err_arr": xpoint_err_arr,
        "bias_arr": np.asarray(histories["bias_norm_hist"], dtype=np.float64),
        "uncertainty_arr": np.asarray(histories["uncertainty_hist"], dtype=np.float64),
        "actuator_bias_hat_arr": np.asarray(
            histories["actuator_bias_hat_hist"],
            dtype=np.float64,
        ),
        "actuator_bias_true_arr": np.asarray(
            histories["actuator_bias_true_hist"],
            dtype=np.float64,
        ),
        "intervention_arr": np.asarray(histories["intervention_hist"], dtype=np.int64),
        "saturation_arr": np.asarray(histories["saturation_hist"], dtype=np.int64),
        "failsafe_arr": np.asarray(histories["failsafe_hist"], dtype=np.int64),
        "degraded_arr": np.asarray(histories["degraded_hist"], dtype=np.int64),
        "diagnostic_dropout_arr": np.asarray(
            histories["diagnostic_dropout_hist"],
            dtype=np.int64,
        ),
        "actuator_dropout_arr": np.asarray(histories["actuator_dropout_hist"], dtype=np.int64),
        "fallback_arr": np.asarray(histories["fallback_hist"], dtype=np.int64),
        "invariant_arr": np.asarray(histories["invariant_hist"], dtype=np.int64),
        "physics_guard_arr": np.asarray(histories["physics_guard_hist"], dtype=np.int64),
        "q95_guard_arr": np.asarray(histories["q95_guard_hist"], dtype=np.int64),
        "beta_guard_arr": np.asarray(histories["beta_guard_hist"], dtype=np.int64),
        "risk_guard_arr": np.asarray(histories["risk_guard_hist"], dtype=np.int64),
        "q95_arr": np.asarray(histories["q95_hist"], dtype=np.float64),
        "beta_n_arr": np.asarray(histories["beta_n_hist"], dtype=np.float64),
        "disruption_risk_arr": np.asarray(histories["disruption_risk_hist"], dtype=np.float64),
        "q95_margin_arr": np.asarray(histories["q95_margin_hist"], dtype=np.float64),
        "beta_margin_arr": np.asarray(histories["beta_margin_hist"], dtype=np.float64),
        "risk_margin_arr": np.asarray(histories["risk_margin_hist"], dtype=np.float64),
        "alert_level_arr": np.asarray(histories["alert_level_hist"], dtype=np.int64),
        "requested_alert_level_arr": np.asarray(
            histories["requested_alert_level_hist"],
            dtype=np.int64,
        ),
        "alert_transition_arr": np.asarray(histories["alert_transition_hist"], dtype=np.int64),
        "recovery_transition_arr": np.asarray(
            histories["recovery_transition_hist"],
            dtype=np.int64,
        ),
        "risk_arr": np.asarray(histories["risk_score_hist"], dtype=np.float64),
        "target_ip_arr": np.asarray(histories["target_ip_hist"], dtype=np.float64),
        "stabilized": stabilized,
    }
    return arrays
