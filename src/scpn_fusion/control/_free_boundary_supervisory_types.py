# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core - Free-Boundary Supervisory Control Types
"""Shared typed records and safety-margin estimators for free-boundary supervision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.control.disruption_risk_runtime import predict_disruption_risk


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

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
    values: Optional[FloatArray | tuple[float, ...] | list[float]],
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
    values: Optional[FloatArray | tuple[float, ...] | list[float]],
    *,
    length: int,
    name: str,
) -> BoolArray:
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
    """Reference magnetic-axis and X-point coordinates for supervisory tracking."""

    r_axis_m: float = 6.0
    z_axis_m: float = 0.0
    x_point_r_m: float = 5.0
    x_point_z_m: float = -3.5

    def as_vector(self) -> FloatArray:
        """Return the target coordinates in controller state-vector order."""

        return np.array(
            [self.r_axis_m, self.z_axis_m, self.x_point_r_m, self.x_point_z_m],
            dtype=np.float64,
        )


@dataclass
class FreeBoundaryEstimate:
    """Bias-corrected observer state used by the supervisory controller."""

    state_hat: FloatArray
    bias_hat: FloatArray
    actuator_bias_hat: FloatArray
    innovation: FloatArray
    corrected_state: FloatArray
    uncertainty_norm: float


@dataclass
class FreeBoundarySafetyMargins:
    """Physics guard margins derived from q95, beta_N, and disruption-risk proxies."""

    signal_scalar: float
    q95: float
    beta_n: float
    disruption_risk: float
    q95_margin: float
    beta_margin: float
    risk_margin: float


@dataclass
class SafetyFilterResult:
    """Full result packet emitted by the supervisory action safety filter."""

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
    corrected_state: FloatArray,
    target_state: FloatArray,
    bias_hat: FloatArray,
    coil_currents: FloatArray,
    target_ip_ma: float,
    q95_floor: float,
    beta_n_ceiling: float,
    disruption_risk_ceiling: float,
    risk_signal_history: Optional[FloatArray | list[float]] = None,
) -> FreeBoundarySafetyMargins:
    """Estimate safety guard margins from geometry, coil currents, and target current."""

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
