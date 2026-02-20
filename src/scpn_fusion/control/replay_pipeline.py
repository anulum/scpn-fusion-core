# ----------------------------------------------------------------------
# SCPN Fusion Core -- Replay Pipeline Contracts
# ----------------------------------------------------------------------
"""Sensor-preprocess and actuator-lag contracts for disruption replay lanes."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


DEFAULT_REPLAY_PIPELINE: dict[str, float | bool] = {
    "sensor_preprocess_enabled": True,
    "sensor_alpha": 0.12,
    "sensor_derivative_gain": 0.08,
    "sensor_clip_sigma": 4.5,
    "actuator_lag_enabled": True,
    "actuator_tau_s": 0.006,
    "actuator_slew_per_s": 260.0,
}


def _require_bool(name: str, value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"{name} must be boolean.")


def _require_finite_float(name: str, value: Any) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite.")
    return out


def _require_finite_float_ge(name: str, value: Any, minimum: float) -> float:
    out = _require_finite_float(name, value)
    if out < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return out


def load_replay_pipeline_config(
    replay_pipeline: Mapping[str, Any] | None = None,
) -> dict[str, float | bool]:
    """Load and validate replay-pipeline settings."""
    cfg = dict(DEFAULT_REPLAY_PIPELINE)
    if replay_pipeline is not None:
        cfg.update(dict(replay_pipeline))

    cfg_norm: dict[str, float | bool] = {
        "sensor_preprocess_enabled": _require_bool(
            "sensor_preprocess_enabled", cfg["sensor_preprocess_enabled"]
        ),
        "sensor_alpha": _require_finite_float_ge("sensor_alpha", cfg["sensor_alpha"], 0.0),
        "sensor_derivative_gain": _require_finite_float_ge(
            "sensor_derivative_gain", cfg["sensor_derivative_gain"], 0.0
        ),
        "sensor_clip_sigma": _require_finite_float_ge(
            "sensor_clip_sigma", cfg["sensor_clip_sigma"], 0.0
        ),
        "actuator_lag_enabled": _require_bool(
            "actuator_lag_enabled", cfg["actuator_lag_enabled"]
        ),
        "actuator_tau_s": _require_finite_float_ge("actuator_tau_s", cfg["actuator_tau_s"], 1e-9),
        "actuator_slew_per_s": _require_finite_float_ge(
            "actuator_slew_per_s", cfg["actuator_slew_per_s"], 1e-9
        ),
    }
    if float(cfg_norm["sensor_alpha"]) > 1.0:
        raise ValueError("sensor_alpha must be <= 1.0.")
    if float(cfg_norm["sensor_derivative_gain"]) > 1.0:
        raise ValueError("sensor_derivative_gain must be <= 1.0.")
    return cfg_norm


def preprocess_sensor_trace(
    signal: NDArray[np.float64],
    *,
    config: Mapping[str, Any] | None = None,
) -> tuple[NDArray[np.float64], float]:
    """Apply deterministic sensor preprocessing to a 1-D signal trace."""
    if signal.ndim != 1:
        raise ValueError("signal must be a 1-D array.")
    if signal.size < 2:
        raise ValueError("signal must contain at least 2 samples.")
    if not np.all(np.isfinite(signal)):
        raise ValueError("signal must contain only finite values.")

    cfg = load_replay_pipeline_config(config)
    if not bool(cfg["sensor_preprocess_enabled"]):
        return signal.copy(), 0.0

    alpha = float(cfg["sensor_alpha"])
    deriv_gain = float(cfg["sensor_derivative_gain"])
    clip_sigma = float(cfg["sensor_clip_sigma"])

    out = np.zeros_like(signal, dtype=np.float64)
    ema = float(signal[0])
    mean = float(signal[0])
    var = 0.0
    last_raw = float(signal[0])
    out[0] = ema
    abs_delta = 0.0

    for i in range(1, signal.size):
        raw = float(signal[i])
        mean = (1.0 - alpha) * mean + alpha * raw
        resid = raw - mean
        var = (1.0 - alpha) * var + alpha * resid * resid
        sigma = float(np.sqrt(max(var, 1e-12)))

        candidate = raw + deriv_gain * (raw - last_raw)
        ema = (1.0 - alpha) * ema + alpha * candidate
        if clip_sigma > 0.0:
            lo = mean - clip_sigma * sigma
            hi = mean + clip_sigma * sigma
            ema = float(np.clip(ema, lo, hi))
        out[i] = ema
        abs_delta += abs(ema - raw)
        last_raw = raw

    return out, float(abs_delta / max(signal.size - 1, 1))


def apply_actuator_lag(
    command_series: NDArray[np.float64],
    *,
    dt_s: float,
    config: Mapping[str, Any] | None = None,
) -> tuple[NDArray[np.float64], float]:
    """Apply first-order actuator lag + slew-rate limit to command series."""
    if command_series.ndim != 1:
        raise ValueError("command_series must be a 1-D array.")
    if command_series.size < 2:
        raise ValueError("command_series must contain at least 2 samples.")
    if not np.all(np.isfinite(command_series)):
        raise ValueError("command_series must contain only finite values.")

    dt = _require_finite_float_ge("dt_s", dt_s, 1e-9)
    cfg = load_replay_pipeline_config(config)
    if not bool(cfg["actuator_lag_enabled"]):
        return command_series.copy(), 0.0

    tau = float(cfg["actuator_tau_s"])
    slew_per_s = float(cfg["actuator_slew_per_s"])
    alpha = dt / (tau + dt)
    max_step = slew_per_s * dt

    out = np.zeros_like(command_series, dtype=np.float64)
    out[0] = float(command_series[0])
    abs_lag = 0.0
    for i in range(1, command_series.size):
        cmd = float(command_series[i])
        prev = float(out[i - 1])
        lagged = prev + alpha * (cmd - prev)
        lagged = float(np.clip(lagged, prev - max_step, prev + max_step))
        out[i] = lagged
        abs_lag += abs(cmd - lagged)

    return out, float(abs_lag / max(command_series.size - 1, 1))


__all__ = [
    "DEFAULT_REPLAY_PIPELINE",
    "apply_actuator_lag",
    "load_replay_pipeline_config",
    "preprocess_sensor_trace",
]

