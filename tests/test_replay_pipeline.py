# ----------------------------------------------------------------------
# SCPN Fusion Core -- Replay Pipeline Contract Tests
# ----------------------------------------------------------------------
"""Tests for scpn_fusion.control.replay_pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.control.replay_pipeline import (
    apply_actuator_lag,
    load_replay_pipeline_config,
    preprocess_sensor_trace,
)


def test_load_replay_pipeline_config_defaults_are_valid() -> None:
    cfg = load_replay_pipeline_config()
    assert cfg["sensor_preprocess_enabled"] is True
    assert cfg["actuator_lag_enabled"] is True
    assert 0.0 <= float(cfg["sensor_alpha"]) <= 1.0


def test_load_replay_pipeline_config_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="sensor_alpha"):
        load_replay_pipeline_config({"sensor_alpha": 1.5})


def test_preprocess_sensor_trace_disabled_is_identity() -> None:
    signal = np.asarray([0.2, 0.3, 0.8, 0.5], dtype=np.float64)
    out, delta = preprocess_sensor_trace(
        signal,
        config={"sensor_preprocess_enabled": False},
    )
    np.testing.assert_allclose(out, signal, rtol=0.0, atol=0.0)
    assert delta == 0.0


def test_preprocess_sensor_trace_enabled_returns_finite_values() -> None:
    signal = np.asarray([0.2, 0.5, 1.1, 0.4, 0.3, 0.7], dtype=np.float64)
    out, delta = preprocess_sensor_trace(signal)
    assert out.shape == signal.shape
    assert np.all(np.isfinite(out))
    assert delta >= 0.0


def test_apply_actuator_lag_disabled_is_identity() -> None:
    cmd = np.asarray([0.0, 0.2, 0.8, 0.4, 0.9], dtype=np.float64)
    out, lag = apply_actuator_lag(
        cmd,
        dt_s=1e-3,
        config={"actuator_lag_enabled": False},
    )
    np.testing.assert_allclose(out, cmd, rtol=0.0, atol=0.0)
    assert lag == 0.0


def test_apply_actuator_lag_enabled_smooths_command() -> None:
    cmd = np.asarray([0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    out, lag = apply_actuator_lag(
        cmd,
        dt_s=1e-3,
        config={
            "actuator_lag_enabled": True,
            "actuator_tau_s": 0.02,
            "actuator_slew_per_s": 80.0,
        },
    )
    assert out[1] < cmd[1]
    assert out[-1] <= cmd[-1]
    assert lag > 0.0


def test_apply_actuator_lag_rejects_nonpositive_dt() -> None:
    cmd = np.asarray([0.0, 0.5], dtype=np.float64)
    with pytest.raises(ValueError, match="dt_s"):
        apply_actuator_lag(cmd, dt_s=0.0)

