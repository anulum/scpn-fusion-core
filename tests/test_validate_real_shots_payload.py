# ----------------------------------------------------------------------
# SCPN Fusion Core -- Real-Shot Payload Validation Tests
# ----------------------------------------------------------------------
"""Tests for disruption payload contracts in validation/validate_real_shots.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "validate_real_shots.py"
SPEC = importlib.util.spec_from_file_location("validate_real_shots", MODULE_PATH)
assert SPEC and SPEC.loader
validate_real_shots = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validate_real_shots
SPEC.loader.exec_module(validate_real_shots)


def _write_npz(path: Path, **payload: Any) -> Path:
    np.savez(path, **payload)
    return path


def test_load_payload_accepts_dbdt_signal_with_defaults(tmp_path: Path) -> None:
    shot_path = _write_npz(
        tmp_path / "shot_valid_dbdt.npz",
        dBdt_gauss_per_s=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
        time_s=np.array([0.0, 0.001, 0.002, 0.003], dtype=np.float64),
    )

    payload = validate_real_shots.load_disruption_shot_payload(shot_path)

    assert payload["is_disruption"] is False
    assert payload["disruption_time_idx"] == -1
    assert payload["n2_amp"] is None
    np.testing.assert_allclose(payload["signal"], payload["n1_amp"])
    assert payload["time_s"] is not None


def test_load_payload_accepts_n1_n2_and_disruption_index(tmp_path: Path) -> None:
    n1 = np.array([0.20, 0.25, 0.35, 0.60, 0.80, 1.00], dtype=np.float64)
    n2 = np.array([0.05, 0.06, 0.08, 0.12, 0.15, 0.20], dtype=np.float64)
    shot_path = _write_npz(
        tmp_path / "shot_valid_toroidal.npz",
        n1_amp=n1,
        n2_amp=n2,
        is_disruption=np.array(True),
        disruption_time_idx=np.array(4),
        time_s=np.array([0.0, 0.001, 0.002, 0.003, 0.004, 0.005], dtype=np.float64),
    )

    payload = validate_real_shots.load_disruption_shot_payload(shot_path)

    assert payload["is_disruption"] is True
    assert payload["disruption_time_idx"] == 4
    np.testing.assert_allclose(payload["signal"], n1)
    np.testing.assert_allclose(payload["n1_amp"], n1)
    np.testing.assert_allclose(payload["n2_amp"], n2)


def test_load_payload_rejects_missing_signal_keys(tmp_path: Path) -> None:
    shot_path = _write_npz(
        tmp_path / "shot_missing_signal.npz",
        n2_amp=np.array([0.1, 0.2, 0.3], dtype=np.float64),
    )
    with pytest.raises(ValueError, match="missing signal key"):
        validate_real_shots.load_disruption_shot_payload(shot_path)


def test_load_payload_rejects_non_finite_signal(tmp_path: Path) -> None:
    shot_path = _write_npz(
        tmp_path / "shot_non_finite.npz",
        dBdt_gauss_per_s=np.array([0.1, np.nan, 0.3], dtype=np.float64),
    )
    with pytest.raises(ValueError, match="signal contains non-finite values"):
        validate_real_shots.load_disruption_shot_payload(shot_path)


def test_load_payload_rejects_length_mismatch_for_n1(tmp_path: Path) -> None:
    shot_path = _write_npz(
        tmp_path / "shot_bad_n1_len.npz",
        dBdt_gauss_per_s=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
        n1_amp=np.array([0.1, 0.2], dtype=np.float64),
    )
    with pytest.raises(ValueError, match="n1_amp length"):
        validate_real_shots.load_disruption_shot_payload(shot_path)


def test_load_payload_rejects_invalid_disruption_index(tmp_path: Path) -> None:
    shot_path = _write_npz(
        tmp_path / "shot_bad_disruption_idx.npz",
        n1_amp=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
        is_disruption=np.array(True),
        disruption_time_idx=np.array(0),
    )
    with pytest.raises(ValueError, match="must satisfy 0 < idx < signal length"):
        validate_real_shots.load_disruption_shot_payload(shot_path)


def test_load_payload_rejects_non_monotonic_timebase(tmp_path: Path) -> None:
    shot_path = _write_npz(
        tmp_path / "shot_bad_timebase.npz",
        n1_amp=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
        time_s=np.array([0.0, 0.001, 0.001, 0.002], dtype=np.float64),
    )
    with pytest.raises(ValueError, match="time_s must be strictly increasing"):
        validate_real_shots.load_disruption_shot_payload(shot_path)


def test_load_disruption_risk_calibration_defaults_when_missing(tmp_path: Path) -> None:
    calibration = validate_real_shots.load_disruption_risk_calibration(
        tmp_path / "missing_calibration.json"
    )
    assert calibration["source"] == "default-v2.1"
    assert calibration["risk_threshold"] == 0.50
    assert calibration["bias_delta"] == 0.0


def test_load_disruption_risk_calibration_reads_selected_values(tmp_path: Path) -> None:
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "version": "diiid-disruption-risk-calibration-v1",
                "selection": {
                    "risk_threshold": 0.58,
                    "bias_delta": -0.12,
                },
                "gates": {"overall_pass": True},
            }
        ),
        encoding="utf-8",
    )
    calibration = validate_real_shots.load_disruption_risk_calibration(calibration_path)
    assert calibration["source"] == "diiid-disruption-risk-calibration-v1"
    assert calibration["risk_threshold"] == 0.58
    assert calibration["bias_delta"] == -0.12
    assert calibration["gates_overall_pass"] is True


def test_load_disruption_risk_calibration_rejects_invalid_threshold(tmp_path: Path) -> None:
    calibration_path = tmp_path / "bad_calibration.json"
    calibration_path.write_text(
        json.dumps({"selection": {"risk_threshold": 1.5, "bias_delta": 0.0}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="risk_threshold"):
        validate_real_shots.load_disruption_risk_calibration(calibration_path)


def test_validate_disruption_reports_pipeline_contract_metadata(tmp_path: Path) -> None:
    n = 160
    t = np.linspace(0.0, 0.159, n, dtype=np.float64)
    base = 0.35 + 0.03 * np.sin(2.0 * np.pi * 4.0 * t)
    spike = 0.35 * np.exp(-(((t - 0.145) / 0.010) ** 2))
    shot_dir = tmp_path / "shots"
    shot_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        shot_dir / "shot_disrupt.npz",
        time_s=t,
        dBdt_gauss_per_s=base + spike,
        n1_amp=0.18 + 0.45 * spike,
        n2_amp=0.05 + 0.15 * spike,
        is_disruption=np.array(True),
        disruption_time_idx=np.array(145),
    )
    np.savez(
        shot_dir / "shot_safe.npz",
        time_s=t,
        dBdt_gauss_per_s=base,
        n1_amp=np.full(n, 0.12, dtype=np.float64),
        n2_amp=np.full(n, 0.04, dtype=np.float64),
        is_disruption=np.array(False),
        disruption_time_idx=np.array(-1),
    )

    disabled = validate_real_shots.validate_disruption(
        shot_dir,
        replay_pipeline={
            "sensor_preprocess_enabled": False,
            "actuator_lag_enabled": False,
        },
    )
    assert disabled["pipeline"]["sensor_preprocess_enabled"] is False
    assert disabled["pipeline"]["actuator_lag_enabled"] is False
    assert float(disabled["pipeline"]["mean_abs_sensor_delta"]) == 0.0
    assert float(disabled["pipeline"]["mean_abs_actuator_lag"]) == 0.0

    enabled = validate_real_shots.validate_disruption(shot_dir)
    assert enabled["pipeline"]["sensor_preprocess_enabled"] is True
    assert enabled["pipeline"]["actuator_lag_enabled"] is True
    assert float(enabled["pipeline"]["mean_abs_sensor_delta"]) >= 0.0
    assert float(enabled["pipeline"]["mean_abs_actuator_lag"]) >= 0.0
