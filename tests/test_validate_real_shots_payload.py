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


def test_load_disruption_risk_calibration_rejects_oversized_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calibration_path = tmp_path / "oversized_calibration.json"
    calibration_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(validate_real_shots, "_MAX_CALIBRATION_JSON_BYTES", 1)
    with pytest.raises(ValueError, match="exceeds max"):
        validate_real_shots.load_disruption_risk_calibration(calibration_path)


def test_load_payload_rejects_oversized_npz_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shot_path = _write_npz(
        tmp_path / "shot_large.npz",
        n1_amp=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
    )
    monkeypatch.setattr(validate_real_shots, "_MAX_SHOT_NPZ_BYTES", 1)
    with pytest.raises(ValueError, match="artifact size"):
        validate_real_shots.load_disruption_shot_payload(shot_path)


def test_load_payload_rejects_oversized_signal_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shot_path = _write_npz(
        tmp_path / "shot_signal_cap.npz",
        n1_amp=np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64),
    )
    monkeypatch.setattr(validate_real_shots, "_MAX_SHOT_SIGNAL_SAMPLES", 4)
    with pytest.raises(ValueError, match="signal length"):
        validate_real_shots.load_disruption_shot_payload(shot_path)


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


def test_validate_transport_reports_uncertainty_envelope_contract(tmp_path: Path) -> None:
    import csv

    fields = [
        "machine",
        "shot",
        "Ip_MA",
        "BT_T",
        "ne19_1e19m3",
        "Ploss_MW",
        "R_m",
        "a_m",
        "kappa",
        "M_AMU",
        "tau_E_s",
    ]
    rows: list[dict[str, str]] = []
    coeffs = validate_real_shots.load_ipb98y2_coefficients()
    scenarios = [
        ("TEST", "001", 8.0, 5.3, 8.0, 50.0, 6.2, 2.0, 1.7, 2.5, 0.00),
        ("TEST", "002", 6.0, 3.2, 5.2, 14.0, 1.8, 0.6, 1.8, 2.0, -0.12),
        ("TEST", "003", 0.8, 2.1, 3.0, 4.0, 0.9, 0.32, 1.6, 2.0, 0.10),
    ]
    for machine, shot, ip_ma, bt_t, ne19, ploss_mw, r_m, a_m, kappa, m_amu, rel_offset in scenarios:
        tau_pred = validate_real_shots.ipb98y2_tau_e(
            ip_ma,
            bt_t,
            ne19,
            ploss_mw,
            r_m,
            kappa,
            a_m / r_m,
            m_amu,
            coefficients=coeffs,
        )
        tau_measured = tau_pred * (1.0 + rel_offset)
        rows.append(
            {
                "machine": machine,
                "shot": shot,
                "Ip_MA": f"{ip_ma:.6f}",
                "BT_T": f"{bt_t:.6f}",
                "ne19_1e19m3": f"{ne19:.6f}",
                "Ploss_MW": f"{ploss_mw:.6f}",
                "R_m": f"{r_m:.6f}",
                "a_m": f"{a_m:.6f}",
                "kappa": f"{kappa:.6f}",
                "M_AMU": f"{m_amu:.6f}",
                "tau_E_s": f"{tau_measured:.10f}",
            }
        )

    itpa_csv = tmp_path / "itpa_test.csv"
    with itpa_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    out = validate_real_shots.validate_transport(itpa_csv)
    env = out["uncertainty_envelope"]
    required = {
        "abs_relative_error_p50",
        "abs_relative_error_p95",
        "residual_s_p05",
        "residual_s_p50",
        "residual_s_p95",
        "sigma_s_p50",
        "sigma_s_p95",
        "zscore_p50",
        "zscore_p95",
        "within_1sigma_fraction",
        "within_2sigma_fraction",
    }
    assert required.issubset(env.keys())
    assert out["within_2sigma_fraction"] == env["within_2sigma_fraction"]
    assert float(env["abs_relative_error_p95"]) >= float(env["abs_relative_error_p50"]) >= 0.0
    assert float(env["sigma_s_p95"]) >= float(env["sigma_s_p50"]) > 0.0
    assert float(env["zscore_p95"]) >= float(env["zscore_p50"]) >= 0.0


def test_dataset_coverage_gate_tracks_observed_vs_required_counts() -> None:
    out = validate_real_shots.evaluate_dataset_coverage(
        {"n_files": 11},
        {"n_shots": 52},
        {"n_shots": 9},
        min_equilibrium_files=10,
        min_transport_shots=52,
        min_disruption_shots=10,
    )

    assert out["checks"]["equilibrium_files"]["passes"] is True
    assert out["checks"]["transport_shots"]["passes"] is True
    assert out["checks"]["disruption_shots"]["passes"] is False
    assert out["passes"] is False


def test_cli_strict_coverage_enforces_default_minima(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, int] = {}

    def fake_main(
        output_json: Path | None = None,
        output_md: Path | None = None,
        *,
        min_equilibrium_files: int = 0,
        min_transport_shots: int = 0,
        min_disruption_shots: int = 0,
    ) -> int:
        del output_json, output_md
        captured["min_equilibrium_files"] = min_equilibrium_files
        captured["min_transport_shots"] = min_transport_shots
        captured["min_disruption_shots"] = min_disruption_shots
        return 0

    monkeypatch.setattr(validate_real_shots, "main", fake_main)

    rc = validate_real_shots.cli(["--strict-coverage"])
    assert rc == 0
    assert captured["min_equilibrium_files"] == validate_real_shots.STRICT_DATASET_MINIMA["equilibrium_files"]
    assert captured["min_transport_shots"] == validate_real_shots.STRICT_DATASET_MINIMA["transport_shots"]
    assert captured["min_disruption_shots"] == validate_real_shots.STRICT_DATASET_MINIMA["disruption_shots"]
