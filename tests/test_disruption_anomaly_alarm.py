# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Disruption Anomaly-Alarm Tests
"""Public-surface tests for hybrid disruption anomaly alarms."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import runpy
import sys
from typing import Any

import numpy as np
import pytest

import scpn_fusion.control.disruption_predictor as dp
from scpn_fusion.control.disruption_predictor import (
    HybridAnomalyDetector,
    predict_disruption_risk,
    predict_disruption_risk_safe,
    run_anomaly_alarm_campaign,
)


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "disruption_anomaly_alarm_validation.py"
SPEC = importlib.util.spec_from_file_location(
    "disruption_anomaly_alarm_validation",
    MODULE_PATH,
)
assert SPEC and SPEC.loader
validation_cli = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validation_cli
SPEC.loader.exec_module(validation_cli)


def _campaign_payload() -> dict[str, Any]:
    return {
        "seed": 0,
        "episodes": 128,
        "window": 64,
        "threshold": 0.5,
        "true_positive_rate": 0.0,
        "false_positive_rate": 0.0,
        "p95_alarm_latency_steps": -1,
        "min_true_positive_rate": 0.9,
        "max_false_positive_rate": 0.1,
        "max_p95_alarm_latency_steps": 24,
        "passes_thresholds": False,
    }


def _current_report(payload: Any = None) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "report_kind": "disruption_anomaly_alarm_validation",
        "generated_at_utc": "2026-08-28T00:00:00+00:00",
        "runtime_seconds": 0.1,
        "disruption_anomaly_alarm_validation": (
            _campaign_payload() if payload is None else payload
        ),
    }


def test_hybrid_detector_score_is_bounded() -> None:
    det = HybridAnomalyDetector(threshold=0.65, ema=0.05)
    signal = np.linspace(0.2, 1.0, 64)
    out = det.score(
        signal,
        {
            "toroidal_n1_amp": 0.2,
            "toroidal_n2_amp": 0.1,
            "toroidal_n3_amp": 0.05,
            "toroidal_asymmetry_index": 0.25,
            "toroidal_radial_spread": 0.03,
        },
    )
    assert 0.0 <= out["supervised_score"] <= 1.0
    assert 0.0 <= out["unsupervised_score"] <= 1.0
    assert 0.0 <= out["anomaly_score"] <= 1.0


def test_hybrid_detector_deterministic_given_same_sequence() -> None:
    sig = np.linspace(0.3, 0.9, 80)
    obs = {
        "toroidal_n1_amp": 0.18,
        "toroidal_n2_amp": 0.09,
        "toroidal_n3_amp": 0.04,
        "toroidal_asymmetry_index": 0.205,
        "toroidal_radial_spread": 0.02,
    }
    d1 = HybridAnomalyDetector()
    d2 = HybridAnomalyDetector()
    o1 = [d1.score(sig[: i + 1], obs)["anomaly_score"] for i in range(32)]
    o2 = [d2.score(sig[: i + 1], obs)["anomaly_score"] for i in range(32)]
    assert np.allclose(o1, o2)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"threshold": -0.1}, "threshold"),
        ({"threshold": 1.1}, "threshold"),
        ({"threshold": float("nan")}, "threshold"),
        ({"threshold": float("inf")}, "threshold"),
        ({"ema": 0.0}, "ema"),
        ({"ema": -0.1}, "ema"),
        ({"ema": 1.1}, "ema"),
        ({"ema": float("nan")}, "ema"),
    ],
)
def test_hybrid_detector_rejects_invalid_constructor_inputs(
    kwargs: dict[str, float], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        HybridAnomalyDetector(**kwargs)


def test_anomaly_alarm_campaign_outputs_expected_metrics() -> None:
    report = run_anomaly_alarm_campaign(seed=23, episodes=8, window=64)
    for key in (
        "true_positive_rate",
        "false_positive_rate",
        "p95_alarm_latency_steps",
        "min_true_positive_rate",
        "max_false_positive_rate",
        "max_p95_alarm_latency_steps",
        "passes_thresholds",
    ):
        assert key in report
    assert 0.0 <= report["true_positive_rate"] <= 1.0
    assert 0.0 <= report["false_positive_rate"] <= 1.0
    assert report["passes_thresholds"] is False


def test_anomaly_alarm_campaign_can_pass_only_public_permissive_thresholds() -> None:
    report = run_anomaly_alarm_campaign(
        seed=0,
        episodes=128,
        window=64,
        threshold=0.02,
        min_true_positive_rate=0.90,
        max_false_positive_rate=1.0,
        max_p95_alarm_latency_steps=24,
    )

    assert report["true_positive_rate"] >= 0.90
    assert report["false_positive_rate"] > 0.10
    assert report["passes_thresholds"] is True


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"seed": -1}, "seed"),
        ({"episodes": 0}, "episodes"),
        ({"episodes": 1.25}, "episodes"),
        ({"window": 8}, "window"),
        ({"window": 32.5}, "window"),
        ({"threshold": -0.1}, "threshold"),
        ({"threshold": 1.1}, "threshold"),
        ({"threshold": float("nan")}, "threshold"),
        ({"min_true_positive_rate": -0.1}, "min_true_positive_rate"),
        ({"min_true_positive_rate": float("nan")}, "min_true_positive_rate"),
        ({"max_false_positive_rate": 1.1}, "max_false_positive_rate"),
        ({"max_false_positive_rate": float("inf")}, "max_false_positive_rate"),
        ({"max_p95_alarm_latency_steps": -1}, "max_p95_alarm_latency_steps"),
        ({"max_p95_alarm_latency_steps": 2.5}, "max_p95_alarm_latency_steps"),
    ],
)
def test_anomaly_alarm_campaign_rejects_invalid_inputs(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        run_anomaly_alarm_campaign(**kwargs)


def test_predict_disruption_risk_safe_fallback_when_checkpoint_missing(
    tmp_path: Path,
) -> None:
    signal = np.linspace(0.25, 0.95, 80)
    toroidal = {
        "toroidal_n1_amp": 0.16,
        "toroidal_n2_amp": 0.07,
        "toroidal_n3_amp": 0.03,
        "toroidal_asymmetry_index": 0.177,
        "toroidal_radial_spread": 0.02,
    }
    expected = predict_disruption_risk(signal, toroidal)
    risk, meta = predict_disruption_risk_safe(
        signal,
        toroidal,
        model_path=tmp_path / "missing_model.pth",
        train_if_missing=False,
    )
    assert abs(risk - expected) < 1e-12
    assert meta["mode"] == "fallback"
    assert meta["risk_source"] == "predict_disruption_risk"
    assert "reason" in meta


def test_predict_disruption_risk_safe_can_disable_fallback(tmp_path: Path) -> None:
    signal = np.linspace(0.25, 0.95, 80)
    toroidal = {
        "toroidal_n1_amp": 0.16,
        "toroidal_n2_amp": 0.07,
        "toroidal_n3_amp": 0.03,
        "toroidal_asymmetry_index": 0.177,
        "toroidal_radial_spread": 0.02,
    }
    with pytest.raises((RuntimeError, FileNotFoundError)):
        predict_disruption_risk_safe(
            signal,
            toroidal,
            model_path=tmp_path / "missing_model.pth",
            train_if_missing=False,
            allow_fallback=False,
        )


def test_load_or_train_predictor_can_return_fallback_metadata_when_missing(
    tmp_path: Path,
) -> None:
    model, meta = dp.load_or_train_predictor(
        model_path=tmp_path / "missing_model.pth",
        seq_len=32,
        train_if_missing=False,
        allow_fallback=True,
    )
    assert model is None
    assert meta["fallback"] is True
    assert meta["reason"] in {"checkpoint_missing", "torch_unavailable"}


@pytest.mark.parametrize("seq_len", [0, 16.5])
def test_load_or_train_predictor_rejects_invalid_seq_len(tmp_path: Path, seq_len: object) -> None:
    with pytest.raises(ValueError, match="seq_len"):
        dp.load_or_train_predictor(
            model_path=tmp_path / "missing_model.pth",
            seq_len=seq_len,  # type: ignore[arg-type]
            train_if_missing=False,
            allow_fallback=True,
        )


# S2-004: Disruption predictor fallback/raise-path coverage


def test_load_or_train_fallback_torch_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(dp, "torch", None)
    model, meta = dp.load_or_train_predictor(
        model_path=tmp_path / "m.pth",
        seq_len=32,
        train_if_missing=False,
        allow_fallback=True,
    )
    assert model is None
    assert meta["fallback"] is True
    assert meta["reason"] == "torch_unavailable"


def test_load_or_train_raises_without_fallback_no_torch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(dp, "torch", None)
    with pytest.raises(RuntimeError, match="Torch is required"):
        dp.load_or_train_predictor(
            model_path=tmp_path / "m.pth",
            seq_len=32,
            train_if_missing=False,
            allow_fallback=False,
        )


@pytest.mark.skipif(getattr(dp, "torch", None) is None, reason="torch required")
def test_load_or_train_raises_without_fallback_missing_checkpoint(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        dp.load_or_train_predictor(
            model_path=tmp_path / "nonexistent.pth",
            seq_len=32,
            train_if_missing=False,
            allow_fallback=False,
        )


@pytest.mark.skipif(getattr(dp, "torch", None) is None, reason="torch required")
def test_load_or_train_fallback_corrupt_checkpoint(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.pth"
    corrupt.write_bytes(b"NOT_A_VALID_CHECKPOINT_123456")
    model, meta = dp.load_or_train_predictor(
        model_path=corrupt,
        seq_len=32,
        train_if_missing=False,
        allow_fallback=True,
    )
    assert model is None
    assert meta["fallback"] is True
    assert "checkpoint_load_failed" in meta["reason"]


def test_predict_safe_fallback_returns_valid_risk_no_torch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(dp, "torch", None)
    signal = np.linspace(0.25, 0.95, 80)
    toroidal = {
        "toroidal_n1_amp": 0.16,
        "toroidal_n2_amp": 0.07,
        "toroidal_n3_amp": 0.03,
        "toroidal_asymmetry_index": 0.177,
        "toroidal_radial_spread": 0.02,
    }
    risk, meta = predict_disruption_risk_safe(
        signal,
        toroidal,
        model_path=tmp_path / "m.pth",
        train_if_missing=False,
    )
    assert isinstance(risk, float)
    assert 0.0 <= risk <= 1.0
    assert meta["mode"] == "fallback"
    assert meta["risk_source"] == "predict_disruption_risk"


def test_predict_safe_raises_without_fallback_no_torch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(dp, "torch", None)
    signal = np.linspace(0.25, 0.95, 80)
    toroidal = {
        "toroidal_n1_amp": 0.16,
        "toroidal_n2_amp": 0.07,
        "toroidal_n3_amp": 0.03,
        "toroidal_asymmetry_index": 0.177,
        "toroidal_radial_spread": 0.02,
    }
    with pytest.raises(RuntimeError, match="Torch is required"):
        predict_disruption_risk_safe(
            signal,
            toroidal,
            model_path=tmp_path / "m.pth",
            train_if_missing=False,
            allow_fallback=False,
        )


def test_validation_reports_default_gate_failure_honestly() -> None:
    report = validation_cli.generate_report(seed=0, episodes=128, window=64)
    payload = validation_cli.validate_report(report)
    markdown = validation_cli.render_markdown(report)

    assert report["schema_version"] == 2
    assert report["report_kind"] == "disruption_anomaly_alarm_validation"
    assert payload["passes_thresholds"] is False
    assert payload["p95_alarm_latency_steps"] == -1
    assert "Overall pass: `NO`" in markdown
    assert "not experimental validation" in markdown


def test_validation_can_pass_only_explicit_permissive_public_gate() -> None:
    report = validation_cli.generate_report(
        seed=0,
        episodes=128,
        window=64,
        threshold=0.02,
        min_true_positive_rate=0.90,
        max_false_positive_rate=1.0,
        max_p95_alarm_latency_steps=24,
    )

    assert validation_cli.validate_report(report)["passes_thresholds"] is True
    assert "Overall pass: `YES`" in validation_cli.render_markdown(report)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schema_version": 1}, "unsupported report schema_version"),
        ({"report_kind": "obsolete"}, "unsupported report_kind"),
        ({"generated_at_utc": None}, "generated_at_utc must be"),
        ({"generated_at_utc": ""}, "generated_at_utc must be"),
        ({"runtime_seconds": True}, "runtime_seconds must be"),
        ({"runtime_seconds": "slow"}, "runtime_seconds must be"),
        ({"runtime_seconds": float("nan")}, "runtime_seconds must be"),
        ({"runtime_seconds": -0.1}, "runtime_seconds must be"),
        ({"disruption_anomaly_alarm_validation": []}, "must be an object"),
        ({"extra": True}, "current descriptive contract"),
    ],
)
def test_report_contract_rejects_invalid_envelopes(change: dict[str, Any], message: str) -> None:
    report = _current_report()
    report.update(change)

    with pytest.raises(ValueError, match=message):
        validation_cli.validate_report(report)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("seed", True, "seed must be"),
        ("seed", -1, "seed must be"),
        ("episodes", 1.5, "episodes must be"),
        ("episodes", 0, "episodes must be"),
        ("window", 15, "window must be"),
        ("threshold", "high", "threshold must be"),
        ("threshold", float("nan"), "threshold must be"),
        ("threshold", 0.0, "strictly between"),
        ("threshold", 1.0, "strictly between"),
        ("true_positive_rate", -0.1, "true_positive_rate must be"),
        ("false_positive_rate", 1.1, "false_positive_rate must be"),
        ("min_true_positive_rate", float("inf"), "min_true_positive_rate must be"),
        ("max_false_positive_rate", True, "max_false_positive_rate must be"),
        ("p95_alarm_latency_steps", -2, "p95_alarm_latency_steps must be"),
        ("max_p95_alarm_latency_steps", -1, "max_p95_alarm_latency_steps must be"),
        ("passes_thresholds", 0, "passes_thresholds must be"),
        ("passes_thresholds", True, "inconsistent with public thresholds"),
    ],
)
def test_report_contract_rejects_invalid_payload_values(key: str, value: Any, message: str) -> None:
    payload = _campaign_payload()
    payload[key] = value

    with pytest.raises(ValueError, match=message):
        validation_cli.validate_report(_current_report(payload))


def test_report_contract_rejects_payload_shape_and_obsolete_coded_payload() -> None:
    missing = _campaign_payload()
    missing.pop("window")
    with pytest.raises(ValueError, match="payload keys"):
        validation_cli.validate_report(_current_report(missing))

    stale_report = {
        "generated_at_utc": "2026-08-28T00:00:00+00:00",
        "runtime_seconds": 0.1,
        "gneu_02": {"passes_thresholds": True},
    }
    with pytest.raises(ValueError, match="current descriptive contract"):
        validation_cli.validate_report(stale_report)


def test_cli_defaults_and_strict_fail_pass(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    defaults = validation_cli.parse_args([])
    assert defaults.output_json.endswith("disruption_anomaly_alarm_validation.json")
    assert defaults.output_md.endswith("disruption_anomaly_alarm_validation.md")

    output_json = tmp_path / "alarm.json"
    output_md = tmp_path / "alarm.md"
    common = [
        "--episodes",
        "128",
        "--output-json",
        str(output_json),
        "--output-md",
        str(output_md),
    ]
    assert validation_cli.main([*common, "--strict"]) == 2
    failed_report = json.loads(output_json.read_text(encoding="utf-8"))
    assert validation_cli.validate_report(failed_report)["passes_thresholds"] is False
    assert "validation complete" in capsys.readouterr().out
    assert validation_cli.main(common) == 0

    permissive = [
        *common,
        "--threshold",
        "0.02",
        "--max-false-positive-rate",
        "1.0",
        "--strict",
    ]
    assert validation_cli.main(permissive) == 0
    passed_report = json.loads(output_json.read_text(encoding="utf-8"))
    assert validation_cli.validate_report(passed_report)["passes_thresholds"] is True


def test_script_entry_point_runs_real_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_json = tmp_path / "script.json"
    output_md = tmp_path / "script.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(MODULE_PATH),
            "--episodes",
            "128",
            "--threshold",
            "0.02",
            "--max-false-positive-rate",
            "1.0",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--strict",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(MODULE_PATH), run_name="__main__")

    assert exc_info.value.code == 0
    assert output_json.is_file()
    assert output_md.is_file()
