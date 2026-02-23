# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — GNEU-02 Anomaly Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for hybrid anomaly detector in disruption predictor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import scpn_fusion.control.disruption_predictor as dp
from scpn_fusion.control.disruption_predictor import (
    HybridAnomalyDetector,
    predict_disruption_risk,
    predict_disruption_risk_safe,
    run_anomaly_alarm_campaign,
)


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
        "passes_thresholds",
    ):
        assert key in report
    assert 0.0 <= report["true_positive_rate"] <= 1.0
    assert 0.0 <= report["false_positive_rate"] <= 1.0


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
    ],
)
def test_anomaly_alarm_campaign_rejects_invalid_inputs(
    kwargs: dict[str, float | int], match: str
) -> None:
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
def test_load_or_train_predictor_rejects_invalid_seq_len(
    tmp_path: Path, seq_len: object
) -> None:
    with pytest.raises(ValueError, match="seq_len"):
        dp.load_or_train_predictor(
            model_path=tmp_path / "missing_model.pth",
            seq_len=seq_len,
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


@pytest.mark.skipif(dp.torch is None, reason="torch required for checkpoint path")
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


@pytest.mark.skipif(dp.torch is None, reason="torch required for checkpoint path")
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
