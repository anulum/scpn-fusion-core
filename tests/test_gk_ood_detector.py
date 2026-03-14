# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — OOD Detector Tests
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.gk_ood_detector import OODDetector, OODResult, _to_input_vector


@pytest.fixture
def detector():
    return OODDetector()


@pytest.fixture
def in_distribution_input():
    """Typical mid-radius tokamak parameters, well within QLKNN training range."""
    return _to_input_vector(
        R_L_Ti=6.0,
        R_L_Te=6.0,
        R_L_ne=2.0,
        q=2.0,
        s_hat=1.0,
        alpha_MHD=0.3,
        Te_Ti=1.0,
        Z_eff=1.5,
        nu_star=0.5,
        beta_e=0.02,
    )


@pytest.fixture
def ood_input():
    """Extreme parameters well outside training bounds."""
    return _to_input_vector(
        R_L_Ti=50.0,
        R_L_Te=50.0,
        R_L_ne=20.0,
        q=10.0,
        s_hat=10.0,
        alpha_MHD=5.0,
        Te_Ti=10.0,
        Z_eff=10.0,
        nu_star=100.0,
        beta_e=1.0,
    )


def test_mahalanobis_in_distribution(detector, in_distribution_input):
    result = detector.check_mahalanobis(in_distribution_input)
    assert result.is_ood is False
    assert result.method == "mahalanobis"
    assert result.confidence < 1.0


def test_mahalanobis_ood(detector, ood_input):
    result = detector.check_mahalanobis(ood_input)
    assert result.is_ood is True
    assert result.confidence == pytest.approx(1.0)


def test_mahalanobis_distance_value(detector, in_distribution_input):
    d = detector.mahalanobis_distance(in_distribution_input)
    assert d >= 0.0
    assert np.isfinite(d)


def test_range_check_in_distribution(detector, in_distribution_input):
    result = detector.check_range(in_distribution_input)
    assert result.is_ood is False
    assert len(result.details["hard_violations"]) == 0


def test_range_check_hard_violation(detector, ood_input):
    result = detector.check_range(ood_input)
    assert result.is_ood is True
    assert len(result.details["hard_violations"]) > 0


def test_range_check_soft_violation(detector):
    # Just outside 2-sigma on R/L_Ti but still within hard bounds
    x = _to_input_vector(
        R_L_Ti=20.0,
        R_L_Te=6.0,
        R_L_ne=2.0,
        q=2.0,
        s_hat=1.0,
        alpha_MHD=0.3,
        Te_Ti=1.0,
        Z_eff=1.5,
        nu_star=0.5,
        beta_e=0.02,
    )
    result = detector.check_range(x)
    assert not result.is_ood  # within hard bounds
    assert len(result.details["soft_violations"]) > 0


def test_ensemble_identical_models():
    detector = OODDetector()
    preds = np.array([[1.0, 0.8, 0.3]] * 5)
    result = detector.check_ensemble(preds)
    assert result.is_ood is False
    assert result.details["max_relative_std"] == pytest.approx(0.0, abs=1e-10)


def test_ensemble_high_disagreement():
    detector = OODDetector()
    preds = np.array(
        [
            [1.0, 0.8, 0.3],
            [5.0, 4.0, 2.0],
            [0.1, 0.1, 0.05],
            [3.0, 2.5, 1.0],
            [8.0, 6.0, 3.0],
        ]
    )
    result = detector.check_ensemble(preds)
    assert result.is_ood is True
    assert result.details["max_relative_std"] > 0.3


def test_ensemble_insufficient_models():
    detector = OODDetector()
    preds = np.array([[1.0, 0.8, 0.3]])
    result = detector.check_ensemble(preds)
    assert result.is_ood is False
    assert result.details["reason"] == "insufficient_models"


def test_combined_check_in_distribution(detector, in_distribution_input):
    result = detector.check(in_distribution_input)
    assert result.is_ood is False
    assert result.method == "combined"
    assert "mahalanobis" in result.details["sub_results"]
    assert "range" in result.details["sub_results"]


def test_combined_check_ood(detector, ood_input):
    result = detector.check(ood_input)
    assert result.is_ood is True


def test_combined_with_ensemble(detector, in_distribution_input):
    preds = np.array([[1.0, 0.8, 0.3]] * 5)
    result = detector.check(in_distribution_input, ensemble_predictions=preds)
    assert result.is_ood is False
    assert "ensemble" in result.details["sub_results"]


def test_custom_thresholds():
    detector = OODDetector(
        mahalanobis_threshold=1.0,
        soft_sigma_threshold=0.5,
        ensemble_disagreement_threshold=0.1,
    )
    x = _to_input_vector(
        R_L_Ti=12.0,
        R_L_Te=12.0,
        R_L_ne=2.0,
        q=2.0,
        s_hat=1.0,
        alpha_MHD=0.3,
        Te_Ti=1.0,
        Z_eff=1.5,
        nu_star=0.5,
        beta_e=0.02,
    )
    result = detector.check_mahalanobis(x)
    # Tighter threshold should trigger more easily
    assert result.details["threshold"] == 1.0


def test_custom_training_stats():
    mean = np.zeros(10)
    cov_inv = np.eye(10)
    detector = OODDetector(training_mean=mean, training_cov_inv=cov_inv)
    x = np.ones(10) * 5.0
    d = detector.mahalanobis_distance(x)
    # sqrt(10 * 25) = sqrt(250) ≈ 15.8
    assert d == pytest.approx(np.sqrt(250.0), rel=1e-6)


def test_ood_result_dataclass():
    r = OODResult(is_ood=True, confidence=0.8, method="test", details={"key": "val"})
    assert r.is_ood is True
    assert r.confidence == 0.8
    assert r.details["key"] == "val"


def test_to_input_vector():
    v = _to_input_vector(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    assert v.shape == (10,)
    assert v[0] == 1.0
    assert v[9] == 10.0
