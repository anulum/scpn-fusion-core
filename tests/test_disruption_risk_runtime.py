# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Direct tests for the disruption-risk runtime helpers and robustness campaigns.

Covers the feature-vector and logit-bias validators, the single-bit fault
injector, the fault-noise robustness campaign (and its input validation), and
the hybrid anomaly detector / alarm-campaign guards.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.control.disruption_risk_runtime import (
    HybridAnomalyDetector,
    apply_bit_flip_fault,
    apply_disruption_logit_bias,
    build_disruption_feature_vector,
    run_anomaly_alarm_campaign,
    run_fault_noise_campaign,
)


class TestBuildFeatureVector:
    """Feature-vector construction validates signal and toroidal inputs."""

    def test_nominal_vector_has_eleven_finite_features(self) -> None:
        """A finite signal yields the documented 11-component feature vector."""
        features = build_disruption_feature_vector([0.1, 0.2, 0.3, 0.4])
        assert features.shape == (11,)
        assert np.all(np.isfinite(features))

    def test_empty_signal_raises(self) -> None:
        """An empty signal is rejected."""
        with pytest.raises(ValueError, match="at least one sample"):
            build_disruption_feature_vector([])

    def test_non_finite_signal_raises(self) -> None:
        """A non-finite signal sample is rejected."""
        with pytest.raises(ValueError, match="signal must be finite"):
            build_disruption_feature_vector([0.1, float("nan")])

    def test_non_finite_toroidal_raises(self) -> None:
        """A non-finite toroidal observable is rejected."""
        with pytest.raises(ValueError, match="toroidal observables must be finite"):
            build_disruption_feature_vector([0.1, 0.2], {"toroidal_n1_amp": float("inf")})


class TestApplyLogitBias:
    """Additive logit-space calibration bias."""

    def test_zero_bias_is_near_identity(self) -> None:
        """A zero bias returns approximately the input risk."""
        assert apply_disruption_logit_bias(0.4, 0.0) == pytest.approx(0.4, abs=1e-9)

    def test_positive_bias_increases_risk(self) -> None:
        """A positive bias shifts the bounded risk upward."""
        assert apply_disruption_logit_bias(0.4, 1.0) > 0.4

    def test_non_finite_risk_raises(self) -> None:
        """A non-finite risk is rejected."""
        with pytest.raises(ValueError, match="risk must be finite"):
            apply_disruption_logit_bias(float("nan"), 0.0)

    def test_non_finite_bias_raises(self) -> None:
        """A non-finite bias delta is rejected."""
        with pytest.raises(ValueError, match="bias_delta must be finite"):
            apply_disruption_logit_bias(0.4, float("inf"))


class TestApplyBitFlipFault:
    """Deterministic single-bit fault injection."""

    def test_flip_changes_or_preserves_value_finitely(self) -> None:
        """A valid bit index returns a finite float."""
        assert np.isfinite(apply_bit_flip_fault(1.0, 5))

    def test_non_finite_result_falls_back_to_value(self) -> None:
        """Flipping the exponent to a non-finite pattern returns the original."""
        # Bit 62 toggles a high exponent bit; the guard returns the input if the
        # result is non-finite.
        assert np.isfinite(apply_bit_flip_fault(1.0, 62))

    @pytest.mark.parametrize("bit", [True, 64, -1])
    def test_invalid_bit_index_raises(self, bit: object) -> None:
        """A boolean or out-of-range bit index is rejected."""
        with pytest.raises(ValueError, match="bit_index must be an integer"):
            apply_bit_flip_fault(1.0, bit)  # type: ignore[arg-type]


class TestFaultNoiseCampaign:
    """Robustness campaign under injected noise and bit flips."""

    def test_small_campaign_returns_summary(self) -> None:
        """A short campaign returns a complete, finite robustness summary."""
        summary = run_fault_noise_campaign(seed=1, episodes=2, window=16, bit_flip_interval=4)
        for key in (
            "mean_abs_risk_error",
            "p95_abs_risk_error",
            "recovery_steps_p95",
            "recovery_success_rate",
            "fault_count",
            "passes_thresholds",
        ):
            assert key in summary
        assert summary["episodes"] == 2
        assert np.isfinite(summary["mean_abs_risk_error"])
        assert summary["fault_count"] >= 0

    def test_deterministic_for_same_seed(self) -> None:
        """The campaign is reproducible for a fixed seed."""
        a = run_fault_noise_campaign(seed=7, episodes=2, window=16)
        b = run_fault_noise_campaign(seed=7, episodes=2, window=16)
        assert a["mean_abs_risk_error"] == b["mean_abs_risk_error"]

    def test_negative_noise_std_raises(self) -> None:
        """A negative noise standard deviation is rejected."""
        with pytest.raises(ValueError, match="noise_std must be finite"):
            run_fault_noise_campaign(seed=1, episodes=2, window=16, noise_std=-1.0)

    def test_non_positive_recovery_epsilon_raises(self) -> None:
        """A non-positive recovery epsilon is rejected."""
        with pytest.raises(ValueError, match="recovery_epsilon must be finite"):
            run_fault_noise_campaign(seed=1, episodes=2, window=16, recovery_epsilon=0.0)


class TestHybridAnomalyDetector:
    """Hybrid anomaly detector configuration guards."""

    def test_invalid_threshold_raises(self) -> None:
        """A threshold outside ``[0, 1]`` is rejected."""
        with pytest.raises(ValueError, match="threshold must be finite and in"):
            HybridAnomalyDetector(threshold=1.5)

    def test_invalid_ema_raises(self) -> None:
        """An EMA rate outside ``(0, 1]`` is rejected."""
        with pytest.raises(ValueError, match="ema must be finite and in"):
            HybridAnomalyDetector(ema=0.0)

    def test_score_returns_fused_scores(self) -> None:
        """Scoring returns supervised, unsupervised, and alarm fields."""
        detector = HybridAnomalyDetector(threshold=0.5, ema=0.1)
        scored = detector.score([0.2, 0.3, 0.4, 0.5])
        for key in ("supervised_score", "unsupervised_score", "anomaly_score", "alarm"):
            assert key in scored


class TestAnomalyAlarmCampaign:
    """Alarm campaign threshold validation."""

    def test_invalid_threshold_raises(self) -> None:
        """A threshold outside ``(0, 1)`` is rejected."""
        with pytest.raises(ValueError, match="threshold must be finite and in"):
            run_anomaly_alarm_campaign(seed=1, episodes=4, window=16, threshold=1.0)
