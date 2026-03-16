# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Bio-Holonomic Controller Tests
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from scpn_fusion.control.bio_holonomic_controller import (
    BioHolonomicController,
    BioTelemetrySnapshot,
    SC_NEUROCORE_HOLONOMIC_AVAILABLE,
)


# ── BioTelemetrySnapshot ────────────────────────────────────────────


class TestBioTelemetrySnapshot:
    def test_fields(self):
        snap = BioTelemetrySnapshot(
            heart_rate_bpm=72.0,
            eeg_coherence_r=0.65,
            galvanic_skin_response=3.2,
        )
        assert snap.heart_rate_bpm == 72.0
        assert snap.eeg_coherence_r == 0.65
        assert snap.galvanic_skin_response == 3.2

    def test_frozen(self):
        snap = BioTelemetrySnapshot(
            heart_rate_bpm=72.0,
            eeg_coherence_r=0.65,
            galvanic_skin_response=3.2,
        )
        with pytest.raises(AttributeError):
            snap.heart_rate_bpm = 80.0  # type: ignore[misc]

    def test_equality(self):
        a = BioTelemetrySnapshot(72.0, 0.65, 3.2)
        b = BioTelemetrySnapshot(72.0, 0.65, 3.2)
        assert a == b

    def test_inequality(self):
        a = BioTelemetrySnapshot(72.0, 0.65, 3.2)
        b = BioTelemetrySnapshot(80.0, 0.65, 3.2)
        assert a != b


# ── BioHolonomicController ──────────────────────────────────────────


class TestBioHolonomicControllerInit:
    def test_raises_without_sc_neurocore(self):
        if SC_NEUROCORE_HOLONOMIC_AVAILABLE:
            pytest.skip("sc-neurocore IS available; cannot test missing-import path")
        with pytest.raises(RuntimeError, match="sc-neurocore"):
            BioHolonomicController()

    def test_raises_with_custom_dt(self):
        if SC_NEUROCORE_HOLONOMIC_AVAILABLE:
            pytest.skip("sc-neurocore IS available")
        with pytest.raises(RuntimeError, match="sc-neurocore"):
            BioHolonomicController(dt_s=0.02, seed=99)


class TestBioHolonomicControllerStep:
    """Test step() using mocked L4/L5 adapters."""

    @pytest.fixture()
    def controller(self):
        if not SC_NEUROCORE_HOLONOMIC_AVAILABLE:
            # Patch the adapters and flag to allow construction
            mock_l4 = MagicMock()
            mock_l4.step_jax.return_value = [0.5, 0.3]
            mock_l4.get_metrics.return_value = {
                "order_parameter": 0.72,
                "avalanche_density": 0.15,
            }
            mock_l4.params = MagicMock()

            mock_l5 = MagicMock()
            mock_l5.step_jax.return_value = None
            mock_l5.get_metrics.return_value = {
                "hrv_coherence_r5": 0.35,
                "emotional_valence": 0.6,
            }

            ctrl = object.__new__(BioHolonomicController)
            ctrl.dt_s = 0.01
            ctrl.l4_adapter = mock_l4
            ctrl.l5_adapter = mock_l5
            return ctrl
        else:
            return BioHolonomicController(dt_s=0.01)

    def test_step_returns_expected_keys(self, controller):
        telemetry = BioTelemetrySnapshot(72.0, 0.65, 3.2)
        result = controller.step(telemetry)
        assert "l4_order_parameter" in result
        assert "l4_avalanche_density" in result
        assert "l5_hrv_coherence" in result
        assert "l5_emotional_valence" in result
        assert "actuator_vibrana_intensity" in result

    def test_vibrana_active_on_low_hrv(self, controller):
        if SC_NEUROCORE_HOLONOMIC_AVAILABLE:
            pytest.skip("Real adapters may not produce exact values")
        telemetry = BioTelemetrySnapshot(72.0, 0.65, 3.2)
        result = controller.step(telemetry)
        # l5 mock returns hrv_coherence_r5=0.35 < 0.4
        assert result["actuator_vibrana_intensity"] > 0.0

    def test_vibrana_zero_on_high_hrv(self):
        mock_l4 = MagicMock()
        mock_l4.step_jax.return_value = [0.5]
        mock_l4.get_metrics.return_value = {
            "order_parameter": 0.8,
            "avalanche_density": 0.1,
        }
        mock_l4.params = MagicMock()

        mock_l5 = MagicMock()
        mock_l5.step_jax.return_value = None
        mock_l5.get_metrics.return_value = {
            "hrv_coherence_r5": 0.6,  # above 0.4 threshold
            "emotional_valence": 0.7,
        }

        ctrl = object.__new__(BioHolonomicController)
        ctrl.dt_s = 0.01
        ctrl.l4_adapter = mock_l4
        ctrl.l5_adapter = mock_l5

        result = ctrl.step(BioTelemetrySnapshot(70.0, 0.8, 2.0))
        assert result["actuator_vibrana_intensity"] == 0.0

    def test_vibrana_intensity_clamped(self):
        mock_l4 = MagicMock()
        mock_l4.step_jax.return_value = [0.5]
        mock_l4.get_metrics.return_value = {
            "order_parameter": 0.5,
            "avalanche_density": 0.2,
        }
        mock_l4.params = MagicMock()

        mock_l5 = MagicMock()
        mock_l5.step_jax.return_value = None
        mock_l5.get_metrics.return_value = {
            "hrv_coherence_r5": 0.0,  # maximally decoherent
            "emotional_valence": 0.1,
        }

        ctrl = object.__new__(BioHolonomicController)
        ctrl.dt_s = 0.01
        ctrl.l4_adapter = mock_l4
        ctrl.l5_adapter = mock_l5

        result = ctrl.step(BioTelemetrySnapshot(100.0, 0.1, 5.0))
        assert result["actuator_vibrana_intensity"] <= 1.0

    def test_eeg_coherence_modulates_coupling(self, controller):
        telemetry_low = BioTelemetrySnapshot(72.0, 0.0, 3.0)
        controller.step(telemetry_low)
        k_low = controller.l4_adapter.params.k_coupling

        telemetry_high = BioTelemetrySnapshot(72.0, 1.0, 3.0)
        controller.step(telemetry_high)
        k_high = controller.l4_adapter.params.k_coupling

        if SC_NEUROCORE_HOLONOMIC_AVAILABLE:
            assert k_low != k_high
