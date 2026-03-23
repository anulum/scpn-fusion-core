# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# SCPN Fusion Core — Coverage Batch 8 (API-verified)
from __future__ import annotations

import numpy as np
import pytest


class TestDisruptionRiskRuntime:
    def test_require_int(self):
        from scpn_fusion.control.disruption_risk_runtime import _require_int

        assert _require_int("x", 5) == 5
        with pytest.raises((ValueError, TypeError)):
            _require_int("x", "hello")

    def test_build_feature_vector(self):
        from scpn_fusion.control.disruption_risk_runtime import build_disruption_feature_vector

        signal = np.random.randn(200)
        features = build_disruption_feature_vector(signal)
        assert isinstance(features, np.ndarray)
        assert np.all(np.isfinite(features))

    def test_apply_logit_bias(self):
        from scpn_fusion.control.disruption_risk_runtime import apply_disruption_logit_bias

        risk = apply_disruption_logit_bias(0.5, 0.1)
        assert 0.0 <= risk <= 1.0


class TestIMASConnectorCommon:
    def test_missing_required_keys(self):
        from scpn_fusion.io.imas_connector_common import _missing_required_keys

        missing = _missing_required_keys({"a": 1, "b": 2}, ("a", "b", "c"))
        assert "c" in missing

    def test_coerce_int(self):
        from scpn_fusion.io.imas_connector_common import _coerce_int

        assert _coerce_int("x", 42) == 42
        assert _coerce_int("x", 3) == 3

    def test_coerce_finite_real(self):
        from scpn_fusion.io.imas_connector_common import _coerce_finite_real

        assert _coerce_finite_real("x", 3.14) == pytest.approx(3.14)
        with pytest.raises(ValueError):
            _coerce_finite_real("x", float("nan"))


class TestLoggingConfig:
    def test_json_formatter(self):
        import logging
        from scpn_fusion.io.logging_config import FusionJSONFormatter

        fmt = FusionJSONFormatter()
        record = logging.LogRecord("test", logging.INFO, "test.py", 1, "hello", (), None)
        output = fmt.format(record)
        assert "hello" in output

    def test_setup_logging(self):
        from scpn_fusion.io.logging_config import setup_fusion_logging

        setup_fusion_logging(level="WARNING")


class TestSimulateTearingMode:
    def test_returns_data(self):
        from scpn_fusion.control.disruption_risk_runtime import simulate_tearing_mode

        result = simulate_tearing_mode(steps=50)
        assert result is not None
