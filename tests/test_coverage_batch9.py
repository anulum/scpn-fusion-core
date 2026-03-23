# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# SCPN Fusion Core — Coverage Batch 9 (API-verified, passing only)
from __future__ import annotations

import numpy as np
import pytest


class TestFederatedDisruption:
    def test_relu(self):
        from scpn_fusion.control.federated_disruption import _relu

        x = np.array([-1.0, 0.0, 1.0])
        result = _relu(x)
        np.testing.assert_array_equal(result, [0.0, 0.0, 1.0])

    def test_sigmoid(self):
        from scpn_fusion.control.federated_disruption import _sigmoid

        x = np.array([0.0])
        assert _sigmoid(x)[0] == pytest.approx(0.5)

    def test_binary_cross_entropy(self):
        from scpn_fusion.control.federated_disruption import _binary_cross_entropy

        bce = _binary_cross_entropy(np.array([0.9]), np.array([1.0]))
        assert bce > 0
        assert np.isfinite(bce)

    def test_init_mlp_weights(self):
        from scpn_fusion.control.federated_disruption import _init_mlp_weights

        rng = np.random.default_rng(42)
        weights = _init_mlp_weights(rng)
        assert isinstance(weights, dict)
        assert len(weights) > 0


class TestHaloREPhysics:
    def test_as_finite_float(self):
        from scpn_fusion.control.halo_re_physics import _as_finite_float

        assert _as_finite_float("x", 3.14) == pytest.approx(3.14)
        with pytest.raises(ValueError):
            _as_finite_float("x", float("nan"))

    def test_as_positive_float(self):
        from scpn_fusion.control.halo_re_physics import _as_positive_float

        assert _as_positive_float("x", 5.0) == 5.0
        with pytest.raises(ValueError):
            _as_positive_float("x", -1.0)

    def test_as_non_negative_float(self):
        from scpn_fusion.control.halo_re_physics import _as_non_negative_float

        assert _as_non_negative_float("x", 0.0) == 0.0
        with pytest.raises(ValueError):
            _as_non_negative_float("x", -0.1)

    def test_as_int(self):
        from scpn_fusion.control.halo_re_physics import _as_int

        assert _as_int("x", 5) == 5
        with pytest.raises(ValueError):
            _as_int("x", -1, minimum=0)
