# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# SCPN Fusion Core — Neural Transport Math Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.neural_transport_math import _relu, _softplus, _gelu, _compute_nustar


class TestActivations:
    def test_relu_positive(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        out = _relu(x)
        np.testing.assert_array_equal(out, [0.0, 0.0, 0.0, 1.0, 2.0])

    def test_softplus_positive(self):
        x = np.array([-5.0, 0.0, 5.0])
        out = _softplus(x)
        assert np.all(out > 0)
        assert out[2] > out[0]

    def test_softplus_clipped_extremes(self):
        out = _softplus(np.array([100.0]))
        assert np.isfinite(out[0])

    def test_gelu_zero(self):
        assert _gelu(np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-10)

    def test_gelu_positive_monotone(self):
        x = np.linspace(0, 5, 100)
        out = _gelu(x)
        assert np.all(np.diff(out) >= 0)

    def test_gelu_matches_approx(self):
        x = np.array([1.0])
        # GELU(1) ≈ 0.8412 (standard approximation)
        assert _gelu(x)[0] == pytest.approx(0.8412, abs=0.01)


class TestComputeNustar:
    def test_iter_baseline(self):
        nu = _compute_nustar(te_kev=10.0, ne_19=10.0, q=1.5, rho=0.5)
        assert nu > 0
        assert np.isfinite(nu)

    def test_collisionality_increases_with_density(self):
        nu_low = _compute_nustar(te_kev=10.0, ne_19=5.0, q=1.5, rho=0.5)
        nu_high = _compute_nustar(te_kev=10.0, ne_19=20.0, q=1.5, rho=0.5)
        assert nu_high > nu_low

    def test_collisionality_decreases_with_temperature(self):
        nu_cold = _compute_nustar(te_kev=1.0, ne_19=10.0, q=1.5, rho=0.5)
        nu_hot = _compute_nustar(te_kev=20.0, ne_19=10.0, q=1.5, rho=0.5)
        assert nu_cold > nu_hot

    def test_zero_rho_safe(self):
        nu = _compute_nustar(te_kev=10.0, ne_19=10.0, q=1.5, rho=0.0)
        assert np.isfinite(nu)

    def test_custom_geometry(self):
        nu = _compute_nustar(
            te_kev=10.0, ne_19=10.0, q=1.5, rho=0.5,
            r_major=1.85, a_minor=0.6, z_eff=2.0,
        )
        assert nu > 0
