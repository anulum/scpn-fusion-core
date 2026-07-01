# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — source/config header compliance
from __future__ import annotations

import numpy as np
import pytest

from types import SimpleNamespace

from scpn_fusion.core.neural_transport_math import (
    _compute_nustar,
    _gelu,
    _mlp_forward,
    _relu,
    _softplus,
)


def _mlp_weights(*, gated, log_transform, gb_scale, n_layers, n_in, n_hidden, n_out_raw):
    """Build a deterministic duck-typed MLP weight bundle for _mlp_forward."""
    dims = [n_in] + [n_hidden] * (n_layers - 1) + [n_out_raw]
    layers_w = [np.full((dims[i], dims[i + 1]), 0.1) for i in range(n_layers)]
    layers_b = [np.zeros(dims[i + 1]) for i in range(n_layers)]
    return SimpleNamespace(
        input_mean=np.zeros(n_in),
        input_std=np.ones(n_in),
        n_layers=n_layers,
        layers_w=layers_w,
        layers_b=layers_b,
        gated=gated,
        output_scale=1.0,
        log_transform=log_transform,
        gb_scale=gb_scale,
    )


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
            te_kev=10.0,
            ne_19=10.0,
            q=1.5,
            rho=0.5,
            r_major=1.85,
            a_minor=0.6,
            z_eff=2.0,
        )
        assert nu > 0


class TestMlpForward:
    def test_gated_with_log_transform_and_gyrobohm_batched(self):
        weights = _mlp_weights(
            gated=True,
            log_transform=True,
            gb_scale=True,
            n_layers=2,
            n_in=4,
            n_hidden=5,
            n_out_raw=6,
        )
        x = np.tile(np.array([1.0, 2.0, 0.5, 0.3]), (7, 1))  # te = column 1 -> array gyro-Bohm
        out = _mlp_forward(x, weights)
        assert out.shape == (7, 3)  # gated collapses to the three flux channels
        assert np.all(np.isfinite(out))
        assert np.all(out >= 0.0)

    def test_scalar_input_uses_scalar_gyrobohm(self):
        weights = _mlp_weights(
            gated=True,
            log_transform=False,
            gb_scale=True,
            n_layers=2,
            n_in=4,
            n_hidden=5,
            n_out_raw=6,
        )
        x = np.array([1.0, 2.0, 0.5, 0.3])  # 1D -> scalar te -> chi_gb.ndim == 0 branch
        out = _mlp_forward(x, weights)
        assert out.shape == (3,)
        assert np.all(np.isfinite(out))

    def test_ungated_plain_softplus_output(self):
        weights = _mlp_weights(
            gated=False,
            log_transform=False,
            gb_scale=False,
            n_layers=3,
            n_in=4,
            n_hidden=5,
            n_out_raw=2,
        )
        x = np.tile(np.array([1.0, 2.0, 0.5, 0.3]), (4, 1))
        out = _mlp_forward(x, weights)
        assert out.shape == (4, 2)
        assert np.all(out >= 0.0)
