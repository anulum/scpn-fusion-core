# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# SCPN Fusion Core — Turbulence Oracle Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.turbulence_oracle import DriftWavePhysics, OracleESN


class TestDriftWavePhysics:
    def test_init_default(self):
        hw = DriftWavePhysics(N=16)
        assert hw.N == 16
        assert hw.phi_k.shape == (16, 16)
        assert hw.n_k.shape == (16, 16)

    def test_bracket_returns_same_shape(self):
        hw = DriftWavePhysics(N=16)
        result = hw.bracket(hw.phi_k, hw.n_k)
        assert result.shape == (16, 16)

    def test_step_returns_real_fields(self):
        hw = DriftWavePhysics(N=16)
        phi, n = hw.step()
        assert phi.shape == (16, 16)
        assert n.shape == (16, 16)
        assert np.all(np.isfinite(phi))
        assert np.all(np.isfinite(n))

    def test_step_multiple_stable(self):
        hw = DriftWavePhysics(N=16)
        for _ in range(20):
            phi, n = hw.step()
        assert np.all(np.isfinite(phi))

    def test_dealiasing_mask_applied(self):
        hw = DriftWavePhysics(N=16)
        # High-k modes should be zeroed by mask
        assert hw.mask.shape == (16, 16)
        assert hw.mask[0, 0] == 1.0  # Low-k preserved


class TestOracleESN:
    def test_init(self):
        esn = OracleESN(input_dim=4, reservoir_size=50)
        assert esn.W_in.shape == (50, 4)
        assert esn.state.shape == (50,)
        assert esn.W_out is None

    def test_train_and_predict(self):
        rng = np.random.default_rng(42)
        n_steps = 100
        dim = 4
        data = rng.normal(size=(n_steps, dim))
        inputs = data[:-1]
        targets = data[1:]

        esn = OracleESN(input_dim=dim, reservoir_size=30)
        esn.train(inputs, targets)
        assert esn.W_out is not None
        assert esn.W_out.shape == (dim, 30)

        preds = esn.predict(data[-1], steps=10)
        assert preds.shape == (10, dim)
        assert np.all(np.isfinite(preds))

    def test_spectral_radius_scaled(self):
        esn = OracleESN(input_dim=2, reservoir_size=20, spectral_radius=0.9)
        eigvals = np.linalg.eigvals(esn.W_res)
        assert np.max(np.abs(eigvals)) <= 1.0
