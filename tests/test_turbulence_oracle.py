# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# SCPN Fusion Core — Turbulence Oracle Tests
from __future__ import annotations

import numpy as np

import scpn_fusion.core.turbulence_oracle as turbulence_oracle
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

    def test_hasegawa_wakatani_uses_fourth_order_hyperviscosity(self):
        hw = DriftWavePhysics(N=16)
        low_k2 = 2.0
        high_k2 = 4.0

        low = hw.spectral_dissipation_multiplier(low_k2)
        high = hw.spectral_dissipation_multiplier(high_k2)

        assert high / low == (high_k2 / low_k2) ** 2

    def test_seeded_initial_conditions_replay_identically(self):
        first = DriftWavePhysics(N=16, seed=1729)
        second = DriftWavePhysics(N=16, seed=1729)

        np.testing.assert_allclose(first.phi_k, second.phi_k, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(first.n_k, second.n_k, rtol=0.0, atol=0.0)

        first_phi, first_n = first.step()
        second_phi, second_n = second.step()

        np.testing.assert_allclose(first_phi, second_phi, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(first_n, second_n, rtol=0.0, atol=0.0)

    def test_distinct_seeds_produce_distinct_initial_spectra(self):
        first = DriftWavePhysics(N=16, seed=1729)
        second = DriftWavePhysics(N=16, seed=1730)

        assert not np.allclose(first.phi_k, second.phi_k, rtol=0.0, atol=0.0)
        assert not np.allclose(first.n_k, second.n_k, rtol=0.0, atol=0.0)

    def test_initial_spectra_reconstruct_real_fields(self):
        hw = DriftWavePhysics(N=16, seed=1729)

        np.testing.assert_allclose(np.fft.ifft2(hw.phi_k).imag, 0.0, atol=1.0e-15)
        np.testing.assert_allclose(np.fft.ifft2(hw.n_k).imag, 0.0, atol=1.0e-15)


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

    def test_degenerate_sparse_reservoir_keeps_finite_weights(self, monkeypatch):
        class ZeroSparseMatrix:
            def toarray(self):
                return np.zeros((4, 4), dtype=float)

        monkeypatch.setattr(turbulence_oracle, "rand", lambda *args, **kwargs: ZeroSparseMatrix())

        esn = OracleESN(input_dim=2, reservoir_size=4, spectral_radius=0.9)

        assert np.all(np.isfinite(esn.W_res))
        assert np.max(np.abs(esn.W_res)) == 0.0

    def test_seeded_reservoir_replay_is_deterministic(self):
        first = OracleESN(input_dim=3, reservoir_size=12, seed=2718)
        second = OracleESN(input_dim=3, reservoir_size=12, seed=2718)

        np.testing.assert_allclose(first.W_in, second.W_in, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(first.W_res, second.W_res, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(first.state, second.state, rtol=0.0, atol=0.0)

    def test_distinct_reservoir_seeds_produce_distinct_weights(self):
        first = OracleESN(input_dim=3, reservoir_size=12, seed=2718)
        second = OracleESN(input_dim=3, reservoir_size=12, seed=2719)

        assert not np.allclose(first.W_in, second.W_in, rtol=0.0, atol=0.0)
        assert not np.allclose(first.W_res, second.W_res, rtol=0.0, atol=0.0)

    def test_sparse_reservoir_weights_are_centred_around_zero(self):
        esn = OracleESN(input_dim=3, reservoir_size=64, seed=2718)
        nonzero_weights = esn.W_res[np.abs(esn.W_res) > 0.0]

        assert np.any(nonzero_weights < 0.0)
        assert np.any(nonzero_weights > 0.0)
