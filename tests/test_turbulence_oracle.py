# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Turbulence Oracle Tests
"""Tests for the Hasegawa-Wakatani drift-wave solver and the ESN turbulence oracle."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import scpn_fusion.core.turbulence_oracle as turbulence_oracle
from scpn_fusion.core.turbulence_oracle import DriftWavePhysics, OracleESN, run_turbulence_oracle


class TestDriftWavePhysics:
    """Spectral Hasegawa-Wakatani drift-wave integrator behaviour."""

    def test_init_default(self) -> None:
        """The solver allocates square spectral fields of the requested size."""
        hw = DriftWavePhysics(N=16)
        assert hw.N == 16
        assert hw.phi_k.shape == (16, 16)
        assert hw.n_k.shape == (16, 16)

    def test_bracket_returns_same_shape(self) -> None:
        """The Poisson bracket preserves the spectral grid shape."""
        hw = DriftWavePhysics(N=16)
        result = hw.bracket(hw.phi_k, hw.n_k)
        assert result.shape == (16, 16)

    def test_step_returns_real_fields(self) -> None:
        """One timestep returns finite real potential and density fields."""
        hw = DriftWavePhysics(N=16)
        phi, n = hw.step()
        assert phi.shape == (16, 16)
        assert n.shape == (16, 16)
        assert np.all(np.isfinite(phi))
        assert np.all(np.isfinite(n))

    def test_step_multiple_stable(self) -> None:
        """Repeated stepping stays finite under the hyperviscous closure."""
        hw = DriftWavePhysics(N=16)
        phi = hw.step()[0]
        for _ in range(20):
            phi, _ = hw.step()
        assert np.all(np.isfinite(phi))

    def test_step_rescales_on_amplitude_overflow(self) -> None:
        """A spectrum above the amplitude ceiling is rescaled to stay bounded."""
        hw = DriftWavePhysics(N=16, seed=1729)
        hw.phi_k = hw.phi_k * 1.0e6
        hw.n_k = hw.n_k * 1.0e6
        hw.step()
        max_amp = max(float(np.max(np.abs(hw.phi_k))), float(np.max(np.abs(hw.n_k))))
        assert np.isfinite(max_amp)

    def test_dealiasing_mask_applied(self) -> None:
        """The 2/3 dealiasing mask preserves the low-k modes."""
        hw = DriftWavePhysics(N=16)
        assert hw.mask.shape == (16, 16)
        assert hw.mask[0, 0] == 1.0

    def test_hasegawa_wakatani_uses_fourth_order_hyperviscosity(self) -> None:
        """Spectral dissipation scales as the square of the wavenumber."""
        hw = DriftWavePhysics(N=16)
        low_k2 = np.asarray(2.0, dtype=np.float64)
        high_k2 = np.asarray(4.0, dtype=np.float64)

        low = hw.spectral_dissipation_multiplier(low_k2)
        high = hw.spectral_dissipation_multiplier(high_k2)

        assert high / low == (high_k2 / low_k2) ** 2

    def test_seeded_initial_conditions_replay_identically(self) -> None:
        """A fixed seed reproduces both the initial spectra and the first step."""
        first = DriftWavePhysics(N=16, seed=1729)
        second = DriftWavePhysics(N=16, seed=1729)

        np.testing.assert_allclose(first.phi_k, second.phi_k, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(first.n_k, second.n_k, rtol=0.0, atol=0.0)

        first_phi, first_n = first.step()
        second_phi, second_n = second.step()

        np.testing.assert_allclose(first_phi, second_phi, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(first_n, second_n, rtol=0.0, atol=0.0)

    def test_distinct_seeds_produce_distinct_initial_spectra(self) -> None:
        """Different seeds give different initial spectra."""
        first = DriftWavePhysics(N=16, seed=1729)
        second = DriftWavePhysics(N=16, seed=1730)

        assert not np.allclose(first.phi_k, second.phi_k, rtol=0.0, atol=0.0)
        assert not np.allclose(first.n_k, second.n_k, rtol=0.0, atol=0.0)

    def test_initial_spectra_reconstruct_real_fields(self) -> None:
        """The seeded spectra inverse-transform to purely real fields."""
        hw = DriftWavePhysics(N=16, seed=1729)

        np.testing.assert_allclose(np.fft.ifft2(hw.phi_k).imag, 0.0, atol=1.0e-15)
        np.testing.assert_allclose(np.fft.ifft2(hw.n_k).imag, 0.0, atol=1.0e-15)

    def test_even_grid_nyquist_wavenumber_uses_numpy_fft_convention(self) -> None:
        """The Nyquist wavenumber follows numpy's negative-frequency convention."""
        hw = DriftWavePhysics(N=16, seed=1729)

        assert hw.k[8] == np.fft.fftfreq(16, d=10.0 / (2 * np.pi * 16))[8]
        assert hw.k[8] < 0.0

    def test_zero_mode_has_no_hyperviscous_dissipation(self) -> None:
        """The k=0 mode carries no hyperviscous dissipation."""
        hw = DriftWavePhysics(N=16, seed=1729)

        assert hw.k2[0, 0] == 0.0
        assert hw.spectral_dissipation_multiplier(hw.k2)[0, 0] == 0.0


class TestOracleESN:
    """Echo-state-network turbulence oracle training and prediction."""

    def test_init(self) -> None:
        """Construction sizes the input/output/reservoir weights consistently."""
        esn = OracleESN(input_dim=4, reservoir_size=50)
        assert esn.W_in.shape == (50, 4)
        assert esn.state.shape == (50,)
        assert esn.W_out is None

    def test_train_and_predict(self) -> None:
        """Training fits readout weights and prediction returns a finite horizon."""
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

    def test_predict_before_training_raises(self) -> None:
        """Predicting before training raises a clear runtime error."""
        esn = OracleESN(input_dim=4, reservoir_size=20)
        with pytest.raises(RuntimeError, match="not trained"):
            esn.predict(np.zeros(4, dtype=np.float64), steps=5)

    def test_training_solves_ill_conditioned_ridge_system_directly(self) -> None:
        """The ridge solve matches the explicit normal-equation solution."""
        rng = np.random.default_rng(42)
        esn = OracleESN(input_dim=1, reservoir_size=20, seed=2718)
        esn.W_res[:] = 0.0
        esn.W_in[:, 0] = 1.0 + 1.0e-8 * np.arange(esn.W_in.shape[0])
        inputs = rng.normal(size=(40, 1))
        targets = rng.normal(size=(40, 3))

        reservoir_states = np.tanh(inputs @ esn.W_in.T)
        system = reservoir_states.T @ reservoir_states + 1.0e-4 * np.eye(reservoir_states.shape[1])
        rhs = targets.T @ reservoir_states

        esn.train(inputs, targets)

        assert esn.W_out is not None
        residual = np.linalg.norm(esn.W_out @ system - rhs) / np.linalg.norm(rhs)
        assert residual < 1.0e-13

    def test_spectral_radius_scaled(self) -> None:
        """The reservoir spectral radius is normalised to the requested value."""
        esn = OracleESN(input_dim=2, reservoir_size=20, spectral_radius=0.9)
        eigvals = np.linalg.eigvals(esn.W_res)
        assert np.max(np.abs(eigvals)) <= 1.0

    def test_degenerate_sparse_reservoir_keeps_finite_weights(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fully-zero sparse reservoir leaves finite (zero) weights, no NaNs."""

        class ZeroSparseMatrix:
            """Stub sparse matrix that densifies to an all-zero array."""

            def toarray(self) -> Any:
                """Return a dense all-zero reservoir."""
                return np.zeros((4, 4), dtype=float)

        monkeypatch.setattr(turbulence_oracle, "rand", lambda *args, **kwargs: ZeroSparseMatrix())

        esn = OracleESN(input_dim=2, reservoir_size=4, spectral_radius=0.9)

        assert np.all(np.isfinite(esn.W_res))
        assert np.max(np.abs(esn.W_res)) == 0.0

    def test_seeded_reservoir_replay_is_deterministic(self) -> None:
        """A fixed seed reproduces the input, reservoir, and state weights."""
        first = OracleESN(input_dim=3, reservoir_size=12, seed=2718)
        second = OracleESN(input_dim=3, reservoir_size=12, seed=2718)

        np.testing.assert_allclose(first.W_in, second.W_in, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(first.W_res, second.W_res, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(first.state, second.state, rtol=0.0, atol=0.0)

    def test_distinct_reservoir_seeds_produce_distinct_weights(self) -> None:
        """Different seeds give different reservoir weights."""
        first = OracleESN(input_dim=3, reservoir_size=12, seed=2718)
        second = OracleESN(input_dim=3, reservoir_size=12, seed=2719)

        assert not np.allclose(first.W_in, second.W_in, rtol=0.0, atol=0.0)
        assert not np.allclose(first.W_res, second.W_res, rtol=0.0, atol=0.0)

    def test_sparse_reservoir_weights_are_centred_around_zero(self) -> None:
        """The sparse reservoir weights span both signs around zero."""
        esn = OracleESN(input_dim=3, reservoir_size=64, seed=2718)
        nonzero_weights = esn.W_res[np.abs(esn.W_res) > 0.0]

        assert np.any(nonzero_weights < 0.0)
        assert np.any(nonzero_weights > 0.0)


def test_run_turbulence_oracle_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """The end-to-end demo runs the solver, trains the oracle, and renders safely."""
    import matplotlib.pyplot as plt

    saved: list[str] = []

    def _capture_savefig(path: object, *args: object, **kwargs: object) -> None:
        saved.append(str(path))

    monkeypatch.setattr(plt, "savefig", _capture_savefig)
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    run_turbulence_oracle()

    assert saved == ["Turbulence_Oracle.png"]
