# ──────────────────────────────────────────────────────────────────────
# Tests — Neural Transport Surrogate
# ──────────────────────────────────────────────────────────────────────
"""Tests for the neural transport module (fallback + MLP inference)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from scpn_fusion.core.neural_transport import (
    MLPWeights,
    NeuralTransportModel,
    TransportFluxes,
    TransportInputs,
    _mlp_forward,
    _relu,
    _softplus,
    critical_gradient_model,
)


# ── Activation functions ──────────────────────────────────────────────


class TestActivations:
    def test_relu_positive(self):
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(_relu(x), x)

    def test_relu_negative(self):
        x = np.array([-1.0, -2.0, 0.0])
        np.testing.assert_array_equal(_relu(x), [0.0, 0.0, 0.0])

    def test_softplus_positive(self):
        x = np.array([0.0, 1.0, 10.0])
        result = _softplus(x)
        assert all(r > 0 for r in result)
        # softplus(0) = ln(2) ≈ 0.693
        assert abs(result[0] - np.log(2)) < 1e-6

    def test_softplus_large_input(self):
        # For large x, softplus(x) ≈ x
        x = np.array([20.0])
        assert abs(_softplus(x)[0] - 20.0) < 0.1


# ── Critical gradient model ──────────────────────────────────────────


class TestCriticalGradient:
    def test_below_threshold_is_stable(self):
        inp = TransportInputs(grad_ti=2.0, grad_te=2.0)
        fluxes = critical_gradient_model(inp)
        assert fluxes.chi_i == 0.0
        assert fluxes.chi_e == 0.0
        assert fluxes.channel == "stable"

    def test_itg_dominant(self):
        inp = TransportInputs(grad_ti=10.0, grad_te=2.0)
        fluxes = critical_gradient_model(inp)
        assert fluxes.chi_i > 0
        assert fluxes.chi_e == 0.0
        assert fluxes.channel == "ITG"

    def test_tem_dominant(self):
        inp = TransportInputs(grad_ti=2.0, grad_te=10.0)
        fluxes = critical_gradient_model(inp)
        assert fluxes.chi_e > 0
        assert fluxes.chi_i == 0.0
        assert fluxes.channel == "TEM"

    def test_monotone_with_gradient(self):
        """Higher gradient → higher transport."""
        chi_prev = 0.0
        for g in [5.0, 8.0, 12.0, 20.0]:
            inp = TransportInputs(grad_ti=g, grad_te=2.0)
            fluxes = critical_gradient_model(inp)
            assert fluxes.chi_i >= chi_prev
            chi_prev = fluxes.chi_i

    def test_particle_diffusivity_relation(self):
        """D_e follows Ware pinch scaling: chi_e * (0.1 + 0.5*sqrt(eps))."""
        inp = TransportInputs(grad_te=10.0)
        fluxes = critical_gradient_model(inp)
        eps = inp.rho / 3.1
        expected = fluxes.chi_e * (0.1 + 0.5 * np.sqrt(eps))
        assert abs(fluxes.d_e - expected) < 1e-12


# ── MLP forward pass ─────────────────────────────────────────────────


class TestMLPForward:
    @staticmethod
    def _make_identity_weights():
        """Weights that approximately pass through (for testing)."""
        hidden = 16
        return MLPWeights(
            w1=np.random.RandomState(42).randn(10, hidden) * 0.1,
            b1=np.zeros(hidden),
            w2=np.random.RandomState(43).randn(hidden, hidden) * 0.1,
            b2=np.zeros(hidden),
            w3=np.random.RandomState(44).randn(hidden, 3) * 0.1,
            b3=np.zeros(3),
            input_mean=np.zeros(10),
            input_std=np.ones(10),
            output_scale=np.ones(3),
        )

    def test_output_shape_single(self):
        w = self._make_identity_weights()
        x = np.random.randn(10)
        out = _mlp_forward(x, w)
        assert out.shape == (3,)

    def test_output_positive(self):
        """Softplus output should always be positive."""
        w = self._make_identity_weights()
        x = np.random.randn(10) * 5
        out = _mlp_forward(x, w)
        assert all(v > 0 for v in out)

    def test_output_scale(self):
        """output_scale should multiply the final output."""
        w = self._make_identity_weights()
        x = np.random.randn(10)

        out1 = _mlp_forward(x, w)
        w.output_scale = np.array([2.0, 3.0, 4.0])
        out2 = _mlp_forward(x, w)

        np.testing.assert_allclose(out2, out1 * [2.0, 3.0, 4.0], rtol=1e-10)

    def test_batch_input(self):
        """Should handle (batch, 10) shaped input."""
        w = self._make_identity_weights()
        x = np.random.randn(5, 10)
        out = _mlp_forward(x, w)
        assert out.shape == (5, 3)


# ── NeuralTransportModel ─────────────────────────────────────────────


class TestNeuralTransportModel:
    def test_fallback_mode(self):
        """Explicitly testing fallback mode by passing None."""
        model = NeuralTransportModel(weights_path=None)
        # Note: In production, it might load default weights if file exists.
        # But we want to ensure it CAN run in fallback.
        # We use a non-existent path to force fallback for this test.
        model = NeuralTransportModel("/tmp/missing.npz")
        assert not model.is_neural
        inp = TransportInputs(grad_ti=8.0)
        fluxes = model.predict(inp)
        assert fluxes.chi_i > 0

    def test_qlknn_mode_loads_by_default(self):
        """Default constructor should now load the shipped QLKNN weights."""
        model = NeuralTransportModel()
        # Since we just shipped weights/neural_transport_qlknn.npz, this should be True
        assert model.is_neural

    def test_missing_weights_file(self):
        model = NeuralTransportModel("/nonexistent/path.npz")
        assert not model.is_neural

    def test_load_weights_from_file(self):
        """Create a temporary .npz with valid weights and load it."""
        hidden = 8
        rng = np.random.RandomState(42)
        weights = {
            "w1": rng.randn(10, hidden),
            "b1": np.zeros(hidden),
            "w2": rng.randn(hidden, hidden),
            "b2": np.zeros(hidden),
            "w3": rng.randn(hidden, 3),
            "b3": np.zeros(3),
            "input_mean": np.zeros(10),
            "input_std": np.ones(10),
            "output_scale": np.ones(3),
        }

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f.name, **weights)
            model = NeuralTransportModel(f.name)

        assert model.is_neural
        inp = TransportInputs(grad_ti=8.0)
        fluxes = model.predict(inp)
        assert fluxes.chi_e > 0 or fluxes.chi_i > 0

    def test_predict_profile(self):
        """Profile prediction should return correct shapes."""
        model = NeuralTransportModel("/tmp/fallback.npz")  # force fallback mode
        n = 50
        rho = np.linspace(0, 1, n)
        te = 10.0 * (1 - rho ** 2) + 0.1
        ti = te.copy()
        ne = 8.0 * (1 - rho ** 2) + 0.5
        q = 1.0 + 2.0 * rho ** 2
        s_hat = 4.0 * rho

        chi_e, chi_i, d_e = model.predict_profile(rho, te, ti, ne, q, s_hat)

        assert chi_e.shape == (n,)
        assert chi_i.shape == (n,)
        assert d_e.shape == (n,)
        assert all(np.isfinite(chi_e))
        assert all(np.isfinite(chi_i))

    def test_fallback_matches_direct(self):
        """Fallback model.predict() should match critical_gradient_model()."""
        model = NeuralTransportModel("/tmp/fallback.npz") # force fallback
        inp = TransportInputs(grad_ti=7.0, grad_te=6.5)

        via_model = model.predict(inp)
        via_direct = critical_gradient_model(inp)

        assert abs(via_model.chi_i - via_direct.chi_i) < 1e-12
        assert abs(via_model.chi_e - via_direct.chi_e) < 1e-12
        assert via_model.channel == via_direct.channel

    def test_vectorised_profile_matches_pointwise(self):
        """Vectorised predict_profile must match point-by-point fallback."""
        model = NeuralTransportModel("/tmp/fallback.npz")  # force fallback mode
        n = 30
        rho = np.linspace(0.01, 0.99, n)
        te = 8.0 * (1 - rho ** 2) + 0.2
        ti = te * 0.95
        ne = 6.0 * (1 - rho ** 2) + 0.5
        q = 1.0 + 2.5 * rho ** 2
        s_hat = 5.0 * rho

        chi_e, chi_i, d_e = model.predict_profile(rho, te, ti, ne, q, s_hat)

        # Point-by-point via predict()
        for i in range(n):
            # Recompute normalised gradients the same way
            dx_te = np.gradient(te, rho)
            safe_te = np.maximum(np.abs(te), 1e-6)
            grad_te_i = float(np.clip(-6.2 * dx_te[i] / safe_te[i], 0, 50))

            dx_ti = np.gradient(ti, rho)
            safe_ti = np.maximum(np.abs(ti), 1e-6)
            grad_ti_i = float(np.clip(-6.2 * dx_ti[i] / safe_ti[i], 0, 50))

            inp = TransportInputs(grad_te=grad_te_i, grad_ti=grad_ti_i, rho=float(rho[i]))
            fluxes = critical_gradient_model(inp)
            np.testing.assert_allclose(chi_e[i], fluxes.chi_e, rtol=1e-10)
            np.testing.assert_allclose(chi_i[i], fluxes.chi_i, rtol=1e-10)

    def test_weight_versioning_accepted(self):
        """Version=1 weights should load successfully."""
        hidden = 8
        rng = np.random.RandomState(99)
        weights = {
            "version": np.array(1),
            "w1": rng.randn(10, hidden),
            "b1": np.zeros(hidden),
            "w2": rng.randn(hidden, hidden),
            "b2": np.zeros(hidden),
            "w3": rng.randn(hidden, 3),
            "b3": np.zeros(3),
            "input_mean": np.zeros(10),
            "input_std": np.ones(10),
            "output_scale": np.ones(3),
        }
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f.name, **weights)
            model = NeuralTransportModel(f.name)

        assert model.is_neural
        assert model.weights_checksum is not None
        assert len(model.weights_checksum) == 16

    def test_weight_versioning_rejected(self):
        """Version=99 weights should be rejected (fallback)."""
        hidden = 8
        rng = np.random.RandomState(99)
        weights = {
            "version": np.array(99),
            "w1": rng.randn(10, hidden),
            "b1": np.zeros(hidden),
            "w2": rng.randn(hidden, hidden),
            "b2": np.zeros(hidden),
            "w3": rng.randn(hidden, 3),
            "b3": np.zeros(3),
            "input_mean": np.zeros(10),
            "input_std": np.ones(10),
            "output_scale": np.ones(3),
        }
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f.name, **weights)
            model = NeuralTransportModel(f.name)

        assert not model.is_neural

    def test_neural_predict_profile_batched(self):
        """Neural-mode predict_profile should use batched forward pass."""
        hidden = 16
        rng = np.random.RandomState(42)
        weights = {
            "version": np.array(1),
            "w1": rng.randn(10, hidden) * 0.1,
            "b1": np.zeros(hidden),
            "w2": rng.randn(hidden, hidden) * 0.1,
            "b2": np.zeros(hidden),
            "w3": rng.randn(hidden, 3) * 0.1,
            "b3": np.zeros(3),
            "input_mean": np.zeros(10),
            "input_std": np.ones(10),
            "output_scale": np.ones(3),
        }
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f.name, **weights)
            model = NeuralTransportModel(f.name)

        assert model.is_neural
        n = 20
        rho = np.linspace(0.01, 0.99, n)
        te = 5.0 * (1 - rho ** 2) + 0.1
        ti = te.copy()
        ne = 5.0 * (1 - rho ** 2) + 0.5
        q = 1.0 + 2.0 * rho ** 2
        s_hat = 4.0 * rho

        chi_e, chi_i, d_e = model.predict_profile(rho, te, ti, ne, q, s_hat)

        assert chi_e.shape == (n,)
        assert chi_i.shape == (n,)
        assert d_e.shape == (n,)
        assert all(chi_e > 0)  # softplus ensures positivity
        assert all(chi_i > 0)
        assert all(d_e > 0)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            (
                {
                    "rho": np.array([0.0, 0.5, 0.4, 1.0]),
                    "te": np.array([5.0, 4.0, 3.0, 2.0]),
                    "ti": np.array([5.0, 4.0, 3.0, 2.0]),
                    "ne": np.array([5.0, 4.0, 3.0, 2.0]),
                    "q_profile": np.array([1.0, 1.2, 1.4, 1.6]),
                    "s_hat_profile": np.array([0.1, 0.2, 0.3, 0.4]),
                },
                "rho must be strictly increasing",
            ),
            (
                {
                    "rho": np.linspace(0.0, 1.0, 10),
                    "te": np.ones(9),
                    "ti": np.ones(10),
                    "ne": np.ones(10),
                    "q_profile": np.ones(10),
                    "s_hat_profile": np.ones(10),
                },
                "identical length",
            ),
            (
                {
                    "rho": np.linspace(0.0, 1.0, 10),
                    "te": np.array([np.nan] * 10),
                    "ti": np.ones(10),
                    "ne": np.ones(10),
                    "q_profile": np.ones(10),
                    "s_hat_profile": np.ones(10),
                },
                "te must contain finite values",
            ),
        ],
    )
    def test_predict_profile_rejects_invalid_profile_inputs(self, kwargs, match):
        model = NeuralTransportModel("/tmp/fallback.npz")
        with pytest.raises(ValueError, match=match):
            model.predict_profile(**kwargs)

    def test_predict_profile_exposes_gradient_clip_telemetry(self):
        model = NeuralTransportModel("/tmp/fallback.npz")
        rho = np.linspace(0.0, 1.0, 64)
        te = 12.0 * np.exp(-15.0 * rho) + 0.05
        ti = 10.0 * np.exp(-13.0 * rho) + 0.05
        ne = 9.0 * np.exp(-10.0 * rho) + 0.2
        q = 1.0 + 2.5 * rho**2
        s_hat = 4.0 * rho
        model.predict_profile(rho, te, ti, ne, q, s_hat)
        counts = model._last_gradient_clip_counts
        assert set(counts) == {"grad_te", "grad_ti", "grad_ne"}
        assert counts["grad_te"] >= 0
        assert counts["grad_ti"] >= 0
        assert counts["grad_ne"] >= 0
        contract = model._last_profile_contract
        assert contract["n_points"] == 64
        assert contract["rho_min"] == pytest.approx(0.0)
        assert contract["rho_max"] == pytest.approx(1.0)
