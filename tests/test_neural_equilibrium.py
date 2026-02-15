# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Tests for Neural Equilibrium Accelerator
# ──────────────────────────────────────────────────────────────────────
"""
Tests for src/scpn_fusion/core/neural_equilibrium.py.

Unit tests (no SPARC data needed):
  - MinimalPCA round-trip
  - SimpleMLP forward/backward shapes
  - NeuralEquilibriumAccelerator save/load round-trip

Integration tests (require SPARC GEQDSK files):
  - Training from real GEQDSK data
  - Inference produces correct grid shapes
  - Benchmark timing
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scpn_fusion.core.neural_equilibrium import (
    MinimalPCA,
    NeuralEqConfig,
    NeuralEquilibriumAccelerator,
    SimpleMLP,
)

ROOT = Path(__file__).resolve().parents[1]
SPARC_DIR = ROOT / "validation" / "reference_data" / "sparc"


# ── Unit tests: MinimalPCA ────────────────────────────────────────────


class TestMinimalPCA:
    """Verify the pure-NumPy PCA implementation."""

    def test_fit_transform_shape(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((50, 100))
        pca = MinimalPCA(n_components=5)
        Z = pca.fit_transform(X)
        assert Z.shape == (50, 5)

    def test_inverse_recovers_with_all_components(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((30, 20))
        pca = MinimalPCA(n_components=20)
        Z = pca.fit_transform(X)
        X_rec = pca.inverse_transform(Z)
        np.testing.assert_allclose(X_rec, X, atol=1e-10)

    def test_explained_variance_sums_to_one_full(self):
        rng = np.random.default_rng(3)
        X = rng.standard_normal((50, 10))
        pca = MinimalPCA(n_components=10)
        pca.fit(X)
        assert abs(pca.explained_variance_ratio_.sum() - 1.0) < 1e-10

    def test_partial_components_lower_variance(self):
        rng = np.random.default_rng(4)
        X = rng.standard_normal((60, 30))
        pca = MinimalPCA(n_components=5)
        pca.fit(X)
        assert pca.explained_variance_ratio_.sum() < 1.0

    def test_components_orthonormal(self):
        rng = np.random.default_rng(5)
        X = rng.standard_normal((40, 20))
        pca = MinimalPCA(n_components=10)
        pca.fit(X)
        # V^T V should be identity
        VTV = pca.components_ @ pca.components_.T
        np.testing.assert_allclose(VTV, np.eye(10), atol=1e-10)


# ── Unit tests: SimpleMLP ────────────────────────────────────────────


class TestSimpleMLP:
    """Verify the pure-NumPy MLP implementation."""

    def test_forward_shape(self):
        mlp = SimpleMLP([8, 64, 32, 20], seed=0)
        x = np.random.default_rng(0).standard_normal((5, 8))
        y = mlp.forward(x)
        assert y.shape == (5, 20)

    def test_single_sample(self):
        mlp = SimpleMLP([4, 16, 8], seed=1)
        x = np.array([[1.0, 2.0, 3.0, 4.0]])
        y = mlp.forward(x)
        assert y.shape == (1, 8)
        assert np.all(np.isfinite(y))

    def test_predict_equals_forward(self):
        mlp = SimpleMLP([3, 10, 5], seed=2)
        x = np.random.default_rng(2).standard_normal((7, 3))
        np.testing.assert_array_equal(mlp.predict(x), mlp.forward(x))

    def test_different_seeds_different_outputs(self):
        x = np.ones((1, 4))
        y1 = SimpleMLP([4, 8], seed=10).forward(x)
        y2 = SimpleMLP([4, 8], seed=20).forward(x)
        assert not np.allclose(y1, y2)

    def test_relu_kills_negatives(self):
        """Hidden layers use ReLU — negative pre-activations → zero."""
        mlp = SimpleMLP([2, 4, 1], seed=3)
        # Force first layer weights very negative
        mlp.weights[0][:] = -100.0
        mlp.biases[0][:] = -100.0
        x = np.array([[1.0, 1.0]])
        y = mlp.forward(x)
        # With ReLU, hidden is all zeros → output = bias only
        np.testing.assert_allclose(y, mlp.biases[1].reshape(1, -1), atol=1e-12)


# ── Unit tests: NeuralEquilibriumAccelerator ─────────────────────────


class TestAcceleratorUnit:
    """Unit tests that don't require SPARC data."""

    def test_predict_before_train_raises(self):
        accel = NeuralEquilibriumAccelerator()
        with pytest.raises(RuntimeError, match="not trained"):
            accel.predict(np.zeros(8))

    def test_save_load_round_trip(self, tmp_path):
        """Train on synthetic data, save, load, verify predictions match."""
        cfg = NeuralEqConfig(
            n_components=3,
            hidden_sizes=(16, 8),
            n_input_features=4,
            grid_shape=(10, 10),
        )
        accel = NeuralEquilibriumAccelerator(cfg)

        # Manually set up a trained model
        rng = np.random.default_rng(42)
        n_samples = 20
        X = rng.standard_normal((n_samples, 4))
        Y = rng.standard_normal((n_samples, 100))  # 10x10 flattened

        accel._input_mean = X.mean(axis=0)
        accel._input_std = X.std(axis=0)
        accel._input_std[accel._input_std < 1e-10] = 1.0

        accel.pca.fit(Y)
        Y_comp = accel.pca.transform(Y)

        accel.mlp = SimpleMLP([4, 16, 8, 3], seed=42)
        accel.is_trained = True

        # Predict before save
        test_feat = rng.standard_normal(4)
        pred_before = accel.predict(test_feat).copy()

        # Save and load
        path = tmp_path / "test_weights.npz"
        accel.save_weights(path)

        accel2 = NeuralEquilibriumAccelerator()
        accel2.load_weights(path)

        pred_after = accel2.predict(test_feat)
        np.testing.assert_allclose(pred_after, pred_before, atol=1e-12)

    def test_batch_predict_shape(self):
        """Batch prediction should return (batch, nh, nw)."""
        cfg = NeuralEqConfig(
            n_components=3,
            hidden_sizes=(8,),
            n_input_features=4,
            grid_shape=(10, 10),
        )
        accel = NeuralEquilibriumAccelerator(cfg)

        rng = np.random.default_rng(7)
        X = rng.standard_normal((20, 4))
        Y = rng.standard_normal((20, 100))

        accel._input_mean = X.mean(axis=0)
        accel._input_std = np.ones(4)
        accel.pca.fit(Y)
        accel.mlp = SimpleMLP([4, 8, 3], seed=7)
        accel.is_trained = True

        batch_feats = rng.standard_normal((5, 4))
        preds = accel.predict(batch_feats)
        assert preds.shape == (5, 10, 10)

    def test_config_defaults(self):
        cfg = NeuralEqConfig()
        assert cfg.n_components == 20
        assert cfg.hidden_sizes == (128, 64, 32)
        assert cfg.n_input_features == 8
        assert cfg.grid_shape == (129, 129)


# ── Integration tests: SPARC training ────────────────────────────────

sparc_available = SPARC_DIR.exists() and (
    list(SPARC_DIR.glob("*.geqdsk")) or list(SPARC_DIR.glob("*.eqdsk"))
)


@pytest.mark.skipif(not sparc_available, reason="SPARC reference data not available")
class TestSPARCTraining:
    """Integration tests that train on real SPARC GEQDSK files."""

    @pytest.fixture(scope="class")
    def trained_model(self, tmp_path_factory):
        """Train once and share across tests in this class."""
        tmp = tmp_path_factory.mktemp("neq")
        files = sorted(SPARC_DIR.glob("*.geqdsk")) + sorted(SPARC_DIR.glob("*.eqdsk"))

        accel = NeuralEquilibriumAccelerator(
            NeuralEqConfig(n_components=10, hidden_sizes=(32, 16))
        )
        result = accel.train_from_geqdsk(
            files[:3],  # Use subset for speed
            n_perturbations=5,
            seed=42,
        )
        weight_path = tmp / "test_sparc.npz"
        accel.save_weights(weight_path)
        return accel, result, weight_path

    def test_training_returns_valid_result(self, trained_model):
        _, result, _ = trained_model
        assert result.n_samples > 0
        assert result.explained_variance > 0.5  # PCA should retain > 50%
        assert np.isfinite(result.final_loss)
        assert result.train_time_s > 0

    def test_predict_shape_matches_grid(self, trained_model):
        accel, _, _ = trained_model
        from scpn_fusion.core.eqdsk import read_geqdsk

        eq = read_geqdsk(next(SPARC_DIR.glob("*.geqdsk")))
        features = np.array([
            eq.current / 1e6, eq.bcentr,
            eq.rmaxis, eq.zmaxis,
            1.0, 1.0, eq.simag, eq.sibry,
        ])
        psi = accel.predict(features)
        assert psi.ndim == 2
        # Grid shape should match what was trained on
        assert psi.shape == accel.cfg.grid_shape

    def test_self_prediction_finite(self, trained_model):
        """Predicting on the same features used for training should be finite."""
        accel, _, _ = trained_model
        from scpn_fusion.core.eqdsk import read_geqdsk

        eq = read_geqdsk(next(SPARC_DIR.glob("*.geqdsk")))
        features = np.array([
            eq.current / 1e6, eq.bcentr,
            eq.rmaxis, eq.zmaxis,
            1.0, 1.0, eq.simag, eq.sibry,
        ])
        psi = accel.predict(features)
        assert np.all(np.isfinite(psi))

    def test_save_load_preserves_prediction(self, trained_model):
        accel, _, weight_path = trained_model
        from scpn_fusion.core.eqdsk import read_geqdsk

        eq = read_geqdsk(next(SPARC_DIR.glob("*.geqdsk")))
        features = np.array([
            eq.current / 1e6, eq.bcentr,
            eq.rmaxis, eq.zmaxis,
            1.0, 1.0, eq.simag, eq.sibry,
        ])

        pred_before = accel.predict(features).copy()

        accel2 = NeuralEquilibriumAccelerator()
        accel2.load_weights(weight_path)
        pred_after = accel2.predict(features)

        np.testing.assert_allclose(pred_after, pred_before, atol=1e-12)

    def test_benchmark_timing(self, trained_model):
        accel, _, _ = trained_model
        features = np.zeros(accel.cfg.n_input_features)
        bench = accel.benchmark(features, n_runs=10)
        assert bench["mean_ms"] > 0
        assert bench["mean_ms"] < 100  # Should be well under 100ms
        assert np.isfinite(bench["p95_ms"])
