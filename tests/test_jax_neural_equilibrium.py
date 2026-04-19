# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests: JAX vs NumPy parity, autodiff, batch inference, weight conversion."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.neural_equilibrium import MinimalPCA, SimpleMLP

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from scpn_fusion.core.jax_neural_equilibrium import (
    _mlp_forward_jax,
    _predict_psi_flat_jax,
    has_jax,
    jax_mlp_forward,
    jax_neural_eq_predict,
    jax_neural_eq_predict_batched,
    jax_pca_inverse,
    numpy_weights_to_jax,
)

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def mlp(rng: np.random.Generator) -> SimpleMLP:
    return SimpleMLP([12, 128, 64, 32, 20], seed=42)


@pytest.fixture()
def pca(rng: np.random.Generator) -> MinimalPCA:
    X = rng.standard_normal((100, 65 * 65))
    pca = MinimalPCA(n_components=20)
    pca.fit(X)
    return pca


@pytest.fixture()
def norm_params(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    input_mean = rng.standard_normal(12)
    input_std = np.abs(rng.standard_normal(12)) + 0.1
    return input_mean, input_std


@pytest.fixture()
def jax_params(
    mlp: SimpleMLP,
    pca: MinimalPCA,
    norm_params: tuple[np.ndarray, np.ndarray],
) -> tuple:
    assert pca.mean_ is not None and pca.components_ is not None
    mlp_w, pca_p, norm_p = numpy_weights_to_jax(
        mlp.weights,
        mlp.biases,
        pca.mean_,
        pca.components_,
        norm_params[0],
        norm_params[1],
    )
    return mlp_w, pca_p, norm_p


# ── Tests ─────────────────────────────────────────────────────────────


class TestHasJax:
    def test_jax_available(self) -> None:
        assert has_jax() is True


class TestMLPForwardParity:
    def test_single_sample(
        self, rng: np.random.Generator, mlp: SimpleMLP, jax_params: tuple
    ) -> None:
        x = rng.standard_normal((1, 12))
        np_out = mlp.forward(x)
        jax_out = jax_mlp_forward(x, jax_params[0])
        np.testing.assert_allclose(jax_out, np_out, atol=1e-10)

    def test_batch(self, rng: np.random.Generator, mlp: SimpleMLP, jax_params: tuple) -> None:
        x = rng.standard_normal((8, 12))
        np_out = mlp.forward(x)
        jax_out = jax_mlp_forward(x, jax_params[0])
        np.testing.assert_allclose(jax_out, np_out, atol=1e-10)

    def test_deterministic(self, rng: np.random.Generator, jax_params: tuple) -> None:
        x = rng.standard_normal((3, 12))
        out1 = jax_mlp_forward(x, jax_params[0])
        out2 = jax_mlp_forward(x, jax_params[0])
        np.testing.assert_array_equal(out1, out2)


class TestPCAInverseParity:
    def test_parity(self, rng: np.random.Generator, pca: MinimalPCA, jax_params: tuple) -> None:
        coeffs = rng.standard_normal((5, 20))
        np_out = pca.inverse_transform(coeffs)
        jax_out = jax_pca_inverse(coeffs, jax_params[1])
        np.testing.assert_allclose(jax_out, np_out, atol=1e-10)


class TestFullPipelineParity:
    def test_single_feature(
        self,
        rng: np.random.Generator,
        mlp: SimpleMLP,
        pca: MinimalPCA,
        norm_params: tuple[np.ndarray, np.ndarray],
        jax_params: tuple,
    ) -> None:
        features = rng.standard_normal(12)
        # NumPy pipeline
        x_norm = (features[np.newaxis, :] - norm_params[0]) / norm_params[1]
        coeffs = mlp.forward(x_norm)
        psi_np = pca.inverse_transform(coeffs).reshape(65, 65)
        # JAX pipeline
        psi_jax = jax_neural_eq_predict(
            features, jax_params[0], jax_params[1], jax_params[2], grid_shape=(65, 65)
        )
        np.testing.assert_allclose(psi_jax, psi_np, atol=1e-10)

    def test_batch_features(
        self,
        rng: np.random.Generator,
        mlp: SimpleMLP,
        pca: MinimalPCA,
        norm_params: tuple[np.ndarray, np.ndarray],
        jax_params: tuple,
    ) -> None:
        features = rng.standard_normal((4, 12))
        # NumPy pipeline
        x_norm = (features - norm_params[0]) / norm_params[1]
        coeffs = mlp.forward(x_norm)
        psi_np = pca.inverse_transform(coeffs).reshape(-1, 65, 65)
        # JAX pipeline
        psi_jax = jax_neural_eq_predict(
            features, jax_params[0], jax_params[1], jax_params[2], grid_shape=(65, 65)
        )
        np.testing.assert_allclose(psi_jax, psi_np, atol=1e-10)


class TestBatchedVmap:
    def test_vmap_matches_loop(self, rng: np.random.Generator, jax_params: tuple) -> None:
        features = rng.standard_normal((6, 12))
        # Loop version
        results = []
        for i in range(6):
            psi = jax_neural_eq_predict(
                features[i], jax_params[0], jax_params[1], jax_params[2], grid_shape=(65, 65)
            )
            results.append(psi)
        psi_loop = np.stack(results)
        # vmap version
        psi_vmap = jax_neural_eq_predict_batched(
            features, jax_params[0], jax_params[1], jax_params[2], grid_shape=(65, 65)
        )
        np.testing.assert_allclose(psi_vmap, psi_loop, atol=1e-10)

    def test_output_shape(self, rng: np.random.Generator, jax_params: tuple) -> None:
        features = rng.standard_normal((10, 12))
        psi = jax_neural_eq_predict_batched(
            features, jax_params[0], jax_params[1], jax_params[2], grid_shape=(65, 65)
        )
        assert psi.shape == (10, 65, 65)


class TestAutodiff:
    def test_grad_through_mlp(self, jax_params: tuple) -> None:
        """jax.grad produces finite, nonzero gradients through MLP."""
        ws, bs = jax_params[0]

        def loss_fn(x: jnp.ndarray) -> jnp.ndarray:
            out = _mlp_forward_jax(x, ws, bs)
            return jnp.sum(out**2)

        x = jnp.ones(12, dtype=jnp.float64)
        grad = jax.grad(loss_fn)(x)
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.any(grad != 0.0)

    def test_grad_through_full_pipeline(self, jax_params: tuple) -> None:
        """jax.grad produces finite, nonzero gradients through full predict."""
        ws, bs = jax_params[0]
        pca_mean, pca_components = jax_params[1]
        input_mean, input_std = jax_params[2]

        def loss_fn(features: jnp.ndarray) -> jnp.ndarray:
            psi = _predict_psi_flat_jax(
                features[jnp.newaxis, :], ws, bs, pca_mean, pca_components, input_mean, input_std
            )
            return jnp.sum(psi**2)

        features = jnp.ones(12, dtype=jnp.float64)
        grad = jax.grad(loss_fn)(features)
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.any(grad != 0.0)
        assert grad.shape == (12,)

    def test_grad_sensitivity_varies_per_input(self, jax_params: tuple) -> None:
        """Different input features produce different gradient magnitudes."""
        ws, bs = jax_params[0]
        pca_mean, pca_components = jax_params[1]
        input_mean, input_std = jax_params[2]

        def loss_fn(features: jnp.ndarray) -> jnp.ndarray:
            psi = _predict_psi_flat_jax(
                features[jnp.newaxis, :], ws, bs, pca_mean, pca_components, input_mean, input_std
            )
            return jnp.sum(psi**2)

        f1 = jnp.ones(12, dtype=jnp.float64)
        f2 = jnp.ones(12, dtype=jnp.float64) * 2.0
        g1 = jax.grad(loss_fn)(f1)
        g2 = jax.grad(loss_fn)(f2)
        # Gradients should differ for different inputs
        assert not jnp.allclose(g1, g2)


class TestWeightConversion:
    def test_roundtrip_shapes(
        self,
        mlp: SimpleMLP,
        pca: MinimalPCA,
        norm_params: tuple[np.ndarray, np.ndarray],
    ) -> None:
        assert pca.mean_ is not None and pca.components_ is not None
        mlp_w, pca_p, norm_p = numpy_weights_to_jax(
            mlp.weights,
            mlp.biases,
            pca.mean_,
            pca.components_,
            norm_params[0],
            norm_params[1],
        )
        ws, bs = mlp_w
        assert len(ws) == len(mlp.weights)
        assert len(bs) == len(mlp.biases)
        for i in range(len(ws)):
            assert ws[i].shape == mlp.weights[i].shape
            assert bs[i].shape == mlp.biases[i].shape
