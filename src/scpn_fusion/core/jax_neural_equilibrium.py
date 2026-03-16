# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — JAX-Accelerated Neural Equilibrium
"""JAX-accelerated PCA + MLP surrogate for Grad-Shafranov equilibrium.

Provides JAX-traced equivalents of the NumPy SimpleMLP and MinimalPCA from
neural_equilibrium.py. When JAX is available, inference runs JIT-compiled
on CPU or GPU with automatic differentiation via jax.grad.

Key functions:
    jax_mlp_forward       — JIT-compiled MLP forward pass (ReLU + linear)
    jax_pca_inverse       — PCA inverse transform (coefficients → psi flat)
    jax_neural_eq_predict — Full pipeline: features → normalize → MLP → PCA⁻¹ → psi
    load_weights_as_jax   — Load .npz weights into JAX-compatible tuple structure

All functions accept and return JAX arrays when JAX is present, or fall back
to the NumPy implementation in neural_equilibrium.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    import jax
    import jax.numpy as jnp

    _HAS_JAX = True
except ImportError:
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    _HAS_JAX = False


def has_jax() -> bool:
    return _HAS_JAX


# Type alias for JAX weight structure: tuple of (weights_list, biases_list)
# Each is a tuple of arrays for JIT compatibility (no Python lists in traced code).
JAXWeights = tuple[tuple[Any, ...], tuple[Any, ...]]

# Type alias for PCA params: (mean, components)
JAXPCAParams = tuple[Any, Any]

# Type alias for normalisation params: (input_mean, input_std)
JAXNormParams = tuple[Any, Any]


def load_weights_as_jax(
    path: str | Path,
) -> tuple[JAXWeights, JAXPCAParams, JAXNormParams, tuple[int, int]]:
    """Load neural equilibrium .npz weights into JAX-compatible structures.

    Returns
    -------
    (mlp_weights, pca_params, norm_params, grid_shape)
        mlp_weights: ((W0, W1, ...), (b0, b1, ...))
        pca_params: (pca_mean, pca_components)
        norm_params: (input_mean, input_std)
        grid_shape: (nh, nw)
    """
    if not _HAS_JAX:
        raise RuntimeError("JAX not available")

    with np.load(path, allow_pickle=False) as data:
        n_layers = int(data["n_layers"][0])
        ws = tuple(jnp.asarray(data[f"w{i}"], dtype=jnp.float64) for i in range(n_layers))
        bs = tuple(jnp.asarray(data[f"b{i}"], dtype=jnp.float64) for i in range(n_layers))

        pca_mean = jnp.asarray(data["pca_mean"], dtype=jnp.float64)
        pca_components = jnp.asarray(data["pca_components"], dtype=jnp.float64)

        input_mean = jnp.asarray(data["input_mean"], dtype=jnp.float64)
        input_std = jnp.asarray(data["input_std"], dtype=jnp.float64)

        grid_shape = (int(data["grid_nh"][0]), int(data["grid_nw"][0]))

    return (ws, bs), (pca_mean, pca_components), (input_mean, input_std), grid_shape


def numpy_weights_to_jax(
    weights: list[NDArray],
    biases: list[NDArray],
    pca_mean: NDArray,
    pca_components: NDArray,
    input_mean: NDArray,
    input_std: NDArray,
) -> tuple[JAXWeights, JAXPCAParams, JAXNormParams]:
    """Convert NumPy weight arrays to JAX-compatible tuples."""
    if not _HAS_JAX:
        raise RuntimeError("JAX not available")

    ws = tuple(jnp.asarray(w, dtype=jnp.float64) for w in weights)
    bs = tuple(jnp.asarray(b, dtype=jnp.float64) for b in biases)
    pm = jnp.asarray(pca_mean, dtype=jnp.float64)
    pc = jnp.asarray(pca_components, dtype=jnp.float64)
    im = jnp.asarray(input_mean, dtype=jnp.float64)
    isd = jnp.asarray(input_std, dtype=jnp.float64)
    return (ws, bs), (pm, pc), (im, isd)


# ── JAX implementations ──────────────────────────────────────────

if _HAS_JAX:

    @jax.jit
    def _mlp_forward_jax(
        x: jnp.ndarray,
        weights: tuple[jnp.ndarray, ...],
        biases: tuple[jnp.ndarray, ...],
    ) -> jnp.ndarray:
        """MLP forward pass: ReLU hidden layers, linear output."""
        h = x
        n_layers = len(weights)
        for i in range(n_layers):
            h = h @ weights[i] + biases[i]
            if i < n_layers - 1:
                h = jax.nn.relu(h)
        return h

    @jax.jit
    def _pca_inverse_jax(
        coeffs: jnp.ndarray,
        pca_mean: jnp.ndarray,
        pca_components: jnp.ndarray,
    ) -> jnp.ndarray:
        """PCA inverse transform: coefficients → reconstructed vector."""
        return coeffs @ pca_components + pca_mean

    @jax.jit
    def _predict_psi_flat_jax(
        features: jnp.ndarray,
        weights: tuple[jnp.ndarray, ...],
        biases: tuple[jnp.ndarray, ...],
        pca_mean: jnp.ndarray,
        pca_components: jnp.ndarray,
        input_mean: jnp.ndarray,
        input_std: jnp.ndarray,
    ) -> jnp.ndarray:
        """Full pipeline: features → normalize → MLP → PCA⁻¹ → psi_flat.

        Parameters
        ----------
        features : (n_features,) or (batch, n_features)

        Returns
        -------
        psi_flat : (n_grid,) or (batch, n_grid)
        """
        x = (features - input_mean) / input_std
        coeffs = _mlp_forward_jax(x, weights, biases)
        result: jnp.ndarray = _pca_inverse_jax(coeffs, pca_mean, pca_components)
        return result


# ── Public API ────────────────────────────────────────────────────


def jax_mlp_forward(
    x: NDArray,
    mlp_weights: JAXWeights,
) -> NDArray:
    """MLP forward pass with JAX acceleration.

    Parameters
    ----------
    x : input features, shape (batch, n_features) or (n_features,)
    mlp_weights : ((W0, W1, ...), (b0, b1, ...)) JAX arrays
    """
    if not _HAS_JAX:
        raise RuntimeError("JAX not available")
    ws, bs = mlp_weights
    result = _mlp_forward_jax(jnp.asarray(x, dtype=jnp.float64), ws, bs)
    return np.asarray(result)


def jax_pca_inverse(
    coeffs: NDArray,
    pca_params: JAXPCAParams,
) -> NDArray:
    """PCA inverse transform with JAX acceleration."""
    if not _HAS_JAX:
        raise RuntimeError("JAX not available")
    pca_mean, pca_components = pca_params
    result = _pca_inverse_jax(
        jnp.asarray(coeffs, dtype=jnp.float64),
        pca_mean,
        pca_components,
    )
    return np.asarray(result)


def jax_neural_eq_predict(
    features: NDArray,
    mlp_weights: JAXWeights,
    pca_params: JAXPCAParams,
    norm_params: JAXNormParams,
    grid_shape: tuple[int, int] = (129, 129),
) -> NDArray:
    """Predict psi(R,Z) from input features using JAX-accelerated pipeline.

    Parameters
    ----------
    features : shape (n_features,) or (batch, n_features)
    mlp_weights : ((W0, W1, ...), (b0, b1, ...))
    pca_params : (pca_mean, pca_components)
    norm_params : (input_mean, input_std)
    grid_shape : (nh, nw) output grid dimensions

    Returns
    -------
    psi : shape (nh, nw) or (batch, nh, nw)
    """
    if not _HAS_JAX:
        raise RuntimeError("JAX not available")

    ws, bs = mlp_weights
    pca_mean, pca_components = pca_params
    input_mean, input_std = norm_params

    feat = jnp.asarray(features, dtype=jnp.float64)
    single = feat.ndim == 1
    if single:
        feat = feat[jnp.newaxis, :]

    psi_flat = _predict_psi_flat_jax(feat, ws, bs, pca_mean, pca_components, input_mean, input_std)
    psi = np.asarray(psi_flat)

    nh, nw = grid_shape
    if single:
        return psi.reshape(nh, nw)
    return psi.reshape(-1, nh, nw)


def jax_neural_eq_predict_batched(
    features_batch: NDArray,
    mlp_weights: JAXWeights,
    pca_params: JAXPCAParams,
    norm_params: JAXNormParams,
    grid_shape: tuple[int, int] = (129, 129),
) -> NDArray:
    """Batched equilibrium prediction via jax.vmap.

    Parameters
    ----------
    features_batch : shape (batch, n_features)

    Returns
    -------
    psi_batch : shape (batch, nh, nw)
    """
    if not _HAS_JAX:
        raise RuntimeError("JAX not available")

    ws, bs = mlp_weights
    pca_mean, pca_components = pca_params
    input_mean, input_std = norm_params

    @jax.vmap
    def predict_single(feat: jnp.ndarray) -> jnp.ndarray:
        x = (feat - input_mean) / input_std
        coeffs = _mlp_forward_jax(x, ws, bs)
        result: jnp.ndarray = _pca_inverse_jax(coeffs, pca_mean, pca_components)
        return result

    feat_j = jnp.asarray(features_batch, dtype=jnp.float64)
    psi_flat = np.asarray(predict_single(feat_j))

    nh, nw = grid_shape
    return psi_flat.reshape(-1, nh, nw)
