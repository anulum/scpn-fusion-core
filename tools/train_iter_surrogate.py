# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — ITER Surrogate Training Tool
"""
Retrains the neural equilibrium surrogate for ITER 6.2 m scenarios.

Generates data by perturbing coil currents in FusionKernel,
performs PCA on resulting Psi fields, and trains a JAX MLP.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, random, value_and_grad, vmap

from scpn_fusion.core.fusion_kernel import FusionKernel
from scpn_fusion.core.neural_equilibrium import MinimalPCA

logger = logging.getLogger(__name__)



class FastNumPyPCA:
    def __init__(self, n_components: int = 20):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ratio_ = None

    def fit_transform(self, X: np.ndarray):
        logger.info("Fitting PCA using NumPy (Gram Matrix Method)...")
        # X is (N, D) = (10000, 16384)
        self.mean_ = X.mean(axis=0)
        logger.info("  Centering data...")
        # Avoid huge intermediate copy: center in-place if possible or chunked
        # But we have 32GB RAM, so one copy of 1.3GB is fine.
        X_c = X - self.mean_
        
        logger.info("  Computing Gram matrix (10000x10000)...")
        G = X_c @ X_c.T
        
        logger.info("  Solving Eigenvalue Decomposition...")
        vals, vecs = np.linalg.eigh(G)
        
        # Sort descending
        idx = np.argsort(vals)[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]
        
        top_vals = vals[:self.n_components]
        top_vecs = vecs[:, :self.n_components]
        
        total_var = np.sum(np.maximum(vals, 0.0))
        self.explained_variance_ratio_ = np.maximum(top_vals, 0.0) / total_var
        
        # Components W = V^T * X_c / sqrt(L)
        inv_sqrt_vals = 1.0 / np.sqrt(np.maximum(top_vals, 1e-15))
        self.components_ = (top_vecs.T @ X_c) * inv_sqrt_vals[:, None]
        
        # Latent projection Z = V * sqrt(L)
        Z = top_vecs * np.sqrt(np.maximum(top_vals, 0.0))
        
        logger.info("  PCA complete.")
        return Z



# ── MLP Hyperparameters ──────────────────────────────────────────────
HIDDEN_SIZES = [256, 128, 64]
LEARNING_RATE = 1e-4
GRAD_CLIP = 0.5
BATCH_SIZE = 32
EPOCHS = 100
PCA_COMPONENTS_TARGET = 20


# ── Model Definition ─────────────────────────────────────────────────


def init_mlp_params(key, input_dim, hidden_sizes, output_dim):
    """Initialise MLP weights with He initialisation."""
    dims = [input_dim] + hidden_sizes + [output_dim]
    params = []
    for i in range(len(dims) - 1):
        key, subkey = random.split(key)
        fan_in, fan_out = dims[i], dims[i + 1]
        std = jnp.sqrt(2.0 / fan_in)
        params.append({
            "W": random.normal(subkey, (fan_in, fan_out)) * std,
            "b": jnp.zeros(fan_out)
        })
    return params


def model_forward(params, x):
    """Forward pass with ReLU activation."""
    h = x
    for i, p in enumerate(params):
        h = jnp.dot(h, p["W"]) + p["b"]
        if i < len(params) - 1:
            h = jax.nn.relu(h)
    return h


def mse_loss(params, x_batch, y_batch):
    """MSE loss for batch."""
    preds = vmap(lambda x: model_forward(params, x))(x_batch)
    return jnp.mean((preds - y_batch) ** 2)


@jit
def update_step(params, m, v, x_batch, y_batch, lr, t):
    """Adam update step with gradient clipping."""
    b1, b2, eps = 0.9, 0.999, 1e-8
    loss, grads = value_and_grad(mse_loss)(params, x_batch, y_batch)

    # Gradient clipping per parameter
    grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -GRAD_CLIP, GRAD_CLIP), grads)

    m = jax.tree_util.tree_map(lambda mi, g: b1 * mi + (1 - b1) * g, m, grads)
    v = jax.tree_util.tree_map(lambda vi, g: b2 * vi + (1 - b2) * (g**2), v, grads)

    m_hat = jax.tree_util.tree_map(lambda mi: mi / (1 - b1**t), m)
    v_hat = jax.tree_util.tree_map(lambda vi: vi / (1 - b2**t), v)

    new_params = jax.tree_util.tree_map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, m_hat, v_hat
    )
    return new_params, m, v, loss


# ── Data Generation ──────────────────────────────────────────────────


def generate_iter_data(n_samples: int, config_path: str | Path, seed: int = 42):
    """Generate training data using FusionKernel by perturbing ITER state."""
    logger.info("Generating %d ITER samples using FusionKernel...", n_samples)
    fk = FusionKernel(config_path)

    # Ensure ITER-like nominals
    fk.cfg["physics"]["B_T"] = 5.3
    fk.cfg["target"] = fk.cfg.get("target", {})
    fk.cfg["target"]["kappa"] = 1.7
    fk.cfg["target"]["R_axis"] = 6.2
    fk.cfg["target"]["Z_axis"] = 0.0

    X = []
    Y = []

    base_currents = [float(c["current"]) for c in fk.cfg["coils"]]
    base_ip = float(fk.cfg["physics"]["plasma_current_target"])

    rng = np.random.default_rng(seed)

    for i in range(n_samples):
        if i > 0 and i % 10 == 0:
            logger.info("Generated %d / %d samples", i, n_samples)

        # Perturb coil currents (+/- 15%)
        for idx, coil in enumerate(fk.cfg["coils"]):
            coil["current"] = base_currents[idx] * rng.uniform(0.85, 1.15)

        # Perturb Ip (+/- 20%)
        ip = base_ip * rng.uniform(0.8, 1.2)
        fk.cfg["physics"]["plasma_current_target"] = ip

        try:
            fk.solve_equilibrium()

            iz, ir, psi_ax = fk._find_magnetic_axis()
            (rx, zx), psi_x = fk.find_x_point(fk.Psi)

            # 12-feature vector (B.1 compatible)
            features = [
                ip / 1e6,
                5.3,  # B_t
                fk.R[ir],  # R_axis
                fk.Z[iz],  # Z_axis
                1.0,  # pprime_scale
                1.0,  # ffprime_scale
                psi_ax,
                psi_x,
                1.7,  # kappa
                0.33, # delta_up
                0.33, # delta_low
                3.0   # q95
            ]

            X.append(features)
            Y.append(fk.Psi.ravel())
        except Exception as e:
            logger.warning("Sample %d failed: %s", i, e)
            continue

    return np.array(X), np.array(Y)


# ── Training Entry Point ─────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Train ITER surrogate")
    parser.add_argument("--config", help="Path to ITER config JSON (required if generating data)")
    parser.add_argument("--data", help="Path to existing .npz dataset")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs")
    parser.add_argument("--out", default="weights/neural_equilibrium_iter_v1.npz", help="Save path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    t_start = time.perf_counter()

    # 1. Load or generate data
    if args.data:
        logger.info("Loading existing dataset from .npy files...")
        X_raw = np.load("/media/anulum/724AA8E84AA8AA75/aaa_God_of_the_Math_Collection/03_CODE/SCPN-FUSION-CORE/data/iter_X.npy")
        Y_raw = np.load("/media/anulum/724AA8E84AA8AA75/aaa_God_of_the_Math_Collection/03_CODE/SCPN-FUSION-CORE/data/iter_Y.npy", mmap_mode='r')
        logger.info("  Loaded X shape: %s, Y shape (mmap): %s", X_raw.shape, Y_raw.shape)
    else:
        if not args.config:
            logger.error("--config is required when generating data.")
            return
        X_raw, Y_raw = generate_iter_data(args.samples, args.config, args.seed)

    n_valid = len(X_raw)
    if n_valid < 2:
        logger.error("Insufficient samples generated. Aborting.")
        return

    # 2. PCA reduction
    n_comp = min(n_valid - 1, PCA_COMPONENTS_TARGET)
    pca = FastNumPyPCA(n_components=n_comp)
    Y_latent = pca.fit_transform(Y_raw)
    explained_var = float(np.sum(pca.explained_variance_ratio_))
    logger.info("PCA: %d components explain %.2f%% variance", n_comp, explained_var * 100)

    # 3. Normalise inputs
    x_mean = X_raw.mean(axis=0)
    x_std = X_raw.std(axis=0)
    x_std = np.where(x_std < 1e-10, 1.0, x_std)
    X_norm = (X_raw - x_mean) / x_std

    
    # 4. Train MLP
    # Normalise Latent Variables for training stability
    z_mean = Y_latent.mean(axis=0)
    z_std = Y_latent.std(axis=0)
    z_std = np.where(z_std < 1e-10, 1.0, z_std)
    Y_norm = (Y_latent - z_mean) / z_std
    
    key = random.PRNGKey(args.seed)
    params = init_mlp_params(key, X_norm.shape[1], HIDDEN_SIZES, n_comp)

    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)

    X_jax = jnp.asarray(X_norm)
    Y_jax = jnp.asarray(Y_norm)

    logger.info("Starting JAX training loop...")
    t_step = 1
    for epoch in range(args.epochs):
        params, m, v, loss = update_step(params, m, v, X_jax, Y_jax, LEARNING_RATE, t_step)
        epoch_loss = float(loss)
        t_step += 1

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            logger.info("Epoch %d: Loss=%.6f", epoch, epoch_loss)

    # 5. Save weights
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    
    payload = {
        "n_components": np.array([n_comp]),
        "grid_nh": np.array([int(np.sqrt(Y_raw.shape[1]))]),
        "grid_nw": np.array([int(np.sqrt(Y_raw.shape[1]))]),
        "n_input_features": np.array([X_raw.shape[1]]),
        "pca_mean": pca.mean_,
        "pca_components": pca.components_,
        "pca_evr": pca.explained_variance_ratio_,
        "input_mean": x_mean,
        "input_std": x_std,
        "latent_mean": z_mean,
        "latent_std": z_std,
        "n_layers": np.array([len(params)]),
    }
    for i, p in enumerate(params):
        payload[f"w{i}"] = np.asarray(p["W"])
        payload[f"b{i}"] = np.asarray(p["b"])

    np.savez(out_path, **payload)

    t_total = time.perf_counter() - t_start
    logger.info("Retraining complete in %.2f s. Saved to %s", t_total, out_path)


if __name__ == "__main__":
    main()
