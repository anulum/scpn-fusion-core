# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FRC Rotating Surrogate Training Tool
"""
Trains a JAX-accelerated MLP surrogate for the Rotating Rigid-Rotor FRC BVP.

Generates data using the newly implemented 1D BVP solver,
performs PCA on resulting profiles, and trains a JAX MLP.
This serves as the secondary (surrogate) lane for 30 kHz control loops.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, random, value_and_grad, vmap

from scpn_fusion.core.frc_rigid_rotor import RigidRotorFRCInputs, solve_frc_equilibrium
from scpn_fusion.core.neural_equilibrium import MinimalPCA

logger = logging.getLogger(__name__)


# ── MLP Hyperparameters ──────────────────────────────────────────────
HIDDEN_SIZES = [64, 32, 16]
LEARNING_RATE = 1e-3
GRAD_CLIP = 1.0
EPOCHS = 100
PCA_COMPONENTS_TARGET = 10


# ── Model Definition ─────────────────────────────────────────────────

def init_mlp_params(key, input_dim, hidden_sizes, output_dim):
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
    h = x
    for i, p in enumerate(params):
        h = jnp.dot(h, p["W"]) + p["b"]
        if i < len(params) - 1:
            h = jax.nn.relu(h)
    return h

def mse_loss(params, x_batch, y_batch):
    preds = vmap(lambda x: model_forward(params, x))(x_batch)
    return jnp.mean((preds - y_batch) ** 2)

@jit
def update_step(params, m, v, x_batch, y_batch, lr, t):
    b1, b2, eps = 0.9, 0.999, 1e-8
    loss, grads = value_and_grad(mse_loss)(params, x_batch, y_batch)
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

def generate_frc_data(n_samples: int, grid_size: int, seed: int = 42):
    logger.info("Generating %d FRC samples...", n_samples)
    X = []
    Y = []
    
    rng = np.random.default_rng(seed)
    
    rho_grid = np.linspace(0.0, 0.5, grid_size)
    
    valid_count = 0
    attempts = 0
    while valid_count < n_samples and attempts < n_samples * 10:
        attempts += 1
        if valid_count > 0 and valid_count % 50 == 0:
            logger.info("Generated %d / %d samples (attempts: %d)", valid_count, n_samples, attempts)
            
        n0 = rng.uniform(1e20, 5e20)
        t_i = rng.uniform(5000, 15000)
        t_e = rng.uniform(2000, 10000)
        theta_dot = 0.0
        r_s = rng.uniform(0.15, 0.3)
        b_ext = rng.uniform(2.0, 8.0)
        delta = rng.uniform(0.01, 0.05)
        
        inputs = RigidRotorFRCInputs(
            n0=n0,
            T_i_eV=t_i,
            T_e_eV=t_e,
            theta_dot=theta_dot,
            R_s=r_s,
            B_ext=b_ext,
            delta=delta
        )
        
        try:
            state = solve_frc_equilibrium(inputs, rho_grid, solver="rust", tolerance=1e-8)
            if not state.converged or not np.all(np.isfinite(state.B_z)):
                continue
            features = [n0/1e20, t_i/1000, t_e/1000, theta_dot/1000, r_s, b_ext, delta]
            X.append(features)
            
            # Predict magnetic field B_z profile
            Y.append(state.B_z)
            valid_count += 1
        except Exception as e:
            continue
            
    return np.array(X), np.array(Y)


# ── Training Entry Point ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train FRC surrogate")
    parser.add_argument("--samples", type=int, default=500, help="Number of samples to generate")
    parser.add_argument("--grid", type=int, default=256, help="Grid size for FRC")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs")
    parser.add_argument("--out", default="weights/frc_rotating_surrogate_v1.npz", help="Save path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t_start = time.perf_counter()

    X_raw, Y_raw = generate_frc_data(args.samples, args.grid, args.seed)
    
    n_valid = len(X_raw)
    if n_valid < 10:
        logger.error("Insufficient samples generated. Aborting.")
        return

    logger.info("Successfully generated %d valid samples.", n_valid)

    n_comp = min(n_valid - 1, PCA_COMPONENTS_TARGET)
    pca = MinimalPCA(n_components=n_comp)
    Y_latent = pca.fit_transform(Y_raw)
    explained_var = float(np.sum(pca.explained_variance_ratio_))
    logger.info("PCA: %d components explain %.2f%% variance", n_comp, explained_var * 100)

    x_mean = X_raw.mean(axis=0)
    x_std = X_raw.std(axis=0)
    x_std = np.where(x_std < 1e-10, 1.0, x_std)
    X_norm = (X_raw - x_mean) / x_std

    key = random.PRNGKey(args.seed)
    params = init_mlp_params(key, X_norm.shape[1], HIDDEN_SIZES, n_comp)
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)

    X_jax = jnp.asarray(X_norm)
    Y_jax = jnp.asarray(Y_latent)

    logger.info("Starting JAX training loop...")
    t_step = 1
    for epoch in range(args.epochs):
        params, m, v, loss = update_step(params, m, v, X_jax, Y_jax, LEARNING_RATE, t_step)
        epoch_loss = float(loss)
        t_step += 1
        if epoch % 20 == 0 or epoch == args.epochs - 1:
            logger.info("Epoch %d: Loss=%.6f", epoch, epoch_loss)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    payload = {
        "n_components": np.array([n_comp]),
        "grid_size": np.array([args.grid]),
        "pca_mean": pca.mean_,
        "pca_components": pca.components_,
        "input_mean": x_mean,
        "input_std": x_std,
    }
    for i, p in enumerate(params):
        payload[f"w{i}"] = np.asarray(p["W"])
        payload[f"b{i}"] = np.asarray(p["b"])

    np.savez(out_path, **payload)
    
    t_total = time.perf_counter() - t_start
    logger.info("Retraining complete in %.2f s. Saved to %s", t_total, out_path)

if __name__ == "__main__":
    main()
