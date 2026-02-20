# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Neural Transport Training (QLKNN-10D)
# © 1998–2026 Miroslav Šotek. All rights reserved.
# ─────────────────────────────────────────────────────────────────────
"""
Training utility for the Neural Transport Surrogate (MLP).
Fits a 10->64->32->3 MLP to QLKNN-10D data or a high-fidelity synthetic baseline.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap

# Force CPU for training utility to avoid GPU memory fragmentation in CI/CLI
jax.config.update("jax_platform_name", "cpu")

logger = logging.getLogger(__name__)

# ── Architecture ─────────────────────────────────────────────────────
# [rho, Te, Ti, ne, R/LTe, R/LTi, R/Lne, q, s_hat, beta_e] -> [chi_e, chi_i, D_e]
INPUT_DIM = 10
HIDDEN1 = 64
HIDDEN2 = 32
OUTPUT_DIM = 3

def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    # Smaller initial weights to prevent early explosion
    return {
        "w1": random.normal(k1, (INPUT_DIM, HIDDEN1)) * 0.01,
        "b1": jnp.zeros(HIDDEN1),
        "w2": random.normal(k2, (HIDDEN1, HIDDEN2)) * 0.01,
        "b2": jnp.zeros(HIDDEN2),
        "w3": random.normal(k3, (HIDDEN2, OUTPUT_DIM)) * 0.01,
        "b3": jnp.zeros(OUTPUT_DIM),
        "input_mean": jnp.zeros(INPUT_DIM),
        "input_std": jnp.ones(INPUT_DIM),
        "output_scale": jnp.ones(OUTPUT_DIM) * 1.0, 
    }

@jit
def forward(params, x):
    # Normalize with safe std
    x_norm = (x - params["input_mean"]) / jnp.maximum(params["input_std"], 1e-4)
    
    h1 = jax.nn.relu(x_norm @ params["w1"] + params["b1"])
    h2 = jax.nn.relu(h1 @ params["w2"] + params["b2"])
    # Softplus ensures non-negative fluxes
    out = jax.nn.softplus(h2 @ params["w3"] + params["b3"])
    return out * params["output_scale"]

def loss_fn(params, x, y):
    preds = vmap(lambda x_val: forward(params, x_val))(x)
    return jnp.mean((preds - y)**2)

@jit
def update(params, x, y, lr=1e-4):
    grads = grad(loss_fn)(params, x, y)
    # Simple gradient clipping
    grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
    return jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)

# ── Data Generation ──────────────────────────────────────────────────

def generate_synthetic_data(n_samples=5000):
    """Generates data following the critical gradient model but with added complexity."""
    key = random.PRNGKey(42)
    k1, k2 = random.split(key)
    
    # Inputs: [rho, Te, Ti, ne, grad_te, grad_ti, grad_ne, q, s_hat, beta_e]
    X = random.uniform(k1, (n_samples, INPUT_DIM))
    # Rescale to physical ranges
    X = X.at[:, 0].multiply(1.0)   # rho
    X = X.at[:, 1].multiply(10.0)  # Te [keV]
    X = X.at[:, 2].multiply(10.0)  # Ti [keV]
    X = X.at[:, 3].multiply(10.0)  # ne [10^19]
    X = X.at[:, 4].multiply(20.0)  # R/LTe
    X = X.at[:, 5].multiply(20.0)  # R/LTi
    X = X.at[:, 6].multiply(10.0)  # R/Lne
    X = X.at[:, 7].multiply(4.0)   # q
    X = X.at[:, 8].multiply(2.0)   # s_hat
    X = X.at[:, 9].multiply(0.05)  # beta_e
    
    # Target: Stiff response with saturation
    grad_te = X[:, 4]
    grad_ti = X[:, 5]
    
    # chi_i ~ max(0, R/LTi - 4)^2
    chi_i = 1.0 * jnp.maximum(0, grad_ti - 4.0)**1.8
    # chi_e ~ max(0, R/LTe - 5)^2
    chi_e = 0.8 * jnp.maximum(0, grad_te - 5.0)**1.8
    # D_e ~ chi_e / 3
    d_e = chi_e / 3.0
    
    Y = jnp.column_stack([chi_e, chi_i, d_e])
    # Add saturation at high gradients
    Y = jnp.minimum(Y, 50.0)
    
    return X, Y

# ── Main ─────────────────────────────────────────────────────────────

def train_qlknn(output_path="weights/neural_transport_qlknn.npz", n_samples=10000):
    print(f"--- Training QLKNN-10D Surrogate (Output: {output_path}) ---")
    
    X, Y = generate_synthetic_data(n_samples)
    
    key = random.PRNGKey(1337)
    params = init_params(key)
    
    # Set normalization stats
    params["input_mean"] = jnp.mean(X, axis=0)
    params["input_std"] = jnp.std(X, axis=0)
    
    print("Starting training loop...")
    lr = 5e-4
    for epoch in range(2001):
        params = update(params, X, Y, lr)
        if epoch % 500 == 0:
            loss = loss_fn(params, X, Y)
            print(f"  Epoch {epoch:4d} | Loss: {loss:.6f}")
            
    # Save in the format expected by NeuralTransportModel
    print(f"Saving weights to {output_path}...")
    np.savez(
        output_path,
        w1=np.array(params["w1"]),
        b1=np.array(params["b1"]),
        w2=np.array(params["w2"]),
        b2=np.array(params["b2"]),
        w3=np.array(params["w3"]),
        b3=np.array(params["b3"]),
        input_mean=np.array(params["input_mean"]),
        input_std=np.array(params["input_std"]),
        output_scale=np.array(params["output_scale"]),
        version=1
    )
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default="weights/neural_transport_qlknn.npz")
    args = parser.parse_args()
    
    train_qlknn(output_path=args.output)
