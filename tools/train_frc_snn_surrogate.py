# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FRC SNN Surrogate Tool
"""
Trains an experimental software SNN surrogate for the no-rotation FRC profile.

This script does not provide FPGA export or MHz control validation. It trains a
JAX-accelerated surrogate-gradient model against generated Steinhauer-limit
samples for offline experiments.
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

from scpn_fusion.core.frc_rigid_rotor import RigidRotorFRCInputs, solve_frc_equilibrium

logger = logging.getLogger(__name__)

# ── SNN Hyperparameters (Neuromorphic Configuration) ─────────────────
N_NEURONS = 256
TAU_RC = 0.02
TAU_REF = 0.002
DT = 0.001
STEPS = 10  # Number of spiking timesteps per control cycle

# ── SNN Primitives (Surrogate Gradient) ──────────────────────────────


@jit
def spike_function(v):
    """Return the hard spike indicator used in the surrogate-gradient path."""
    return (v >= 1.0).astype(jnp.float32)


@jit
def surrogate_spike(v):
    """Fast Sigmoid surrogate for backprop through spikes."""
    return jax.nn.sigmoid(10.0 * (v - 1.0))


def lif_step(v, z, x, weights, bias, tau_rc, dt):
    """Single-step LIF dynamics with surrogate gradients."""
    # current J = W*x + b
    j = jnp.dot(weights, x) + bias
    # update voltage
    v_next = v + (dt / tau_rc) * (j - v) - z * 1.0  # reset via subtraction
    v_next = jnp.clip(v_next, 0.0, 2.0)
    # determine spike
    z_next = jax.lax.stop_gradient(spike_function(v_next)) + (
        surrogate_spike(v_next) - jax.lax.stop_gradient(surrogate_spike(v_next))
    )
    return v_next, z_next


# ── Model Training ───────────────────────────────────────────────────


def model_forward_snn(params, x):
    """Unroll SNN for N steps."""
    v = jnp.zeros(N_NEURONS)
    z = jnp.zeros(N_NEURONS)

    # Simple 1-layer SNN encoding
    def body_fun(carry, _):
        v, z = carry
        v, z = lif_step(v, z, x, params["W"], params["b"], TAU_RC, DT)
        return (v, z), z

    (v_final, z_final), spikes = jax.lax.scan(body_fun, (v, z), None, length=STEPS)
    # Decode: mean firing rate * output weights
    out_spikes = jnp.mean(spikes, axis=0)
    return jnp.dot(out_spikes, params["W_out"]) + params["b_out"]


# ── Data Generation ──────────────────────────────────────────────────


def generate_frc_data(n_samples: int, grid_size: int, seed: int = 42):
    """Generate no-rotation FRC training samples for the software SNN surrogate."""
    logger.info("Generating %d FRC samples for SNN training...", n_samples)
    X, Y = [], []
    rng = np.random.default_rng(seed)
    rho_grid = np.linspace(0.0, 0.5, grid_size)

    for _ in range(n_samples):
        inputs = RigidRotorFRCInputs(
            n0=rng.uniform(1e20, 5e20),
            T_i_eV=rng.uniform(5000, 15000),
            T_e_eV=rng.uniform(2000, 10000),
            theta_dot=0.0,
            R_s=rng.uniform(0.15, 0.3),
            B_ext=rng.uniform(2.0, 8.0),
            delta=rng.uniform(0.01, 0.05),
        )
        try:
            state = solve_frc_equilibrium(inputs, rho_grid, solver="rust")
            if not state.converged:
                continue
            X.append(
                [
                    inputs.n0 / 1e20,
                    inputs.T_i_eV / 1000,
                    inputs.T_e_eV / 1000,
                    inputs.R_s,
                    inputs.B_ext,
                    inputs.delta,
                ]
            )
            Y.append(state.B_z)
        except Exception:
            continue
    return np.array(X), np.array(Y)


def main():
    """Train the experimental no-rotation FRC software SNN surrogate."""
    parser = argparse.ArgumentParser(description="Train experimental FRC SNN surrogate")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--out", default="weights/frc_snn_surrogate_v1.npz")
    parser.add_argument(
        "--export-verilog",
        action="store_true",
        help="Fail-closed placeholder for future RTL export",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    X_raw, Y_raw = generate_frc_data(args.samples, args.grid)

    if len(X_raw) < 10:
        return

    # PCA
    from scpn_fusion.core.neural_equilibrium import MinimalPCA

    n_comp = 10
    pca = MinimalPCA(n_components=n_comp)
    Y_latent = pca.fit_transform(Y_raw)

    # Train SNN params
    key = random.PRNGKey(42)
    in_dim = X_raw.shape[1]
    params = {
        "W": random.normal(key, (N_NEURONS, in_dim)) * 0.1,
        "b": jnp.zeros(N_NEURONS),
        "W_out": random.normal(key, (N_NEURONS, n_comp)) * 0.1,
        "b_out": jnp.zeros(n_comp),
    }

    # Simplified training loop
    X_jax = jnp.asarray(X_raw)
    Y_jax = jnp.asarray(Y_latent)

    @jit
    def loss_fn(p, x, y):
        preds = vmap(lambda xi: model_forward_snn(p, xi))(x)
        return jnp.mean((preds - y) ** 2)

    @jit
    def update(p, x, y, lr=1e-2):
        loss, grads = value_and_grad(loss_fn)(p, x, y)
        return jax.tree_util.tree_map(lambda p_i, g_i: p_i - lr * g_i, p, grads), loss

    logger.info("Training software SNN surrogate...")
    for i in range(50):
        params, loss = update(params, X_jax, Y_jax)
        if i % 10 == 0:
            logger.info("Epoch %d: Loss=%.6f", i, loss)

    # Benchmark Latency
    t0 = time.perf_counter()
    x_single = X_jax[0]
    for _ in range(100):
        _ = model_forward_snn(params, x_single).block_until_ready()
    lat = (time.perf_counter() - t0) / 100 * 1e6
    logger.info(
        "Software SNN inference latency on this host: %.2f us (%.1f kHz)", lat, 1000 / lat * 1000
    )

    # Save
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **params, pca_mean=pca.mean_, pca_components=pca.components_)
    if args.export_verilog:
        export_to_verilog(params, "build/frc_snn_core.v")
    logger.info("SNN surrogate saved for software experiments.")


def export_to_verilog(params, out_path):
    """Fail closed until real RTL generation is implemented."""
    raise NotImplementedError("FRC SNN Verilog export is not implemented")


if __name__ == "__main__":
    main()
