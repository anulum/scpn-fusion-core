# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — JAX Fourier Neural Operator (FNO) Training
# © 1998–2026 Miroslav Šotek. All rights reserved.
# ─────────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad, random
import numpy as np
from pathlib import Path
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── FNO Hyperparameters ──────────────────────────────────────────────
MODES = 16
WIDTH = 64
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
EPOCHS = 100

# ── Model Definition ─────────────────────────────────────────────────

def init_fno_params(key, modes, width):
    k1, k2, k3, k4 = random.split(key, 4)

    # Simple 2-layer FNO
    params = {
        "w1_real": random.normal(k1, (width, width, modes, modes)) * 0.02,
        "w1_imag": random.normal(k2, (width, width, modes, modes)) * 0.02,
        "b1": jnp.zeros(width),
        "linear1": random.normal(k3, (width, width)) * 0.02,

        "fc1": random.normal(k4, (width, 128)) * 0.02,
        "fc2": random.normal(random.split(k4)[0], (128, 1)) * 0.02,
    }
    return params

@jit
def fno_layer(x, w_real, w_imag, linear, b):
    # x: (grid, grid, width)
    # 1. Fourier Transform
    x_ft = jnp.fft.rfft2(x, axes=(0, 1))

    # 2. Spectral Convolution
    # x_ft shape: (grid, grid//2 + 1, width)
    # weights shape: (width, width, modes, modes)

    # We want out_ft[width_out, i, j] = sum_width_in x_ft[width_in, i, j] * w[width_out, width_in, i, j]
    # For simplicity, we use einsum for the spectral multiplication
    # We only take the top modes
    modes = w_real.shape[2]

    # Extract modes from x_ft
    # rfft2 result is (N, N//2 + 1, C)
    x_ft_low = x_ft[:modes, :modes, :]

    # weights shape: (width_out, width_in, modes, modes)
    # x_ft_low shape: (modes, modes, width_in)
    # Target shape: (modes, modes, width_out)
    # Logic: for each (m1, m2), we do Matrix-Vector multiply (W_out, W_in) @ (W_in)
    weights = w_real + 1j * w_imag

    # Correct einsum for (O, I, M1, M2) x (M1, M2, I) -> (M1, M2, O)
    out_ft_low = jnp.einsum("oimj,mji->mjo", weights, x_ft_low)
    # Pad back to full Fourier size

    out_ft = jnp.zeros_like(x_ft)
    out_ft = out_ft.at[:modes, :modes, :].set(out_ft_low)

    # 3. Inverse Fourier
    x_out = jnp.fft.irfft2(out_ft, axes=(0, 1), s=x.shape[:2])

    # 4. Spatial linear (channel mixing)
    return jax.nn.gelu(jnp.dot(x, linear) + b + x_out)
def model_forward(params, x):
    # x: (grid, grid, 1) -> input turbulence field
    # Project to width
    x = jnp.repeat(x, WIDTH, axis=-1)

    # FNO Layer 1
    x = fno_layer(x, params["w1_real"], params["w1_imag"], params["linear1"], params["b1"])

    # Global average pooling for turbulence intensity prediction
    z = jnp.mean(x, axis=(0, 1))

    # MLP head
    z = jax.nn.relu(jnp.dot(z, params["fc1"]))
    z = jnp.dot(z, params["fc2"])
    return z[0]

# ── Training Loop ────────────────────────────────────────────────────

def mse_loss(params, x_batch, y_batch):
    preds = vmap(lambda x: model_forward(params, x))(x_batch)
    return jnp.mean((preds - y_batch)**2)

@jit
def update_step(params, opt_state, x_batch, y_batch, lr):
    loss, grads = value_and_grad(mse_loss)(params, x_batch, y_batch)
    # Simple SGD-like update for brevity
    new_params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
    return new_params, loss

def train_fno_jax():
    logger.info("Generating synthetic turbulence dataset (JAX)...")
    key = random.PRNGKey(42)

    # Generate 1000 samples (vs 60 in NumPy version)
    n_samples = 1000
    grid_size = 64
    X = random.normal(key, (n_samples, grid_size, grid_size, 1))
    # Synthetic target: turbulence intensity correlated with spatial variance
    Y = jnp.std(X, axis=(1, 2, 3)) * 5.0 + random.normal(key, (n_samples,)) * 0.1

    params = init_fno_params(key, MODES, WIDTH)

    logger.info("Starting training loop...")
    for epoch in range(EPOCHS):
        t0 = time.perf_counter()

        # Mini-batching
        idx = np.random.permutation(n_samples)
        epoch_loss = 0
        for i in range(0, n_samples, BATCH_SIZE):
            batch_idx = idx[i:i+BATCH_SIZE]
            params, loss = update_step(params, None, X[batch_idx], Y[batch_idx], LEARNING_RATE)
            epoch_loss += loss

        if epoch % 10 == 0:
            dt = time.perf_counter() - t0
            logger.info(f"Epoch {epoch}: Loss={epoch_loss/n_samples:.6f} ({dt:.2f}s)")

    logger.info("Training complete. Saving JAX-FNO weights...")
    np.savez("weights/fno_turbulence_jax.npz", **params)

if __name__ == "__main__":
    train_fno_jax()

