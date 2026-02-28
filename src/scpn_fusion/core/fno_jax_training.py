# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — JAX Fourier Neural Operator (FNO) Training
# © 1998–2026 Miroslav Šotek. All rights reserved.
# ─────────────────────────────────────────────────────────────────────
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad, random
import numpy as np
from pathlib import Path
import time
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── FNO Hyperparameters ──────────────────────────────────────────────
MODES = 16
WIDTH = 64
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCHS = 50

# ── Model Definition ─────────────────────────────────────────────────

def init_fno_params(key, modes, width):
    k1, k2, k3, k4 = random.split(key, 4)
    
    def xavier(k, shape):
        return random.normal(k, shape) * jnp.sqrt(2.0 / (shape[0] + shape[1]))

    params = {
        "w1_real": xavier(k1, (width, width, modes, modes)) * 0.1,
        "w1_imag": xavier(k2, (width, width, modes, modes)) * 0.1,
        "b1": jnp.zeros(width),
        "linear1": xavier(k3, (width, width)),
        
        "fc1": xavier(k4, (width, 128)),
        "fc2": random.normal(random.split(k4)[0], (128, 1)) * 0.01,
    }
    return params

@jit
def fno_layer(x, w_real, w_imag, linear, b):
    # x: (grid, grid, width)
    x_ft = jnp.fft.rfft2(x, axes=(0, 1))
    modes = w_real.shape[2]
    
    # Extract modes
    x_ft_low = x_ft[:modes, :modes, :]
    weights = w_real + 1j * w_imag
    
    # Spectral convolution
    out_ft_low = jnp.einsum("oimj,mji->mjo", weights, x_ft_low)
    
    # Pad back
    out_ft = jnp.zeros_like(x_ft)
    out_ft = out_ft.at[:modes, :modes, :].set(out_ft_low)
    
    x_out = jnp.fft.irfft2(out_ft, axes=(0, 1), s=x.shape[:2])
    return jax.nn.gelu(jnp.dot(x, linear) + b + x_out)

def model_forward(params, x):
    # x: (grid, grid, 1)
    x = jnp.repeat(x, WIDTH, axis=-1)
    x = fno_layer(x, params["w1_real"], params["w1_imag"], params["linear1"], params["b1"])
    
    z = jnp.mean(x, axis=(0, 1))
    z = jax.nn.relu(jnp.dot(z, params["fc1"]))
    z = jnp.dot(z, params["fc2"])
    return z[0]

# ── Training Loop (Adam) ─────────────────────────────────────────────

def mse_loss(params, x_batch, y_batch):
    preds = vmap(lambda x: model_forward(params, x))(x_batch)
    return jnp.mean((preds - y_batch)**2)

@jit
def update_step(params, m, v, x_batch, y_batch, lr, t):
    b1, b2, eps = 0.9, 0.999, 1e-8
    loss, grads = value_and_grad(mse_loss)(params, x_batch, y_batch)
    
    # Clipping
    grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
    
    m = jax.tree_util.tree_map(lambda mi, g: b1 * mi + (1 - b1) * g, m, grads)
    v = jax.tree_util.tree_map(lambda vi, g: b2 * vi + (1 - b2) * (g**2), v, grads)
    
    m_hat = jax.tree_util.tree_map(lambda mi: mi / (1 - b1**t), m)
    v_hat = jax.tree_util.tree_map(lambda vi: vi / (1 - b2**t), v)
    
    new_params = jax.tree_util.tree_map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps),
        params, m_hat, v_hat
    )
    return new_params, m, v, loss

# ── Data Generation (Physics-Informed GENE-like) ──────────────────────

def load_gene_binary(file_path: str):
    """
    Placeholder for actual GENE binary output ingestion.
    Use this to replace synthetic fields with real gyrokinetic data (Roadmap G4).
    """
    raise NotImplementedError("Actual GENE binary ingestion requires the GA-TGLF/GENE tools.")

def generate_gene_like_field(grid_size, regime, key):
    """
    Generates a turbulent field with GENE-like structures (Blobs/Streamers).
    
    NOTE: As of v3.8.2, this is a physics-informed synthetic generator modeling
    GENE spectral character. It is not the direct output of a GENE binary run.
    """
    k = jnp.fft.fftfreq(grid_size)
    kx, ky = jnp.meshgrid(k, k)
    k_mag = jnp.sqrt(kx**2 + ky**2)
    
    if regime == "ITG":
        alpha, anisotropy = 3.5, 1.0
    elif regime == "ETG":
        alpha, anisotropy = 2.2, 4.0
    else: # TEM
        alpha, anisotropy = 2.8, 1.5

    # Safe effective k
    k_eff = jnp.sqrt(kx**2 + (ky/anisotropy)**2)
    k_eff = jnp.maximum(k_eff, 1e-4) # Avoid zero
    
    spectrum = (k_eff ** -alpha) * jnp.exp(-(k_mag**2) / 0.5)
    
    k1, k2 = random.split(key)
    noise = random.normal(k1, (grid_size, grid_size)) + 1j * random.normal(k2, (grid_size, grid_size))
    field_ft = noise * spectrum
    
    # Back to spatial domain
    field = jnp.abs(jnp.fft.ifft2(field_ft))
    
    # Use direct peak normalization + small noise floor
    field = field / (jnp.max(field) + 1e-6)
    return field.reshape(grid_size, grid_size, 1)

def train_fno_jax(
    data_path: str | None = None,
    *,
    n_samples: int = 2000,
    epochs: int = 101,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    save_path: str = "weights/fno_turbulence_jax.npz",
    seed: int = 42,
) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO)

    if epochs < 1:
        raise ValueError("epochs must be >= 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1.")

    data_source = "synthetic"
    if data_path and Path(data_path).exists():
        logger.info(f"Loading REAL GENE/TGLF dataset from {data_path}...")
        with np.load(data_path, allow_pickle=False) as data:
            if "X" not in data or "Y" not in data:
                raise ValueError("Dataset .npz must contain 'X' and 'Y' arrays.")
            x_np = np.asarray(data["X"], dtype=np.float32)
            y_np = np.asarray(data["Y"], dtype=np.float32)
        if x_np.ndim != 4 or x_np.shape[-1] != 1:
            raise ValueError("X must have shape [n_samples, grid, grid, 1].")
        if y_np.ndim > 2:
            raise ValueError("Y must be 1D or 2D with singleton second dimension.")
        y_np = y_np.reshape(-1)
        if x_np.shape[0] != y_np.shape[0]:
            raise ValueError("X and Y must have the same number of samples.")
        if x_np.shape[0] < 1:
            raise ValueError("Dataset must contain at least one sample.")
        if x_np.shape[1] < MODES or x_np.shape[2] < MODES:
            raise ValueError(
                f"Grid must be >= {MODES} in both spatial dims; "
                f"got ({x_np.shape[1]}, {x_np.shape[2]})."
            )
        if not np.all(np.isfinite(x_np)) or not np.all(np.isfinite(y_np)):
            raise ValueError("X and Y must be finite.")
        X = jnp.asarray(x_np)
        Y = jnp.asarray(y_np)
        n_samples_eff = int(x_np.shape[0])
        data_source = "real"
    else:
        logger.info("Generating Physics-Informed GENE-like dataset (G4 Roadmap)...")
        grid_size = 64
        key = random.PRNGKey(seed)
        x_list = []
        y_list = []

        for i in range(n_samples):
            key, subkey = random.split(key)
            regime = ["ITG", "TEM", "ETG"][i % 3]
            field = generate_gene_like_field(grid_size, regime, subkey)
            gamma = 0.35 if regime == "ITG" else (0.25 if regime == "TEM" else 0.15)
            gamma *= (0.9 + 0.2 * random.uniform(subkey))
            x_list.append(field * gamma)
            y_list.append(gamma)

        X = jnp.asarray(x_list)
        Y = jnp.asarray(y_list)
        n_samples_eff = int(n_samples)

    key = random.PRNGKey(seed)
    params = init_fno_params(key, MODES, WIDTH)
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)

    logger.info("Starting training loop (Adam)...")
    t_step = 1
    rng = np.random.default_rng(seed + 17)
    losses: list[float] = []
    batch_n = min(int(batch_size), n_samples_eff)
    for epoch in range(int(epochs)):
        t0 = time.perf_counter()
        idx = rng.permutation(n_samples_eff)
        epoch_losses: list[float] = []
        for i in range(0, n_samples_eff, batch_n):
            batch_idx = idx[i : i + batch_n]
            if batch_idx.size == 0:
                continue
            params, m, v, loss = update_step(
                params,
                m,
                v,
                X[batch_idx],
                Y[batch_idx],
                float(learning_rate),
                t_step,
            )
            epoch_losses.append(float(loss))
            t_step += 1

        epoch_mean_loss = float(np.mean(np.asarray(epoch_losses, dtype=np.float64)))
        losses.append(epoch_mean_loss)
        if epoch % 5 == 0 or epoch == epochs - 1:
            dt = time.perf_counter() - t0
            logger.info(f"Epoch {epoch}: Loss={epoch_mean_loss:.6f} ({dt:.2f}s)")

    logger.info("Training complete. Saving JAX-FNO weights...")
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **{k: np.asarray(v) for k, v in params.items()})
    return {
        "data_source": data_source,
        "n_samples": n_samples_eff,
        "epochs_completed": int(epochs),
        "final_loss": float(losses[-1] if losses else float("nan")),
        "save_path": str(out),
    }

def fno_predict_jit(params, x):
    """JIT-compiled single-sample FNO inference.

    Parameters
    ----------
    params : dict
        FNO parameter dict (as returned by ``init_fno_params`` or loaded from .npz).
    x : jax array, shape (grid, grid, 1)
        Input field.

    Returns
    -------
    float — predicted scalar output.
    """
    return jit(model_forward)(params, x)


def load_fno_params(path="weights/fno_turbulence_jax.npz"):
    """Load FNO params from .npz into JAX arrays."""
    with np.load(path, allow_pickle=False) as data:
        return {k: jnp.asarray(data[k]) for k in data.files}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Path to .npz dataset")
    args = parser.parse_args()
    train_fno_jax(data_path=args.data)
