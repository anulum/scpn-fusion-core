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

def train_fno_jax(data_path=None):
    logging.basicConfig(level=logging.INFO)
    
    if data_path and Path(data_path).exists():
        logger.info(f"Loading REAL GENE/TGLF dataset from {data_path}...")
        data = np.load(data_path)
        X, Y = jnp.array(data["X"]), jnp.array(data["Y"])
        n_samples = X.shape[0]
    else:
        logger.info("Generating Physics-Informed GENE-like dataset (G4 Roadmap)...")
        n_samples = 2000 # Increased for better fidelity
        grid_size = 64
        
        X_list = []
        Y_list = []
        key = random.PRNGKey(42)
        
        for i in range(n_samples):
            key, subkey = random.split(key)
            # Match regimes to TGLF validation order/magnitudes
            regime = ["ITG", "TEM", "ETG"][i % 3]
            field = generate_gene_like_field(grid_size, regime, subkey)

            # Target intensity: ITG (high) -> TEM (mid) -> ETG (low)
            gamma = 0.35 if regime == "ITG" else (0.25 if regime == "TEM" else 0.15)
            # Add some physical noise
            gamma *= (0.9 + 0.2 * random.uniform(subkey))

            # Scale field by gamma to make it learnable from amplitude
            field = field * gamma

            X_list.append(field)

            Y_list.append(gamma)

        X = jnp.array(X_list)
        Y = jnp.array(Y_list)

        key = random.PRNGKey(42)
        params = init_fno_params(key, MODES, WIDTH)
        m = jax.tree_util.tree_map(jnp.zeros_like, params)
        v = jax.tree_util.tree_map(jnp.zeros_like, params)

    logger.info("Starting training loop (Adam)...")
    t_step = 1
    # Increase epochs for G4 fidelity
    for epoch in range(101):
        t0 = time.perf_counter()
        idx = np.random.permutation(n_samples)
        epoch_loss = 0
        for i in range(0, n_samples, BATCH_SIZE):
            batch_idx = idx[i:i+BATCH_SIZE]
            if len(batch_idx) < BATCH_SIZE: continue
            params, m, v, loss = update_step(params, m, v, X[batch_idx], Y[batch_idx], LEARNING_RATE, t_step)
            epoch_loss += loss
            t_step += 1
            
        if epoch % 5 == 0:
            dt = time.perf_counter() - t0
            logger.info(f"Epoch {epoch}: Loss={epoch_loss/(n_samples/BATCH_SIZE):.6f} ({dt:.2f}s)")

    logger.info("Training complete. Saving JAX-FNO weights...")
    np.savez("weights/fno_turbulence_jax.npz", **params)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Path to .npz dataset")
    args = parser.parse_args()
    train_fno_jax(data_path=args.data)
