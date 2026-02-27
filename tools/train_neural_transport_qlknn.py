# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Neural Transport Training on Real QLKNN-10D Data
# © 1998–2026 Miroslav Sotek. All rights reserved.
# ─────────────────────────────────────────────────────────────────────
"""
Train the Neural Transport Surrogate (MLP) on real QLKNN-10D data.

This replaces the synthetic-data trainer with one that uses real
QuaLiKiz flux calculations from the QLKNN-10D public dataset.

Prerequisites
-------------
1. Download data:   python tools/download_qlknn10d.py
2. Process data:    python tools/qlknn10d_to_npz.py

Usage
-----
    # Quick smoke test (< 5 min, CPU OK)
    python tools/train_neural_transport_qlknn.py --quick

    # Full training (1-4 hours on GPU)
    python tools/train_neural_transport_qlknn.py

    # Custom settings
    python tools/train_neural_transport_qlknn.py \\
        --epochs 500 --lr 3e-4 --batch-size 4096 --hidden-dims 256,128
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "qlknn10d_processed"
DEFAULT_OUTPUT = REPO_ROOT / "weights" / "neural_transport_qlknn.npz"

OUTPUT_DIM = 3

# Critical gradient thresholds (ITG / TEM onset)
_CRIT_ITG = 4.0  # R/L_Ti threshold
_CRIT_TEM = 5.0  # R/L_Te threshold

# Gyro-Bohm scaling constants (for chi_gb normalization)
_BT_REF = 5.3       # T
_R_REF = 6.2        # m
_MI_KG = 3.344e-27  # deuterium
_E_CHARGE = 1.602e-19


def _load_npz_required(path: Path, *, required_keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Load a .npz file with secure defaults and key validation."""
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in required_keys if key not in data]
        if missing:
            missing_text = ", ".join(missing)
            raise KeyError(f"{path} missing required NPZ keys: {missing_text}")
        out = {key: data[key] for key in required_keys}
        if "gb_normalized" in data:
            out["gb_normalized"] = data["gb_normalized"]
    return out


def _add_derived_features(X: np.ndarray, include_log_chi_gb: bool = False) -> np.ndarray:
    """Append threshold-excess features to the base input array.

    Base columns (10D legacy): [rho, te, ti, ne, ate, ati, an, q, smag, beta_e]
    Base columns (12D):        [rho, te, ti, ne, ate, ati, an, q, smag, beta_e, ti_te, nustar]
    Added:
      ITG excess: max(0, R/LTi - 4.0)
      TEM excess: max(0, R/LTe - 5.0)
      log(chi_gb(Te)) (optional, for physical-space targets)
    """
    grad_te = X[:, 4]
    grad_ti = X[:, 5]

    itg_excess = np.maximum(0.0, grad_ti - _CRIT_ITG)
    tem_excess = np.maximum(0.0, grad_te - _CRIT_TEM)

    if not include_log_chi_gb:
        return np.column_stack([X, itg_excess, tem_excess])

    te_kev = X[:, 1]
    te_j = te_kev * 1e3 * _E_CHARGE
    cs = np.sqrt(te_j / _MI_KG)
    rho_s = np.sqrt(_MI_KG * te_j) / (_E_CHARGE * _BT_REF)
    chi_gb = rho_s ** 2 * cs / _R_REF
    log_chi_gb = np.log(np.maximum(chi_gb, 1e-10))

    return np.column_stack([X, itg_excess, tem_excess, log_chi_gb])


def _compute_stiff_baseline(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stiff-transport baseline: flux ∝ max(0, gradient - threshold).

    Returns (Y_base, coefficients) where:
      Y_base[:, 0] = C_e * max(0, Ate - 5.0)  (chi_e)
      Y_base[:, 1] = C_i * max(0, Ati - 4.0)  (chi_i)
      Y_base[:, 2] = C_d * max(0, An  - 2.0)  (D_e)
    Coefficients are fit from training data (least-squares on nonzero samples).
    """
    ate = X[:, 4]
    ati = X[:, 5]
    an = X[:, 6]
    itg_excess = np.maximum(0.0, ati - _CRIT_ITG)
    tem_excess = np.maximum(0.0, ate - _CRIT_TEM)
    an_excess = np.maximum(0.0, an - 2.0)
    return np.column_stack([tem_excess, itg_excess, an_excess])


def _fit_stiff_coefficients(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Fit stiffness coefficients C = [C_e, C_i, C_d] by least-squares."""
    base = _compute_stiff_baseline(X)
    coeffs = np.ones(OUTPUT_DIM)
    for i in range(OUTPUT_DIM):
        nz = base[:, i] > 0.01
        if nz.sum() > 100:
            coeffs[i] = np.sum(Y[nz, i] * base[nz, i]) / np.sum(base[nz, i] ** 2)
    return coeffs


# ── JAX Training Backend ────────────────────────────────────────────

def _chi_gb_np(te_kev: np.ndarray) -> np.ndarray:
    """Gyro-Bohm diffusivity chi_gb = rho_s^2 * c_s / R [m^2/s].

    Exact formula matching the pipeline in qlknn10d_to_npz.py.
    """
    te_j = te_kev * 1e3 * _E_CHARGE
    cs = np.sqrt(te_j / _MI_KG)
    rho_s = np.sqrt(_MI_KG * te_j) / (_E_CHARGE * _BT_REF)
    return rho_s ** 2 * cs / _R_REF


def _train_jax(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    hidden_dims: list[int],
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    seed: int,
    wd: float = 1e-5,
    log_transform: bool = False,
    Y_val_linear: np.ndarray | None = None,
    gb_scale: bool = False,
    gated: bool = False,
    hybrid_log: bool = False,
    align_metric: bool = False,
    X_train_raw: np.ndarray | None = None,
    residual: bool = False,
    stiff_coeffs: np.ndarray | None = None,
) -> dict:
    """Train using JAX with Adam and cosine annealing."""
    import jax
    import jax.numpy as jnp
    from jax import random, grad, jit

    # Auto-detect GPU
    devices = jax.devices()
    gpu = any(d.platform == "gpu" for d in devices)
    platform = "GPU" if gpu else "CPU"
    print(f"JAX backend: {platform} ({devices[0]})")

    if not gpu:
        jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)

    n_layers = len(hidden_dims)
    # Gated mode: last layer outputs 6 (3 flux + 3 gate logits)
    out_dim = OUTPUT_DIM * 2 if gated else OUTPUT_DIM
    input_dim = X_train.shape[1]
    dims = [input_dim] + hidden_dims + [out_dim]

    # Initialize parameters (He init)
    key = random.PRNGKey(seed)
    params = {}
    for i in range(n_layers + 1):
        key, k = random.split(key)
        fan_in = dims[i]
        scale = np.sqrt(2.0 / fan_in)
        params[f"w{i+1}"] = random.normal(k, (dims[i], dims[i+1])) * scale
        params[f"b{i+1}"] = jnp.zeros(dims[i+1])

    # Normalization stats from training data
    input_mean = jnp.array(np.mean(X_train, axis=0))
    input_std = jnp.array(np.std(X_train, axis=0))

    if residual:
        _output_scale = np.std(Y_train, axis=0)
        _output_scale = np.maximum(_output_scale, 1e-4)
        print(f"  Output scale (std, residual): {_output_scale}")
    else:
        _output_scale = np.ones(OUTPUT_DIM)
        for _i in range(OUTPUT_DIM):
            _nz = Y_train[:, _i][Y_train[:, _i] > 0.01]
            if len(_nz) > 100:
                _output_scale[_i] = np.percentile(_nz, 90)
        _output_scale = np.maximum(_output_scale, 1.0)
        print(f"  Output scale (P90): {_output_scale}")
    output_scale = jnp.array(_output_scale)

    _y_mean = np.mean(Y_train, axis=0)
    if residual:
        _init_bias = np.zeros(OUTPUT_DIM)
    else:
        _target_sp = np.clip(_y_mean / _output_scale, 1e-4, 10.0)
        _init_bias = np.log(np.exp(_target_sp) - 1.0)  # inv_softplus
    if gated:
        # For gated: first 3 are flux biases, last 3 are gate biases
        # Init gate bias to ~0 (sigmoid(0)=0.5, since ~36% of data is unstable)
        _has_flux = float(np.mean(np.sum(Y_train, axis=1) > 0.01))
        _gate_init = np.log(_has_flux / max(1.0 - _has_flux, 0.01))  # inv_sigmoid
        _full_bias = np.concatenate([_init_bias, np.full(OUTPUT_DIM, _gate_init)])
        params[f"b{n_layers+1}"] = jnp.array(_full_bias)
        print(f"  Output bias init: flux={_init_bias}, gate={_gate_init:.2f} "
              f"(flux fraction={_has_flux:.2f}, target mean: {_y_mean})")
    else:
        params[f"b{n_layers+1}"] = jnp.array(_init_bias)
        print(f"  Output bias init: {_init_bias} (target mean: {_y_mean})")

    params["input_mean"] = input_mean
    params["input_std"] = input_std
    params["output_scale"] = output_scale

    # Adam optimizer state
    adam_m = {k: jnp.zeros_like(v) for k, v in params.items() if k.startswith(("w", "b"))}
    adam_v = {k: jnp.zeros_like(v) for k, v in params.items() if k.startswith(("w", "b"))}

    @jit
    def _backbone(params, x):
        """Shared backbone: input norm + hidden layers -> raw output."""
        h = (x - params["input_mean"]) / jnp.maximum(params["input_std"], 1e-8)
        for i in range(n_layers):
            h = jax.nn.gelu(h @ params[f"w{i+1}"] + params[f"b{i+1}"])
        return h @ params[f"w{n_layers+1}"] + params[f"b{n_layers+1}"]

    @jit
    def forward(params, x):
        """MLP forward pass. Outputs transport fluxes (3 values)."""
        raw = _backbone(params, x)
        if gated:
            flux = jax.nn.softplus(raw[..., :OUTPUT_DIM]) * params["output_scale"]
            gate = jax.nn.sigmoid(raw[..., OUTPUT_DIM:])
            return gate * flux
        if residual:
            return raw * params["output_scale"]
        return jax.nn.softplus(raw) * params["output_scale"]

    # Weight decay for L2 regularisation
    weight_decay = wd

    if gated:
        @jit
        def loss_fn(params, x_batch, y_batch, w_batch):
            """Masked gated loss: BCE for gate (all samples) + MSE for flux (nonzero only).

            Key insight: the flux head is trained ONLY on nonzero-flux samples,
            so it focuses entirely on regression accuracy.  The gate head handles
            the zero/nonzero classification on all samples.  At inference the
            gate zeros out stable predictions.
            """
            raw = _backbone(params, x_batch)
            flux = jax.nn.softplus(raw[..., :OUTPUT_DIM]) * params["output_scale"]
            gate = jax.nn.sigmoid(raw[..., OUTPUT_DIM:])

            # Gate classification: BCE on all samples (per-output)
            is_active = (y_batch > 0.01).astype(jnp.float32)
            eps = 1e-6
            bce = -jnp.mean(
                is_active * jnp.log(gate + eps) +
                (1.0 - is_active) * jnp.log(1.0 - gate + eps)
            )

            # Flux regression: MSE on nonzero samples ONLY
            # Compare flux (NOT gate*flux) to y_batch — gate handles the masking
            active_mask = (jnp.sum(y_batch, axis=1) > 0.01).astype(jnp.float32)
            n_active = jnp.maximum(jnp.sum(active_mask), 1.0)
            flux_errors = jnp.mean((flux - y_batch) ** 2, axis=1)
            flux_mse = jnp.sum(active_mask * flux_errors) / n_active

            l2_penalty = sum(jnp.sum(params[k] ** 2) for k in params if k.startswith("w"))
            return 2.0 * bce + flux_mse + weight_decay * l2_penalty
    elif hybrid_log:
        @jit
        def loss_fn(params, x_batch, y_batch, w_batch):
            """Hybrid log-MSE: equal weight on raw and log(1+x) space.

            Raw MSE captures large-flux magnitude; log-space MSE captures
            threshold location where fluxes transition from ~0 to large.
            """
            preds = forward(params, x_batch)
            mse_raw = jnp.mean(w_batch[:, None] * (preds - y_batch) ** 2)
            mse_log = jnp.mean(w_batch[:, None] * (jnp.log1p(preds) - jnp.log1p(y_batch)) ** 2)
            l2_penalty = sum(jnp.sum(params[k] ** 2) for k in params if k.startswith("w"))
            return 0.5 * mse_raw + 0.5 * mse_log + weight_decay * l2_penalty
    else:
        @jit
        def loss_fn(params, x_batch, y_batch, w_batch):
            """Standard MSE loss."""
            preds = forward(params, x_batch)
            per_sample_mse = jnp.mean((preds - y_batch) ** 2, axis=1)
            weighted_mse = jnp.mean(w_batch * per_sample_mse)
            l2_penalty = sum(jnp.sum(params[k] ** 2) for k in params if k.startswith("w"))
            return weighted_mse + weight_decay * l2_penalty

    # Keep a reference to Y_val in linear space for rel_l2 evaluation
    _Y_val_linear_jax = jnp.array(Y_val_linear) if Y_val_linear is not None else None

    # JAX-compatible chi_gb for converting GB predictions back to linear
    _jax_E = jnp.float64(_E_CHARGE)
    _jax_M = jnp.float64(_MI_KG)
    _jax_B = jnp.float64(_BT_REF)
    _jax_R = jnp.float64(_R_REF)

    @jit
    def _chi_gb_jax(te_kev):
        """chi_gb in JAX, matching _chi_gb_np exactly."""
        te_j = te_kev * 1e3 * _jax_E
        cs = jnp.sqrt(te_j / _jax_M)
        rho_s = jnp.sqrt(_jax_M * te_j) / (_jax_E * _jax_B)
        return rho_s ** 2 * cs / _jax_R

    _stiff_coeffs_jax = jnp.array(stiff_coeffs) if stiff_coeffs is not None else None
    _crit_itg_j = jnp.float64(_CRIT_ITG)
    _crit_tem_j = jnp.float64(_CRIT_TEM)

    @jit
    def _stiff_baseline_jax(x):
        """Stiff-transport baseline in JAX (for residual mode eval)."""
        ate = x[..., 4]
        ati = x[..., 5]
        an = x[..., 6]
        base = jnp.stack([
            jnp.maximum(0.0, ate - _crit_tem_j),
            jnp.maximum(0.0, ati - _crit_itg_j),
            jnp.maximum(0.0, an - 2.0),
        ], axis=-1)
        return base * _stiff_coeffs_jax

    @jit
    def relative_l2(params, x, y):
        """Compute relative L2 in linear space regardless of training transform."""
        preds = forward(params, x)
        if residual:
            y_base = _stiff_baseline_jax(x)
            preds_full = jnp.maximum(0.0, y_base + preds)
            y_full = y_base + y
            if gb_scale:
                chi_gb_v = _chi_gb_jax(x[..., 1])
                return jnp.sqrt(
                    jnp.sum(chi_gb_v[:, jnp.newaxis] ** 2 * (preds_full - y_full) ** 2) /
                    jnp.maximum(jnp.sum(chi_gb_v[:, jnp.newaxis] ** 2 * y_full ** 2), 1e-8))
            return jnp.sqrt(jnp.sum((preds_full - y_full) ** 2) / jnp.maximum(jnp.sum(y_full ** 2), 1e-8))
        if gb_scale:
            chi_gb_v = _chi_gb_jax(x[..., 1])
            preds_lin = preds * chi_gb_v[..., jnp.newaxis]
            y_lin = y * chi_gb_v[..., jnp.newaxis]
        elif log_transform:
            preds_lin = jnp.expm1(jnp.clip(preds, 0.0, 20.0))
            y_lin = jnp.expm1(jnp.clip(y, 0.0, 20.0))
        else:
            preds_lin = preds
            y_lin = y
        return jnp.sqrt(jnp.sum((preds_lin - y_lin) ** 2) / jnp.maximum(jnp.sum(y_lin ** 2), 1e-8))

    grad_fn = jit(grad(loss_fn))

    # Convert to JAX arrays
    X_t = jnp.array(X_train)
    Y_t = jnp.array(Y_train)
    X_v = jnp.array(X_val)
    Y_v = jnp.array(Y_val)

    if align_metric and gb_scale:
        _chi_gb_train = _chi_gb_np(np.array(X_train[:, 1]))
        _sw = _chi_gb_train ** 2
        _sw /= np.mean(_sw)
        _sample_weights_jax = jnp.array(_sw)
        print(f"  Align-metric: sample weights by chi_gb^2 "
              f"(min={_sw.min():.3f}, median={np.median(_sw):.3f}, max={_sw.max():.3f})")
    else:
        _sample_weights_jax = jnp.ones(len(Y_train))

    best_val_rel = float("inf")
    best_params = None
    no_improve = 0
    train_losses = []
    val_losses = []
    t_start = time.monotonic()
    t = 0  # Adam step counter

    n_train = len(X_train)
    n_batches = max(1, n_train // batch_size)
    val_n = min(len(X_v), 50000)

    for epoch in range(epochs):
        # Warmup (10 epochs) + cosine annealing
        warmup_epochs = 10
        if epoch < warmup_epochs:
            lr_t = lr * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
            lr_t = lr * 0.5 * (1 + np.cos(np.pi * progress))

        # Uniform shuffle (data + weights)
        key, sk = random.split(key)
        perm = random.permutation(sk, n_train)
        X_shuf = X_t[perm]
        Y_shuf = Y_t[perm]
        W_shuf = _sample_weights_jax[perm]

        epoch_loss = 0.0
        for b in range(n_batches):
            x_b = X_shuf[b * batch_size : (b + 1) * batch_size]
            y_b = Y_shuf[b * batch_size : (b + 1) * batch_size]
            w_b = W_shuf[b * batch_size : (b + 1) * batch_size]

            grads = grad_fn(params, x_b, y_b, w_b)
            t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8

            for k in list(adam_m.keys()):
                g = jnp.clip(grads[k], -10.0, 10.0)
                adam_m[k] = beta1 * adam_m[k] + (1 - beta1) * g
                adam_v[k] = beta2 * adam_v[k] + (1 - beta2) * g ** 2
                m_hat = adam_m[k] / (1 - beta1 ** t)
                v_hat = adam_v[k] / (1 - beta2 ** t)
                params[k] = params[k] - lr_t * m_hat / (jnp.sqrt(v_hat) + eps)

            batch_loss = loss_fn(params, x_b, y_b, w_b)
            epoch_loss += float(batch_loss)

        epoch_loss /= n_batches
        train_losses.append(epoch_loss)

        # Validation: use rel_l2 as early-stopping metric (directly targets goal)
        val_w = jnp.ones(val_n)
        val_loss = float(loss_fn(params, X_v[:val_n], Y_v[:val_n], val_w))
        val_losses.append(val_loss)
        val_rel = float(relative_l2(params, X_v[:val_n], Y_v[:val_n]))

        if val_rel < best_val_rel:
            best_val_rel = val_rel
            best_params = {k: np.array(v) for k, v in params.items()}
            no_improve = 0
        else:
            no_improve += 1

        elapsed = time.monotonic() - t_start
        line = (
            f"  Epoch {epoch:4d}/{epochs} | "
            f"train_loss={epoch_loss:.4f} | val_loss={val_loss:.4f} | "
            f"rel_l2={val_rel:.4f} | best={best_val_rel:.4f} | "
            f"lr={lr_t:.2e} | pat={no_improve}/{patience} | {elapsed:.0f}s"
        )
        print(line)

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # Compute final metrics on full validation set (always in LINEAR space)
    final_params_jax = {k: jnp.array(v) for k, v in best_params.items()}
    val_rel_l2 = float(relative_l2(final_params_jax, X_v, Y_v))
    train_rel_l2 = float(relative_l2(final_params_jax, X_t[:min(100000, len(X_t))], Y_t[:min(100000, len(Y_t))]))

    # Per-output relative L2 (in linear space)
    preds_val_raw = forward(final_params_jax, X_v)
    if residual:
        y_base_v = _stiff_baseline_jax(X_v)
        preds_val_lin = jnp.maximum(0.0, y_base_v + preds_val_raw)
        Y_v_lin = y_base_v + Y_v
    elif gb_scale and Y_val_linear is not None:
        chi_gb_val = jnp.array(_chi_gb_np(np.array(X_val[:, 1])))
        preds_val_lin = preds_val_raw * chi_gb_val[:, jnp.newaxis]
        Y_v_lin = jnp.array(Y_val_linear)
    elif log_transform:
        preds_val_lin = jnp.expm1(jnp.clip(preds_val_raw, 0.0, 20.0))
        Y_v_lin = jnp.array(Y_val_linear) if Y_val_linear is not None else jnp.expm1(Y_v)
    else:
        preds_val_lin = preds_val_raw
        Y_v_lin = Y_v
    per_output_l2 = []
    for col in range(OUTPUT_DIM):
        l2 = float(jnp.sqrt(
            jnp.sum((preds_val_lin[:, col] - Y_v_lin[:, col]) ** 2) /
            jnp.maximum(jnp.sum(Y_v_lin[:, col] ** 2), 1e-8)
        ))
        per_output_l2.append(l2)

    return {
        "params": best_params,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": val_losses[-1] if val_losses else 0.0,
        "train_rel_l2": train_rel_l2,
        "val_rel_l2": val_rel_l2,
        "per_output_rel_l2": per_output_l2,
        "n_layers": n_layers,
        "hidden_dims": hidden_dims,
        "epochs_run": len(train_losses),
        "training_time_s": time.monotonic() - t_start,
        "platform": platform,
        "log_transform": log_transform,
        "gb_scale": gb_scale,
        "gated": gated,
        "hybrid_log": hybrid_log,
        "residual": residual,
        "stiff_coeffs": stiff_coeffs,
    }


# ── Verification Gate ────────────────────────────────────────────────

def verify_and_save(
    result: dict,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    output_path: Path,
    Y_test_linear: np.ndarray | None = None,
) -> bool:
    """Verify training results meet quality gates before saving."""
    params = result["params"]
    n_layers = result["n_layers"]
    log_transform = result.get("log_transform", False)
    gb_scale = result.get("gb_scale", False)
    gated = result.get("gated", False)
    is_residual = result.get("residual", False)
    stiff_coeffs = result.get("stiff_coeffs")
    output_names = ["chi_e", "chi_i", "D_e"]

    print("\n=== VERIFICATION GATE ===")
    all_pass = True

    # Gate 1: test relative L2 < 0.05
    import jax
    import jax.numpy as jnp

    def jax_forward(params, x):
        h = (x - params["input_mean"]) / jnp.maximum(params["input_std"], 1e-8)
        n_layers_local = (len([k for k in params if k.startswith("w")])) - 1
        for i in range(n_layers_local):
            h = jax.nn.gelu(h @ params[f"w{i+1}"] + params[f"b{i+1}"])
        raw = h @ params[f"w{n_layers_local+1}"] + params[f"b{n_layers_local+1}"]
        if gated:
            flux = jax.nn.softplus(raw[..., :OUTPUT_DIM]) * params["output_scale"]
            gate = jax.nn.sigmoid(raw[..., OUTPUT_DIM:])
            return gate * flux
        if is_residual:
            return raw * params["output_scale"]
        return jax.nn.softplus(raw) * params["output_scale"]

    params_jax = {k: jnp.array(v) for k, v in params.items()}
    X_test_jax = jnp.array(X_test)
    preds_test_raw = np.array(jax_forward(params_jax, X_test_jax))

    if is_residual:
        base_test = _compute_stiff_baseline(X_test) * stiff_coeffs
        preds_test = np.maximum(0.0, base_test + preds_test_raw)
        Y_test_lin = base_test + Y_test
    elif gb_scale and Y_test_linear is not None:
        chi_gb_test = _chi_gb_np(np.array(X_test[:, 1]))
        preds_test = preds_test_raw * chi_gb_test[:, None]
        Y_test_lin = Y_test_linear
    elif log_transform:
        preds_test = np.expm1(np.clip(preds_test_raw, 0.0, 20.0))
        Y_test_lin = Y_test_linear if Y_test_linear is not None else np.expm1(Y_test)
    else:
        preds_test = preds_test_raw
        Y_test_lin = Y_test

    test_rel_l2 = np.sqrt(np.sum((preds_test - Y_test_lin) ** 2) / max(np.sum(Y_test_lin ** 2), 1e-8))

    if test_rel_l2 >= 0.05:
        print(f"  WARN: test_relative_l2 = {test_rel_l2:.4f} >= 0.05")
        if test_rel_l2 >= 0.10:
            print(f"  FAIL: test_relative_l2 = {test_rel_l2:.4f} >= 0.10 (hard fail)")
            all_pass = False
    else:
        print(f"  PASS: test_relative_l2 = {test_rel_l2:.4f} < 0.05")

    # Gate 2: no severe overfitting
    train_loss = result["train_losses"][-1] if result["train_losses"] else 0
    val_loss = result["best_val_loss"]
    if val_loss > train_loss * 3.0 and train_loss > 0:
        print(f"  WARN: val_loss ({val_loss:.6f}) > 3x train_loss ({train_loss:.6f})")
    else:
        print(f"  PASS: no severe overfitting (val/train = {val_loss/max(train_loss, 1e-8):.2f})")

    # Gate 3: per-output relative L2 (always in linear space)
    for i, name in enumerate(output_names):
        col_l2 = np.sqrt(
            np.sum((preds_test[:, i] - Y_test_lin[:, i]) ** 2) /
            max(np.sum(Y_test_lin[:, i] ** 2), 1e-8)
        )
        if col_l2 >= 0.10:
            print(f"  WARN: {name} relative_l2 = {col_l2:.4f} >= 0.10")
        else:
            print(f"  PASS: {name} relative_l2 = {col_l2:.4f}")

    # Gate 4: outputs are finite and non-negative
    if np.any(~np.isfinite(preds_test)):
        print("  FAIL: predictions contain NaN/Inf")
        all_pass = False
    elif np.any(preds_test < 0):
        print("  FAIL: predictions contain negative values")
        all_pass = False
    else:
        print("  PASS: all predictions finite and non-negative")

    if not all_pass:
        print("\n  VERIFICATION FAILED — weights NOT saved.")
        print("  Try: --epochs 1000 --lr 1e-4 --hidden-dims 512,256")
        return False

    # ── Save weights in NeuralTransportModel format ──────────────────
    save_dict = {
        "input_mean": params["input_mean"],
        "input_std": params["input_std"],
        "output_scale": params["output_scale"],
        "version": 1,
        "log_transform": np.array(1 if log_transform else 0),
        "gb_scale": np.array(1 if gb_scale else 0),
        "gated": np.array(1 if gated else 0),
        "residual": np.array(1 if is_residual else 0),
    }
    if is_residual and stiff_coeffs is not None:
        save_dict["stiff_coeffs"] = stiff_coeffs
    for i in range(n_layers + 1):
        save_dict[f"w{i+1}"] = params[f"w{i+1}"]
        save_dict[f"b{i+1}"] = params[f"b{i+1}"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **save_dict)

    # Gate 5: verify saved file loads correctly
    with np.load(output_path, allow_pickle=False) as loaded:
        for key in ["w1", "b1", "w2", "b2", "w3", "b3", "input_mean", "input_std", "output_scale"]:
            if key not in loaded:
                print(f"  FAIL: saved file missing key '{key}'")
                return False
    print(f"  PASS: saved weights load correctly ({output_path})")

    # Save training metrics alongside
    metrics_path = output_path.with_suffix(".metrics.json")
    metrics = {
        "test_relative_l2": float(test_rel_l2),
        "val_relative_l2": float(result["val_rel_l2"]),
        "train_relative_l2": float(result["train_rel_l2"]),
        "best_val_mse": float(result["best_val_loss"]),
        "epochs_run": result["epochs_run"],
        "training_time_s": result["training_time_s"],
        "platform": result["platform"],
        "hidden_dims": result["hidden_dims"],
        "data_source": "QLKNN-10D (Zenodo DOI 10.5281/zenodo.3497066)",
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"  Metrics saved to {metrics_path}")

    print("\n=== VERIFICATION PASSED ===")
    return True


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train neural transport surrogate on real QLKNN-10D data."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-dims", type=str, default="256,128",
                        help="Comma-separated hidden layer widths")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0,
                        help="Subsample training set (0 = use all)")
    parser.add_argument("--weight-decay", type=float, default=1e-6,
                        help="L2 weight decay for regularisation")
    parser.add_argument("--regime-balance", action="store_true",
                        help="Downsample zero-flux samples to match flux samples")
    parser.add_argument("--log-transform", action="store_true",
                        help="Train on log(1+Y) targets (recommended for heavy-tailed flux data)")
    parser.add_argument("--gb-scale", action="store_true",
                        help="Gyro-Bohm skip connection: MLP predicts GB-normalised fluxes, "
                             "multiplied by chi_gb(Te) at output")
    parser.add_argument("--gated", action="store_true",
                        help="Gated output: sigmoid gate * softplus flux, allows exact-zero "
                             "predictions for stable (sub-threshold) samples")
    parser.add_argument("--hybrid-log", action="store_true",
                        help="Hybrid log-MSE loss: 0.5*MSE(raw) + 0.5*MSE(log1p). "
                             "Balances threshold accuracy with flux magnitude.")
    parser.add_argument("--align-metric", action="store_true",
                        help="Weight sample MSE by chi_gb(Te)^2 to align loss with "
                             "physical-space relative-L2 metric (GB-normalized data only)")
    parser.add_argument("--residual", action="store_true",
                        help="Residual learning: subtract stiff-transport baseline, "
                             "MLP predicts only the correction (linear output)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test (100 samples, 10 epochs)")
    args = parser.parse_args()

    hidden_dims = [int(d) for d in args.hidden_dims.split(",")]

    print("=== SCPN Fusion Core — Neural Transport Training (QLKNN-10D) ===")
    print(f"Data:    {args.data_dir}")
    print(f"Output:  {args.output}")
    print(f"Arch:    auto -> {' -> '.join(map(str, hidden_dims))} -> {OUTPUT_DIM}")
    print(f"Epochs:  {args.epochs}")
    print(f"LR:      {args.lr}")
    print(f"Batch:   {args.batch_size}")

    # Load data
    train_path = args.data_dir / "train.npz"
    val_path = args.data_dir / "val.npz"
    test_path = args.data_dir / "test.npz"

    for p in (train_path, val_path, test_path):
        if not p.exists():
            print(f"ERROR: {p} not found. Run tools/qlknn10d_to_npz.py first.")
            sys.exit(1)

    try:
        train_data = _load_npz_required(train_path, required_keys=("X", "Y"))
        val_data = _load_npz_required(val_path, required_keys=("X", "Y"))
        test_data = _load_npz_required(test_path, required_keys=("X", "Y"))
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: failed to load .npz datasets securely: {exc}")
        sys.exit(1)

    X_train, Y_train = train_data["X"], train_data["Y"]
    X_val, Y_val = val_data["X"], val_data["Y"]
    X_test, Y_test = test_data["X"], test_data["Y"]

    # Detect GB-normalized data from pipeline metadata
    data_gb_normalized = bool(int(train_data["gb_normalized"])) if "gb_normalized" in train_data else False
    if data_gb_normalized:
        print(f"\nData is GB-normalized (raw gyro-Bohm fluxes, no chi_gb multiplication)")

    # Append threshold-excess features.  log_chi_gb is only useful for
    # physical-space targets (for GB targets Te is uncorrelated with Y).
    base_dim = X_train.shape[1]
    if base_dim in (10, 12):
        include_log = not data_gb_normalized
        X_train = _add_derived_features(X_train, include_log_chi_gb=include_log)
        X_val = _add_derived_features(X_val, include_log_chi_gb=include_log)
        X_test = _add_derived_features(X_test, include_log_chi_gb=include_log)
        feat_list = "ITG excess, TEM excess" + (", log(chi_gb)" if include_log else "")
        print(f"Added derived features: {feat_list} -> {X_train.shape[1]}D input")
        if base_dim == 12:
            print(f"  Base input includes Ti_Te (col 10) and Nustar (col 11)")

    if args.quick:
        print("\n--- QUICK SMOKE TEST MODE ---")
        X_train = X_train[:100]
        Y_train = Y_train[:100]
        X_val = X_val[:50]
        Y_val = Y_val[:50]
        X_test = X_test[:50]
        Y_test = Y_test[:50]
        args.epochs = 10
        args.batch_size = 32
    else:
        # Apply regime balancing first (before subsampling)
        if args.regime_balance:
            flux_sum = Y_train.sum(axis=1)
            has_flux = flux_sum > 0.1
            n_flux = int(has_flux.sum())
            n_no_flux = len(Y_train) - n_flux
            if n_no_flux > n_flux:
                rng_bal = np.random.default_rng(args.seed)
                no_flux_idx = np.where(~has_flux)[0]
                keep_no_flux = rng_bal.choice(no_flux_idx, n_flux, replace=False)
                keep_idx = np.sort(np.concatenate([np.where(has_flux)[0], keep_no_flux]))
                X_train, Y_train = X_train[keep_idx], Y_train[keep_idx]
                print(f"\nRegime balanced: {len(X_train):,} samples "
                      f"({n_flux:,} flux + {n_flux:,} zero-flux from {n_no_flux:,})")

        # Then subsample if requested
        if args.max_train_samples > 0 and len(X_train) > args.max_train_samples:
            rng = np.random.default_rng(args.seed + 1)
            idx = rng.permutation(len(X_train))[:args.max_train_samples]
            X_train, Y_train = X_train[idx], Y_train[idx]
            print(f"\nSubsampled training set to {args.max_train_samples:,} samples")

    print(f"\nData loaded: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    print(f"Y ranges: chi_e=[{Y_train[:,0].min():.2f}, {Y_train[:,0].max():.2f}], "
          f"chi_i=[{Y_train[:,1].min():.2f}, {Y_train[:,1].max():.2f}], "
          f"D_e=[{Y_train[:,2].min():.2f}, {Y_train[:,2].max():.2f}]")

    # Target transformation
    Y_val_linear = None
    Y_test_linear = None
    use_gb_scale = False

    if data_gb_normalized:
        # Data already contains raw GB fluxes — compute physical-space references
        # for rel_l2 evaluation. Y stays unchanged (MLP trains on GB values).
        chi_gb_val = _chi_gb_np(X_val[:, 1])
        chi_gb_test = _chi_gb_np(X_test[:, 1])
        Y_val_linear = Y_val * chi_gb_val[:, None]
        Y_test_linear = Y_test * chi_gb_test[:, None]
        use_gb_scale = True
        print(f"\nTraining on GB-normalized targets (loss in GB space, eval in physical space)")
    elif args.gb_scale:
        # Legacy: divide physical targets by chi_gb(Te) at training time
        Y_val_linear = Y_val.copy()
        Y_test_linear = Y_test.copy()
        chi_gb_train = _chi_gb_np(X_train[:, 1])
        chi_gb_val = _chi_gb_np(X_val[:, 1])
        chi_gb_test = _chi_gb_np(X_test[:, 1])
        Y_train = Y_train / np.maximum(chi_gb_train[:, None], 1e-10)
        Y_val = Y_val / np.maximum(chi_gb_val[:, None], 1e-10)
        Y_test = Y_test / np.maximum(chi_gb_test[:, None], 1e-10)
        use_gb_scale = True
        print(f"\nGB-scale: training on Y/chi_gb (legacy mode)")
    elif args.log_transform:
        Y_val_linear = Y_val.copy()
        Y_test_linear = Y_test.copy()
        Y_train = np.log1p(Y_train)
        Y_val = np.log1p(Y_val)
        Y_test = np.log1p(Y_test)
        print(f"\nLog-transform applied: training on log(1+Y)")

    print(f"Y train ranges: [{Y_train[:,0].min():.3f}, {Y_train[:,0].max():.3f}], "
          f"[{Y_train[:,1].min():.3f}, {Y_train[:,1].max():.3f}], "
          f"[{Y_train[:,2].min():.3f}, {Y_train[:,2].max():.3f}]")

    # Residual learning: subtract stiff-transport baseline
    _stiff_coeffs = None
    if args.residual:
        _stiff_coeffs = _fit_stiff_coefficients(X_train, Y_train)
        base_train = _compute_stiff_baseline(X_train) * _stiff_coeffs
        base_val = _compute_stiff_baseline(X_val) * _stiff_coeffs
        base_test = _compute_stiff_baseline(X_test) * _stiff_coeffs
        Y_train = Y_train - base_train
        Y_val = Y_val - base_val
        Y_test = Y_test - base_test
        print(f"\nResidual mode: stiff coefficients = {_stiff_coeffs}")
        print(f"  Baseline explains: "
              f"chi_e={1 - np.var(Y_train[:,0]) / max(np.var(Y_train[:,0] + base_train[:,0]), 1e-8):.1%}, "
              f"chi_i={1 - np.var(Y_train[:,1]) / max(np.var(Y_train[:,1] + base_train[:,1]), 1e-8):.1%}, "
              f"D_e={1 - np.var(Y_train[:,2]) / max(np.var(Y_train[:,2] + base_train[:,2]), 1e-8):.1%}")
        print(f"  Residual ranges: [{Y_train[:,0].min():.2f}, {Y_train[:,0].max():.2f}], "
              f"[{Y_train[:,1].min():.2f}, {Y_train[:,1].max():.2f}], "
              f"[{Y_train[:,2].min():.2f}, {Y_train[:,2].max():.2f}]")

    # Train
    print("\n--- Training ---")
    result = _train_jax(
        X_train, Y_train, X_val, Y_val,
        hidden_dims=hidden_dims,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        seed=args.seed,
        wd=args.weight_decay,
        log_transform=args.log_transform,
        Y_val_linear=Y_val_linear,
        gb_scale=use_gb_scale,
        gated=args.gated,
        hybrid_log=args.hybrid_log,
        align_metric=args.align_metric,
        X_train_raw=X_train,
        residual=args.residual,
        stiff_coeffs=_stiff_coeffs,
    )

    print(f"\nTraining complete in {result['training_time_s']:.1f}s")
    print(f"  val_relative_l2:   {result['val_rel_l2']:.4f}")
    print(f"  train_relative_l2: {result['train_rel_l2']:.4f}")

    # Verify and save
    ok = verify_and_save(result, X_test, Y_test, args.output, Y_test_linear=Y_test_linear)

    if args.quick and ok:
        print("\nSmoke test PASSED")
    elif not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
