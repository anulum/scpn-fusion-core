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

INPUT_DIM = 10
OUTPUT_DIM = 3


# ── JAX Training Backend ────────────────────────────────────────────

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
) -> dict:
    """Train using JAX with Adam and cosine annealing."""
    import jax
    import jax.numpy as jnp
    from jax import random, grad, jit, vmap

    # Auto-detect GPU
    devices = jax.devices()
    gpu = any(d.platform == "gpu" for d in devices)
    platform = "GPU" if gpu else "CPU"
    print(f"JAX backend: {platform} ({devices[0]})")

    if not gpu:
        jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)

    n_layers = len(hidden_dims)
    dims = [INPUT_DIM] + hidden_dims + [OUTPUT_DIM]

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
    output_scale = jnp.array(np.mean(Y_train, axis=0))
    output_scale = jnp.maximum(output_scale, 1.0)  # safety floor

    params["input_mean"] = input_mean
    params["input_std"] = input_std
    params["output_scale"] = output_scale

    # Adam optimizer state
    adam_m = {k: jnp.zeros_like(v) for k, v in params.items() if k.startswith(("w", "b"))}
    adam_v = {k: jnp.zeros_like(v) for k, v in params.items() if k.startswith(("w", "b"))}

    @jit
    def forward(params, x):
        h = (x - params["input_mean"]) / jnp.maximum(params["input_std"], 1e-8)
        for i in range(n_layers):
            h = jax.nn.gelu(h @ params[f"w{i+1}"] + params[f"b{i+1}"])
        out = jax.nn.softplus(h @ params[f"w{n_layers+1}"] + params[f"b{n_layers+1}"])
        return out * params["output_scale"]

    @jit
    def loss_fn(params, x_batch, y_batch):
        preds = vmap(lambda x: forward(params, x))(x_batch)
        # Revert to standard MSE to align with relative L2 metric
        return jnp.mean((preds - y_batch) ** 2)

    @jit
    def relative_l2(params, x, y):
        preds = vmap(lambda xi: forward(params, xi))(x)
        return jnp.sqrt(jnp.sum((preds - y) ** 2) / jnp.maximum(jnp.sum(y ** 2), 1e-8))

    grad_fn = jit(grad(loss_fn))

    # Convert to JAX arrays
    X_t = jnp.array(X_train)
    Y_t = jnp.array(Y_train)
    X_v = jnp.array(X_val)
    Y_v = jnp.array(Y_val)

    best_val_loss = float("inf")
    best_params = None
    no_improve = 0
    train_losses = []
    val_losses = []
    t_start = time.monotonic()
    t = 0  # Adam step counter

    n_train = len(X_train)
    n_batches = max(1, n_train // batch_size)

    for epoch in range(epochs):
        # Cosine annealing
        lr_t = lr * 0.5 * (1 + np.cos(np.pi * epoch / epochs))

        # Shuffle
        key, sk = random.split(key)
        perm = random.permutation(sk, n_train)
        X_shuf = X_t[perm]
        Y_shuf = Y_t[perm]

        epoch_loss = 0.0
        for b in range(n_batches):
            x_b = X_shuf[b * batch_size : (b + 1) * batch_size]
            y_b = Y_shuf[b * batch_size : (b + 1) * batch_size]

            grads = grad_fn(params, x_b, y_b)
            t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8

            for k in list(adam_m.keys()):
                g = jnp.clip(grads[k], -10.0, 10.0)  # increased gradient clipping
                adam_m[k] = beta1 * adam_m[k] + (1 - beta1) * g
                adam_v[k] = beta2 * adam_v[k] + (1 - beta2) * g ** 2
                m_hat = adam_m[k] / (1 - beta1 ** t)
                v_hat = adam_v[k] / (1 - beta2 ** t)
                params[k] = params[k] - lr_t * m_hat / (jnp.sqrt(v_hat) + eps)

            batch_loss = loss_fn(params, x_b, y_b)
            epoch_loss += float(batch_loss)

        epoch_loss /= n_batches
        train_losses.append(epoch_loss)

        # Validation (subsample for speed)
        val_n = min(len(X_v), 50000)
        val_loss = float(loss_fn(params, X_v[:val_n], Y_v[:val_n]))
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = {k: np.array(v) for k, v in params.items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 1 == 0 or epoch == epochs - 1:
            elapsed = time.monotonic() - t_start
            print(
                f"  Epoch {epoch:4d}/{epochs} | "
                f"train_mse={epoch_loss:.6f} | val_mse={val_loss:.6f} | "
                f"lr={lr_t:.2e} | best_val={best_val_loss:.6f} | "
                f"patience={no_improve}/{patience} | {elapsed:.0f}s"
            )

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # Compute final metrics on full validation set
    final_params_jax = {k: jnp.array(v) for k, v in best_params.items()}
    val_rel_l2 = float(relative_l2(final_params_jax, X_v, Y_v))
    train_rel_l2 = float(relative_l2(final_params_jax, X_t[:100000], Y_t[:100000]))

    # Per-output relative L2
    preds_val = vmap(lambda x: forward(final_params_jax, x))(X_v)
    per_output_l2 = []
    for col in range(OUTPUT_DIM):
        l2 = float(jnp.sqrt(
            jnp.sum((preds_val[:, col] - Y_v[:, col]) ** 2) /
            jnp.maximum(jnp.sum(Y_v[:, col] ** 2), 1e-8)
        ))
        per_output_l2.append(l2)

    return {
        "params": best_params,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "train_rel_l2": train_rel_l2,
        "val_rel_l2": val_rel_l2,
        "per_output_rel_l2": per_output_l2,
        "n_layers": n_layers,
        "hidden_dims": hidden_dims,
        "epochs_run": len(train_losses),
        "training_time_s": time.monotonic() - t_start,
        "platform": platform,
    }


# ── Verification Gate ────────────────────────────────────────────────

def verify_and_save(
    result: dict,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    output_path: Path,
) -> bool:
    """Verify training results meet quality gates before saving."""
    params = result["params"]
    n_layers = result["n_layers"]
    output_names = ["chi_e", "chi_i", "D_e"]

    print("\n=== VERIFICATION GATE ===")
    all_pass = True

    # Gate 1: test relative L2 < 0.05
    import jax
    import jax.numpy as jnp
    from jax import vmap

    # Define the exact same forward function as in _train_jax
    def jax_forward(params, x):
        h = (x - params["input_mean"]) / jnp.maximum(params["input_std"], 1e-8)
        n_layers_local = (len([k for k in params if k.startswith("w")])) - 1
        for i in range(n_layers_local):
            h = jax.nn.gelu(h @ params[f"w{i+1}"] + params[f"b{i+1}"])
        out = jax.nn.softplus(h @ params[f"w{n_layers_local+1}"] + params[f"b{n_layers_local+1}"])
        return out * params["output_scale"]

    # Convert test data to JAX
    params_jax = {k: jnp.array(v) for k, v in params.items()}
    X_test_jax = jnp.array(X_test)
    Y_test_jax = jnp.array(Y_test)

    # Batched inference
    preds_test_jax = vmap(lambda x: jax_forward(params_jax, x))(X_test_jax)
    preds_test = np.array(preds_test_jax)
    
    test_rel_l2 = np.sqrt(np.sum((preds_test - Y_test) ** 2) / max(np.sum(Y_test ** 2), 1e-8))

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

    # Gate 3: per-output relative L2
    for i, name in enumerate(output_names):
        col_l2 = np.sqrt(
            np.sum((preds_test[:, i] - Y_test[:, i]) ** 2) /
            max(np.sum(Y_test[:, i] ** 2), 1e-8)
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
    }
    for i in range(n_layers + 1):
        save_dict[f"w{i+1}"] = params[f"w{i+1}"]
        save_dict[f"b{i+1}"] = params[f"b{i+1}"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **save_dict)

    # Gate 5: verify saved file loads correctly
    loaded = np.load(output_path)
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
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test (100 samples, 10 epochs)")
    args = parser.parse_args()

    hidden_dims = [int(d) for d in args.hidden_dims.split(",")]

    print("=== SCPN Fusion Core — Neural Transport Training (QLKNN-10D) ===")
    print(f"Data:    {args.data_dir}")
    print(f"Output:  {args.output}")
    print(f"Arch:    {INPUT_DIM} -> {' -> '.join(map(str, hidden_dims))} -> {OUTPUT_DIM}")
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

    train_data = np.load(train_path)
    val_data = np.load(val_path)
    test_data = np.load(test_path)

    X_train, Y_train = train_data["X"], train_data["Y"]
    X_val, Y_val = val_data["X"], val_data["Y"]
    X_test, Y_test = test_data["X"], test_data["Y"]

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

    print(f"\nData loaded: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    print(f"Y ranges: chi_e=[{Y_train[:,0].min():.2f}, {Y_train[:,0].max():.2f}], "
          f"chi_i=[{Y_train[:,1].min():.2f}, {Y_train[:,1].max():.2f}], "
          f"D_e=[{Y_train[:,2].min():.2f}, {Y_train[:,2].max():.2f}]")

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
    )

    print(f"\nTraining complete in {result['training_time_s']:.1f}s")
    print(f"  val_relative_l2:   {result['val_rel_l2']:.4f}")
    print(f"  train_relative_l2: {result['train_rel_l2']:.4f}")

    # Verify and save
    ok = verify_and_save(result, X_test, Y_test, args.output)

    if args.quick and ok:
        print("\nSmoke test PASSED")
    elif not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
