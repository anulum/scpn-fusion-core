# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — ITER Surrogate Training Tool
"""Retrain the neural equilibrium surrogate for ITER 6.2 m scenarios.

Generates data by perturbing coil currents in FusionKernel,
performs PCA on resulting Psi fields, and trains a JAX MLP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, random, value_and_grad, vmap
from numpy.typing import NDArray

from scpn_fusion.core.fusion_kernel import FusionKernel
from scpn_fusion.core.neural_equilibrium import NeuralEquilibriumAccelerator

logger = logging.getLogger(__name__)
FloatArray = NDArray[np.float64]
Layer = dict[str, jax.Array]
Params = list[Layer]
CHECKPOINT_VERSION = 1


def default_iter_dataset_paths(data_dir: str | Path = "data") -> tuple[Path, Path]:
    """Return repository-local default ITER feature and field arrays."""
    root = Path(data_dir)
    return root / "iter_X.npy", root / "iter_Y.npy"


def load_iter_dataset(data_path: str | Path) -> tuple[FloatArray, FloatArray]:
    """Load ITER surrogate arrays from an NPZ file or a directory of NPY files."""
    path = Path(data_path).expanduser()
    if path.is_dir():
        x_path, y_path = default_iter_dataset_paths(path)
        return (
            cast(FloatArray, np.load(x_path, allow_pickle=False)),
            cast(FloatArray, np.load(y_path, mmap_mode="r", allow_pickle=False)),
        )

    if path.suffix == ".npz":
        data = np.load(path, mmap_mode="r", allow_pickle=False)
        return cast(FloatArray, data["X"]), cast(FloatArray, data["Y"])

    raise ValueError(
        "--data must point to an .npz file or a directory containing iter_X.npy and iter_Y.npy"
    )


def inspect_iter_dataset(
    X: FloatArray,
    Y: FloatArray,
    *,
    min_full_fidelity_samples: int = 50_000,
    expected_features: int = 12,
) -> dict[str, Any]:
    """Return shape and evidence status for an ITER surrogate dataset."""
    report: dict[str, Any] = {
        "n_samples": int(X.shape[0]) if X.ndim >= 1 else 0,
        "x_shape": list(X.shape),
        "y_shape": list(Y.shape),
        "expected_features": expected_features,
        "min_full_fidelity_samples": min_full_fidelity_samples,
        "x_finite": bool(np.all(np.isfinite(X))),
        "y_finite": bool(np.all(np.isfinite(Y))),
    }
    if X.ndim != 2 or X.shape[1] != expected_features:
        report["status"] = "blocked_invalid_feature_shape"
    elif Y.ndim != 2 or Y.shape[0] != X.shape[0]:
        report["status"] = "blocked_invalid_field_shape"
    elif not report["x_finite"] or not report["y_finite"]:
        report["status"] = "blocked_nonfinite_values"
    elif X.shape[0] < min_full_fidelity_samples:
        report["status"] = "development_dataset_below_full_fidelity_sample_count"
    else:
        report["status"] = "full_fidelity_iter_dataset_ready"
    return report


def write_iter_dataset_report(path: Path, report: dict[str, Any]) -> None:
    """Write an ITER dataset evidence report."""
    _atomic_write_text(path, json.dumps(report, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a dataset or checkpoint file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, content: str) -> None:
    """Replace a text artifact only after its complete payload reaches disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_savez(path: Path, payload: dict[str, Any]) -> None:
    """Replace an NPZ artifact atomically after a complete temporary write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".npz",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez(temporary, **payload)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class FastNumPyPCA:
    """Memory-aware NumPy PCA using the sample Gram matrix."""

    def __init__(self, n_components: int = 20):
        """Configure the number of retained principal components."""
        self.n_components = n_components
        self.mean_: FloatArray | None = None
        self.components_: FloatArray | None = None
        self.explained_variance_ratio_: FloatArray | None = None

    def fit_transform(self, X: FloatArray) -> FloatArray:
        """Fit PCA components and return projected samples."""
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

        top_vals = vals[: self.n_components]
        top_vecs = vecs[:, : self.n_components]

        total_var = np.sum(np.maximum(vals, 0.0))
        self.explained_variance_ratio_ = np.maximum(top_vals, 0.0) / total_var

        # Components W = V^T * X_c / sqrt(L)
        inv_sqrt_vals = 1.0 / np.sqrt(np.maximum(top_vals, 1e-15))
        self.components_ = (top_vecs.T @ X_c) * inv_sqrt_vals[:, None]

        # Latent projection Z = V * sqrt(L)
        Z = top_vecs * np.sqrt(np.maximum(top_vals, 0.0))

        logger.info("  PCA complete.")
        return np.asarray(Z, dtype=np.float64)

    def transform(self, X: FloatArray) -> FloatArray:
        """Project samples into the fitted PCA basis."""
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA must be fitted before transform")
        return np.asarray((X - self.mean_) @ self.components_.T, dtype=np.float64)

    def inverse_transform(self, Z: FloatArray) -> FloatArray:
        """Reconstruct field samples from PCA coordinates."""
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA must be fitted before inverse_transform")
        return np.asarray(Z @ self.components_ + self.mean_, dtype=np.float64)


# ── MLP Hyperparameters ──────────────────────────────────────────────
HIDDEN_SIZES = [256, 128, 64]
LEARNING_RATE = 1e-4
GRAD_CLIP = 0.5
BATCH_SIZE = 32
EPOCHS = 100
PCA_COMPONENTS_TARGET = 20


# ── Model Definition ─────────────────────────────────────────────────


def init_mlp_params(
    key: jax.Array,
    input_dim: int,
    hidden_sizes: list[int],
    output_dim: int,
) -> Params:
    """Initialise MLP weights with He initialisation."""
    dims = [input_dim] + hidden_sizes + [output_dim]
    params: Params = []
    for i in range(len(dims) - 1):
        key, subkey = random.split(key)
        fan_in, fan_out = dims[i], dims[i + 1]
        std = jnp.sqrt(2.0 / fan_in)
        params.append(
            {"W": random.normal(subkey, (fan_in, fan_out)) * std, "b": jnp.zeros(fan_out)}
        )
    return params


def model_forward(params: Params, x: jax.Array) -> jax.Array:
    """Forward pass with ReLU activation."""
    h = x
    for i, p in enumerate(params):
        h = jnp.dot(h, p["W"]) + p["b"]
        if i < len(params) - 1:
            h = jax.nn.relu(h)
    return h


def mse_loss(params: Params, x_batch: jax.Array, y_batch: jax.Array) -> jax.Array:
    """MSE loss for batch."""
    preds = vmap(lambda x: model_forward(params, x))(x_batch)
    return jnp.mean((preds - y_batch) ** 2)


@jit
def update_step(
    params: Params,
    m: Params,
    v: Params,
    x_batch: jax.Array,
    y_batch: jax.Array,
    lr: float,
    t: int,
) -> tuple[Params, Params, Params, jax.Array]:
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
    return cast(Params, new_params), m, v, cast(jax.Array, loss)


# ── Data Generation ──────────────────────────────────────────────────


def generate_iter_data(
    n_samples: int,
    config_path: str | Path,
    seed: int = 42,
) -> tuple[FloatArray, FloatArray]:
    """Generate training data using FusionKernel by perturbing ITER state."""
    logger.info("Generating %d ITER samples using FusionKernel...", n_samples)
    fk = FusionKernel(config_path)

    # Ensure ITER-like nominals
    fk.cfg["physics"]["B_T"] = 5.3
    fk.cfg["target"] = fk.cfg.get("target", {})
    fk.cfg["target"]["kappa"] = 1.7
    fk.cfg["target"]["R_axis"] = 6.2
    fk.cfg["target"]["Z_axis"] = 0.0

    X: list[list[float]] = []
    Y: list[FloatArray] = []

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
                float(ip / 1e6),
                5.3,  # B_t
                float(fk.R[ir]),  # R_axis
                float(fk.Z[iz]),  # Z_axis
                1.0,  # pprime_scale
                1.0,  # ffprime_scale
                float(psi_ax),
                float(psi_x),
                1.7,  # kappa
                0.33,  # delta_up
                0.33,  # delta_low
                3.0,  # q95
            ]

            X.append(features)
            Y.append(np.asarray(fk.Psi, dtype=np.float64).ravel())
        except Exception as e:
            logger.warning("Sample %d failed: %s", i, e)
            continue

    return np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64)


# ── Training Entry Point ─────────────────────────────────────────────


def deterministic_split(
    n_samples: int,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return deterministic disjoint training and validation indices."""
    if n_samples < 3:
        raise ValueError("At least three samples are required for a held-out split.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between zero and one.")
    n_validation = max(1, int(round(n_samples * validation_fraction)))
    if n_validation >= n_samples:
        raise ValueError("validation_fraction leaves no training samples.")
    indices = np.random.default_rng(seed).permutation(n_samples)
    return (
        np.sort(indices[n_validation:]).astype(np.int64),
        np.sort(indices[:n_validation]).astype(np.int64),
    )


def _checkpoint_identity(
    *,
    x_sha256: str,
    y_sha256: str,
    seed: int,
    validation_fraction: float,
) -> dict[str, Any]:
    """Build immutable fields that bind recovery state to one dataset and split."""
    return {
        "checkpoint_version": np.array([CHECKPOINT_VERSION], dtype=np.int64),
        "x_sha256": np.array([x_sha256]),
        "y_sha256": np.array([y_sha256]),
        "seed": np.array([seed], dtype=np.int64),
        "validation_fraction": np.array([validation_fraction], dtype=np.float64),
        "hidden_sizes": np.asarray(HIDDEN_SIZES, dtype=np.int64),
        "learning_rate": np.array([LEARNING_RATE], dtype=np.float64),
        "gradient_clip": np.array([GRAD_CLIP], dtype=np.float64),
    }


def _validate_checkpoint_identity(checkpoint: Any, identity: dict[str, Any]) -> None:
    """Reject recovery state created for different data or split settings."""
    for name, value in identity.items():
        if name not in checkpoint or not np.array_equal(checkpoint[name], value):
            raise ValueError(f"Checkpoint identity mismatch for {name}.")


def _save_optimizer_checkpoint(
    path: Path,
    *,
    params: Params,
    first_moment: Params,
    second_moment: Params,
    completed_epochs: int,
    final_loss: float,
    identity: dict[str, Any],
) -> None:
    """Persist model and Adam state required for an exact epoch-boundary resume."""
    payload = dict(identity)
    payload["completed_epochs"] = np.array([completed_epochs], dtype=np.int64)
    payload["final_loss"] = np.array([final_loss], dtype=np.float64)
    payload["n_layers"] = np.array([len(params)], dtype=np.int64)
    for index, (parameter, moment_1, moment_2) in enumerate(
        zip(params, first_moment, second_moment, strict=True)
    ):
        for name in ("W", "b"):
            payload[f"param_{index}_{name}"] = np.asarray(parameter[name])
            payload[f"m_{index}_{name}"] = np.asarray(moment_1[name])
            payload[f"v_{index}_{name}"] = np.asarray(moment_2[name])
    _atomic_savez(path, payload)


def _load_optimizer_checkpoint(
    path: Path,
    *,
    identity: dict[str, Any],
) -> tuple[Params, Params, Params, int, float]:
    """Load an optimizer checkpoint after verifying its immutable identity."""
    with np.load(path, allow_pickle=False) as checkpoint:
        _validate_checkpoint_identity(checkpoint, identity)
        n_layers = int(checkpoint["n_layers"][0])
        completed_epochs = int(checkpoint["completed_epochs"][0])
        final_loss = float(checkpoint["final_loss"][0])
        params: Params = []
        first_moment: Params = []
        second_moment: Params = []
        for index in range(n_layers):
            params.append(
                {name: jnp.asarray(checkpoint[f"param_{index}_{name}"]) for name in ("W", "b")}
            )
            first_moment.append(
                {name: jnp.asarray(checkpoint[f"m_{index}_{name}"]) for name in ("W", "b")}
            )
            second_moment.append(
                {name: jnp.asarray(checkpoint[f"v_{index}_{name}"]) for name in ("W", "b")}
            )
    return params, first_moment, second_moment, completed_epochs, final_loss


def _held_out_metrics(
    *,
    pca: FastNumPyPCA,
    params: Params,
    x_validation: FloatArray,
    y_validation: FloatArray,
    x_mean: FloatArray,
    x_std: FloatArray,
    z_mean: FloatArray,
    z_std: FloatArray,
    chunk_size: int = 128,
) -> dict[str, float]:
    """Evaluate latent and reconstructed-field errors on unseen samples in chunks."""
    squared_field_error = 0.0
    field_elements = 0
    relative_l2: list[float] = []
    squared_latent_error = 0.0
    latent_elements = 0
    for start in range(0, len(x_validation), chunk_size):
        stop = min(start + chunk_size, len(x_validation))
        x_chunk = np.asarray((x_validation[start:stop] - x_mean) / x_std, dtype=np.float64)
        y_chunk = np.asarray(y_validation[start:stop], dtype=np.float64)
        z_true = pca.transform(y_chunk)
        z_pred_norm = np.asarray(vmap(lambda x: model_forward(params, x))(jnp.asarray(x_chunk)))
        z_pred = z_pred_norm * z_std + z_mean
        latent_delta = z_pred - z_true
        squared_latent_error += float(np.sum(latent_delta**2))
        latent_elements += latent_delta.size
        delta = pca.inverse_transform(z_pred) - y_chunk
        squared_field_error += float(np.sum(delta**2))
        field_elements += delta.size
        numerator = np.linalg.norm(delta, axis=1)
        denominator = np.maximum(np.linalg.norm(y_chunk, axis=1), 1e-15)
        relative_l2.extend(np.asarray(numerator / denominator, dtype=np.float64).tolist())
    return {
        "latent_rmse": float(np.sqrt(squared_latent_error / max(latent_elements, 1))),
        "field_rmse": float(np.sqrt(squared_field_error / max(field_elements, 1))),
        "mean_relative_l2": float(np.mean(relative_l2)),
        "p95_relative_l2": float(np.percentile(relative_l2, 95.0)),
        "max_relative_l2": float(np.max(relative_l2)),
    }


def run_training(
    *,
    data_path: Path,
    out_path: Path,
    report_path: Path,
    checkpoint_dir: Path,
    epochs: int,
    seed: int,
    validation_fraction: float = 0.2,
    checkpoint_every: int = 10,
    min_full_fidelity_samples: int = 50_000,
    strict_full_fidelity: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Train a recoverable ITER surrogate candidate from an existing dataset."""
    if epochs < 1 or checkpoint_every < 1:
        raise ValueError("epochs and checkpoint_every must be positive.")
    if not data_path.is_dir():
        raise ValueError(
            "Recoverable training requires a directory with iter_X.npy and iter_Y.npy."
        )
    x_path, y_path = default_iter_dataset_paths(data_path)
    x_sha256 = sha256_file(x_path)
    y_sha256 = sha256_file(y_path)
    X_raw, Y_raw = load_iter_dataset(data_path)
    dataset_report = inspect_iter_dataset(
        X_raw, Y_raw, min_full_fidelity_samples=min_full_fidelity_samples
    )
    if strict_full_fidelity and dataset_report["status"] != "full_fidelity_iter_dataset_ready":
        raise SystemExit(f"strict full-fidelity gate failed: {dataset_report['status']}")
    train_indices, validation_indices = deterministic_split(
        len(X_raw), validation_fraction=validation_fraction, seed=seed
    )
    identity = _checkpoint_identity(
        x_sha256=x_sha256, y_sha256=y_sha256, seed=seed, validation_fraction=validation_fraction
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pca_path = checkpoint_dir / "pca_checkpoint.npz"
    optimizer_path = checkpoint_dir / "optimizer_checkpoint.npz"
    if resume and optimizer_path.exists() and not pca_path.exists():
        raise ValueError("Optimizer checkpoint exists without its PCA checkpoint.")
    report: dict[str, Any] = {
        **dataset_report,
        "claim_class": "high_resolution_synthetic_solver_surrogate_candidate",
        "facility_validated": False,
        "status": "running",
        "dataset": {
            "x_path": str(x_path),
            "y_path": str(y_path),
            "x_sha256": x_sha256,
            "y_sha256": y_sha256,
            "original_generation_provenance": "unknown_preexisting_local_arrays",
        },
        "split": {
            "seed": seed,
            "validation_fraction": validation_fraction,
            "training_samples": int(train_indices.size),
            "validation_samples": int(validation_indices.size),
        },
        "recovery": {
            "checkpoint_version": CHECKPOINT_VERSION,
            "pca_checkpoint": str(pca_path),
            "optimizer_checkpoint": str(optimizer_path),
            "checkpoint_every_epochs": checkpoint_every,
            "resume_requested": resume,
        },
    }
    write_iter_dataset_report(report_path, report)
    started = time.perf_counter()
    pca = FastNumPyPCA()
    if resume and pca_path.exists():
        with np.load(pca_path, allow_pickle=False) as checkpoint:
            _validate_checkpoint_identity(checkpoint, identity)
            if not np.array_equal(checkpoint["train_indices"], train_indices) or not np.array_equal(
                checkpoint["validation_indices"], validation_indices
            ):
                raise ValueError("PCA checkpoint split mismatch.")
            pca.mean_ = np.asarray(checkpoint["pca_mean"], dtype=np.float64)
            pca.components_ = np.asarray(checkpoint["pca_components"], dtype=np.float64)
            pca.explained_variance_ratio_ = np.asarray(checkpoint["pca_evr"], dtype=np.float64)
            pca.n_components = int(pca.components_.shape[0])
            y_train_latent = np.asarray(checkpoint["y_train_latent"], dtype=np.float64)
        logger.info("Resumed completed PCA from %s", pca_path)
    else:
        pca = FastNumPyPCA(n_components=min(train_indices.size - 1, PCA_COMPONENTS_TARGET))
        y_train_latent = pca.fit_transform(np.asarray(Y_raw[train_indices], dtype=np.float64))
        if pca.mean_ is None or pca.components_ is None or pca.explained_variance_ratio_ is None:
            raise RuntimeError("PCA fit completed without learned components")
        _atomic_savez(
            pca_path,
            {
                **identity,
                "train_indices": train_indices,
                "validation_indices": validation_indices,
                "pca_mean": pca.mean_,
                "pca_components": pca.components_,
                "pca_evr": pca.explained_variance_ratio_,
                "y_train_latent": y_train_latent,
            },
        )
        logger.info("Saved recoverable PCA state to %s", pca_path)
    x_train = np.asarray(X_raw[train_indices], dtype=np.float64)
    x_mean = x_train.mean(axis=0)
    x_raw_std = x_train.std(axis=0)
    x_std = np.where(x_raw_std < 1e-10, 1.0, x_raw_std)
    x_train_norm = np.asarray((x_train - x_mean) / x_std, dtype=np.float64)
    z_mean = y_train_latent.mean(axis=0)
    z_raw_std = y_train_latent.std(axis=0)
    z_std = np.where(z_raw_std < 1e-10, 1.0, z_raw_std)
    y_train_norm = np.asarray((y_train_latent - z_mean) / z_std, dtype=np.float64)
    if resume and optimizer_path.exists():
        params, first_moment, second_moment, completed_epochs, final_loss = (
            _load_optimizer_checkpoint(optimizer_path, identity=identity)
        )
        if completed_epochs > epochs:
            raise ValueError("Optimizer checkpoint is newer than the requested epoch target.")
    else:
        params = init_mlp_params(
            random.PRNGKey(seed), x_train_norm.shape[1], HIDDEN_SIZES, pca.n_components
        )
        first_moment = cast(Params, jax.tree_util.tree_map(jnp.zeros_like, params))
        second_moment = cast(Params, jax.tree_util.tree_map(jnp.zeros_like, params))
        completed_epochs = 0
        final_loss = float("nan")
    x_jax = jnp.asarray(x_train_norm)
    y_jax = jnp.asarray(y_train_norm)
    for epoch in range(completed_epochs, epochs):
        step = epoch + 1
        params, first_moment, second_moment, loss = update_step(
            params, first_moment, second_moment, x_jax, y_jax, LEARNING_RATE, step
        )
        final_loss = float(loss)
        completed_epochs = step
        if completed_epochs % checkpoint_every == 0 or completed_epochs == epochs:
            _save_optimizer_checkpoint(
                optimizer_path,
                params=params,
                first_moment=first_moment,
                second_moment=second_moment,
                completed_epochs=completed_epochs,
                final_loss=final_loss,
                identity=identity,
            )
            logger.info("Epoch %d: loss=%.6f; checkpoint saved", completed_epochs, final_loss)
    if pca.mean_ is None or pca.components_ is None or pca.explained_variance_ratio_ is None:
        raise RuntimeError("PCA state unavailable before final artifact save")
    if not np.isfinite(final_loss):
        raise RuntimeError("Training completed without a finite loss.")
    metrics = _held_out_metrics(
        pca=pca,
        params=params,
        x_validation=np.asarray(X_raw[validation_indices], dtype=np.float64),
        y_validation=np.asarray(Y_raw[validation_indices], dtype=np.float64),
        x_mean=np.asarray(x_mean, dtype=np.float64),
        x_std=np.asarray(x_std, dtype=np.float64),
        z_mean=np.asarray(z_mean, dtype=np.float64),
        z_std=np.asarray(z_std, dtype=np.float64),
    )
    grid_width = int(np.sqrt(Y_raw.shape[1]))
    if grid_width * grid_width != Y_raw.shape[1]:
        raise ValueError("ITER field width must describe a square grid.")
    payload: dict[str, Any] = {
        "n_components": np.array([pca.n_components]),
        "grid_nh": np.array([grid_width]),
        "grid_nw": np.array([grid_width]),
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
    for index, parameter in enumerate(params):
        payload[f"w{index}"] = np.asarray(parameter["W"])
        payload[f"b{index}"] = np.asarray(parameter["b"])
    _atomic_savez(out_path, payload)
    accelerator = NeuralEquilibriumAccelerator()
    accelerator.load_weights(out_path)
    runtime_prediction = accelerator.predict(np.asarray(x_mean, dtype=np.float64))
    reference_latent_normalized = np.asarray(
        model_forward(params, jnp.zeros(x_mean.shape[0], dtype=jnp.float64))
    )
    reference_prediction = pca.inverse_transform(
        (reference_latent_normalized * z_std + z_mean)[np.newaxis, :]
    ).reshape(grid_width, grid_width)
    if runtime_prediction.shape != (grid_width, grid_width) or not np.all(
        np.isfinite(runtime_prediction)
    ):
        raise RuntimeError("Candidate failed its production runtime load/predict check.")
    runtime_parity_max_abs = float(np.max(np.abs(runtime_prediction - reference_prediction)))
    if runtime_parity_max_abs > 1e-8:
        raise RuntimeError(
            "Candidate production runtime does not preserve latent normalization "
            f"(max abs error {runtime_parity_max_abs:.3e})."
        )
    elapsed = time.perf_counter() - started
    report.update(
        {
            "status": "completed_candidate_below_full_fidelity_claim_threshold"
            if dataset_report["status"] != "full_fidelity_iter_dataset_ready"
            else "completed_candidate_meeting_configured_sample_threshold",
            "training": {
                "backend": jax.default_backend(),
                "devices": [str(device) for device in jax.devices()],
                "epochs": epochs,
                "final_training_loss": final_loss,
                "pca_components": pca.n_components,
                "pca_explained_variance_sum": float(np.sum(pca.explained_variance_ratio_)),
                "elapsed_seconds": elapsed,
            },
            "held_out_validation": metrics,
            "artifact": {
                "path": str(out_path),
                "sha256": sha256_file(out_path),
                "promotion_status": "candidate_not_promoted",
                "runtime_load_predict_finite": True,
                "runtime_prediction_shape": list(runtime_prediction.shape),
                "runtime_training_path_parity_max_abs": runtime_parity_max_abs,
            },
        }
    )
    write_iter_dataset_report(report_path, report)
    logger.info("Retraining complete in %.2f s. Candidate saved to %s", elapsed, out_path)
    return report


def main() -> None:
    """Train or evaluate the ITER equilibrium surrogate workflow."""
    parser = argparse.ArgumentParser(description="Train ITER surrogate")
    parser.add_argument("--data", required=True, help="Directory with iter_X.npy and iter_Y.npy")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs")
    parser.add_argument("--out", required=True, help="Candidate NPZ output path")
    parser.add_argument("--report", required=True, help="Training evidence report path")
    parser.add_argument("--checkpoint-dir", required=True, help="Recovery checkpoint directory")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--min-full-fidelity-samples", type=int, default=50_000)
    parser.add_argument("--strict-full-fidelity", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_training(
        data_path=Path(args.data),
        out_path=Path(args.out),
        report_path=Path(args.report),
        checkpoint_dir=Path(args.checkpoint_dir),
        epochs=args.epochs,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        checkpoint_every=args.checkpoint_every,
        min_full_fidelity_samples=args.min_full_fidelity_samples,
        strict_full_fidelity=args.strict_full_fidelity,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
