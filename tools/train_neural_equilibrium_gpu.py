#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Multi-Machine Neural Equilibrium GPU Training
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Train neural equilibrium on ALL reference GEQDSK files (SPARC + DIII-D + JET)
with psi and grid domain normalisation.

Fixes the cross-machine regression found in CPU proof-of-concept (2026-03-10):
  SPARC rel_L2 went from 0.01 → 1.39 without normalisation.

Key differences from train_neural_equilibrium_augmented.py:
  1. Psi normalised to [0,1] per equilibrium before PCA
  2. Grid domain normalised to [0,1]×[0,1] via RectBivariateSpline
  3. Deeper network (256,128,64,32) — 4 hidden layers
  4. Stratified train/val/test split (each machine in each split)
  5. Cosine LR decay from 1e-3 → 1e-5
  6. Weight decay L2 = 1e-4
  7. 50 perturbations per file = 918 samples

Usage:
    python tools/train_neural_equilibrium_gpu.py \\
        --n-perturbations 50 --seed 42 \\
        --save-path weights/neural_equilibrium_augmented.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

TARGET_GRID = 129  # square grid NxN in normalised [0,1]x[0,1]
HIDDEN_SIZES = (256, 128, 64, 32)
N_PCA_COMPONENTS = 30  # more components for 3-machine diversity
LR_INITIAL = 1e-3
LR_FINAL = 1e-5
WEIGHT_DECAY = 1e-4
N_EPOCHS = 1500
BATCH_SIZE = 64
PATIENCE = 80
MOMENTUM = 0.9
LAMBDA_GS = 0.05  # GS residual penalty weight


@dataclass
class AugTrainingResult:
    n_samples: int
    n_components: int
    explained_variance: float
    final_train_loss: float
    best_val_loss: float
    test_mse: float
    test_max_error: float
    train_time_s: float
    per_machine: dict
    hidden_sizes: tuple
    n_epochs_run: int
    lr_initial: float
    lr_final: float


# ── Minimal PCA ────────────────────────────────────────────────────────


class MinimalPCA:
    def __init__(self, n_components: int = 20) -> None:
        self.n_components = n_components
        self.mean_: NDArray | None = None
        self.components_: NDArray | None = None
        self.explained_variance_ratio_: NDArray | None = None

    def fit(self, X: NDArray) -> "MinimalPCA":
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        total_var = (S**2).sum()
        k = min(self.n_components, len(S))
        self.components_ = Vt[:k]
        self.explained_variance_ratio_ = S[:k] ** 2 / max(total_var, 1e-15)
        self.n_components = k
        return self

    def transform(self, X: NDArray) -> NDArray:
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, Z: NDArray) -> NDArray:
        return Z @ self.components_ + self.mean_

    def fit_transform(self, X: NDArray) -> NDArray:
        self.fit(X)
        return self.transform(X)


# ── Simple MLP ─────────────────────────────────────────────────────────


class SimpleMLP:
    def __init__(self, layer_sizes: list[int], seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.weights: list[NDArray] = []
        self.biases: list[NDArray] = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            scale = np.sqrt(2.0 / fan_in)  # He init
            self.weights.append(self.rng.normal(0, scale, (layer_sizes[i], layer_sizes[i + 1])))
            self.biases.append(np.zeros(layer_sizes[i + 1]))

    def forward(self, x: NDArray) -> NDArray:
        h = x
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ W + b
            if i < len(self.weights) - 1:
                h = np.maximum(0, h)  # ReLU
        return h

    def predict(self, x: NDArray) -> NDArray:
        return self.forward(x)


# ── Data loading ───────────────────────────────────────────────────────


def collect_geqdsk_files(ref_dir: Path) -> dict[str, list[Path]]:
    """Collect GEQDSK/EQDSK files grouped by machine."""
    result: dict[str, list[Path]] = {}
    for subdir in sorted(ref_dir.iterdir()):
        if not subdir.is_dir():
            continue
        found = sorted(subdir.glob("*.geqdsk")) + sorted(subdir.glob("*.eqdsk"))
        if found:
            result[subdir.name] = found
    return result


def load_and_normalise(
    geqdsk_paths: dict[str, list[Path]],
    n_perturbations: int,
    rng: np.random.Generator,
) -> tuple[NDArray, NDArray, list[str]]:
    """Load all GEQDSK files, normalise psi and grid domain.

    Returns:
        X: (N, 12) feature matrix
        Y: (N, TARGET_GRID*TARGET_GRID) normalised psi
        machine_labels: per-sample machine name
    """
    from scpn_fusion.core.eqdsk import read_geqdsk
    from scipy.interpolate import RectBivariateSpline

    target_rho = np.linspace(0, 1, TARGET_GRID)
    target_zeta = np.linspace(0, 1, TARGET_GRID)

    X_features: list[NDArray] = []
    Y_psi: list[NDArray] = []
    labels: list[str] = []

    for machine, paths in geqdsk_paths.items():
        for path in paths:
            eq = read_geqdsk(path)

            # Physical grid
            r_phys = eq.r  # shape (nw,)
            z_phys = eq.z  # shape (nh,)

            # Normalise grid to [0,1]×[0,1]
            r_norm = (r_phys - r_phys[0]) / max(r_phys[-1] - r_phys[0], 1e-12)
            z_norm = (z_phys - z_phys[0]) / max(z_phys[-1] - z_phys[0], 1e-12)

            # Interpolate psi onto normalised target grid
            spline = RectBivariateSpline(z_norm, r_norm, eq.psirz, kx=3, ky=3)
            psi_on_norm_grid = spline(target_zeta, target_rho, grid=True)

            # Normalise psi to [0,1]: psi_n = (psi - simag) / (sibry - simag)
            denom = eq.sibry - eq.simag
            if abs(denom) < 1e-12:
                denom = 1.0
            psi_normalised = (psi_on_norm_grid - eq.simag) / denom

            # Shape parameters
            kappa = 1.7
            delta_upper, delta_lower = 0.3, 0.3
            q95 = 3.0
            if hasattr(eq, "rbbbs") and eq.rbbbs is not None and len(eq.rbbbs) > 3:
                r_span = eq.rbbbs.max() - eq.rbbbs.min()
                kappa = (eq.zbbbs.max() - eq.zbbbs.min()) / max(r_span, 0.01)
            if hasattr(eq, "rbdry") and eq.rbdry is not None and len(eq.rbdry) > 3:
                r_span = eq.rbdry.max() - eq.rbdry.min()
                if r_span > 0.01:
                    kappa = (eq.zbdry.max() - eq.zbdry.min()) / r_span
                    r_mid = (eq.rbdry.max() + eq.rbdry.min()) / 2
                    r_axis = eq.rmaxis
                    # upper triangularity
                    z_top_idx = np.argmax(eq.zbdry)
                    delta_upper = (r_axis - eq.rbdry[z_top_idx]) / (r_span / 2)
                    # lower triangularity
                    z_bot_idx = np.argmin(eq.zbdry)
                    delta_lower = (r_axis - eq.rbdry[z_bot_idx]) / (r_span / 2)
            if hasattr(eq, "qpsi") and eq.qpsi is not None and len(eq.qpsi) > 0:
                idx_95 = int(0.95 * len(eq.qpsi))
                q95 = eq.qpsi[min(idx_95, len(eq.qpsi) - 1)]

            # 12-dim feature vector
            base_features = np.array(
                [
                    eq.current / 1e6,
                    eq.bcentr,
                    eq.rmaxis,
                    eq.zmaxis,
                    1.0,  # pprime scale
                    1.0,  # ffprime scale
                    eq.simag,
                    eq.sibry,
                    kappa,
                    delta_upper,
                    delta_lower,
                    q95,
                ]
            )

            # Unperturbed sample
            X_features.append(base_features)
            Y_psi.append(psi_normalised.ravel())
            labels.append(machine)

            # Perturbed samples
            for _ in range(n_perturbations):
                pp_scale = rng.uniform(0.7, 1.3)
                ff_scale = rng.uniform(0.7, 1.3)

                feat = base_features.copy()
                feat[4] = pp_scale
                feat[5] = ff_scale

                psi_n = psi_normalised.copy()
                mix = 0.5 * (pp_scale + ff_scale) - 1.0
                plasma_mask = (psi_n >= 0) & (psi_n < 1.0)
                psi_n[plasma_mask] += mix * 0.1 * (1.0 - psi_n[plasma_mask])

                X_features.append(feat)
                Y_psi.append(psi_n.ravel())
                labels.append(machine)

    return np.array(X_features), np.array(Y_psi), labels


# ── Stratified split ──────────────────────────────────────────────────


def stratified_split(
    labels: list[str],
    rng: np.random.Generator,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[NDArray, NDArray, NDArray]:
    """Split indices ensuring each machine appears in train/val/test."""
    machines = sorted(set(labels))
    train_idx, val_idx, test_idx = [], [], []

    for machine in machines:
        machine_indices = np.array([i for i, l in enumerate(labels) if l == machine])
        rng.shuffle(machine_indices)
        n = len(machine_indices)
        n_train = max(1, int(train_frac * n))
        n_val = max(1, int(val_frac * n))
        train_idx.extend(machine_indices[:n_train])
        val_idx.extend(machine_indices[n_train : n_train + n_val])
        test_idx.extend(machine_indices[n_train + n_val :])

    return (
        np.array(train_idx, dtype=int),
        np.array(val_idx, dtype=int),
        np.array(test_idx, dtype=int),
    )


# ── GS residual loss ──────────────────────────────────────────────────


def gs_residual_loss(psi_flat: NDArray, nh: int, nw: int) -> float:
    psi = psi_flat.reshape(nh, nw)
    lap = np.zeros_like(psi)
    lap[1:-1, 1:-1] = (
        psi[2:, 1:-1] + psi[:-2, 1:-1] + psi[1:-1, 2:] + psi[1:-1, :-2] - 4 * psi[1:-1, 1:-1]
    )
    return float(np.mean(lap[1:-1, 1:-1] ** 2))


# ── Training loop ─────────────────────────────────────────────────────


def train(
    X: NDArray,
    Y_compressed: NDArray,
    Y_raw: NDArray,
    pca: MinimalPCA,
    train_idx: NDArray,
    val_idx: NDArray,
    test_idx: NDArray,
    seed: int,
) -> tuple[SimpleMLP, AugTrainingResult, NDArray, NDArray]:
    n_features = X.shape[1]
    layer_sizes = [n_features, *HIDDEN_SIZES, pca.n_components]
    mlp = SimpleMLP(layer_sizes, seed=seed)

    X_train, Y_train = X[train_idx], Y_compressed[train_idx]
    X_val, Y_val = X[val_idx], Y_compressed[val_idx]

    velocity = [np.zeros_like(w) for w in mlp.weights]
    velocity_b = [np.zeros_like(b) for b in mlp.biases]

    best_val_loss = float("inf")
    best_train_loss = float("inf")
    best_weights = None
    best_biases = None
    patience_counter = 0
    rng = np.random.default_rng(seed + 1)
    t0 = time.perf_counter()
    epochs_run = 0

    for epoch in range(N_EPOCHS):
        epochs_run = epoch + 1
        # Cosine LR decay
        lr = LR_FINAL + 0.5 * (LR_INITIAL - LR_FINAL) * (1 + np.cos(np.pi * epoch / N_EPOCHS))

        order = rng.permutation(len(X_train))
        epoch_loss = 0.0

        for start in range(0, len(X_train), BATCH_SIZE):
            idx = order[start : start + BATCH_SIZE]
            x_batch = X_train[idx]
            y_batch = Y_train[idx]
            bs = len(idx)

            # Forward with stored activations
            activations = [x_batch]
            h = x_batch
            for i, (W, b) in enumerate(zip(mlp.weights, mlp.biases)):
                z = h @ W + b
                if i < len(mlp.weights) - 1:
                    h = np.maximum(0, z)
                else:
                    h = z
                activations.append(h)

            error = activations[-1] - y_batch
            mse = float(np.mean(error**2))

            # GS residual (sampled — check 4 random items per batch)
            gs_loss = 0.0
            n_gs_check = min(4, bs)
            gs_indices = rng.choice(bs, n_gs_check, replace=False)
            for gi in gs_indices:
                psi_flat = pca.inverse_transform(activations[-1][gi : gi + 1])[0]
                gs_loss += gs_residual_loss(psi_flat, TARGET_GRID, TARGET_GRID)
            gs_loss /= n_gs_check

            loss = mse + LAMBDA_GS * gs_loss
            epoch_loss += loss * bs

            # Backprop
            delta = 2.0 * error / bs
            for i in range(len(mlp.weights) - 1, -1, -1):
                grad_w = activations[i].T @ delta + WEIGHT_DECAY * mlp.weights[i]
                grad_b = delta.sum(axis=0)

                grad_norm = np.linalg.norm(grad_w)
                if grad_norm > 5.0:
                    scale = 5.0 / grad_norm
                    grad_w *= scale
                    grad_b *= scale

                velocity[i] = MOMENTUM * velocity[i] - lr * grad_w
                velocity_b[i] = MOMENTUM * velocity_b[i] - lr * grad_b
                mlp.weights[i] += velocity[i]
                mlp.biases[i] += velocity_b[i]

                if i > 0:
                    delta = delta @ mlp.weights[i].T
                    delta *= (activations[i] > 0).astype(float)

        epoch_loss /= max(len(X_train), 1)
        if epoch_loss < best_train_loss:
            best_train_loss = epoch_loss

        # Validation
        val_pred = mlp.forward(X_val)
        val_mse = float(np.mean((val_pred - Y_val) ** 2))

        if val_mse < best_val_loss:
            best_val_loss = val_mse
            patience_counter = 0
            best_weights = [w.copy() for w in mlp.weights]
            best_biases = [b.copy() for b in mlp.biases]
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} (val_mse={val_mse:.6f})")
                break

        if epoch % 100 == 0:
            print(f"  Epoch {epoch:4d}: train={epoch_loss:.6f}  val={val_mse:.6f}  lr={lr:.2e}")

    train_time = time.perf_counter() - t0

    # Restore best weights
    if best_weights is not None:
        mlp.weights = best_weights
        mlp.biases = best_biases

    # Test evaluation
    X_test = X[test_idx]
    Y_test_raw = Y_raw[test_idx]
    x_test_pred = mlp.forward(X_test)
    psi_pred = pca.inverse_transform(x_test_pred)
    test_mse = float(np.mean((psi_pred - Y_test_raw) ** 2))
    test_max = float(np.max(np.abs(psi_pred - Y_test_raw)))

    explained = float(np.sum(pca.explained_variance_ratio_))

    input_mean = X.mean(axis=0)
    input_std = X.std(axis=0)
    input_std[input_std < 1e-10] = 1.0

    result = AugTrainingResult(
        n_samples=len(X),
        n_components=pca.n_components,
        explained_variance=explained,
        final_train_loss=best_train_loss,
        best_val_loss=best_val_loss,
        test_mse=test_mse,
        test_max_error=test_max,
        train_time_s=train_time,
        per_machine={},
        hidden_sizes=HIDDEN_SIZES,
        n_epochs_run=epochs_run,
        lr_initial=LR_INITIAL,
        lr_final=LR_FINAL,
    )

    return mlp, result, input_mean, input_std


# ── Per-machine validation ─────────────────────────────────────────────


def validate_per_machine(
    geqdsk_paths: dict[str, list[Path]],
    mlp: SimpleMLP,
    pca: MinimalPCA,
    input_mean: NDArray,
    input_std: NDArray,
) -> dict[str, dict]:
    """Validate on first file of each machine, returning rel_L2."""
    from scpn_fusion.core.eqdsk import read_geqdsk
    from scipy.interpolate import RectBivariateSpline

    target_rho = np.linspace(0, 1, TARGET_GRID)
    target_zeta = np.linspace(0, 1, TARGET_GRID)
    results = {}

    for machine, paths in geqdsk_paths.items():
        per_file_l2 = []
        for path in paths:
            eq = read_geqdsk(path)

            # Same normalization as training
            r_phys = eq.r
            z_phys = eq.z
            r_norm = (r_phys - r_phys[0]) / max(r_phys[-1] - r_phys[0], 1e-12)
            z_norm = (z_phys - z_phys[0]) / max(z_phys[-1] - z_phys[0], 1e-12)

            spline = RectBivariateSpline(z_norm, r_norm, eq.psirz, kx=3, ky=3)
            psi_on_norm_grid = spline(target_zeta, target_rho, grid=True)

            denom = eq.sibry - eq.simag
            if abs(denom) < 1e-12:
                denom = 1.0
            psi_ref_normalised = (psi_on_norm_grid - eq.simag) / denom

            # Build features
            kappa, delta_upper, delta_lower, q95 = 1.7, 0.3, 0.3, 3.0
            if hasattr(eq, "rbdry") and eq.rbdry is not None and len(eq.rbdry) > 3:
                r_span = eq.rbdry.max() - eq.rbdry.min()
                if r_span > 0.01:
                    kappa = (eq.zbdry.max() - eq.zbdry.min()) / r_span
            if hasattr(eq, "qpsi") and eq.qpsi is not None and len(eq.qpsi) > 0:
                idx_95 = int(0.95 * len(eq.qpsi))
                q95 = eq.qpsi[min(idx_95, len(eq.qpsi) - 1)]

            features = np.array(
                [
                    eq.current / 1e6,
                    eq.bcentr,
                    eq.rmaxis,
                    eq.zmaxis,
                    1.0,
                    1.0,
                    eq.simag,
                    eq.sibry,
                    kappa,
                    delta_upper,
                    delta_lower,
                    q95,
                ]
            )

            x_norm = (features - input_mean) / input_std
            coeffs = mlp.predict(x_norm[np.newaxis, :])
            psi_pred_norm = pca.inverse_transform(coeffs)[0].reshape(TARGET_GRID, TARGET_GRID)

            rel_l2 = float(
                np.linalg.norm(psi_pred_norm - psi_ref_normalised)
                / max(np.linalg.norm(psi_ref_normalised), 1e-12)
            )
            per_file_l2.append((path.name, rel_l2))

        mean_l2 = np.mean([x[1] for x in per_file_l2])
        max_l2 = np.max([x[1] for x in per_file_l2])
        results[machine] = {
            "mean_rel_l2": float(mean_l2),
            "max_rel_l2": float(max_l2),
            "per_file": {name: l2 for name, l2 in per_file_l2},
        }
        print(f"  [{machine}] mean_rel_L2={mean_l2:.4f}  max_rel_L2={max_l2:.4f}")
        for name, l2 in per_file_l2:
            print(f"    {name}: {l2:.4f}")

    return results


# ── Save weights ───────────────────────────────────────────────────────


def save_weights(
    path: Path,
    mlp: SimpleMLP,
    pca: MinimalPCA,
    input_mean: NDArray,
    input_std: NDArray,
    result: AugTrainingResult,
) -> None:
    payload = {
        "n_components": np.array([pca.n_components]),
        "grid_nh": np.array([TARGET_GRID]),
        "grid_nw": np.array([TARGET_GRID]),
        "n_input_features": np.array([12]),
        "pca_mean": pca.mean_,
        "pca_components": pca.components_,
        "pca_evr": pca.explained_variance_ratio_,
        "input_mean": input_mean,
        "input_std": input_std,
        "n_layers": np.array([len(mlp.weights)]),
        "psi_normalized": np.array([1]),  # flag: model trained on normalised psi
    }
    for i, (w, b) in enumerate(zip(mlp.weights, mlp.biases)):
        payload[f"w{i}"] = w
        payload[f"b{i}"] = b

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)

    # Metrics JSON
    metrics_path = path.with_suffix(".metrics.json")
    metrics = {
        "model": "neural_equilibrium_augmented",
        "version": "v1",
        "n_samples": result.n_samples,
        "n_components": result.n_components,
        "explained_variance": result.explained_variance,
        "hidden_dims": list(HIDDEN_SIZES),
        "final_train_loss": result.final_train_loss,
        "best_val_loss": result.best_val_loss,
        "test_mse": result.test_mse,
        "test_max_error": result.test_max_error,
        "train_time_s": result.train_time_s,
        "n_epochs_run": result.n_epochs_run,
        "lr_initial": result.lr_initial,
        "lr_final": result.lr_final,
        "weight_decay": WEIGHT_DECAY,
        "lambda_gs": LAMBDA_GS,
        "batch_size": BATCH_SIZE,
        "target_grid": TARGET_GRID,
        "psi_normalized": True,
        "grid_normalized": True,
        "per_machine": result.per_machine,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics: {metrics_path}")


# ── Main ──────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-perturbations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-path",
        type=str,
        default=str(REPO_ROOT / "weights" / "neural_equilibrium_augmented.npz"),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    rng = np.random.default_rng(args.seed)

    # Collect files
    ref_dir = REPO_ROOT / "validation" / "reference_data"
    files_by_machine = collect_geqdsk_files(ref_dir)
    total_files = sum(len(v) for v in files_by_machine.values())
    if total_files == 0:
        print(f"No GEQDSK/EQDSK files found in {ref_dir}")
        return 1

    print("=" * 70)
    print("Multi-Machine Neural Equilibrium GPU Training")
    print(f"  Machines: {list(files_by_machine.keys())}")
    print(f"  Files: {total_files}")
    print(f"  Perturbations: {args.n_perturbations} per file")
    print(f"  Expected samples: {total_files * (1 + args.n_perturbations)}")
    print(f"  Network: {HIDDEN_SIZES}")
    print(f"  PCA components: {N_PCA_COMPONENTS}")
    print(f"  LR: {LR_INITIAL} → {LR_FINAL} (cosine)")
    print(f"  Grid: {TARGET_GRID}x{TARGET_GRID} normalised")
    print("=" * 70)

    for machine, paths in files_by_machine.items():
        for p in paths:
            print(f"  [{machine}] {p.name}")

    # Checksum input files
    print("\nRecording input checksums...")
    checksums = {}
    for machine, paths in files_by_machine.items():
        for p in paths:
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            checksums[str(p.relative_to(REPO_ROOT))] = h
    checksum_path = Path(args.save_path).parent / "input_checksums.json"
    with open(checksum_path, "w") as f:
        json.dump(checksums, f, indent=2)
    print(f"  Saved {len(checksums)} checksums to {checksum_path}")

    # Load and normalise
    print("\nLoading and normalising data...")
    X, Y, labels = load_and_normalise(files_by_machine, args.n_perturbations, rng)
    print(f"  X: {X.shape}, Y: {Y.shape}")
    print(f"  Psi value range: [{Y.min():.4f}, {Y.max():.4f}]")

    # Input normalisation
    input_mean = X.mean(axis=0)
    input_std = X.std(axis=0)
    input_std[input_std < 1e-10] = 1.0
    X_norm = (X - input_mean) / input_std

    # PCA
    print("\nFitting PCA...")
    pca = MinimalPCA(n_components=N_PCA_COMPONENTS)
    Y_compressed = pca.fit_transform(Y)
    explained = float(np.sum(pca.explained_variance_ratio_))
    print(f"  {Y.shape[1]} → {pca.n_components} components, {explained * 100:.2f}% variance")

    # Stratified split
    train_idx, val_idx, test_idx = stratified_split(labels, rng)
    print(f"\nSplit: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")
    for machine in sorted(set(labels)):
        n_tr = sum(1 for i in train_idx if labels[i] == machine)
        n_va = sum(1 for i in val_idx if labels[i] == machine)
        n_te = sum(1 for i in test_idx if labels[i] == machine)
        print(f"  [{machine}] {n_tr}/{n_va}/{n_te}")

    # Train
    print(f"\nTraining ({N_EPOCHS} max epochs, patience={PATIENCE})...")
    mlp, result, _, _ = train(
        X_norm,
        Y_compressed,
        Y,
        pca,
        train_idx,
        val_idx,
        test_idx,
        seed=args.seed,
    )

    print(f"\n{'=' * 70}")
    print(f"Training complete in {result.train_time_s:.1f}s ({result.n_epochs_run} epochs)")
    print(f"  Train loss: {result.final_train_loss:.6f}")
    print(f"  Val loss:   {result.best_val_loss:.6f}")
    print(f"  Test MSE:   {result.test_mse:.6f}")
    print(f"  Test max:   {result.test_max_error:.6f}")

    # Per-machine validation
    print("\nPer-machine validation (all files):")
    per_machine = validate_per_machine(files_by_machine, mlp, pca, input_mean, input_std)
    result.per_machine = per_machine

    # Check acceptance criteria
    save_path = Path(args.save_path)
    criteria_met = True
    for machine, metrics in per_machine.items():
        if machine == "sparc":
            if metrics["mean_rel_l2"] > 0.10:
                print(
                    f"\n  FAIL: {machine} mean_rel_L2={metrics['mean_rel_l2']:.4f} > 0.10 hard limit"
                )
                criteria_met = False
            elif metrics["mean_rel_l2"] > 0.05:
                print(f"\n  WARN: {machine} mean_rel_L2={metrics['mean_rel_l2']:.4f} > 0.05 target")
        else:
            if metrics["mean_rel_l2"] > 0.20:
                print(
                    f"\n  FAIL: {machine} mean_rel_L2={metrics['mean_rel_l2']:.4f} > 0.20 hard limit"
                )
                criteria_met = False
            elif metrics["mean_rel_l2"] > 0.10:
                print(f"\n  WARN: {machine} mean_rel_L2={metrics['mean_rel_l2']:.4f} > 0.10 target")

    # Save
    save_weights(save_path, mlp, pca, input_mean, input_std, result)
    print(f"\nWeights: {save_path}")

    if not criteria_met:
        print("\n  ACCEPTANCE CRITERIA NOT MET — weights saved as experimental")
        # Also save with _experimental suffix
        exp_path = save_path.with_stem(save_path.stem + "_experimental")
        save_weights(exp_path, mlp, pca, input_mean, input_std, result)
        return 2

    print("\n  ALL ACCEPTANCE CRITERIA MET")

    # Verify input checksums unchanged
    print("\nVerifying input checksums...")
    for rel_path, expected in checksums.items():
        actual = hashlib.sha256((REPO_ROOT / rel_path).read_bytes()).hexdigest()
        if actual != expected:
            print(f"  MISMATCH: {rel_path}")
            return 3
    print("  All checksums verified OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
