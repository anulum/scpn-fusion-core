# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FRC Linear Surrogate Training Tool
"""
Trains a Linear/Ridge surrogate for the no-rotation FRC profile.

Generates data using the Steinhauer analytical limit, performs PCA on the
profiles, and trains a purely linear mapping matrix for offline experiments.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from scpn_fusion.core.frc_rigid_rotor import RigidRotorFRCInputs, solve_frc_equilibrium
from scpn_fusion.core.neural_equilibrium import MinimalPCA

logger = logging.getLogger(__name__)

PCA_COMPONENTS_TARGET = 10

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
            n0=n0, T_i_eV=t_i, T_e_eV=t_e, theta_dot=theta_dot,
            R_s=r_s, B_ext=b_ext, delta=delta
        )

        try:
            state = solve_frc_equilibrium(inputs, rho_grid, solver="rust", tolerance=1e-8)
            if not state.converged or not np.all(np.isfinite(state.B_z)):
                continue
            features = [n0/1e20, t_i/1000, t_e/1000, theta_dot/1000, r_s, b_ext, delta]
            X.append(features)
            Y.append(state.B_z)
            valid_count += 1
        except Exception:
            continue

    return np.array(X), np.array(Y)

# ── Training Entry Point ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train FRC linear surrogate")
    parser.add_argument("--samples", type=int, default=500, help="Number of samples to generate")
    parser.add_argument("--grid", type=int, default=256, help="Grid size for FRC")
    parser.add_argument("--out", default="weights/frc_linear_surrogate_v1.npz", help="Save path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--l2", type=float, default=1e-4, help="Ridge regression L2 penalty")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t_start = time.perf_counter()

    X_raw, Y_raw = generate_frc_data(args.samples, args.grid, args.seed)

    n_valid = len(X_raw)
    if n_valid < 10:
        logger.error("Insufficient samples generated. Aborting.")
        return

    logger.info("Successfully generated %d valid samples.", n_valid)

    # PCA reduction
    n_comp = min(n_valid - 1, PCA_COMPONENTS_TARGET)
    pca = MinimalPCA(n_components=n_comp)
    Y_latent = pca.fit_transform(Y_raw)
    explained_var = float(np.sum(pca.explained_variance_ratio_))
    logger.info("PCA: %d components explain %.2f%% variance", n_comp, explained_var * 100)

    # Normalise inputs
    x_mean = X_raw.mean(axis=0)
    x_std = X_raw.std(axis=0)
    x_std = np.where(x_std < 1e-10, 1.0, x_std)
    X_norm = (X_raw - x_mean) / x_std

    # Linear Regression (Ridge)
    logger.info("Solving linear system (Ridge Regression, L2=%.1e)...", args.l2)
    # Append bias column to X
    X_bias = np.hstack([X_norm, np.ones((n_valid, 1))])

    # Normal equation: W = (X^T X + alpha I)^-1 X^T Y
    # W shape will be (features + 1, n_comp)
    I = np.eye(X_bias.shape[1])
    I[-1, -1] = 0.0  # Don't penalize the bias term

    W_full = np.linalg.solve(X_bias.T @ X_bias + args.l2 * I, X_bias.T @ Y_latent)
    W_linear = W_full[:-1, :]
    b_linear = W_full[-1, :]

    # Evaluate
    Y_pred_latent = X_norm @ W_linear + b_linear
    mse = np.mean((Y_pred_latent - Y_latent)**2)
    logger.info("Linear Mapping MSE on Latent Space: %.6f", mse)

    # Save weights
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_path,
        n_components=np.array([n_comp]),
        grid_size=np.array([args.grid]),
        pca_mean=pca.mean_,
        pca_components=pca.components_,
        input_mean=x_mean,
        input_std=x_std,
        w_linear=W_linear,
        b_linear=b_linear
    )

    t_total = time.perf_counter() - t_start
    logger.info("Linear surrogate training complete in %.2f s. Saved to %s", t_total, out_path)

if __name__ == "__main__":
    main()
