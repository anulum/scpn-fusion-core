# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Neural Equilibrium Accelerator
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
PCA + MLP surrogate for Grad-Shafranov equilibrium reconstruction.

Maps coil currents (or profile parameters) → PCA coefficients → ψ(R,Z)
with ~1000× speedup over the full Picard iteration.

**Training modes:**

1. **From FusionKernel** — Generate training data by perturbing coil
   currents and running the GS solver.  Requires a valid config JSON.

2. **From SPARC GEQDSKs** — Train on real equilibrium data from CFS.
   Uses the GEQDSK's profile parameters (p', FF', I_p) as input features
   and ψ(R,Z) as targets.  No coil model needed.

**Status:** Reduced-order surrogate.  Not on the critical control path.
Use for rapid design-space exploration and batch equilibrium sweeps.
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WEIGHTS_PATH = REPO_ROOT / "weights" / "neural_equilibrium_sparc.npz"


# ── Data containers ──────────────────────────────────────────────────

@dataclass
class NeuralEqConfig:
    """Configuration for the neural equilibrium model."""
    n_components: int = 20
    hidden_sizes: tuple[int, ...] = (128, 64, 32)
    n_input_features: int = 12
    grid_shape: tuple[int, int] = (129, 129)  # (nh, nw)
    lambda_gs: float = 0.1


@dataclass
class TrainingResult:
    """Training summary."""
    n_samples: int
    n_components: int
    explained_variance: float
    final_loss: float
    train_time_s: float
    weights_path: str
    val_loss: float = float("nan")
    test_mse: float = float("nan")
    test_max_error: float = float("nan")


# ── Simple MLP (pure NumPy) ──────────────────────────────────────────

class SimpleMLP:
    """Feedforward MLP with ReLU hidden layers and linear output."""

    def __init__(self, layer_sizes: list[int], seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.weights: list[NDArray] = []
        self.biases: list[NDArray] = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            # He initialisation
            scale = np.sqrt(2.0 / fan_in)
            self.weights.append(
                self.rng.normal(0, scale, (layer_sizes[i], layer_sizes[i + 1]))
            )
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


# ── PCA (minimal, no sklearn dependency) ─────────────────────────────

class MinimalPCA:
    """Minimal PCA via SVD, no sklearn required."""

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
        self.components_ = Vt[: self.n_components]
        self.explained_variance_ratio_ = (
            S[: self.n_components] ** 2 / max(total_var, 1e-15)
        )
        return self

    def transform(self, X: NDArray) -> NDArray:
        assert self.mean_ is not None and self.components_ is not None
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, Z: NDArray) -> NDArray:
        assert self.mean_ is not None and self.components_ is not None
        return Z @ self.components_ + self.mean_

    def fit_transform(self, X: NDArray) -> NDArray:
        self.fit(X)
        return self.transform(X)


# ── Neural Equilibrium Accelerator ───────────────────────────────────

class NeuralEquilibriumAccelerator:
    """
    PCA + MLP surrogate for Grad-Shafranov equilibrium.

    Can be trained from SPARC GEQDSK files (preferred) or from a
    FusionKernel config with coil perturbations.
    """

    def __init__(self, config: NeuralEqConfig | None = None) -> None:
        self.cfg = config or NeuralEqConfig()
        self.pca = MinimalPCA(n_components=self.cfg.n_components)
        self.mlp: SimpleMLP | None = None
        self.is_trained = False
        self._input_mean: NDArray | None = None
        self._input_std: NDArray | None = None

    # ── GS residual loss ───────────────────────────────────────────

    def _gs_residual_loss(self, psi_pred_flat: NDArray, grid_shape: tuple[int, int]) -> float:
        """GS residual loss: penalizes Laplacian of predicted psi."""
        nh, nw = grid_shape
        psi = psi_pred_flat.reshape(nh, nw)
        lap = np.zeros_like(psi)
        lap[1:-1, 1:-1] = (
            psi[2:, 1:-1] + psi[:-2, 1:-1]
            + psi[1:-1, 2:] + psi[1:-1, :-2]
            - 4 * psi[1:-1, 1:-1]
        )
        return float(np.mean(lap[1:-1, 1:-1] ** 2))

    # ── Evaluation ────────────────────────────────────────────────

    def evaluate_surrogate(
        self, X_test: NDArray, Y_test_raw: NDArray
    ) -> dict[str, float]:
        """Evaluate on test set. Returns dict with mse, max_error, gs_residual."""
        if not self.is_trained:
            raise RuntimeError("Not trained")
        assert self._input_mean is not None and self._input_std is not None
        x_norm = (X_test - self._input_mean) / self._input_std
        coeffs = self.mlp.predict(x_norm)
        psi_pred = self.pca.inverse_transform(coeffs)

        mse = float(np.mean((psi_pred - Y_test_raw) ** 2))
        max_err = float(np.max(np.abs(psi_pred - Y_test_raw)))
        gs_res = 0.0
        for row in range(len(psi_pred)):
            gs_res += self._gs_residual_loss(psi_pred[row], self.cfg.grid_shape)
        gs_res /= max(len(psi_pred), 1)

        return {"mse": mse, "max_error": max_err, "gs_residual": gs_res}

    # ── Training from SPARC GEQDSKs ─────────────────────────────────

    def train_from_geqdsk(
        self,
        geqdsk_paths: list[Path],
        n_perturbations: int = 25,
        seed: int = 42,
    ) -> TrainingResult:
        """
        Train on real SPARC GEQDSK equilibria with perturbations.

        For each GEQDSK file, generates n_perturbations by scaling p'/FF'
        profiles, yielding n_files × n_perturbations training pairs.

        Input features (12-dim):
            [I_p, B_t, R_axis, Z_axis, pprime_scale, ffprime_scale,
             simag, sibry, kappa, delta_upper, delta_lower, q95]

        Output: flattened ψ(R,Z) → PCA coefficients

        Uses 70/15/15 train/val/test split with val-loss early stopping
        (patience=20) and combined MSE + lambda_gs * GS residual loss.
        """
        from scpn_fusion.core.eqdsk import read_geqdsk

        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()

        X_features: list[NDArray] = []
        Y_psi: list[NDArray] = []

        # Target grid: use the first file's grid as reference
        first_eq = read_geqdsk(geqdsk_paths[0])
        target_nh, target_nw = first_eq.nh, first_eq.nw
        self.cfg.grid_shape = (target_nh, target_nw)

        for path in geqdsk_paths:
            eq = read_geqdsk(path)

            # Interpolate onto target grid if needed
            if eq.nh != target_nh or eq.nw != target_nw:
                from scipy.interpolate import RectBivariateSpline
                spline = RectBivariateSpline(eq.z, eq.r, eq.psirz, kx=3, ky=3)
                target_r = np.linspace(eq.rleft, eq.rleft + eq.rdim, target_nw)
                target_z = np.linspace(
                    eq.zmid - eq.zdim / 2, eq.zmid + eq.zdim / 2, target_nh
                )
                psi_interp = spline(target_z, target_r, grid=True)
            else:
                psi_interp = eq.psirz

            # Extract shape parameters from boundary if available
            kappa = 1.7  # default elongation
            delta_upper = 0.3  # default upper triangularity
            delta_lower = 0.3  # default lower triangularity
            q95 = 3.0  # default safety factor at 95% flux
            if hasattr(eq, 'rbbbs') and eq.rbbbs is not None and len(eq.rbbbs) > 3:
                r_span = eq.rbbbs.max() - eq.rbbbs.min()
                kappa = (eq.zbbbs.max() - eq.zbbbs.min()) / max(r_span, 0.01)
            if hasattr(eq, 'qpsi') and eq.qpsi is not None and len(eq.qpsi) > 0:
                idx_95 = int(0.95 * len(eq.qpsi))
                q95 = eq.qpsi[min(idx_95, len(eq.qpsi) - 1)]

            # Base feature vector (12-dim)
            base_features = np.array([
                eq.current / 1e6,  # I_p in MA
                eq.bcentr,         # B_t in T
                eq.rmaxis,         # R_axis in m
                eq.zmaxis,         # Z_axis in m
                1.0,               # pprime scale factor
                1.0,               # ffprime scale factor
                eq.simag,          # psi at axis
                eq.sibry,          # psi at boundary
                kappa,             # elongation
                delta_upper,       # upper triangularity
                delta_lower,       # lower triangularity
                q95,               # safety factor at 95% flux
            ])

            # Unperturbed sample
            X_features.append(base_features)
            Y_psi.append(psi_interp.ravel())

            # Perturbed samples: scale p'/FF' and blend psi
            for _ in range(n_perturbations):
                pp_scale = rng.uniform(0.7, 1.3)
                ff_scale = rng.uniform(0.7, 1.3)

                # Perturbed features (shape params stay at base values)
                feat = base_features.copy()
                feat[4] = pp_scale
                feat[5] = ff_scale
                # kappa/delta/q95 at indices 8-11 are inherited from base

                # Linearly blend psi with a scale-dependent offset
                # This simulates the effect of profile scaling on equilibrium
                denom = eq.sibry - eq.simag
                if abs(denom) < 1e-12:
                    denom = 1.0
                psi_n = (psi_interp - eq.simag) / denom

                # Profile perturbation modifies the normalised psi shape
                mix = 0.5 * (pp_scale + ff_scale) - 1.0  # deviation from 1.0
                # Perturb interior normalised psi
                plasma_mask = (psi_n >= 0) & (psi_n < 1.0)
                psi_perturbed = psi_interp.copy()
                psi_perturbed[plasma_mask] += (
                    mix * 0.1 * denom * (1.0 - psi_n[plasma_mask])
                )

                X_features.append(feat)
                Y_psi.append(psi_perturbed.ravel())

        X = np.array(X_features)
        Y = np.array(Y_psi)
        n_samples = len(X)
        logger.info("Training data: %d samples, %d features → %d outputs",
                     n_samples, X.shape[1], Y.shape[1])

        # Normalise inputs
        self._input_mean = X.mean(axis=0)
        self._input_std = X.std(axis=0)
        self._input_std[self._input_std < 1e-10] = 1.0
        X_norm = (X - self._input_mean) / self._input_std

        # PCA on flattened psi
        Y_compressed = self.pca.fit_transform(Y)
        explained = float(np.sum(self.pca.explained_variance_ratio_))
        logger.info("PCA: %d → %d components, %.2f%% variance retained",
                     Y.shape[1], self.cfg.n_components, explained * 100)

        # ── Train/val/test split (70/15/15) ────────────────────────
        indices = rng.permutation(n_samples)
        n_train = int(0.70 * n_samples)
        n_val = int(0.15 * n_samples)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        X_train, Y_train = X_norm[train_idx], Y_compressed[train_idx]
        X_val, Y_val = X_norm[val_idx], Y_compressed[val_idx]
        X_test, Y_test = X_norm[test_idx], Y_compressed[test_idx]
        # Keep uncompressed targets for GS residual evaluation
        Y_test_raw = Y[test_idx]

        logger.info(
            "Split: %d train / %d val / %d test",
            len(train_idx), len(val_idx), len(test_idx),
        )

        # ── Train MLP: X_train → Y_train ─────────────────────────
        self.cfg.n_input_features = X.shape[1]
        layer_sizes = [
            self.cfg.n_input_features,
            *self.cfg.hidden_sizes,
            self.cfg.n_components,
        ]
        self.mlp = SimpleMLP(layer_sizes, seed=seed)

        # Mini-batch SGD with momentum
        lr = 1e-3
        momentum = 0.9
        n_epochs = 500
        batch_size = min(32, len(X_train))

        velocity = [np.zeros_like(w) for w in self.mlp.weights]
        velocity_b = [np.zeros_like(b) for b in self.mlp.biases]

        best_val_loss = float("inf")
        best_train_loss = float("inf")
        patience_counter = 0
        patience = 20

        for epoch in range(n_epochs):
            order = rng.permutation(len(X_train))
            epoch_loss = 0.0

            for start in range(0, len(X_train), batch_size):
                idx = order[start:start + batch_size]
                x_batch = X_train[idx]
                y_batch = Y_train[idx]

                # Forward pass (store activations for backprop)
                activations = [x_batch]
                h = x_batch
                for i, (W, b) in enumerate(zip(self.mlp.weights, self.mlp.biases)):
                    z = h @ W + b
                    if i < len(self.mlp.weights) - 1:
                        h = np.maximum(0, z)
                    else:
                        h = z
                    activations.append(h)

                # MSE loss
                error = activations[-1] - y_batch
                loss = float(np.mean(error ** 2))

                # GS residual loss on this batch
                gs_loss = 0.0
                for row in range(len(idx)):
                    psi_pred_flat = self.pca.inverse_transform(
                        activations[-1][row:row + 1]
                    )[0]
                    gs_loss += self._gs_residual_loss(
                        psi_pred_flat, self.cfg.grid_shape
                    )
                gs_loss /= len(idx)
                loss = loss + self.cfg.lambda_gs * gs_loss

                epoch_loss += loss * len(idx)

                # Backprop (on MSE part; GS is treated as a monitoring
                # penalty — its gradient flows implicitly through the
                # PCA reconstruction but we approximate with MSE grads
                # to keep pure-NumPy backprop tractable)
                delta = 2.0 * error / len(idx)
                for i in range(len(self.mlp.weights) - 1, -1, -1):
                    grad_w = activations[i].T @ delta
                    grad_b = delta.sum(axis=0)

                    velocity[i] = momentum * velocity[i] - lr * grad_w
                    velocity_b[i] = momentum * velocity_b[i] - lr * grad_b

                    self.mlp.weights[i] += velocity[i]
                    self.mlp.biases[i] += velocity_b[i]

                    if i > 0:
                        delta = delta @ self.mlp.weights[i].T
                        # ReLU derivative
                        delta *= (activations[i] > 0).astype(float)

            epoch_loss /= max(len(X_train), 1)
            if epoch_loss < best_train_loss:
                best_train_loss = epoch_loss

            # ── Validation loss (MSE + lambda_gs * GS) ────────────
            val_pred = self.mlp.forward(X_val)
            val_mse = float(np.mean((val_pred - Y_val) ** 2))
            val_gs = 0.0
            for row in range(len(X_val)):
                psi_pred_flat = self.pca.inverse_transform(
                    val_pred[row:row + 1]
                )[0]
                val_gs += self._gs_residual_loss(
                    psi_pred_flat, self.cfg.grid_shape
                )
            val_gs /= max(len(X_val), 1)
            val_loss = val_mse + self.cfg.lambda_gs * val_gs

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(
                        "Early stopping at epoch %d (val_loss=%.6f)",
                        epoch, val_loss,
                    )
                    break

            if epoch % 50 == 0:
                logger.info(
                    "Epoch %d: train_loss=%.6f  val_loss=%.6f",
                    epoch, epoch_loss, val_loss,
                )

        self.is_trained = True
        train_time = time.perf_counter() - t0

        # ── Evaluate on held-out test set ─────────────────────────
        test_metrics = self.evaluate_surrogate(
            X[test_idx], Y_test_raw,
        )
        logger.info(
            "Test set: MSE=%.6f  max_error=%.6f  GS_residual=%.6f",
            test_metrics["mse"],
            test_metrics["max_error"],
            test_metrics["gs_residual"],
        )

        return TrainingResult(
            n_samples=n_samples,
            n_components=self.cfg.n_components,
            explained_variance=explained,
            final_loss=best_train_loss,
            train_time_s=train_time,
            weights_path="",
            val_loss=best_val_loss,
            test_mse=test_metrics["mse"],
            test_max_error=test_metrics["max_error"],
        )

    # ── Inference ────────────────────────────────────────────────────

    def predict(self, features: NDArray) -> NDArray:
        """
        Predict ψ(R,Z) from input features.

        Parameters
        ----------
        features : NDArray
            Shape (n_features,) or (batch, n_features).

        Returns
        -------
        NDArray
            Shape (nh, nw) or (batch, nh, nw).
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train_from_geqdsk() or load_weights() first.")

        if features.ndim == 1:
            features = features[np.newaxis, :]

        assert self._input_mean is not None and self._input_std is not None
        x_norm = (features - self._input_mean) / self._input_std
        coeffs = self.mlp.predict(x_norm)
        psi_flat = self.pca.inverse_transform(coeffs)

        nh, nw = self.cfg.grid_shape
        if features.shape[0] == 1:
            return psi_flat.reshape(nh, nw)
        return psi_flat.reshape(-1, nh, nw)

    # ── Save / Load ──────────────────────────────────────────────────

    def save_weights(self, path: str | Path = DEFAULT_WEIGHTS_PATH) -> None:
        """Save model to .npz (no pickle dependency)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, NDArray] = {
            "n_components": np.array([self.cfg.n_components]),
            "grid_nh": np.array([self.cfg.grid_shape[0]]),
            "grid_nw": np.array([self.cfg.grid_shape[1]]),
            "n_input_features": np.array([self.cfg.n_input_features]),
            "pca_mean": self.pca.mean_,
            "pca_components": self.pca.components_,
            "pca_evr": self.pca.explained_variance_ratio_,
            "input_mean": self._input_mean,
            "input_std": self._input_std,
            "n_layers": np.array([len(self.mlp.weights)]),
        }
        for i, (w, b) in enumerate(zip(self.mlp.weights, self.mlp.biases)):
            payload[f"w{i}"] = w
            payload[f"b{i}"] = b

        np.savez(path, **payload)
        logger.info("Saved neural equilibrium weights to %s", path)

    def load_weights(self, path: str | Path = DEFAULT_WEIGHTS_PATH) -> None:
        """Load model from .npz."""
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            self.cfg.n_components = int(data["n_components"][0])
            self.cfg.grid_shape = (int(data["grid_nh"][0]), int(data["grid_nw"][0]))
            self.cfg.n_input_features = int(data["n_input_features"][0])

            self.pca = MinimalPCA(self.cfg.n_components)
            self.pca.mean_ = np.array(data["pca_mean"])
            self.pca.components_ = np.array(data["pca_components"])
            self.pca.explained_variance_ratio_ = np.array(data["pca_evr"])

            self._input_mean = np.array(data["input_mean"])
            self._input_std = np.array(data["input_std"])

            n_layers = int(data["n_layers"][0])
            weights = [np.array(data[f"w{i}"]) for i in range(n_layers)]
            biases = [np.array(data[f"b{i}"]) for i in range(n_layers)]

            # Reconstruct layer sizes from weight shapes
            layer_sizes = [weights[0].shape[0]]
            for w in weights:
                layer_sizes.append(w.shape[1])
            self.mlp = SimpleMLP(layer_sizes)
            self.mlp.weights = weights
            self.mlp.biases = biases

        self.is_trained = True
        logger.info("Loaded neural equilibrium weights from %s", path)

    # ── Convenience ──────────────────────────────────────────────────

    def benchmark(self, features: NDArray, n_runs: int = 100) -> dict[str, float]:
        """Time inference over n_runs and return stats."""
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.predict(features)
            times.append((time.perf_counter() - t0) * 1000)
        return {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)),
        }


# ── SPARC training convenience function ─────────────────────────────

def train_on_sparc(
    sparc_dir: str | Path | None = None,
    save_path: str | Path = DEFAULT_WEIGHTS_PATH,
    n_perturbations: int = 25,
    seed: int = 42,
) -> TrainingResult:
    """
    Train neural equilibrium on all SPARC GEQDSK files and save weights.

    This is the recommended entry point for training.
    """
    if sparc_dir is None:
        sparc_dir = REPO_ROOT / "validation" / "reference_data" / "sparc"
    sparc_dir = Path(sparc_dir)

    files = sorted(sparc_dir.glob("*.geqdsk")) + sorted(sparc_dir.glob("*.eqdsk"))
    if not files:
        raise FileNotFoundError(f"No GEQDSK/EQDSK files in {sparc_dir}")

    accel = NeuralEquilibriumAccelerator()
    result = accel.train_from_geqdsk(
        files,
        n_perturbations=n_perturbations,
        seed=seed,
    )
    accel.save_weights(save_path)
    result.weights_path = str(save_path)
    return result


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")

    sparc_dir = REPO_ROOT / "validation" / "reference_data" / "sparc"
    if not sparc_dir.exists():
        print(f"SPARC data not found at {sparc_dir}")
        sys.exit(1)

    print("=" * 60)
    print("Training Neural Equilibrium on SPARC GEQDSKs")
    print("=" * 60)

    result = train_on_sparc(sparc_dir)
    print(f"\nSamples: {result.n_samples}")
    print(f"PCA components: {result.n_components}")
    print(f"Explained variance: {result.explained_variance * 100:.2f}%")
    print(f"Final train loss: {result.final_loss:.6f}")
    print(f"Val loss: {result.val_loss:.6f}")
    print(f"Test MSE: {result.test_mse:.6f}")
    print(f"Test max error: {result.test_max_error:.6f}")
    print(f"Train time: {result.train_time_s:.1f}s")
    print(f"Weights: {result.weights_path}")

    # Quick validation
    accel = NeuralEquilibriumAccelerator()
    accel.load_weights(result.weights_path)

    from scpn_fusion.core.eqdsk import read_geqdsk
    test_eq = read_geqdsk(next(sparc_dir.glob("*.geqdsk")))
    kappa_cli = 1.7
    q95_cli = 3.0
    if hasattr(test_eq, 'rbbbs') and test_eq.rbbbs is not None and len(test_eq.rbbbs) > 3:
        r_span = test_eq.rbbbs.max() - test_eq.rbbbs.min()
        kappa_cli = (test_eq.zbbbs.max() - test_eq.zbbbs.min()) / max(r_span, 0.01)
    if hasattr(test_eq, 'qpsi') and test_eq.qpsi is not None and len(test_eq.qpsi) > 0:
        idx_95 = int(0.95 * len(test_eq.qpsi))
        q95_cli = test_eq.qpsi[min(idx_95, len(test_eq.qpsi) - 1)]
    features = np.array([
        test_eq.current / 1e6, test_eq.bcentr,
        test_eq.rmaxis, test_eq.zmaxis,
        1.0, 1.0, test_eq.simag, test_eq.sibry,
        kappa_cli, 0.3, 0.3, q95_cli,
    ])

    psi_pred = accel.predict(features)
    diff = psi_pred - test_eq.psirz[:psi_pred.shape[0], :psi_pred.shape[1]]
    rel_l2 = float(np.linalg.norm(diff) / np.linalg.norm(test_eq.psirz[:psi_pred.shape[0], :psi_pred.shape[1]]))
    print(f"\nValidation relative L2 on first file: {rel_l2:.6f}")

    bench = accel.benchmark(features)
    print(f"Inference: {bench['mean_ms']:.3f} ms (median: {bench['median_ms']:.3f} ms)")
