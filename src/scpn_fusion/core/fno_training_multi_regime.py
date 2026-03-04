# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — FNO Multi-Regime Training Runtime
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""Multi-regime turbulence data generation and training helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── SPARC-relevant turbulence regime parameters ──────────────────────
#
# Each regime maps to a range of physical parameters used to construct
# the modified Hasegawa-Wakatani spectral time-stepper.
SPARC_REGIMES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "itg": {
        "alpha": (0.1, 0.5),
        "kappa": (5.0, 15.0),
        "nu": (0.001, 0.01),
        "damp": (0.05, 0.15),
        "k_cut": (4.0, 8.0),
    },
    "tem": {
        "alpha": (0.5, 2.0),
        "kappa": (2.0, 8.0),
        "nu": (0.005, 0.05),
        "damp": (0.10, 0.25),
        "k_cut": (6.0, 12.0),
    },
    "etg": {
        "alpha": (1.0, 3.0),
        "kappa": (3.0, 12.0),
        "nu": (0.01, 0.1),
        "damp": (0.15, 0.30),
        "k_cut": (10.0, 20.0),
    },
}


def _relative_l2(pred: np.ndarray, target: np.ndarray) -> float:
    denom = np.linalg.norm(target) + 1e-8
    return float(np.linalg.norm(pred - target) / denom)


def _sample_regime_params(
    rng: np.random.Generator,
    regime: str,
) -> Dict[str, float]:
    """Sample a random parameter vector from a given turbulence regime."""
    bounds = SPARC_REGIMES[regime]
    return {k: rng.uniform(lo, hi) for k, (lo, hi) in bounds.items()}


def _generate_multi_regime_pairs(
    n_samples: int,
    grid_size: int,
    seed: int,
    regime_weights: Dict[str, float] | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    """
    Generate training data spanning ITG / TEM / ETG regimes.

    Each sample uses a randomly sampled parameter set from one of the
    three turbulence regimes.
    """
    rng = np.random.default_rng(seed)
    regimes = list(SPARC_REGIMES.keys())

    if regime_weights is None:
        probs = np.ones(len(regimes)) / len(regimes)
    else:
        probs = np.array([regime_weights.get(r, 1.0) for r in regimes])
        probs /= probs.sum()

    x = np.empty((n_samples, grid_size, grid_size), dtype=np.float64)
    y = np.empty_like(x)
    metadata: List[Dict[str, object]] = []

    # Pre-compute wavenumber grids
    kx = np.fft.fftfreq(grid_size) * grid_size
    ky = np.fft.fftfreq(grid_size) * grid_size
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k2 = kx_grid**2 + ky_grid**2
    k4 = k2**2
    k2_safe = k2.copy()
    k2_safe[0, 0] = 1.0  # avoid div-by-zero

    dt = 0.01

    for i in range(n_samples):
        regime = rng.choice(regimes, p=probs)
        params = _sample_regime_params(rng, regime)

        alpha = params["alpha"]
        kappa = params["kappa"]
        nu = params["nu"]
        damp = params["damp"]
        k_cut = params["k_cut"]

        # Modified H-W dispersion: ω = α·ky/(α+k²)
        denom = alpha + k2_safe
        omega = alpha * ky_grid / denom

        # Linear growth rate: γ = κ·ky·k²/(α+k²)² - ν·k⁴
        growth = kappa * ky_grid * k2 / (denom**2) - nu * k4

        # Spectral cutoff: exponential damping above k_cut
        spectral_filter = np.exp(-((k2 / k_cut**2) ** 2))

        # Phase rotation and growth/damping over dt
        phase_shift = np.exp(-1j * omega * dt)
        amplitude = np.exp(growth * dt) * spectral_filter * (1.0 - damp)

        # Low-k forcing mask (drives the instability)
        mask_low_k = (k2 < (k_cut * 0.5) ** 2).astype(np.float64)

        # Initial field: filtered noise with regime-dependent spectrum
        field = rng.standard_normal((grid_size, grid_size)) * 0.1
        field_k = np.fft.fft2(field) * spectral_filter

        # Forcing: random low-k injection (simulates gradient drive)
        forcing_r = rng.standard_normal((grid_size, grid_size))
        forcing_i = rng.standard_normal((grid_size, grid_size))
        forcing_k = np.fft.fft2(forcing_r + 1j * forcing_i) * mask_low_k
        forcing_k *= kappa * 0.5  # Scale forcing by gradient drive

        # Time step
        next_k = (field_k * phase_shift * amplitude) + forcing_k * dt

        x[i] = np.fft.ifft2(field_k).real
        y[i] = np.fft.ifft2(next_k).real

        metadata.append(
            {
                "regime": regime,
                "alpha": alpha,
                "kappa": kappa,
                "nu": nu,
                "damp": damp,
                "k_cut": k_cut,
            }
        )

    return x, y, metadata


def train_fno_multi_regime(
    *,
    n_samples: int,
    epochs: int,
    lr: float,
    modes: int,
    width: int,
    save_path: str | Path,
    batch_size: int,
    seed: int,
    patience: int,
    regime_weights: Dict[str, float] | None = None,
) -> Dict[str, object]:
    """Train FNO on multi-regime SPARC-parameterized turbulence data."""
    # Local import avoids module cycle with fno_training.py re-exports.
    from .fno_training import AdamOptimizer, MultiLayerFNO

    logger.info("Generating %d multi-regime samples (ITG/TEM/ETG)...", n_samples)
    x, y, meta = _generate_multi_regime_pairs(
        n_samples=n_samples,
        grid_size=64,
        seed=seed,
        regime_weights=regime_weights,
    )

    split = max(1, int(0.9 * n_samples))
    x_train, y_train = x[:split], y[:split]
    x_val, y_val = x[split:], y[split:]
    meta_val = meta[split:]

    # Count regimes
    regime_counts: Dict[str, int] = {}
    for m in meta:
        r = m["regime"]
        regime_counts[r] = regime_counts.get(r, 0) + 1
    logger.info("Regime distribution: %s", regime_counts)

    model = MultiLayerFNO(modes=modes, width=width, n_layers=4, seed=seed)
    optimizer = AdamOptimizer()
    rng = np.random.default_rng(seed + 456)

    history: Dict[str, object] = {
        "train_loss": [],
        "val_loss": [],
        "best_epoch": 0,
        "best_val_loss": float("inf"),
        "trained_parameters": "project_head_only",
        "samples": n_samples,
        "epochs_requested": epochs,
        "regime_counts": regime_counts,
        "data_mode": "multi_regime_sparc",
    }

    best_project_w = model.project_w.copy()
    best_project_b = model.project_b
    best_val = float("inf")
    wait = 0

    for epoch in range(epochs):
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), batch_size):
            batch_idx = order[start : start + batch_size]
            grad_w = np.zeros_like(model.project_w)
            grad_b = 0.0

            for i in batch_idx:
                pred, hidden = model.forward_with_hidden(x_train[i])
                target = y_train[i]
                target_energy = float(np.mean(target * target) + 1e-8)
                error = pred - target

                grad_y = (2.0 / error.size) * error / target_energy
                grad_w += np.tensordot(hidden, grad_y, axes=([0, 1], [0, 1]))
                grad_b += float(np.sum(grad_y))

            if len(batch_idx) == 0:
                continue

            grad_w /= len(batch_idx)
            grad_b /= len(batch_idx)

            params = {
                "project_w": model.project_w,
                "project_b": np.array([model.project_b], dtype=np.float64),
            }
            grads = {
                "project_w": grad_w,
                "project_b": np.array([grad_b], dtype=np.float64),
            }
            optimizer.step(params, grads, lr=lr)
            model.project_b = float(params["project_b"][0])

        # Keep validation path unchanged from legacy implementation.
        def _evaluate_loss(data_x: np.ndarray, data_y: np.ndarray, max_samples: int = 16) -> float:
            n = min(max_samples, len(data_x))
            if n == 0:
                return 0.0
            idx = np.arange(n)
            losses = []
            for i in idx:
                pred = model.forward(data_x[i])
                losses.append(_relative_l2(pred, data_y[i]))
            return float(np.mean(losses))

        train_loss = _evaluate_loss(x_train, y_train)
        val_loss = _evaluate_loss(x_val, y_val)
        history["train_loss"].append(train_loss)  # type: ignore[attr-defined]
        history["val_loss"].append(val_loss)  # type: ignore[attr-defined]

        if val_loss < best_val:
            best_val = val_loss
            best_project_w = model.project_w.copy()
            best_project_b = model.project_b
            history["best_epoch"] = epoch + 1
            history["best_val_loss"] = val_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

        if epoch % 50 == 0:
            logger.info("Epoch %d: train=%.4f val=%.4f", epoch, train_loss, val_loss)

    model.project_w = best_project_w
    model.project_b = best_project_b
    model.save_weights(save_path)

    # Per-regime validation breakdown
    regime_val_losses: Dict[str, List[float]] = {}
    for j in range(len(x_val)):
        r = meta_val[j]["regime"]
        pred = model.forward(x_val[j])
        loss_j = _relative_l2(pred, y_val[j])
        regime_val_losses.setdefault(r, []).append(loss_j)

    regime_summary = {
        r: {"mean": float(np.mean(v)), "n": len(v)}
        for r, v in regime_val_losses.items()
    }
    history["regime_val_losses"] = regime_summary

    history["saved_path"] = str(Path(save_path))
    history["epochs_completed"] = len(history["train_loss"])  # type: ignore[arg-type]
    history["final_train_loss"] = float(history["train_loss"][-1]) if history["train_loss"] else None  # type: ignore[index]
    history["final_val_loss"] = float(history["val_loss"][-1]) if history["val_loss"] else None  # type: ignore[index]

    logger.info(
        "Multi-regime training complete: %d epochs, best_val=%.4f",
        history["epochs_completed"],
        best_val,
    )
    for r, s in regime_summary.items():
        logger.info("  %s: mean_val_loss=%.4f (n=%d)", r, s["mean"], s["n"])

    return history

