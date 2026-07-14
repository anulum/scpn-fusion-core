# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Pretrained Eurofusion/JET FNO Surrogate
"""Pretrained FNO surrogate trained on synthetic Eurofusion/JET ψ(R,Z) fields.

Builds an augmented dataset from JET GEQDSK equilibria and trains/evaluates a
Fourier Neural Operator surrogate:

- :func:`_build_jet_fno_dataset` — resize + augment ψ(R,Z) into (x, y) pairs.
- :func:`_train_fno_on_jet` — train the FNO and persist its weights.
- :func:`evaluate_pretrained_fno` — relative-L2 metrics on held-out samples.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scpn_fusion.core._surrogate_utils import AdamOptimizer
from scpn_fusion.core.eqdsk import read_geqdsk
from scpn_fusion.core.fno_training import MultiLayerFNO

from ._pretrained_surrogate_config import DEFAULT_FNO_PATH, DEFAULT_JET_DIR, FloatArray


def _resize_2d(arr: FloatArray, out_h: int = 64, out_w: int = 64) -> FloatArray:
    src = np.asarray(arr, dtype=np.float64)
    if src.ndim != 2:
        raise ValueError("Expected 2D array.")

    in_h, in_w = src.shape
    x_in = np.linspace(0.0, 1.0, in_w, dtype=np.float64)
    x_out = np.linspace(0.0, 1.0, out_w, dtype=np.float64)
    row_interp = np.empty((in_h, out_w), dtype=np.float64)
    for i in range(in_h):
        row_interp[i] = np.interp(x_out, x_in, src[i])

    y_in = np.linspace(0.0, 1.0, in_h, dtype=np.float64)
    y_out = np.linspace(0.0, 1.0, out_h, dtype=np.float64)
    out = np.empty((out_h, out_w), dtype=np.float64)
    for j in range(out_w):
        out[:, j] = np.interp(y_out, y_in, row_interp[:, j])
    return out


def _build_jet_fno_dataset(
    *,
    jet_dir: Path = DEFAULT_JET_DIR,
    seed: int = 42,
    augment_per_file: int = 12,
) -> tuple[FloatArray, FloatArray]:
    rng = np.random.default_rng(int(seed))
    x_rows: list[FloatArray] = []
    y_rows: list[FloatArray] = []

    files = sorted(Path(jet_dir).glob("*.geqdsk"))
    if not files:
        raise ValueError(f"No JET GEQDSK files found in {jet_dir}.")

    for path in files:
        eq = read_geqdsk(path)
        psi = np.asarray(eq.psirz, dtype=np.float64)
        psi_norm = (psi - float(np.min(psi))) / (float(np.max(psi) - np.min(psi)) + 1e-12)
        base = _resize_2d(psi_norm, out_h=64, out_w=64)
        for _ in range(int(augment_per_file)):
            perturb = rng.normal(0.0, 0.008, size=base.shape)
            rolled = np.roll(base, int(rng.integers(-2, 3)), axis=0)
            rolled = np.roll(rolled, int(rng.integers(-2, 3)), axis=1)
            x = np.clip(rolled + perturb, 0.0, 1.2)

            lap = (
                np.roll(x, 1, axis=0)
                + np.roll(x, -1, axis=0)
                + np.roll(x, 1, axis=1)
                + np.roll(x, -1, axis=1)
                - 4.0 * x
            )
            y = np.clip(x + 0.06 * lap - 0.010 * (x * x), 0.0, 1.2)
            x_rows.append(x.astype(np.float64))
            y_rows.append(y.astype(np.float64))

    return np.asarray(x_rows, dtype=np.float64), np.asarray(y_rows, dtype=np.float64)


def _relative_l2(pred: FloatArray, target: FloatArray) -> float:
    denom = float(np.linalg.norm(target) + 1e-8)
    return float(np.linalg.norm(pred - target) / denom)


def _train_fno_on_jet(
    *,
    save_path: Path,
    seed: int = 42,
    modes: int = 8,
    width: int = 16,
    epochs: int = 24,
    batch_size: int = 8,
    augment_per_file: int = 12,
    jet_dir: Path = DEFAULT_JET_DIR,
) -> dict[str, float]:
    x, y = _build_jet_fno_dataset(jet_dir=jet_dir, seed=seed, augment_per_file=augment_per_file)
    n = x.shape[0]
    split = max(4, int(0.85 * n))
    x_train, y_train = x[:split], y[:split]
    x_val, y_val = x[split:], y[split:]

    model = MultiLayerFNO(modes=modes, width=width, n_layers=4, seed=seed)
    optim = AdamOptimizer()
    rng = np.random.default_rng(seed + 991)
    best_val = float("inf")
    best_w = model.project_w.copy()
    best_b = model.project_b

    for _ in range(int(epochs)):
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), int(batch_size)):
            batch = order[start : start + int(batch_size)]
            if batch.size == 0:
                continue
            grad_w = np.zeros_like(model.project_w)
            grad_b = 0.0
            for i in batch:
                pred, hidden = model.forward_with_hidden(x_train[int(i)])
                target = y_train[int(i)]
                scale = float(np.mean(target * target) + 1e-8)
                err = pred - target
                grad_y = (2.0 / err.size) * err / scale
                grad_w += np.tensordot(hidden, grad_y, axes=([0, 1], [0, 1]))
                grad_b += float(np.sum(grad_y))
            grad_w /= float(batch.size)
            grad_b /= float(batch.size)

            params = {
                "project_w": model.project_w,
                "project_b": np.array([model.project_b], dtype=np.float64),
            }
            grads = {
                "project_w": grad_w,
                "project_b": np.array([grad_b], dtype=np.float64),
            }
            optim.step(params, grads, lr=8e-4)
            model.project_b = float(params["project_b"][0])

        val_losses = [_relative_l2(model.forward(x_val[i]), y_val[i]) for i in range(len(x_val))]
        val = float(np.mean(np.asarray(val_losses, dtype=np.float64)))
        if val < best_val:
            best_val = val
            best_w = model.project_w.copy()
            best_b = model.project_b

    model.project_w = best_w
    model.project_b = best_b
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(save_path)

    train_losses = [
        _relative_l2(model.forward(x_train[i]), y_train[i]) for i in range(min(24, len(x_train)))
    ]
    val_losses = [_relative_l2(model.forward(x_val[i]), y_val[i]) for i in range(len(x_val))]
    return {
        "train_relative_l2": float(np.mean(np.asarray(train_losses, dtype=np.float64))),
        "val_relative_l2": float(np.mean(np.asarray(val_losses, dtype=np.float64))),
        "dataset_samples": float(n),
    }


def evaluate_pretrained_fno(
    *,
    fno_path: Path = DEFAULT_FNO_PATH,
    jet_dir: Path = DEFAULT_JET_DIR,
    seed: int = 47,
    augment_per_file: int = 8,
    max_samples: int = 16,
) -> dict[str, float]:
    """Evaluate a bundled FNO surrogate on synthetic Eurofusion-style references.

    Parameters
    ----------
    fno_path
        Path to the pretrained FNO weights file.
    jet_dir
        Directory containing GEQDSK cases used for evaluation sample generation.
    seed
        Random seed for deterministic augmentation in the synthetic dataset stage.
    augment_per_file
        Number of augmentations produced per source GEQDSK file.
    max_samples
        Maximum number of samples to include in evaluation; must be positive.

    Returns
    -------
    dict[str, float]
        Dictionary containing mean and 95th-percentile relative L2 errors and
        evaluated sample count.
    """
    if int(max_samples) <= 0:
        raise ValueError("max_samples must be > 0.")
    x, y = _build_jet_fno_dataset(jet_dir=jet_dir, seed=seed, augment_per_file=augment_per_file)
    if x.size == 0:
        raise ValueError("FNO evaluation dataset is empty.")
    m = MultiLayerFNO()
    m.load_weights(fno_path)
    n = min(int(max_samples), int(x.shape[0]))
    losses = [_relative_l2(m.forward(x[i]), y[i]) for i in range(n)]
    return {
        "eval_relative_l2_mean": float(np.mean(np.asarray(losses, dtype=np.float64))),
        "eval_relative_l2_p95": float(np.percentile(np.asarray(losses, dtype=np.float64), 95)),
        "eval_samples": float(n),
    }
