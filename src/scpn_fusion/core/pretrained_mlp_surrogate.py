# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Pretrained ITPA MLP Surrogate
"""Pretrained MLP surrogate for ITPA H-mode confinement-time prediction.

Trains a compact two-layer MLP on the ITPA confinement dataset and provides
serialisation and evaluation without external ML dependencies:

- :class:`PretrainedMLPSurrogate` — in-memory weights + normalisation metadata.
- :func:`save_pretrained_mlp` / :func:`load_pretrained_mlp` — NumPy archive I/O.
- :func:`evaluate_pretrained_mlp` — RMSE metrics against the ITPA reference.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scpn_fusion.io.safe_loaders import checked_np_load

from ._pretrained_surrogate_config import (
    DEFAULT_BUNDLE_MLP_PATH,
    DEFAULT_ITPA_CSV,
    DEFAULT_MLP_PATH,
    FloatArray,
)


@dataclass(frozen=True)
class PretrainedMLPSurrogate:
    """In-memory two-layer MLP surrogate and normalisation metadata.

    The weights are stored in a compact numpy format suitable for
    :func:`numpy.savez` serialization and later reloading without external ML
    dependencies.
    """

    feature_mean: FloatArray
    feature_std: FloatArray
    w1: FloatArray
    b1: FloatArray
    w2: FloatArray
    b2: FloatArray
    target_mean: float
    target_std: float

    def predict(self, features: FloatArray) -> FloatArray:
        """Predict confinement time from feature matrix.

        Parameters
        ----------
        features
            Two-dimensional feature matrix with shape ``(n_samples, n_features)``.

        Returns
        -------
        FloatArray
            Predicted confinement times in seconds, one entry per sample. Inputs
            with shape ``(n_features,)`` are treated as a single-sample batch.
        """
        x = np.asarray(features, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != self.feature_mean.size:
            raise ValueError(f"Expected feature width {self.feature_mean.size}, got {x.shape[1]}.")
        x_norm = (x - self.feature_mean[None, :]) / self.feature_std[None, :]
        h = np.tanh(x_norm @ self.w1 + self.b1[None, :])
        y_norm = h @ self.w2 + self.b2
        y = y_norm * self.target_std + self.target_mean
        return np.asarray(np.maximum(y.reshape(-1), 1e-6), dtype=np.float64)


def _load_itpa_training_data(csv_path: Path = DEFAULT_ITPA_CSV) -> tuple[FloatArray, FloatArray]:
    features: list[list[float]] = []
    targets: list[float] = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            features.append(
                [
                    float(row["Ip_MA"]),
                    float(row["BT_T"]),
                    float(row["ne19_1e19m3"]),
                    float(row["Ploss_MW"]),
                    float(row["R_m"]),
                    float(row["a_m"]),
                    float(row["kappa"]),
                    float(row["delta"]),
                    float(row["M_AMU"]),
                ]
            )
            targets.append(float(row["tau_E_s"]))
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 8:
        raise ValueError("ITPA dataset must contain at least 8 rows.")
    return x, y


def _train_itpa_mlp(
    *,
    seed: int = 42,
    hidden: int = 32,
    epochs: int = 1200,
    lr: float = 1.2e-2,
    l2: float = 5.0e-4,
    csv_path: Path = DEFAULT_ITPA_CSV,
) -> tuple[PretrainedMLPSurrogate, dict[str, float]]:
    x, y = _load_itpa_training_data(csv_path)
    rng = np.random.default_rng(int(seed))

    x_mu = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    x_n = (x - x_mu[None, :]) / x_std[None, :]

    y_mu = float(np.mean(y))
    y_std = float(np.std(y)) if float(np.std(y)) > 1e-8 else 1.0
    y_n = (y - y_mu) / y_std

    n, d = x_n.shape
    h = int(hidden)
    w1 = rng.normal(0.0, 0.22, size=(d, h))
    b1 = np.zeros((h,), dtype=np.float64)
    w2 = rng.normal(0.0, 0.18, size=(h,))
    b2 = np.array(0.0, dtype=np.float64)

    for _ in range(int(epochs)):
        h_act = np.tanh(x_n @ w1 + b1[None, :])
        pred = h_act @ w2 + b2
        err = pred - y_n

        grad_pred = (2.0 / n) * err
        grad_w2 = h_act.T @ grad_pred + l2 * w2
        grad_b2 = float(np.sum(grad_pred))
        grad_h = grad_pred[:, None] * w2[None, :]
        grad_z1 = grad_h * (1.0 - h_act * h_act)
        grad_w1 = x_n.T @ grad_z1 + l2 * w1
        grad_b1 = np.sum(grad_z1, axis=0)

        w2 -= lr * grad_w2
        b2 -= lr * grad_b2
        w1 -= lr * grad_w1
        b1 -= lr * grad_b1

    model = PretrainedMLPSurrogate(
        feature_mean=x_mu.astype(np.float64),
        feature_std=x_std.astype(np.float64),
        w1=w1.astype(np.float64),
        b1=b1.astype(np.float64),
        w2=w2.astype(np.float64),
        b2=np.asarray(b2, dtype=np.float64),
        target_mean=y_mu,
        target_std=y_std,
    )
    pred_tau = model.predict(x)
    rmse = float(np.sqrt(np.mean((pred_tau - y) ** 2)))
    rmse_pct = float(100.0 * rmse / (np.mean(np.abs(y)) + 1e-12))
    return model, {"train_rmse_s": rmse, "train_rmse_pct": rmse_pct}


def save_pretrained_mlp(
    model: PretrainedMLPSurrogate, path: Path = DEFAULT_BUNDLE_MLP_PATH
) -> None:
    """Persist a pretrained MLP surrogate to a compressed NumPy archive.

    Parameters
    ----------
    model
        Trained :class:`PretrainedMLPSurrogate` instance.
    path
        Target path where the ``.npz`` artifact will be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        feature_mean=model.feature_mean,
        feature_std=model.feature_std,
        w1=model.w1,
        b1=model.b1,
        w2=model.w2,
        b2=model.b2,
        target_mean=np.asarray([model.target_mean], dtype=np.float64),
        target_std=np.asarray([model.target_std], dtype=np.float64),
    )


def load_pretrained_mlp(path: Path = DEFAULT_MLP_PATH) -> PretrainedMLPSurrogate:
    """Load a persisted pretrained MLP surrogate from disk.

    Parameters
    ----------
    path
        Path to the saved ``.npz`` bundle.

    Returns
    -------
    PretrainedMLPSurrogate
        Reconstructed surrogate ready for deterministic inference.
    """
    with checked_np_load(path, allow_pickle=False) as data:
        return PretrainedMLPSurrogate(
            feature_mean=np.asarray(data["feature_mean"], dtype=np.float64),
            feature_std=np.asarray(data["feature_std"], dtype=np.float64),
            w1=np.asarray(data["w1"], dtype=np.float64),
            b1=np.asarray(data["b1"], dtype=np.float64),
            w2=np.asarray(data["w2"], dtype=np.float64),
            b2=np.asarray(data["b2"], dtype=np.float64),
            target_mean=float(np.asarray(data["target_mean"], dtype=np.float64).reshape(-1)[0]),
            target_std=float(np.asarray(data["target_std"], dtype=np.float64).reshape(-1)[0]),
        )


def evaluate_pretrained_mlp(
    *,
    model_path: Path = DEFAULT_MLP_PATH,
    csv_path: Path = DEFAULT_ITPA_CSV,
    max_samples: int = 0,
) -> dict[str, float]:
    """Evaluate a pretrained MLP surrogate on ITPA reference samples.

    Parameters
    ----------
    model_path
        Path to the pretrained MLP archive.
    csv_path
        Input CSV dataset used for reference metrics.
    max_samples
        If set to a positive integer, only this many rows are used.

    Returns
    -------
    dict[str, float]
        RMSE metrics in seconds and percent plus evaluated sample count.
    """
    model = load_pretrained_mlp(path=model_path)
    x, y = _load_itpa_training_data(csv_path)
    if max_samples > 0:
        x = x[:max_samples]
        y = y[:max_samples]
    pred = model.predict(x)
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    rmse_pct = float(100.0 * rmse / (np.mean(np.abs(y)) + 1e-12))
    return {"rmse_s": rmse, "rmse_pct": rmse_pct, "samples": float(x.shape[0])}
