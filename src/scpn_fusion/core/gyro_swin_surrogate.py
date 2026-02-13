# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — GyroSwin-Like Turbulence Surrogate (GAI-01)
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Deterministic GyroSwin-like surrogate for core-turbulence benchmarking.

This module intentionally uses synthetic data and an offline, zero-dependency
training path (NumPy only). It is scoped for CI and reproducibility while
providing a measurable speed/accuracy comparison against a slower
"GENE-like proxy" baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np


@dataclass(frozen=True)
class TurbulenceDataset:
    """Synthetic core-turbulence dataset bundle."""

    features: np.ndarray
    chi_i: np.ndarray


@dataclass(frozen=True)
class SpeedBenchmark:
    """Per-sample timing comparison against a slower proxy solver."""

    gene_proxy_s_per_sample: float
    surrogate_s_per_sample: float
    speedup: float


def _as_2d(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != 10:
        raise ValueError("Expected feature matrix with shape (N, 10)")
    return x


def synthetic_core_turbulence_target(features: np.ndarray) -> np.ndarray:
    """Reference synthetic turbulence law used as the CI benchmark target."""

    x = _as_2d(features)
    rho, q, s_hat, beta_e, grad_ti, grad_te, coll, kappa, delta, shear = x.T

    drive_i = np.maximum(0.0, grad_ti - 4.0)
    drive_e = np.maximum(0.0, grad_te - 4.7)
    shape_factor = 1.0 + 0.28 * (kappa - 1.3) + 0.12 * delta
    shear_suppression = np.exp(-1.6 * np.abs(shear))
    mode_mix = 0.15 * np.sin(np.pi * rho * q * 0.35) ** 2 + 0.10 * np.tanh(22.0 * beta_e * s_hat)

    chi_i = (
        0.25
        + 0.13 * drive_i ** 1.18
        + 0.10 * drive_e ** 1.10
        + 0.05 * coll
        + mode_mix
    ) * shape_factor * shear_suppression
    return np.maximum(chi_i, 1e-6)


def generate_synthetic_gyrokinetic_dataset(seed: int, samples: int) -> TurbulenceDataset:
    """Generate deterministic synthetic "JET/ITPA-like" core turbulence samples."""

    if samples < 8:
        raise ValueError("samples must be >= 8")
    rng = np.random.default_rng(seed)
    rho = rng.uniform(0.05, 0.95, samples)
    q = rng.uniform(1.0, 3.5, samples)
    s_hat = rng.uniform(0.2, 2.2, samples)
    beta_e = rng.uniform(0.003, 0.045, samples)
    grad_ti = rng.uniform(3.0, 11.0, samples)
    grad_te = rng.uniform(3.0, 10.0, samples)
    coll = rng.uniform(0.05, 1.5, samples)
    kappa = rng.uniform(1.2, 2.0, samples)
    delta = rng.uniform(0.08, 0.5, samples)
    shear = rng.uniform(-0.45, 0.45, samples)
    features = np.column_stack([rho, q, s_hat, beta_e, grad_ti, grad_te, coll, kappa, delta, shear])
    chi_i = synthetic_core_turbulence_target(features)
    return TurbulenceDataset(features=features, chi_i=chi_i)


class GyroSwinLikeSurrogate:
    """Lightweight random-feature surrogate emulating a transformer-style map."""

    def __init__(self, hidden_dim: int = 64, ridge: float = 5e-4, seed: int = 42) -> None:
        if hidden_dim < 8:
            raise ValueError("hidden_dim must be >= 8")
        self.hidden_dim = int(hidden_dim)
        self.ridge = float(max(ridge, 1e-10))
        rng = np.random.default_rng(seed)
        self._omega = rng.normal(0.0, 0.65, size=(10, self.hidden_dim))
        self._phase = rng.uniform(-np.pi, np.pi, size=self.hidden_dim)
        self._weights: np.ndarray | None = None

    def _feature_map(self, features: np.ndarray) -> np.ndarray:
        x = _as_2d(features)
        rho, q, s_hat, beta_e, grad_ti, grad_te, coll, kappa, delta, shear = x.T
        z = x @ self._omega + self._phase
        rff = np.concatenate([np.sin(z), np.cos(z)], axis=1)
        physics_terms = np.column_stack(
            [
                np.maximum(0.0, grad_ti - 4.0),
                np.maximum(0.0, grad_te - 4.7),
                beta_e * s_hat * 22.0,
                rho * q,
                (kappa - 1.0),
                delta,
                shear * shear,
                np.exp(-np.abs(shear)),
            ]
        )
        return np.column_stack([np.ones(x.shape[0]), x, physics_terms, rff])

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        phi = self._feature_map(features)
        y = np.asarray(targets, dtype=np.float64).reshape(-1)
        if phi.shape[0] != y.shape[0]:
            raise ValueError("feature/target rows must match")
        lhs = phi.T @ phi + self.ridge * np.eye(phi.shape[1], dtype=np.float64)
        rhs = phi.T @ y
        self._weights = np.linalg.solve(lhs, rhs)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("Surrogate is not fit. Call fit() first.")
        phi = self._feature_map(features)
        out = phi @ self._weights
        return np.maximum(out, 1e-6)


def gene_proxy_predict(features: np.ndarray, iterations: int = 800) -> np.ndarray:
    """Slow "GENE-like" reference proxy for speed benchmarking.

    The proxy intentionally performs per-sample iterative updates to emulate
    heavier solver cost, while remaining deterministic and bounded for CI.
    """

    x = _as_2d(features)
    base = synthetic_core_turbulence_target(x)
    out = np.empty(x.shape[0], dtype=np.float64)

    for i, row in enumerate(x):
        state = row.copy()
        accum = 0.0
        for _ in range(iterations):
            rolled = np.roll(state, 1)
            state = np.tanh(0.84 * state + 0.14 * rolled + 0.02 * state * state)
            accum += (
                0.22 * state[4] * state[4]
                + 0.18 * state[5] * state[5]
                + 0.06 * abs(state[3])
                + 0.04 * state[2] * state[2]
            )
        correction = 1.0 + 0.015 * np.tanh(accum / max(iterations, 1) - 0.5)
        out[i] = max(base[i] * correction, 1e-6)
    return out


def rmse_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_p = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_t.size == 0 or y_t.shape != y_p.shape:
        raise ValueError("y_true/y_pred must be non-empty and same shape")
    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
    scale = float(max(np.mean(np.abs(y_t)), 1e-9))
    return 100.0 * rmse / scale


def benchmark_speedup(
    features: np.ndarray,
    surrogate: GyroSwinLikeSurrogate,
    min_baseline_s: float = 0.15,
    min_surrogate_s: float = 0.02,
) -> SpeedBenchmark:
    """Measure per-sample speedup of surrogate vs slow baseline proxy."""

    x = _as_2d(features)

    baseline_loops = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(baseline_loops):
            _ = gene_proxy_predict(x)
        baseline_total = time.perf_counter() - t0
        if baseline_total >= min_baseline_s or baseline_loops >= 32:
            break
        baseline_loops *= 2

    surrogate_loops = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(surrogate_loops):
            _ = surrogate.predict(x)
        surrogate_total = time.perf_counter() - t0
        if surrogate_total >= min_surrogate_s or surrogate_loops >= 4096:
            break
        surrogate_loops *= 2

    n_samples = x.shape[0]
    gene_s_per = baseline_total / (baseline_loops * n_samples)
    surrogate_s_per = surrogate_total / (surrogate_loops * n_samples)
    speedup = gene_s_per / max(surrogate_s_per, 1e-12)
    return SpeedBenchmark(
        gene_proxy_s_per_sample=gene_s_per,
        surrogate_s_per_sample=surrogate_s_per,
        speedup=speedup,
    )
