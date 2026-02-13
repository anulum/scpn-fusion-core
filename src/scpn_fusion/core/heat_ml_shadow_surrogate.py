# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — HEAT-ML Shadow Surrogate (GAI-03)
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Deterministic HEAT-ML surrogate for magnetic-shadow divertor optimization."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np


@dataclass(frozen=True)
class ShadowDataset:
    features: np.ndarray
    shadow_fraction: np.ndarray


def _as_2d(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != 7:
        raise ValueError("Expected shape (N, 7): [R, B_pol, P_sol, fx, kappa, delta, xpt_z]")
    return x


def synthetic_shadow_reference(features: np.ndarray) -> np.ndarray:
    """Synthetic reference law for divertor magnetic-shadow fraction."""

    x = _as_2d(features)
    r, b_pol, p_sol, fx, kappa, delta, xpt_z = x.T

    field_term = 0.20 + 0.24 * np.tanh(0.65 * (b_pol - 1.3))
    geometry_term = 0.16 * np.tanh(0.20 * (fx - 10.0)) + 0.10 * np.tanh(1.1 * (kappa - 1.5))
    xpt_term = 0.08 * np.exp(-((xpt_z + 1.7) ** 2) / 0.28)
    power_penalty = 0.18 * np.tanh(0.020 * (p_sol - 70.0))
    shaping_bonus = 0.06 * delta

    shadow = field_term + geometry_term + xpt_term + shaping_bonus - power_penalty
    return np.clip(shadow, 0.03, 0.82)


def generate_shadow_dataset(seed: int, samples: int) -> ShadowDataset:
    if samples < 8:
        raise ValueError("samples must be >= 8")
    rng = np.random.default_rng(seed)
    r = rng.uniform(1.0, 3.5, samples)
    b_pol = rng.uniform(0.8, 3.8, samples)
    p_sol = rng.uniform(20.0, 180.0, samples)
    fx = rng.uniform(6.0, 24.0, samples)
    kappa = rng.uniform(1.2, 2.2, samples)
    delta = rng.uniform(0.05, 0.65, samples)
    xpt_z = rng.uniform(-2.6, -1.1, samples)
    feats = np.column_stack([r, b_pol, p_sol, fx, kappa, delta, xpt_z])
    return ShadowDataset(features=feats, shadow_fraction=synthetic_shadow_reference(feats))


class HeatMLShadowSurrogate:
    """Compact polynomial HEAT-ML surrogate with deterministic ridge fit."""

    def __init__(self, ridge: float = 1e-4) -> None:
        self.ridge = float(max(ridge, 1e-10))
        self._weights: np.ndarray | None = None

    def _feature_map(self, features: np.ndarray) -> np.ndarray:
        x = _as_2d(features)
        r, b_pol, p_sol, fx, kappa, delta, xpt_z = x.T
        phi = np.column_stack(
            [
                np.ones(x.shape[0]),
                x,
                b_pol * fx,
                p_sol / np.maximum(fx, 1e-6),
                kappa * delta,
                np.exp(-((xpt_z + 1.7) ** 2) / 0.30),
                np.tanh(0.02 * (p_sol - 70.0)),
                np.tanh(0.20 * (fx - 10.0)),
                np.tanh(0.65 * (b_pol - 1.3)),
            ]
        )
        return phi

    def fit(self, features: np.ndarray, target: np.ndarray) -> None:
        phi = self._feature_map(features)
        y = np.asarray(target, dtype=np.float64).reshape(-1)
        if y.shape[0] != phi.shape[0]:
            raise ValueError("features and target row count mismatch")
        lhs = phi.T @ phi + self.ridge * np.eye(phi.shape[1], dtype=np.float64)
        rhs = phi.T @ y
        self._weights = np.linalg.solve(lhs, rhs)

    def fit_synthetic(self, seed: int = 42, samples: int = 2048) -> None:
        ds = generate_shadow_dataset(seed=seed, samples=samples)
        self.fit(ds.features, ds.shadow_fraction)

    def predict_shadow_fraction(self, features: np.ndarray) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("Model is not fit. Call fit() first.")
        phi = self._feature_map(features)
        out = phi @ self._weights
        return np.clip(out, 0.0, 0.85)

    def predict_divertor_flux(
        self,
        q_div_baseline_w_m2: np.ndarray | float,
        features: np.ndarray,
    ) -> np.ndarray:
        q = np.asarray(q_div_baseline_w_m2, dtype=np.float64)
        shadow = self.predict_shadow_fraction(features)
        # Map shadow fraction to effective heat-load attenuation.
        atten = 1.0 - 0.58 * shadow
        return np.maximum(q * atten, 1e3)


def rmse_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if yt.size == 0 or yt.shape != yp.shape:
        raise ValueError("y_true/y_pred must be non-empty and same shape")
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    denom = float(max(np.mean(np.abs(yt)), 1e-9))
    return 100.0 * rmse / denom


def benchmark_inference_seconds(model: HeatMLShadowSurrogate, samples: int = 200_000) -> float:
    rng = np.random.default_rng(123)
    feats = np.column_stack(
        [
            rng.uniform(1.0, 3.5, samples),
            rng.uniform(0.8, 3.8, samples),
            rng.uniform(20.0, 180.0, samples),
            rng.uniform(6.0, 24.0, samples),
            rng.uniform(1.2, 2.2, samples),
            rng.uniform(0.05, 0.65, samples),
            rng.uniform(-2.6, -1.1, samples),
        ]
    )
    t0 = time.perf_counter()
    _ = model.predict_shadow_fraction(feats)
    return float(time.perf_counter() - t0)
