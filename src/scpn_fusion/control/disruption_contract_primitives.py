# ----------------------------------------------------------------------
# SCPN Fusion Core -- Disruption Contract Primitives
# ----------------------------------------------------------------------
"""Small, reusable primitives shared by disruption contract lanes."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def require_finite_float(name: str, value: Any) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite.")
    return out


def require_positive_float(name: str, value: Any) -> float:
    out = require_finite_float(name, value)
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0.")
    return out


def require_int(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    out = int(value)
    if out < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return out


def require_fraction(name: str, value: Any) -> float:
    out = require_finite_float(name, value)
    if out < 0.0 or out > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1].")
    return out


def require_1d_array(
    name: str,
    value: Any,
    *,
    minimum_size: int = 1,
    expected_size: int | None = None,
) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if arr.size < minimum_size:
        raise ValueError(f"{name} must have at least {minimum_size} samples.")
    if expected_size is not None and arr.size != expected_size:
        raise ValueError(f"{name} must have {expected_size} samples (got {arr.size}).")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def gaussian_interval(
    *,
    mean: float,
    sigma: float,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    z_score: float = 1.96,
) -> tuple[float, float]:
    mean_f = require_finite_float("mean", mean)
    sigma_f = abs(require_finite_float("sigma", sigma))
    z_f = require_positive_float("z_score", z_score)
    lo = float(mean_f - z_f * sigma_f)
    hi = float(mean_f + z_f * sigma_f)
    if lower_bound is not None:
        lo = max(lo, float(lower_bound))
    if upper_bound is not None:
        hi = min(hi, float(upper_bound))
    if lo > hi:
        lo = hi
    return lo, hi


def synthetic_disruption_signal(
    *,
    rng: np.random.Generator,
    disturbance: float,
    window: int = 220,
) -> tuple[NDArray[np.float64], dict[str, float]]:
    """Generate a lightweight synthetic disruption precursor waveform."""
    t = np.linspace(0.0, 1.0, int(window), dtype=np.float64)
    base = 0.68 + 0.10 * np.sin(2.0 * np.pi * 2.4 * t + rng.uniform(-0.4, 0.4))
    elm = disturbance * (0.30 * np.exp(-(((t - 0.78) / 0.10) ** 2)))
    signal = np.clip(base + elm + rng.normal(0.0, 0.018, size=t.shape), 0.01, None)
    n1 = float(0.08 + 0.55 * disturbance + rng.uniform(0.00, 0.05))
    n2 = float(0.05 + 0.32 * disturbance + rng.uniform(0.00, 0.04))
    n3 = float(0.02 + 0.15 * disturbance + rng.uniform(0.00, 0.03))
    toroidal = {
        "toroidal_n1_amp": n1,
        "toroidal_n2_amp": n2,
        "toroidal_n3_amp": n3,
        "toroidal_asymmetry_index": float(np.sqrt(n1 * n1 + n2 * n2 + n3 * n3)),
        "toroidal_radial_spread": float(0.02 + 0.08 * disturbance),
        "proxy_signal_noise_sigma": 0.018,
        "proxy_disturbance_min": 0.0,
        "proxy_disturbance_max": 1.0,
        "proxy_is_synthetic": 1.0,
    }
    return signal, toroidal


__all__ = [
    "gaussian_interval",
    "require_1d_array",
    "require_finite_float",
    "require_fraction",
    "require_int",
    "require_positive_float",
    "synthetic_disruption_signal",
]

