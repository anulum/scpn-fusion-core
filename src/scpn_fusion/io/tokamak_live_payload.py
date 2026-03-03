# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Tokamak Live Payload Helpers
# ──────────────────────────────────────────────────────────────────────
"""Small payload-coercion helpers extracted from tokamak_archive."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def to_scalar(value: Any) -> float:
    """Convert scalar/array payload to finite float (last finite entry)."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Empty MDSplus node payload.")
    flat = arr.reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        raise ValueError("No finite values in MDSplus payload.")
    return float(finite[-1])


def to_trace(value: Any, points: int, *, resample_1d: Any) -> NDArray[np.float64]:
    """Convert payload to finite 1-D trace and resample to requested points."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.repeat(arr.reshape(1), 8)
    elif arr.ndim > 1:
        arr = arr.reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        raise ValueError("Trace payload must contain at least 2 finite values.")
    return resample_1d(arr, points)

