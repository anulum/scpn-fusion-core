"""Shared numeric helpers for FusionKernel solver paths."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

NUMERIC_SANITIZE_CAP = 1.0e250


def sanitize_numeric_array(arr: FloatArray, *, cap: float = NUMERIC_SANITIZE_CAP) -> FloatArray:
    """Return finite array with values clipped to ``[-cap, cap]``."""
    out = np.nan_to_num(np.asarray(arr, dtype=np.float64), nan=0.0, posinf=cap, neginf=-cap)
    if np.max(np.abs(out), initial=0.0) > cap:
        out = np.clip(out, -cap, cap)
    return out


def stable_rms(arr: FloatArray) -> float:
    """Compute RMS without overflow from squaring very large magnitudes."""
    vals = sanitize_numeric_array(arr)
    if vals.size == 0:
        return 0.0
    max_abs = float(np.max(np.abs(vals), initial=0.0))
    if max_abs <= 0.0:
        return 0.0
    scaled = vals / max_abs
    return float(max_abs * np.sqrt(np.mean(scaled * scaled)))
