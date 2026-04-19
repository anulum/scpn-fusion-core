# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Integrated Transport Solver Contracts
from __future__ import annotations

from typing import Any

import numpy as np

_RUST_CHANG_HINTON_DEFAULTS: dict[str, float] = {
    "R0": 6.2,
    "a": 2.0,
    "B0": 5.3,
    "A_ion": 2.0,
    "Z_eff": 1.5,
}
_RUST_CHANG_HINTON_PARAM_ATOL = 1e-12


def require_positive_finite_scalar(name: str, value: Any) -> float:
    """Validate finite-positive scalar inputs for transport kernels."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc
    if (not np.isfinite(parsed)) or parsed <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return parsed


def coerce_matching_1d_profiles(**profiles: Any) -> dict[str, np.ndarray]:
    """Coerce profile-like inputs to matching 1-D float64 arrays."""
    out: dict[str, np.ndarray] = {}
    expected_shape: tuple[int, ...] | None = None
    for name, values in profiles.items():
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be a 1-D array, got shape {arr.shape}")
        if arr.size == 0:
            raise ValueError(f"{name} must not be empty")
        if expected_shape is None:
            expected_shape = arr.shape
        elif arr.shape != expected_shape:
            raise ValueError(
                f"All profiles must have the same shape; {name} has {arr.shape}, "
                f"expected {expected_shape}"
            )
        out[name] = arr
    return out


def rust_chang_hinton_params_match_defaults(
    *,
    R0: float,
    a: float,
    B0: float,
    A_ion: float,
    Z_eff: float,
) -> bool:
    """Return whether Chang-Hinton inputs match Rust default-parameter contract."""
    params = {
        "R0": float(R0),
        "a": float(a),
        "B0": float(B0),
        "A_ion": float(A_ion),
        "Z_eff": float(Z_eff),
    }
    for key, value in params.items():
        if not np.isclose(
            value,
            _RUST_CHANG_HINTON_DEFAULTS[key],
            rtol=0.0,
            atol=_RUST_CHANG_HINTON_PARAM_ATOL,
        ):
            return False
    return True


__all__ = [
    "_RUST_CHANG_HINTON_DEFAULTS",
    "_RUST_CHANG_HINTON_PARAM_ATOL",
    "require_positive_finite_scalar",
    "coerce_matching_1d_profiles",
    "rust_chang_hinton_params_match_defaults",
]
