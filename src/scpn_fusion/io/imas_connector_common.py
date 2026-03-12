# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — IMAS Connector Common
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Shared validation/coercion helpers for IMAS connector modules."""

from __future__ import annotations

from collections.abc import Sequence
import math
import numbers
from typing import Any, Mapping

REQUIRED_IDS_KEYS = (
    "schema",
    "machine",
    "time_slice",
    "equilibrium",
    "performance",
)

REQUIRED_DIGITAL_TWIN_SUMMARY_KEYS = (
    "steps",
    "final_islands_px",
    "final_reward",
    "reward_mean_last_50",
    "final_avg_temp",
)

REQUIRED_PROFILE_1D_KEYS = (
    "rho_norm",
    "electron_temp_keV",
    "electron_density_1e20_m3",
)

REQUIRED_DIGITAL_TWIN_STATE_KEYS = REQUIRED_DIGITAL_TWIN_SUMMARY_KEYS + REQUIRED_PROFILE_1D_KEYS


def _missing_required_keys(mapping: Mapping[str, Any], required: tuple[str, ...]) -> list[str]:
    return [key for key in required if key not in mapping]


def _coerce_int(
    name: str,
    value: Any,
    *,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer.")
    parsed = int(value)
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return parsed


def _coerce_finite_real(
    name: str,
    value: Any,
    *,
    minimum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite number.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite number.")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return parsed


def _coerce_finite_real_sequence(
    name: str,
    value: Any,
    *,
    minimum_len: int = 1,
    minimum: float | None = None,
    maximum: float | None = None,
    strictly_increasing: bool = False,
) -> list[float]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of finite numbers.")
    parsed: list[float] = []
    for idx, item in enumerate(value):
        parsed_item = _coerce_finite_real(f"{name}[{idx}]", item, minimum=minimum)
        if maximum is not None and parsed_item > maximum:
            raise ValueError(f"{name}[{idx}] must be <= {maximum}.")
        parsed.append(parsed_item)
    if len(parsed) < minimum_len:
        raise ValueError(f"{name} must contain at least {minimum_len} values.")
    if strictly_increasing:
        for idx in range(1, len(parsed)):
            if parsed[idx] <= parsed[idx - 1]:
                raise ValueError(f"{name} must be strictly increasing.")
    return parsed


def _coerce_profiles_1d(
    payload: Mapping[str, Any],
    *,
    name: str,
) -> dict[str, list[float]]:
    missing = _missing_required_keys(payload, REQUIRED_PROFILE_1D_KEYS)
    if missing:
        raise ValueError(f"{name} missing keys: {missing}")

    rho = _coerce_finite_real_sequence(
        f"{name}.rho_norm",
        payload.get("rho_norm"),
        minimum_len=2,
        minimum=0.0,
        maximum=1.0,
        strictly_increasing=True,
    )
    temp = _coerce_finite_real_sequence(
        f"{name}.electron_temp_keV",
        payload.get("electron_temp_keV"),
        minimum_len=2,
        minimum=0.0,
    )
    dens = _coerce_finite_real_sequence(
        f"{name}.electron_density_1e20_m3",
        payload.get("electron_density_1e20_m3"),
        minimum_len=2,
        minimum=0.0,
    )
    n = len(rho)
    if len(temp) != n:
        raise ValueError(f"{name}.electron_temp_keV length must match {name}.rho_norm.")
    if len(dens) != n:
        raise ValueError(f"{name}.electron_density_1e20_m3 length must match {name}.rho_norm.")

    return {
        "rho_norm": rho,
        "electron_temp_keV": temp,
        "electron_density_1e20_m3": dens,
    }


__all__ = [
    "REQUIRED_IDS_KEYS",
    "REQUIRED_DIGITAL_TWIN_SUMMARY_KEYS",
    "REQUIRED_PROFILE_1D_KEYS",
    "REQUIRED_DIGITAL_TWIN_STATE_KEYS",
    "_missing_required_keys",
    "_coerce_int",
    "_coerce_finite_real",
    "_coerce_finite_real_sequence",
    "_coerce_profiles_1d",
]
