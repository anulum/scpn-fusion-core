# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Strict array ABI admission
"""Shared strict array and scalar admission for public numerical kernels."""

from __future__ import annotations

import math
import sys
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
INDEX_LIMIT = min((1 << 63) - 1, sys.maxsize, int(np.iinfo(np.uintp).max))


def array_input(value: object, name: str) -> FloatArray:
    """Require a native float64 matrix and strip subclass behavior without coercion."""
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy ndarray")
    array = np.ndarray.view(value, np.ndarray)
    if array.dtype != np.dtype(np.float64) or not array.dtype.isnative:
        raise TypeError(f"{name} must have native float64 dtype")
    if array.ndim != 2:
        raise ValueError(f"{name} must have rank two")
    return array


def scalar_kind(value: object, name: str, *, integer: bool) -> None:
    """Check delivered scalar kinds before any scalar domain checks or conversion."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must not be Boolean")
    if isinstance(value, (int, np.integer)):
        return
    if not integer and (
        isinstance(value, float)
        or (isinstance(value, np.floating) and np.generic.dtype.__get__(value).itemsize <= 8)
    ):
        return
    raise TypeError(f"{name} has an unsupported {'integer' if integer else 'real'} scalar kind")


def integer_value(value: int | np.integer[Any], name: str, minimum: int) -> int:
    """Read an admitted mathematical integer before checking its native index domain."""
    if isinstance(value, int):
        result = int.__int__(value)
    else:
        result = np.integer.__int__(value)
    if result < minimum or result > INDEX_LIMIT:
        raise ValueError(f"{name} must be between {minimum} and {INDEX_LIMIT}")
    return result


def real_value(value: int | np.integer[Any] | float | np.floating[Any], name: str) -> float:
    """Read an admitted finite real, preserving mathematical integer exactness."""
    if isinstance(value, int):
        integer = int.__int__(value)
    elif isinstance(value, np.integer):
        integer = np.integer.__int__(value)
    else:
        integer = None
    try:
        if integer is not None:
            result = float(integer)
            if not math.isfinite(result) or int(result) != integer:
                raise ValueError(f"{name} must be exactly representable in binary64")
        elif isinstance(value, float):
            result = float.__float__(value)
        else:
            result = np.floating.__float__(cast(np.floating[Any], value))
    except OverflowError as error:
        raise ValueError(f"{name} must be finite binary64") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def checked_grid(nr: int, nz: int) -> None:
    """Refuse logical element and binary64 buffer sizes outside native address space."""
    if nr > INDEX_LIMIT // nz or nr * nz > INDEX_LIMIT // 8:
        raise ValueError("grid element or byte count exceeds native address space")


def grid_spacing(lower: float, upper: float, count: int, name: str) -> float:
    """Require ordered endpoints and a finite positive nominal grid spacing."""
    spacing = (upper - lower) / (count - 1)
    if not lower < upper or not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError(f"{name} bounds must give a finite positive spacing")
    return spacing


def finite_array(array: FloatArray, name: str) -> None:
    """Require every logical input cell to be finite before owned copies or numerical work."""
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
