# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Typed physical case admission
"""TOML 1.0 admission for the shared fixed-boundary physical case.

Limits bound software admission. They do not qualify numerical accuracy or
authorise expensive execution. File input supplies every field, including mu0.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TypedDict

import numpy as np

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


class CaseMapping(TypedDict):
    """The thirteen typed solver arguments, with SI physical scalars."""

    R_min: float
    R_max: float
    Z_min: float
    Z_max: float
    NR: int
    NZ: int
    Ip_target: float
    mu0: float
    n_picard: int
    n_jacobi: int
    alpha: float
    omega_j: float
    beta_mix: float


def _integer(value: object, field: str) -> int:
    """Read an actual Python integer without coercion or subclass conversion."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field} requires Integer, excluding Boolean and integral Float")
    number = int.__int__(value)
    if not -(1 << 63) <= number < (1 << 63):
        raise ValueError(f"{field} integer exceeds signed64")
    return number


def _real(value: object, field: str) -> float:
    """Read finite binary64 or a mathematically exact signed64 integer."""
    if isinstance(value, int):
        integer = _integer(value, field)
        number = float(integer)
        if int(number) != integer:
            raise ValueError(f"{field} integer is not exactly representable in binary64")
    elif isinstance(value, float):
        number = float.__float__(value)
    else:
        raise TypeError(f"{field} requires Float or exact Integer")
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


@dataclass(frozen=True)
class GradShafranovCase:
    """Validated physical input shared by file admission and direct NumPy solves.

    Radii/heights are metres, signed target current is amperes and mu0 is H/m.
    Counts are actual signed64 integers. Grid counts are 3:1025, iteration
    counts 1:10000 and their checked product is at most 100000000.
    """

    R_min: float
    R_max: float
    Z_min: float
    Z_max: float
    NR: int
    NZ: int
    Ip_target: float
    mu0: float
    n_picard: int
    n_jacobi: int
    alpha: float
    omega_j: float
    beta_mix: float

    def __post_init__(self) -> None:
        """Validate delivered kinds, domains, work bounds and actual mesh axes."""
        count_names = {"NR", "NZ", "n_picard", "n_jacobi"}
        for field in fields(self):
            value: object = getattr(self, field.name)
            normalized = (
                _integer(value, field.name)
                if field.name in count_names
                else _real(value, field.name)
            )
            object.__setattr__(self, field.name, normalized)
        if not 0.0 < self.R_min < self.R_max or not self.Z_min < self.Z_max:
            raise ValueError("require positive ordered R and ordered Z bounds")
        if not (3 <= self.NR <= 1025 and 3 <= self.NZ <= 1025):
            raise ValueError("grid counts must be in [3,1025]")
        if not (1 <= self.n_picard <= 10000 and 1 <= self.n_jacobi <= 10000):
            raise ValueError("iteration counts must be in [1,10000]")
        work = 1
        for count in (self.NR, self.NZ, self.n_picard, self.n_jacobi):
            if count > 100000000 // work:
                raise ValueError("case exceeds 100000000 point iterations")
            work *= count
        if self.mu0 <= 0.0:
            raise ValueError("mu0 must be positive")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0,1]")
        if not 0.0 < self.omega_j < 2.0:
            raise ValueError("omega_j must be in (0,2)")
        if not 0.0 <= self.beta_mix <= 1.0:
            raise ValueError("beta_mix must be in [0,1]")
        for start, stop, count in (
            (self.R_min, self.R_max, self.NR),
            (self.Z_min, self.Z_max, self.NZ),
        ):
            step = (stop - start) / (count - 1)
            if not math.isfinite(step) or step <= 0.0:
                raise ValueError("grid spacing must be finite and positive")
            # Check the formula used by native counterparts as well as NumPy's
            # actual endpoint-inclusive linspace used by the public solver.
            previous = start
            for index in range(1, count):
                current = start + step * index
                if not math.isfinite(current) or current <= previous:
                    raise ValueError("grid axis has nonfinite or collapsed adjacent nodes")
                previous = current
            axis = np.linspace(start, stop, count, dtype=np.float64)
            if not np.all(np.isfinite(axis)) or not np.all(np.diff(axis) > 0.0):
                raise ValueError("actual NumPy grid axis is nonfinite or collapsed")

    def as_mapping(self) -> CaseMapping:
        """Return independent typed keyword arguments for the public solver.

        Returns
        -------
        CaseMapping
            All thirteen normalized physical and iteration values.
        """
        return CaseMapping(
            R_min=self.R_min,
            R_max=self.R_max,
            Z_min=self.Z_min,
            Z_max=self.Z_max,
            NR=self.NR,
            NZ=self.NZ,
            Ip_target=self.Ip_target,
            mu0=self.mu0,
            n_picard=self.n_picard,
            n_jacobi=self.n_jacobi,
            alpha=self.alpha,
            omega_j=self.omega_j,
            beta_mix=self.beta_mix,
        )


def case_from_toml(path: str | Path) -> GradShafranovCase:
    """Parse a complete TOML 1.0 physical case with no implicit file defaults.

    Parameters
    ----------
    path : str or pathlib.Path
        UTF-8 TOML file containing only the grad_shafranov table.

    Returns
    -------
    GradShafranovCase
        Typed and validated SI physical input.

    Raises
    ------
    TypeError
        A field has an unsupported delivered scalar kind.
    ValueError
        Syntax, schema, exactness, finite domain or admission limits fail.
    OSError
        The file cannot be read.
    """
    with Path(path).open("rb") as stream:
        root = tomllib.load(stream)
    table: object = root.get("grad_shafranov")
    if set(root) != {"grad_shafranov"} or not isinstance(table, dict):
        raise ValueError("expected only the grad_shafranov table")
    expected = {field.name for field in fields(GradShafranovCase)}
    missing = expected - set(table)
    if missing:
        raise ValueError(
            "missing required Grad-Shafranov case field: " + ", ".join(sorted(missing))
        )
    if set(table) != expected:
        raise ValueError("expected exactly thirteen required case fields")
    return GradShafranovCase(**table)
