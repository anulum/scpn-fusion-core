# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — IMAS ``equilibrium`` IDS bridge (solver fields ⇄ OMAS ODS ⇄ JSON)
"""IMAS ``equilibrium`` IDS bridge: solver equilibrium fields ⇄ the IMAS data dictionary.

The collaboration's exchange contract delivers equilibria as IMAS IDSs. This module maps a solved
free-boundary equilibrium (the SI fields the solvers in this package produce) into the IMAS
``equilibrium`` IDS via `OMAS <https://gafusion.github.io/omas/>`_ — whose :class:`~omas.ODS`
validates every path against the IMAS data dictionary on assignment — and back, plus JSON file
persistence (OMAS's backend-free serialisation, no ITER access layer required).

Conventions (fail-closed where they could silently corrupt):

* The solver's ψ is ``(NZ, NR)`` (rows = Z); the IMAS ``profiles_2d`` grid is
  ``[dim1, dim2] = [R, Z]``, so ψ is stored **transposed** — both directions are covered by an
  orientation test that cannot self-cancel.
* Only rectangular grids are handled (``grid_type.index = 1``); reading any other grid type raises.
* Shape mismatches between ψ and the grids raise instead of writing a malformed IDS.
* **COCOS is handled, not assumed** (audited 2026-07-21). The solver's native frame is
  **COCOS 3**: its Green's function carries an extra ``1/2π`` over the Maxwell mutual-inductance
  form, so ψ is the flux *per radian* (``exp_Bp = 0``), and ``Ip > 0`` gives ``Δ*ψ < 0`` — ψ
  *peaked* at the axis (``σ_Bp = −1``) with an effective ``p' > 0`` — matching the in-package
  Sauter table entry for COCOS 3 exactly. The IMAS data dictionary is COCOS 11, so every write
  and read goes through :func:`omas.omas_environment` with ``cocosio=solver_cocos`` and OMAS's
  own verified machinery applies the transform (measured: ψ ↦ −2π·ψ with ``ip`` unchanged for
  3 → 11; ψ ↦ +2π·ψ with ``ip ↦ −ip`` for 4 → 11, where σ_RφZ differs). The φ-handedness
  (odd/even COCOS pair) is *not observable* by an axisymmetric 2-D solver, so ``solver_cocos``
  is a parameter (default ``3``); a partner asserting the opposite handedness passes ``4``.
  **Only the audited pair {3, 4} is accepted** — any other frame (including 11 itself, which
  would silently pass IMAS-frame ψ through untransformed) is rejected until a general contract
  is separately proven. Optional quantities that were not computed are simply absent from the
  IDS (never fabricated as zeros).
* **Field validation is fail-closed in both directions**: ψ, the grids, ``time`` and every
  present optional scalar must be finite, and the coordinate vectors must be 1-D, ≥ 2 points
  and strictly increasing (the solver grid convention — descending or duplicated coordinates
  are rejected, not silently reordered). The same invariants are enforced on IDS read, so a
  corrupted or foreign IDS cannot leak non-physical fields into the solvers.

Scope: this is the structural I/O bridge (fields ⇄ IDS ⇄ file). Fetching *measured* equilibria from
a facility MDSplus/IMAS database is a separate, authorisation-gated concern.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from omas import ODS, load_omas_json, omas_environment, save_omas_json

_TS = "equilibrium.time_slice.0."
_RECTANGULAR = 1  # IMAS grid_type.index for a rectangular R,Z grid
DEFAULT_SOLVER_COCOS = 3  # audited native frame of the solvers (see module docstring)
_SUPPORTED_SOLVER_COCOS = (3, 4)  # the audited frame and its unobservable handedness partner


def _check_solver_cocos(solver_cocos: int) -> None:
    """Reject any frame outside the audited pair — fail closed, never silently mislabel."""
    if solver_cocos not in _SUPPORTED_SOLVER_COCOS:
        raise ValueError(
            f"solver_cocos must be one of {_SUPPORTED_SOLVER_COCOS} (the audited native frame "
            f"and its φ-handedness partner); got {solver_cocos}. Other frames — including 11, "
            "which would pass IMAS-frame ψ through untransformed — have no verified contract "
            "in this bridge."
        )


def _check_axis(name: str, axis: NDArray[np.float64]) -> None:
    """A coordinate vector must be 1-D, ≥ 2 points, finite and strictly increasing."""
    if axis.ndim != 1 or axis.size < 2:
        raise ValueError(
            f"{name} must be a 1-D coordinate vector with at least 2 points; got shape {axis.shape}"
        )
    if not np.all(np.isfinite(axis)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.all(np.diff(axis) > 0.0):
        raise ValueError(
            f"{name} must be strictly increasing (solver grid convention); descending or "
            "duplicated coordinates are rejected, not silently reordered"
        )


def _check_fields(
    psi: NDArray[np.float64],
    r: NDArray[np.float64],
    z: NDArray[np.float64],
    time: float,
    scalars: dict[str, float | None],
) -> None:
    """Shared fail-closed validation for both bridge directions (write and read)."""
    _check_axis("R_grid", r)
    _check_axis("Z_grid", z)
    if psi.shape != (z.size, r.size):
        raise ValueError(
            f"psi shape {psi.shape} does not match solver convention (NZ, NR) = ({z.size}, {r.size})"
        )
    if not np.all(np.isfinite(psi)):
        raise ValueError("psi contains non-finite values")
    if not np.isfinite(time):
        raise ValueError(f"time must be finite; got {time}")
    for name, value in scalars.items():
        if value is not None and not np.isfinite(value):
            raise ValueError(f"{name} must be finite when present; got {value}")


@dataclass(frozen=True)
class EquilibriumSlice:
    """One equilibrium time slice in solver convention (ψ shape ``(NZ, NR)``, SI units)."""

    psi: NDArray[np.float64]
    R_grid: NDArray[np.float64]
    Z_grid: NDArray[np.float64]
    ip: float | None = None
    r_axis: float | None = None
    z_axis: float | None = None
    psi_axis: float | None = None
    psi_boundary: float | None = None
    time: float = 0.0


def equilibrium_to_ods(
    eq: EquilibriumSlice, ods: ODS | None = None, solver_cocos: int = DEFAULT_SOLVER_COCOS
) -> ODS:
    """Write an :class:`EquilibriumSlice` into an IMAS ``equilibrium`` IDS (a new ODS by default).

    ψ is validated against the grids and stored transposed to the IMAS ``[dim1, dim2] = [R, Z]``
    layout. Optional quantities (``ip``, axis, ψ levels) are written only when present — absent
    values stay absent in the IDS. Every assignment is schema-checked by OMAS against the IMAS
    data dictionary.
    """
    _check_solver_cocos(solver_cocos)
    psi = np.asarray(eq.psi, dtype=np.float64)
    r = np.asarray(eq.R_grid, dtype=np.float64)
    z = np.asarray(eq.Z_grid, dtype=np.float64)
    _check_fields(
        psi,
        r,
        z,
        float(eq.time),
        {
            "ip": eq.ip,
            "r_axis": eq.r_axis,
            "z_axis": eq.z_axis,
            "psi_axis": eq.psi_axis,
            "psi_boundary": eq.psi_boundary,
        },
    )
    out = ODS() if ods is None else ods
    with omas_environment(out, cocosio=solver_cocos):
        _write_slice(out, eq, psi, r, z)
    return out


def _write_slice(
    out: ODS,
    eq: EquilibriumSlice,
    psi: NDArray[np.float64],
    r: NDArray[np.float64],
    z: NDArray[np.float64],
) -> None:
    out["equilibrium.time"] = np.array([eq.time], dtype=np.float64)
    out[_TS + "time"] = float(eq.time)
    out[_TS + "profiles_2d.0.grid_type.index"] = _RECTANGULAR
    out[_TS + "profiles_2d.0.grid.dim1"] = r
    out[_TS + "profiles_2d.0.grid.dim2"] = z
    out[_TS + "profiles_2d.0.psi"] = psi.T  # (NZ, NR) → IMAS [dim1, dim2] = [R, Z]
    if eq.ip is not None:
        out[_TS + "global_quantities.ip"] = float(eq.ip)
    if eq.r_axis is not None:
        out[_TS + "global_quantities.magnetic_axis.r"] = float(eq.r_axis)
    if eq.z_axis is not None:
        out[_TS + "global_quantities.magnetic_axis.z"] = float(eq.z_axis)
    if eq.psi_axis is not None:
        out[_TS + "global_quantities.psi_axis"] = float(eq.psi_axis)
    if eq.psi_boundary is not None:
        out[_TS + "global_quantities.psi_boundary"] = float(eq.psi_boundary)


def _optional(ods: ODS, path: str) -> float | None:
    """A scalar at ``path`` if present in the IDS, else ``None`` (absence is not an error)."""
    try:
        return float(ods[path])
    except (LookupError, ValueError):
        return None


def ods_to_equilibrium(ods: ODS, solver_cocos: int = DEFAULT_SOLVER_COCOS) -> EquilibriumSlice:
    """Read the first ``equilibrium`` time slice back into solver convention.

    Fails closed on a non-rectangular ``grid_type``, on a ψ whose shape does not match the
    stored grids, and on any field violating the bridge invariants (non-finite values,
    non-monotonic coordinate vectors) — a corrupted or foreign IDS is rejected, not passed
    through into the solvers.
    """
    _check_solver_cocos(solver_cocos)
    with omas_environment(ods, cocosio=solver_cocos):
        return _read_slice(ods)


def _read_slice(ods: ODS) -> EquilibriumSlice:
    grid_type = int(ods[_TS + "profiles_2d.0.grid_type.index"])
    if grid_type != _RECTANGULAR:
        raise ValueError(
            f"only rectangular grids are supported (grid_type.index=1), got {grid_type}"
        )
    r = np.asarray(ods[_TS + "profiles_2d.0.grid.dim1"], dtype=np.float64)
    z = np.asarray(ods[_TS + "profiles_2d.0.grid.dim2"], dtype=np.float64)
    psi_rz = np.asarray(ods[_TS + "profiles_2d.0.psi"], dtype=np.float64)
    if psi_rz.shape != (r.size, z.size):
        raise ValueError(
            f"stored psi shape {psi_rz.shape} does not match IMAS [dim1, dim2] = ({r.size}, {z.size})"
        )
    eq = EquilibriumSlice(
        psi=psi_rz.T,  # IMAS [R, Z] → solver (NZ, NR)
        R_grid=r,
        Z_grid=z,
        ip=_optional(ods, _TS + "global_quantities.ip"),
        r_axis=_optional(ods, _TS + "global_quantities.magnetic_axis.r"),
        z_axis=_optional(ods, _TS + "global_quantities.magnetic_axis.z"),
        psi_axis=_optional(ods, _TS + "global_quantities.psi_axis"),
        psi_boundary=_optional(ods, _TS + "global_quantities.psi_boundary"),
        time=float(ods[_TS + "time"]),
    )
    _check_fields(
        eq.psi,
        r,
        z,
        eq.time,
        {
            "ip": eq.ip,
            "r_axis": eq.r_axis,
            "z_axis": eq.z_axis,
            "psi_axis": eq.psi_axis,
            "psi_boundary": eq.psi_boundary,
        },
    )
    return eq


def save_equilibrium_ids(ods: ODS, path: str) -> None:
    """Persist the IDS as OMAS JSON (backend-free, no ITER access layer)."""
    save_omas_json(ods, path)


def load_equilibrium_ids(path: str) -> ODS:
    """Load an OMAS-JSON IDS written by :func:`save_equilibrium_ids`."""
    result = load_omas_json(path)
    if not isinstance(result, ODS):
        raise TypeError(f"expected an ODS from {path!r}, got {type(result).__name__}")
    return result
