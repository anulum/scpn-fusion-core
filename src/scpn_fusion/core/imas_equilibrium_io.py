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
* Values are written **as solver SI** (ψ [Wb], Ip [A], lengths [m]) with no COCOS sign/2π
  transformation — a COCOS audit against the partner's convention is an explicit follow-up, not
  silently claimed here. Optional quantities that were not computed are simply absent from the IDS
  (never fabricated as zeros).

Scope: this is the structural I/O bridge (fields ⇄ IDS ⇄ file). Fetching *measured* equilibria from
a facility MDSplus/IMAS database is a separate, authorisation-gated concern.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from omas import ODS, load_omas_json, save_omas_json

_TS = "equilibrium.time_slice.0."
_RECTANGULAR = 1  # IMAS grid_type.index for a rectangular R,Z grid


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


def equilibrium_to_ods(eq: EquilibriumSlice, ods: ODS | None = None) -> ODS:
    """Write an :class:`EquilibriumSlice` into an IMAS ``equilibrium`` IDS (a new ODS by default).

    ψ is validated against the grids and stored transposed to the IMAS ``[dim1, dim2] = [R, Z]``
    layout. Optional quantities (``ip``, axis, ψ levels) are written only when present — absent
    values stay absent in the IDS. Every assignment is schema-checked by OMAS against the IMAS
    data dictionary.
    """
    psi = np.asarray(eq.psi, dtype=np.float64)
    r = np.asarray(eq.R_grid, dtype=np.float64)
    z = np.asarray(eq.Z_grid, dtype=np.float64)
    if psi.shape != (z.size, r.size):
        raise ValueError(
            f"psi shape {psi.shape} does not match solver convention (NZ, NR) = ({z.size}, {r.size})"
        )
    out = ODS() if ods is None else ods
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
    return out


def _optional(ods: ODS, path: str) -> float | None:
    """A scalar at ``path`` if present in the IDS, else ``None`` (absence is not an error)."""
    try:
        return float(ods[path])
    except (LookupError, ValueError):
        return None


def ods_to_equilibrium(ods: ODS) -> EquilibriumSlice:
    """Read the first ``equilibrium`` time slice back into solver convention.

    Fails closed on a non-rectangular ``grid_type`` and on a ψ whose shape does not match the
    stored grids, rather than returning silently mis-oriented fields.
    """
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
    return EquilibriumSlice(
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


def save_equilibrium_ids(ods: ODS, path: str) -> None:
    """Persist the IDS as OMAS JSON (backend-free, no ITER access layer)."""
    save_omas_json(ods, path)


def load_equilibrium_ids(path: str) -> ODS:
    """Load an OMAS-JSON IDS written by :func:`save_equilibrium_ids`."""
    result = load_omas_json(path)
    if not isinstance(result, ODS):
        raise TypeError(f"expected an ODS from {path!r}, got {type(result).__name__}")
    return result
