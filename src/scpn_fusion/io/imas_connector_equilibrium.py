# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — IMAS Connector Equilibrium
"""IMAS DD v3 equilibrium converters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

import numpy as np

from scpn_fusion.core.eqdsk import GEqdsk
from scpn_fusion.io.imas_connector_common import (
    _coerce_finite_real_sequence,
    _missing_required_keys,
)


IMAS_DD_EQUILIBRIUM_KEYS = (
    "ids_properties",
    "time",
    "time_slice",
)


def validate_imas_equilibrium_payload(ids: Mapping[str, Any]) -> None:
    """Validate the bounded IMAS equilibrium schema used by this adapter."""
    missing = _missing_required_keys(ids, IMAS_DD_EQUILIBRIUM_KEYS)
    if missing:
        raise ValueError(f"IMAS equilibrium IDS missing keys: {missing}")

    time_slices = ids["time_slice"]
    if isinstance(time_slices, (str, bytes, bytearray)) or not isinstance(time_slices, Sequence):
        raise ValueError("IMAS equilibrium time_slice must be a non-empty sequence.")
    if len(time_slices) == 0:
        raise ValueError("IMAS equilibrium must have at least one time_slice.")
    if len(time_slices) > 1024:
        raise ValueError("IMAS equilibrium time_slice exceeds safety limit 1024.")
    first_slice = time_slices[0]
    if not isinstance(first_slice, Mapping):
        raise ValueError("IMAS equilibrium time_slice[0] must be a mapping.")

    p2d_list = first_slice.get("profiles_2d", [])
    if isinstance(p2d_list, (str, bytes, bytearray)) or not isinstance(p2d_list, Sequence):
        raise ValueError("IMAS equilibrium profiles_2d must be a sequence.")
    if len(p2d_list) == 0:
        raise ValueError("IMAS equilibrium must have at least one profiles_2d entry.")
    if len(p2d_list) > 64:
        raise ValueError("IMAS equilibrium profiles_2d exceeds safety limit 64.")
    first_profile = p2d_list[0]
    if not isinstance(first_profile, Mapping):
        raise ValueError("IMAS equilibrium profiles_2d[0] must be a mapping.")
    grid = first_profile.get("grid", {})
    if not isinstance(grid, Mapping):
        raise ValueError("IMAS equilibrium profiles_2d[0].grid must be a mapping.")

    r_grid = _coerce_finite_real_sequence(
        "IMAS profiles_2d[0].grid.dim1",
        grid.get("dim1", []),
        minimum_len=2,
        strictly_increasing=True,
    )
    z_grid = _coerce_finite_real_sequence(
        "IMAS profiles_2d[0].grid.dim2",
        grid.get("dim2", []),
        minimum_len=2,
        strictly_increasing=True,
    )
    psi_rows = first_profile.get("psi", [])
    if isinstance(psi_rows, (str, bytes, bytearray)) or not isinstance(psi_rows, Sequence):
        raise ValueError("IMAS profiles_2d[0].psi must be a 2-D sequence.")
    if len(psi_rows) != len(z_grid):
        raise ValueError("IMAS profiles_2d[0].psi row count must match grid dim2.")
    for row_idx, row in enumerate(psi_rows):
        parsed = _coerce_finite_real_sequence(
            f"IMAS profiles_2d[0].psi[{row_idx}]",
            row,
            minimum_len=2,
        )
        if len(parsed) != len(r_grid):
            raise ValueError("IMAS profiles_2d[0].psi column count must match grid dim1.")


def geqdsk_to_imas_equilibrium(
    eq: GEqdsk,
    *,
    time_s: float = 0.0,
    shot: int = 0,
    run: int = 0,
) -> dict[str, Any]:
    """Convert a GEqdsk equilibrium to an IMAS Data Dictionary ``equilibrium`` IDS."""
    if eq.nw < 2 or eq.nh < 2:
        raise ValueError("GEqdsk must have nw >= 2 and nh >= 2 for IMAS conversion.")
    if eq.psirz.size == 0:
        raise ValueError("GEqdsk psirz must be non-empty for IMAS conversion.")

    time_val = float(time_s)
    r_grid = eq.r.tolist()
    z_grid = eq.z.tolist()
    psi_norm = eq.psi_norm.tolist()

    midplane_idx = eq.nh // 2
    psi_1d = eq.psirz[midplane_idx, :].tolist()

    return {
        "ids_properties": {
            "homogeneous_time": 1,
            "comment": f"SCPN Fusion Core IMAS export (shot={shot}, run={run})",
        },
        "time": [time_val],
        "time_slice": [
            {
                "time": time_val,
                "global_quantities": {
                    "ip": float(eq.current),
                    "magnetic_axis": {
                        "r": float(eq.rmaxis),
                        "z": float(eq.zmaxis),
                    },
                    "psi_axis": float(eq.simag),
                    "psi_boundary": float(eq.sibry),
                    "vacuum_toroidal_field": {
                        "r0": float(eq.rcentr),
                        "b0": float(eq.bcentr),
                    },
                },
                "profiles_1d": {
                    "psi": psi_1d,
                    "q": eq.qpsi.tolist() if eq.qpsi.size > 0 else [],
                    "pressure": eq.pres.tolist() if eq.pres.size > 0 else [],
                    "f": eq.fpol.tolist() if eq.fpol.size > 0 else [],
                    "psi_norm": psi_norm,
                },
                "profiles_2d": [
                    {
                        "psi": eq.psirz.tolist(),
                        "grid": {
                            "dim1": r_grid,
                            "dim2": z_grid,
                        },
                        "grid_type": {"index": 1, "name": "rectangular"},
                    }
                ],
                "boundary": {
                    "outline": {
                        "r": eq.rbdry.tolist() if eq.rbdry.size > 0 else [],
                        "z": eq.zbdry.tolist() if eq.zbdry.size > 0 else [],
                    }
                },
            }
        ],
        "code": {
            "name": "SCPN-Fusion-Core",
            "version": "2.0.0",
        },
    }


def imas_equilibrium_to_geqdsk(ids: Mapping[str, Any]) -> GEqdsk:
    """Convert an IMAS Data Dictionary ``equilibrium`` IDS back to a GEqdsk."""
    validate_imas_equilibrium_payload(ids)

    time_slices = ids["time_slice"]
    # RE-DEADGUARD: validate_imas_equilibrium_payload already proved time_slice is a
    # non-empty Sequence (see the isinstance/len guards above); unreachable single-threaded.
    if not isinstance(time_slices, Sequence) or len(time_slices) == 0:  # pragma: no cover
        raise ValueError("IMAS equilibrium must have at least one time_slice.")

    ts = time_slices[0]
    gq = ts.get("global_quantities", {})
    p1d = ts.get("profiles_1d", {})
    p2d_list = ts.get("profiles_2d", [])
    boundary = ts.get("boundary", {})

    axis = gq.get("magnetic_axis", {})
    vtf = gq.get("vacuum_toroidal_field", {})

    # RE-DEADGUARD: validate already proved profiles_2d is a non-empty sequence; unreachable.
    if not p2d_list:  # pragma: no cover
        raise ValueError("IMAS equilibrium must have at least one profiles_2d entry.")
    p2d = p2d_list[0]
    grid = p2d.get("grid", {})
    r_grid = np.asarray(grid.get("dim1", []), dtype=np.float64)
    z_grid = np.asarray(grid.get("dim2", []), dtype=np.float64)
    psirz = np.asarray(p2d.get("psi", []), dtype=np.float64)

    nw = r_grid.size
    nh = z_grid.size
    # RE-DEADGUARD: validate coerced dim1/dim2 with minimum_len=2, so nw, nh >= 2; unreachable.
    if nw < 2 or nh < 2:  # pragma: no cover
        raise ValueError("IMAS profiles_2d grid must have at least 2 points per dimension.")
    # RE-DEADGUARD: validate proved psi has nh rows each of nw finite columns, so the numpy
    # array is exactly (nh, nw); the shape mismatch is unreachable after validation.
    if psirz.ndim != 2 or psirz.shape != (nh, nw):  # pragma: no cover
        raise ValueError(
            f"IMAS profiles_2d psi shape {psirz.shape} does not match grid ({nh}, {nw})."
        )

    rdim = float(r_grid[-1] - r_grid[0])
    zdim = float(z_grid[-1] - z_grid[0])
    rleft = float(r_grid[0])
    zmid = float(0.5 * (z_grid[0] + z_grid[-1]))
    outline = boundary.get("outline", {})

    return GEqdsk(
        description="IMAS import",
        nw=nw,
        nh=nh,
        rdim=rdim,
        zdim=zdim,
        rcentr=float(vtf.get("r0", 0.0)),
        rleft=rleft,
        zmid=zmid,
        rmaxis=float(axis.get("r", 0.0)),
        zmaxis=float(axis.get("z", 0.0)),
        simag=float(gq.get("psi_axis", 0.0)),
        sibry=float(gq.get("psi_boundary", 0.0)),
        bcentr=float(vtf.get("b0", 0.0)),
        current=float(gq.get("ip", 0.0)),
        fpol=np.asarray(p1d.get("f", []), dtype=np.float64),
        pres=np.asarray(p1d.get("pressure", []), dtype=np.float64),
        ffprime=np.zeros(nw, dtype=np.float64),
        pprime=np.zeros(nw, dtype=np.float64),
        qpsi=np.asarray(p1d.get("q", []), dtype=np.float64),
        psirz=psirz,
        rbdry=np.asarray(outline.get("r", []), dtype=np.float64),
        zbdry=np.asarray(outline.get("z", []), dtype=np.float64),
        rlim=np.array([], dtype=np.float64),
        zlim=np.array([], dtype=np.float64),
    )


__all__ = [
    "IMAS_DD_EQUILIBRIUM_KEYS",
    "geqdsk_to_imas_equilibrium",
    "imas_equilibrium_to_geqdsk",
    "validate_imas_equilibrium_payload",
]
