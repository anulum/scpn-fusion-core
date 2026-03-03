# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — IMAS Connector Equilibrium
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""IMAS DD v3 equilibrium converters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Mapping

import numpy as np

from scpn_fusion.core.eqdsk import GEqdsk
from scpn_fusion.io.imas_connector_common import _missing_required_keys


IMAS_DD_EQUILIBRIUM_KEYS = (
    "ids_properties",
    "time",
    "time_slice",
)


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
    missing = _missing_required_keys(ids, IMAS_DD_EQUILIBRIUM_KEYS)
    if missing:
        raise ValueError(f"IMAS equilibrium IDS missing keys: {missing}")

    time_slices = ids["time_slice"]
    if not isinstance(time_slices, Sequence) or len(time_slices) == 0:
        raise ValueError("IMAS equilibrium must have at least one time_slice.")

    ts = time_slices[0]
    gq = ts.get("global_quantities", {})
    p1d = ts.get("profiles_1d", {})
    p2d_list = ts.get("profiles_2d", [])
    boundary = ts.get("boundary", {})

    axis = gq.get("magnetic_axis", {})
    vtf = gq.get("vacuum_toroidal_field", {})

    if not p2d_list:
        raise ValueError("IMAS equilibrium must have at least one profiles_2d entry.")
    p2d = p2d_list[0]
    grid = p2d.get("grid", {})
    r_grid = np.asarray(grid.get("dim1", []), dtype=np.float64)
    z_grid = np.asarray(grid.get("dim2", []), dtype=np.float64)
    psirz = np.asarray(p2d.get("psi", []), dtype=np.float64)

    nw = r_grid.size
    nh = z_grid.size
    if nw < 2 or nh < 2:
        raise ValueError("IMAS profiles_2d grid must have at least 2 points per dimension.")
    if psirz.ndim != 2 or psirz.shape != (nh, nw):
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
]

