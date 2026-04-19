# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — IMAS Connector OMAS Bridge
"""OMAS compatibility layer for equilibrium IDS payloads."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

try:
    import omas  # type: ignore[import-untyped]

    HAS_OMAS = True
except ImportError:
    omas = None
    HAS_OMAS = False


def ids_to_omas_equilibrium(ids_dict: Mapping[str, Any]) -> Any:
    """Convert an IMAS equilibrium IDS dict to an OMAS ODS."""
    if not HAS_OMAS or omas is None:
        raise ImportError(
            "The 'omas' package is required for OMAS conversion. Install with: pip install omas"
        )

    ods = omas.ODS()

    props = ids_dict.get("ids_properties", {})
    ods["equilibrium.ids_properties.homogeneous_time"] = props.get("homogeneous_time", 1)
    ods["equilibrium.ids_properties.comment"] = props.get("comment", "")

    time_arr = ids_dict.get("time", [0.0])
    ods["equilibrium.time"] = np.asarray(time_arr, dtype=np.float64)

    for i, ts in enumerate(ids_dict.get("time_slice", [])):
        prefix = f"equilibrium.time_slice.{i}"
        ods[f"{prefix}.time"] = ts.get("time", 0.0)

        gq = ts.get("global_quantities", {})
        ods[f"{prefix}.global_quantities.ip"] = gq.get("ip", 0.0)
        axis = gq.get("magnetic_axis", {})
        ods[f"{prefix}.global_quantities.magnetic_axis.r"] = axis.get("r", 0.0)
        ods[f"{prefix}.global_quantities.magnetic_axis.z"] = axis.get("z", 0.0)
        ods[f"{prefix}.global_quantities.psi_axis"] = gq.get("psi_axis", 0.0)
        ods[f"{prefix}.global_quantities.psi_boundary"] = gq.get("psi_boundary", 0.0)

        vtf = gq.get("vacuum_toroidal_field", {})
        ods[f"{prefix}.global_quantities.vacuum_toroidal_field.r0"] = vtf.get("r0", 0.0)
        ods[f"{prefix}.global_quantities.vacuum_toroidal_field.b0"] = vtf.get("b0", 0.0)

        p1d = ts.get("profiles_1d", {})
        if p1d:
            ods[f"{prefix}.profiles_1d.psi"] = np.asarray(p1d.get("psi", []), dtype=np.float64)
            ods[f"{prefix}.profiles_1d.q"] = np.asarray(p1d.get("q", []), dtype=np.float64)
            ods[f"{prefix}.profiles_1d.pressure"] = np.asarray(
                p1d.get("pressure", []), dtype=np.float64
            )
            ods[f"{prefix}.profiles_1d.f"] = np.asarray(p1d.get("f", []), dtype=np.float64)

        bdry = ts.get("boundary", {})
        outline = bdry.get("outline", {})
        if outline:
            ods[f"{prefix}.boundary.outline.r"] = np.asarray(outline.get("r", []), dtype=np.float64)
            ods[f"{prefix}.boundary.outline.z"] = np.asarray(outline.get("z", []), dtype=np.float64)

    return ods


def omas_equilibrium_to_ids(ods: Any) -> dict[str, Any]:
    """Convert OMAS ODS equilibrium data back to an IDS dict."""
    if not HAS_OMAS:
        raise ImportError("The 'omas' package is required for OMAS conversion.")

    ids_dict: dict[str, Any] = {}
    ids_dict["ids_properties"] = {
        "homogeneous_time": int(ods.get("equilibrium.ids_properties.homogeneous_time", 1)),
        "comment": str(ods.get("equilibrium.ids_properties.comment", "")),
    }

    time_val = ods.get("equilibrium.time", np.array([0.0]))
    if hasattr(time_val, "tolist"):
        ids_dict["time"] = time_val.tolist()
    else:
        ids_dict["time"] = [float(time_val)]

    time_slices = []
    i = 0
    while True:
        prefix = f"equilibrium.time_slice.{i}"
        try:
            t = float(ods[f"{prefix}.time"])
        except (KeyError, IndexError, TypeError):
            break

        ts: dict[str, Any] = {"time": t}
        gq: dict[str, Any] = {
            "ip": float(ods.get(f"{prefix}.global_quantities.ip", 0.0)),
            "magnetic_axis": {
                "r": float(ods.get(f"{prefix}.global_quantities.magnetic_axis.r", 0.0)),
                "z": float(ods.get(f"{prefix}.global_quantities.magnetic_axis.z", 0.0)),
            },
            "psi_axis": float(ods.get(f"{prefix}.global_quantities.psi_axis", 0.0)),
            "psi_boundary": float(ods.get(f"{prefix}.global_quantities.psi_boundary", 0.0)),
            "vacuum_toroidal_field": {
                "r0": float(ods.get(f"{prefix}.global_quantities.vacuum_toroidal_field.r0", 0.0)),
                "b0": float(ods.get(f"{prefix}.global_quantities.vacuum_toroidal_field.b0", 0.0)),
            },
        }
        ts["global_quantities"] = gq

        p1d: dict[str, Any] = {}
        for key in ("psi", "q", "pressure", "f"):
            val = ods.get(f"{prefix}.profiles_1d.{key}", None)
            if val is not None:
                p1d[key] = np.asarray(val).tolist() if hasattr(val, "tolist") else val
        if p1d:
            ts["profiles_1d"] = p1d

        bdry_r = ods.get(f"{prefix}.boundary.outline.r", None)
        bdry_z = ods.get(f"{prefix}.boundary.outline.z", None)
        if bdry_r is not None and bdry_z is not None:
            ts["boundary"] = {
                "outline": {
                    "r": np.asarray(bdry_r).tolist() if hasattr(bdry_r, "tolist") else bdry_r,
                    "z": np.asarray(bdry_z).tolist() if hasattr(bdry_z, "tolist") else bdry_z,
                }
            }

        time_slices.append(ts)
        i += 1

    ids_dict["time_slice"] = time_slices
    return ids_dict


__all__ = [
    "HAS_OMAS",
    "ids_to_omas_equilibrium",
    "omas_equilibrium_to_ids",
]
