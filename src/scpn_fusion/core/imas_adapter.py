# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — IMAS/OMAS Equilibrium Adapter
"""
Thin adapter between IMAS/OMAS equilibrium IDS and scpn-control solvers.

Reads/writes the ``equilibrium`` IDS (time_slice → profiles_2d) via the
OMAS Python package when available, or falls back to raw dict construction
for environments without IMAS access.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

ITER_B0_VACUUM_T = 5.3  # T, ITER Design Description Document §2.1


@dataclass
class EquilibriumIDS:
    """Minimal subset of IMAS equilibrium IDS for scpn-control interop.

    Fields match ``equilibrium.time_slice[0].profiles_2d[0]``.
    """

    r: FloatArray  # R grid [m], shape (nr,)
    z: FloatArray  # Z grid [m], shape (nz,)
    psi: FloatArray  # poloidal flux [Wb], shape (nz, nr)
    j_tor: FloatArray  # toroidal current density [A/m²], shape (nz, nr)
    ip: float  # plasma current [A]
    b0: float  # vacuum toroidal field at R0 [T]
    r0: float  # geometric center R [m]
    time: float = 0.0  # time of slice [s]


def from_kernel(kernel: Any, time: float = 0.0) -> EquilibriumIDS:
    """Extract EquilibriumIDS from a solved FusionKernel instance."""
    cfg = kernel.cfg if hasattr(kernel, "cfg") else {}
    physics = cfg.get("physics", {})
    dims = cfg.get("dimensions", {})

    ip = physics.get("plasma_current_target", 15e6)
    b0 = physics.get("B0", 5.3)
    r0 = dims.get("R0", 6.2)

    return EquilibriumIDS(
        r=np.asarray(kernel.R, dtype=np.float64),
        z=np.asarray(kernel.Z, dtype=np.float64),
        psi=np.asarray(kernel.Psi, dtype=np.float64),
        j_tor=np.asarray(kernel.J_phi, dtype=np.float64),
        ip=float(ip),
        b0=float(b0),
        r0=float(r0),
        time=float(time),
    )


def to_kernel_arrays(ids: EquilibriumIDS) -> dict:
    """Convert EquilibriumIDS to arrays compatible with FusionKernel init."""
    return {
        "R": ids.r.copy(),
        "Z": ids.z.copy(),
        "Psi": ids.psi.copy(),
        "J_phi": ids.j_tor.copy(),
        "ip": ids.ip,
        "b0": ids.b0,
        "r0": ids.r0,
    }


def to_omas(ids: EquilibriumIDS) -> Any:
    """Export to OMAS ODS object (requires ``pip install omas``).

    Returns None if omas is not installed.
    """
    try:
        from omas import ODS  # type: ignore[import-untyped]
    except ImportError:
        return None

    ods = ODS()
    eq = ods["equilibrium"]["time_slice"][0]
    eq["time"] = ids.time
    eq["global_quantities"]["ip"] = ids.ip
    eq["global_quantities"]["magnetic_axis"]["r"] = ids.r0

    p2d = eq["profiles_2d"][0]
    p2d["grid"]["dim1"] = ids.r
    p2d["grid"]["dim2"] = ids.z
    p2d["psi"] = ids.psi
    p2d["j_tor"] = ids.j_tor

    return ods


def from_omas(ods: Any, time_index: int = 0) -> EquilibriumIDS:
    """Import from OMAS ODS object.

    Parameters
    ----------
    ods : omas.ODS
        OMAS data structure with equilibrium IDS populated.
    time_index : int
        Index into ``time_slice`` array.
    """
    eq = ods["equilibrium"]["time_slice"][time_index]
    p2d = eq["profiles_2d"][0]

    return EquilibriumIDS(
        r=np.asarray(p2d["grid"]["dim1"], dtype=np.float64),
        z=np.asarray(p2d["grid"]["dim2"], dtype=np.float64),
        psi=np.asarray(p2d["psi"], dtype=np.float64),
        j_tor=np.asarray(p2d["j_tor"], dtype=np.float64),
        ip=float(eq["global_quantities"]["ip"]),
        b0=ITER_B0_VACUUM_T,
        r0=float(eq["global_quantities"]["magnetic_axis"]["r"]),
        time=float(eq.get("time", 0.0)),
    )


def from_geqdsk(filepath: str) -> EquilibriumIDS:
    """Import from GEQDSK file via scpn_fusion.core.eqdsk."""
    from scpn_fusion.core.eqdsk import read_geqdsk

    g = read_geqdsk(filepath)
    return EquilibriumIDS(
        r=g.r,
        z=g.z,
        psi=g.psirz,
        j_tor=np.zeros_like(g.psirz),
        ip=float(g.current),
        b0=float(g.bcentr),
        r0=float(g.rcentr),
    )
