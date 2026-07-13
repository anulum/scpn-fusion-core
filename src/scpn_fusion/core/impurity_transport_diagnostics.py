# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Impurity Transport Diagnostics
"""Neoclassical impurity pinch and tungsten-accumulation diagnostics.

This cluster holds the Hirshman & Sigmar neoclassical impurity pinch velocity
and the core/edge tungsten-accumulation danger diagnostic. It depends only on
the data contracts (:mod:`impurity_transport_contracts`).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scpn_fusion.core.impurity_transport_contracts import FloatArray


def neoclassical_impurity_pinch(
    Z: int,
    ne: FloatArray,
    Te_eV: FloatArray,
    Ti_eV: FloatArray,
    q: FloatArray,
    rho: FloatArray,
    R0: float,
    a: float,
    epsilon: FloatArray,
) -> FloatArray:
    """V_neo [m/s] (negative = inward).

    Hirshman & Sigmar, Nucl. Fusion 21, 1079 (1981).
    V_neo = -D_neo [Z/L_n + (Z/2 - H_Z)/L_Ti]
    with inverse scale lengths 1/L_x = -d ln(x)/dr.
    """
    if Z < 1:
        raise ValueError("Z must be positive")
    rho_arr = np.asarray(rho, dtype=float)
    ne_arr = np.asarray(ne, dtype=float)
    ti_arr = np.asarray(Ti_eV, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    eps_arr = np.asarray(epsilon, dtype=float)
    arrays = (rho_arr, ne_arr, ti_arr, q_arr, eps_arr)
    if any(arr.shape != rho_arr.shape for arr in arrays):
        raise ValueError("rho, ne, Ti_eV, q, and epsilon must have matching shapes")
    if rho_arr.size < 3:
        raise ValueError("rho must contain at least three points")
    if not np.all(np.isfinite(rho_arr)) or not np.all(np.diff(rho_arr) > 0.0):
        raise ValueError("rho must be finite and strictly increasing")
    if not np.all(np.isfinite(ne_arr)) or np.any(ne_arr <= 0.0):
        raise ValueError("ne must be finite and positive")
    if not np.all(np.isfinite(ti_arr)) or np.any(ti_arr <= 0.0):
        raise ValueError("Ti_eV must be finite and positive")
    if not np.all(np.isfinite(q_arr)) or np.any(q_arr <= 0.0):
        raise ValueError("q must be finite and positive")
    if not np.all(np.isfinite(eps_arr)) or np.any(eps_arr < 0.0):
        raise ValueError("epsilon must be finite and non-negative")
    if not np.isfinite(R0) or R0 <= 0.0:
        raise ValueError("R0 must be finite and positive")
    if not np.isfinite(a) or a <= 0.0:
        raise ValueError("a must be finite and positive")

    dr = (rho_arr[1] - rho_arr[0]) * a
    inv_Ln = -np.gradient(np.log(ne_arr), dr)
    inv_LTi = -np.gradient(np.log(ti_arr), dr)

    D_NEO = 0.1  # m²/s, banana-regime nominal scale
    D_neo = D_NEO * q_arr**2 / np.sqrt(np.maximum(eps_arr, 0.05))

    H_Z = 0.5  # screening factor, banana regime trace impurities

    V_neo = -D_neo * (Z * inv_Ln + (Z / 2.0 - H_Z) * inv_LTi)
    return np.asarray(V_neo)


def tungsten_accumulation_diagnostic(n_W: FloatArray, ne: FloatArray) -> dict[str, Any]:
    """Return core/edge tungsten concentration and accumulation danger level."""
    c_W_core = float(n_W[0] / max(ne[0], 1e-6))
    c_W_edge = float(n_W[-1] / max(ne[-1], 1e-6))

    peaking_factor = c_W_core / max(c_W_edge, 1e-12)

    if c_W_core < 1e-5:
        danger = "safe"
    elif c_W_core < 5e-5:
        danger = "warning"
    else:
        danger = "critical"

    return {
        "c_W_core": c_W_core,
        "c_W_edge": c_W_edge,
        "peaking_factor": peaking_factor,
        "danger_level": danger,
    }
