# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Impurity Cooling and Radiated Power
"""Parametric impurity cooling curves and total radiated power.

This cluster holds the element cooling curve :math:`L_Z(T_e)` and the
volume-integrated total radiated power. It depends only on the data contracts
(:mod:`impurity_transport_contracts`).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scpn_fusion.core.impurity_transport_contracts import FloatArray


class CoolingCurve:
    """Parametric cooling rate L_Z(Te) [W m^3]."""

    def __init__(self, element: str):
        self.element = element

    def L_z(self, Te_eV: FloatArray) -> FloatArray:
        """Evaluate the element cooling curve for electron temperatures."""
        Te = np.asarray(Te_eV, dtype=float)
        valid = np.isfinite(Te) & (Te > 0.0)
        if not np.any(valid):
            return np.zeros_like(Te, dtype=float)

        log_Te = np.zeros_like(Te, dtype=float)
        log_Te[valid] = np.log(Te[valid])
        if self.element == "W":
            # Putterich et al. 2010 fit — peaks near 1500 eV and 50 eV
            L = 1e-31 * np.exp(-(((log_Te - np.log(1500.0)) / 1.5) ** 2))
            L += 3e-33 * np.exp(-(((log_Te - np.log(50.0)) / 1.0) ** 2))
            L[~valid] = 0.0
            return np.asarray(L)
        # Carbon and neon coronal cooling rates computed from the OpenADAS adf11
        # *96 dataset: line (PLT) + recombination/continuum (PRB) power summed over
        # the coronal charge-state balance. Carbon peaks at 5.84e-32 W m^3 near
        # 7 eV and neon at 5.74e-32 W m^3 near 30 eV, both line-radiation
        # dominated. These ADAS96 values are below the older Post & Jensen 1977
        # estimate (~2e-31). See tools/compute_coronal_lz_from_adas.py and
        # validation/reference_data/openadas_coronal_lz_manifest.json.
        if self.element == "C":
            L = 5.84e-32 * np.exp(-(((log_Te - np.log(7.0)) / 0.5) ** 2))
            L[~valid] = 0.0
            return np.asarray(L)
        if self.element == "Ar":
            # Argon coronal Lz is bimodal. The global peak from the OpenADAS adf11
            # *89 computation is 1.98e-31 W m^3 near 20 eV (L-shell line emission);
            # there is a weaker high-Te feature near 300 eV (Mavrin 2018 coronal
            # fit ~1.65e-31, valid above ~100 eV). The single Gaussian tracks the
            # dominant low-Te peak relevant to edge/divertor radiation.
            L = 1.98e-31 * np.exp(-(((log_Te - np.log(20.0)) / 0.6) ** 2))
            L[~valid] = 0.0
            return np.asarray(L)
        if self.element == "Ne":
            L = 5.74e-32 * np.exp(-(((log_Te - np.log(30.0)) / 0.85) ** 2))
            L[~valid] = 0.0
            return np.asarray(L)
        return np.zeros_like(Te)


def total_radiated_power(
    ne: FloatArray,
    n_impurity: dict[str, FloatArray],
    Te_eV: FloatArray,
    rho: FloatArray,
    R0: float,
    a: float,
) -> float:
    """P_rad in MW."""
    p_rad_density = np.zeros_like(rho)

    for element, n_Z in n_impurity.items():
        curve = CoolingCurve(element)
        L = curve.L_z(Te_eV)
        # p_rad = n_e * n_z * L_z
        p_rad_density += ne * n_Z * L

    # Integrate over volume: dV = 4 pi^2 R0 a^2 rho drho
    vol_element = 4.0 * np.pi**2 * R0 * a**2 * rho
    _trapz: Any = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    P_rad_W = _trapz(p_rad_density * vol_element, rho)

    return float(P_rad_W / 1e6)
