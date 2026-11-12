# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Impurity Cooling and Radiated Power
"""Impurity cooling curves and total radiated power.

This cluster holds the element cooling curve :math:`L_Z(T_e)` and the
volume-integrated total radiated power. It depends only on the data contracts
(:mod:`impurity_transport_contracts`).
"""

from __future__ import annotations

from typing import Final

import numpy as np

from scpn_fusion.core.impurity_transport_contracts import FloatArray

#: Tungsten cooling factor L_Z [W m^3] in coronal equilibrium against electron
#: temperature [eV]: T. Pütterich et al., Nucl. Fusion 50 (2010) 025012,
#: Table 2 (all 31 rows, 30 eV to 40 keV).
TUNGSTEN_COOLING_TABLE_PUETTERICH_2010: Final[tuple[tuple[float, float], ...]] = (
    (30.0, 2.32e-31),
    (40.0, 2.56e-31),
    (50.0, 2.55e-31),
    (60.0, 2.08e-31),
    (70.0, 1.73e-31),
    (100.0, 1.67e-31),
    (150.0, 1.57e-31),
    (200.0, 1.65e-31),
    (300.0, 1.99e-31),
    (400.0, 2.42e-31),
    (500.0, 2.91e-31),
    (600.0, 3.23e-31),
    (800.0, 3.79e-31),
    (1000.0, 4.21e-31),
    (1500.0, 4.59e-31),
    (2000.0, 4.38e-31),
    (2300.0, 3.94e-31),
    (2700.0, 2.88e-31),
    (3000.0, 2.28e-31),
    (3500.0, 1.96e-31),
    (4000.0, 1.88e-31),
    (5000.0, 1.87e-31),
    (6000.0, 1.79e-31),
    (7000.0, 1.67e-31),
    (10000.0, 1.33e-31),
    (12000.0, 1.15e-31),
    (15000.0, 9.47e-32),
    (20000.0, 7.30e-32),
    (25000.0, 6.11e-32),
    (30000.0, 5.47e-32),
    (40000.0, 4.95e-32),
)
_W_LOG_TE: Final = np.log(np.array([row[0] for row in TUNGSTEN_COOLING_TABLE_PUETTERICH_2010]))
_W_LOG_LZ: Final = np.log(np.array([row[1] for row in TUNGSTEN_COOLING_TABLE_PUETTERICH_2010]))


def tungsten_cooling_factor(te_ev: FloatArray) -> FloatArray:
    """Return the tungsten cooling factor L_Z(T_e) of Pütterich et al. (2010).

    The paper states that interpolating log L_Z linearly on the log T_e grid of
    its Table 2 reproduces the calculated curve within 3 % over 30-40 000 eV;
    this function does exactly that. Outside the tabulated range the end values
    are held constant, which is a declared extension, not a printed result.

    Parameters
    ----------
    te_ev
        Electron temperature in eV. Non-finite or non-positive entries give 0.

    Returns
    -------
    FloatArray
        Cooling factor in W m^3, same shape as ``te_ev``.
    """
    te = np.asarray(te_ev, dtype=float)
    valid = np.isfinite(te) & (te > 0.0)
    out = np.zeros_like(te, dtype=float)
    if np.any(valid):
        log_te = np.clip(np.log(te[valid]), _W_LOG_TE[0], _W_LOG_TE[-1])
        out[valid] = np.exp(np.interp(log_te, _W_LOG_TE, _W_LOG_LZ))
    return np.asarray(out)


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
            # Pütterich et al. 2010, Table 2, log-log interpolation.
            return tungsten_cooling_factor(Te)
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
    P_rad_W = np.trapezoid(p_rad_density * vol_element, rho)

    return float(P_rad_W / 1e6)
