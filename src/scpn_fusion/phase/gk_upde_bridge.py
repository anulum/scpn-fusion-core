# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — GK → UPDE Phase Dynamics Bridge
"""
Bridge between gyrokinetic transport fluxes and the 8-layer UPDE
Kuramoto phase dynamics system.

Maps GK-computed growth rates and diffusivities into adaptive K_nm
coupling modulation for layers P0 (microturbulence), P1 (zonal flows),
P4 (transport barrier), and P5 (current profile).

Reference layer mappings:
  P0 ← max(gamma_ITG, gamma_TEM): turbulence drive
  P1 ← chi_e suppression ratio: zonal flow damping of transport
  P4 ← chi_i pedestal / chi_i core: transport barrier strength
  P5 ← bootstrap current contribution (via pressure gradient)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.core.gk_interface import GKOutput


def adaptive_knm(
    K_base: NDArray[np.float64],
    gk_output: GKOutput,
    chi_i_profile: NDArray[np.float64] | None = None,
    gamma_ref: float = 0.2,
    chi_ref: float = 1.0,
) -> NDArray[np.float64]:
    """Modulate K_nm based on GK fluxes.

    Parameters
    ----------
    K_base : array, shape (L, L)
        Baseline coupling matrix from build_knm_plasma().
    gk_output : GKOutput
        GK solver output (growth rates, fluxes).
    chi_i_profile : array or None
        Full chi_i(rho) profile for pedestal ratio calculation.
    gamma_ref : float
        Reference growth rate for tanh scaling [c_s/a].
    chi_ref : float
        Reference chi_e for transport modulation [m^2/s].
    """
    K = K_base.copy()
    L = K.shape[0]
    if L < 6:
        return K

    # P0↔P1: microturbulence ↔ zonal flows
    max_gamma = float(np.max(gk_output.gamma)) if len(gk_output.gamma) > 0 else 0.0
    K[0, 1] = K_base[0, 1] * (1.0 + 0.5 * np.tanh(max_gamma / max(gamma_ref, 1e-10)))
    K[1, 0] = K[0, 1]

    # P1↔P4: zonal flow ↔ transport barrier
    mean_chi_e = max(gk_output.chi_e, 1e-10)
    K[1, 4] = K_base[1, 4] * (1.0 + 0.3 * np.clip(mean_chi_e / chi_ref, 0, 2))
    K[4, 1] = K[1, 4]

    # P3↔P4: sawtooth/ELM ↔ transport barrier (pedestal ratio)
    if chi_i_profile is not None and len(chi_i_profile) > 5:
        chi_core = max(float(np.mean(chi_i_profile[: len(chi_i_profile) // 3])), 1e-10)
        chi_ped = max(float(np.mean(chi_i_profile[-len(chi_i_profile) // 5 :])), 1e-10)
        K[3, 4] = K_base[3, 4] * (1.0 + 0.4 * (chi_ped / chi_core - 1.0))
        K[4, 3] = K[3, 4]

    return K


def gk_natural_frequencies(
    omega_base: NDArray[np.float64],
    gk_output: GKOutput,
    gamma_scale: float = 0.1,
) -> NDArray[np.float64]:
    """Adjust layer-0 natural frequency based on GK growth rate.

    The turbulence layer's effective frequency increases with the
    dominant instability growth rate.
    """
    omega = omega_base.copy()
    max_gamma = float(np.max(gk_output.gamma)) if len(gk_output.gamma) > 0 else 0.0
    omega[0] += gamma_scale * max_gamma
    return omega
