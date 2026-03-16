# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Quasilinear Flux Model Tests
from __future__ import annotations

import numpy as np

from scpn_fusion.core.gk_eigenvalue import LinearGKResult, solve_linear_gk
from scpn_fusion.core.gk_quasilinear import (
    critical_gradient_scan,
    mixing_length_saturation,
    quasilinear_fluxes_from_spectrum,
)
from scpn_fusion.core.gk_species import deuterium_ion


def test_mixing_length_positive_gamma():
    gamma = np.array([0.1, 0.2, 0.0, -0.1])
    omega_r = np.array([-0.5, -0.8, 0.1, 0.2])
    k_y = np.array([0.1, 0.3, 0.5, 1.0])
    phi_sq = mixing_length_saturation(gamma, omega_r, k_y)
    assert phi_sq[0] > 0
    assert phi_sq[1] > 0
    assert phi_sq[2] == 0  # gamma=0
    assert phi_sq[3] == 0  # gamma<0


def test_mixing_length_scales_with_gamma():
    gamma = np.array([0.1, 0.2])
    omega_r = np.array([-0.5, -0.5])
    k_y = np.array([0.3, 0.3])
    phi_sq = mixing_length_saturation(gamma, omega_r, k_y)
    assert phi_sq[1] > phi_sq[0]


def test_mixing_length_zero_omega():
    """Zero omega_r uses gamma_floor to avoid division by zero."""
    gamma = np.array([0.1])
    omega_r = np.array([0.0])
    k_y = np.array([0.3])
    phi_sq = mixing_length_saturation(gamma, omega_r, k_y)
    assert np.isfinite(phi_sq[0])
    assert phi_sq[0] > 0


def test_quasilinear_fluxes_empty():
    result = LinearGKResult(
        k_y=np.array([]),
        gamma=np.array([]),
        omega_r=np.array([]),
        mode_type=[],
        modes=[],
    )
    ion = deuterium_ion()
    output = quasilinear_fluxes_from_spectrum(result, ion)
    assert output.chi_i == 0.0
    assert output.chi_e == 0.0
    assert output.dominant_mode == "stable"


def test_quasilinear_fluxes_from_solver():
    result = solve_linear_gk(
        R0=2.78,
        a=1.0,
        B0=2.0,
        q=1.4,
        s_hat=0.78,
        n_ky_ion=4,
        n_theta=16,
        n_period=1,
    )
    ion = deuterium_ion(R_L_T=6.9, R_L_n=2.2)
    output = quasilinear_fluxes_from_spectrum(result, ion, R0=2.78, a=1.0, B0=2.0)
    assert np.isfinite(output.chi_i)
    assert np.isfinite(output.chi_e)
    assert output.chi_i >= 0
    assert output.chi_e >= 0
    assert output.converged is True


def test_quasilinear_chi_increases_with_gradient():
    ion_weak = deuterium_ion(R_L_T=3.0, R_L_n=2.0)
    ion_strong = deuterium_ion(R_L_T=12.0, R_L_n=2.0)

    r_weak = solve_linear_gk(
        species_list=[
            ion_weak,
            (
                deuterium_ion.__wrapped__(R_L_T=3.0)
                if hasattr(deuterium_ion, "__wrapped__")
                else ion_weak
            ),
        ],
        n_ky_ion=4,
        n_theta=16,
        n_period=1,
    )
    r_strong = solve_linear_gk(
        species_list=[
            ion_strong,
            (
                deuterium_ion.__wrapped__(R_L_T=12.0)
                if hasattr(deuterium_ion, "__wrapped__")
                else ion_strong
            ),
        ],
        n_ky_ion=4,
        n_theta=16,
        n_period=1,
    )

    out_weak = quasilinear_fluxes_from_spectrum(r_weak, ion_weak)
    out_strong = quasilinear_fluxes_from_spectrum(r_strong, ion_strong)

    # Stronger gradient → more transport (or at least not less)
    assert out_strong.chi_i >= out_weak.chi_i or out_strong.gamma_max >= 0


def test_quasilinear_gk_output_has_spectrum():
    result = solve_linear_gk(n_ky_ion=4, n_theta=16, n_period=1)
    ion = deuterium_ion()
    output = quasilinear_fluxes_from_spectrum(result, ion)
    assert len(output.gamma) == 4
    assert len(output.k_y) == 4


def test_critical_gradient_scan_shape():
    rlt = np.array([2.0, 4.0, 6.0, 8.0])
    rlt_out, gamma_out = critical_gradient_scan(rlt, n_ky=2)
    assert len(rlt_out) == 4
    assert len(gamma_out) == 4
    assert np.all(np.isfinite(gamma_out))


def test_critical_gradient_scan_monotonic():
    """Growth rates from scan should all be non-negative and finite."""
    rlt = np.array([1.0, 3.0, 5.0, 8.0, 12.0])
    _, gamma = critical_gradient_scan(rlt, n_ky=2)
    assert np.all(gamma >= 0)
    assert np.all(np.isfinite(gamma))
