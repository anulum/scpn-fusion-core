# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Gyrokinetic Transport Tests
from __future__ import annotations

import numpy as np

from scpn_fusion.core.gyrokinetic_transport import (
    GyrokineticsParams,
    GyrokineticTransportModel,
    compute_spectrum,
    quasilinear_fluxes,
    solve_dispersion,
)


def test_zero_gradients():
    params = GyrokineticsParams(
        R_L_Ti=0.0,
        R_L_Te=0.0,
        R_L_ne=0.0,
        q=1.5,
        s_hat=1.0,
        alpha_MHD=0.0,
        Te_Ti=1.0,
        Z_eff=1.5,
        nu_star=0.1,
        beta_e=0.01,
        epsilon=0.1,
    )
    g, w, mt = solve_dispersion(params, 0.5, etg_scale=False)
    assert g == 0.0

    spec = compute_spectrum(params)
    fluxes = quasilinear_fluxes(params, spec)
    assert fluxes.chi_i == 0.0
    assert fluxes.chi_e == 0.0
    assert fluxes.D_e == 0.0


def test_sub_critical_itg():
    params = GyrokineticsParams(
        R_L_Ti=1.0,  # low
        R_L_Te=1.0,
        R_L_ne=1.0,
        q=1.5,
        s_hat=1.0,
        alpha_MHD=0.0,
        Te_Ti=1.0,
        Z_eff=1.5,
        nu_star=0.1,
        beta_e=0.01,
        epsilon=0.1,
    )
    g, w, mt = solve_dispersion(params, 0.5, etg_scale=False)
    # Should not be ITG, might be TEM if driven, but let's just check mt != 1
    assert mt != 1


def test_super_critical_itg():
    params = GyrokineticsParams(
        R_L_Ti=10.0,  # high
        R_L_Te=1.0,
        R_L_ne=1.0,
        q=1.5,
        s_hat=1.0,
        alpha_MHD=0.0,
        Te_Ti=1.0,
        Z_eff=1.5,
        nu_star=0.1,
        beta_e=0.01,
        epsilon=0.1,
    )
    g, w, mt = solve_dispersion(params, 0.5, etg_scale=False)
    assert mt == 1
    assert g > 0.0


def test_tem_regime():
    params = GyrokineticsParams(
        R_L_Ti=1.0,
        R_L_Te=1.0,
        R_L_ne=10.0,  # high density gradient
        q=1.5,
        s_hat=1.0,
        alpha_MHD=0.0,
        Te_Ti=1.0,
        Z_eff=1.5,
        nu_star=0.1,
        beta_e=0.01,
        epsilon=0.1,
    )
    g, w, mt = solve_dispersion(params, 0.5, etg_scale=False)
    assert mt == 2
    assert g > 0.0
    assert w > 0.0


def test_etg_scale():
    params = GyrokineticsParams(
        R_L_Ti=1.0,
        R_L_Te=20.0,  # high Te gradient
        R_L_ne=1.0,
        q=1.5,
        s_hat=1.0,
        alpha_MHD=0.0,
        Te_Ti=1.0,
        Z_eff=1.5,
        nu_star=0.1,
        beta_e=0.01,
        epsilon=0.1,
    )
    g, w, mt = solve_dispersion(params, 5.0, etg_scale=True)
    assert mt == 3
    assert g > 0.0


def test_quasilinear_fluxes():
    params = GyrokineticsParams(
        R_L_Ti=10.0,
        R_L_Te=10.0,
        R_L_ne=10.0,
        q=1.5,
        s_hat=1.0,
        alpha_MHD=0.0,
        Te_Ti=1.0,
        Z_eff=1.5,
        nu_star=0.1,
        beta_e=0.01,
        epsilon=0.1,
    )
    spec = compute_spectrum(params, n_modes=10, include_etg=True)
    fluxes = quasilinear_fluxes(params, spec)
    assert fluxes.chi_i > 0.0
    assert fluxes.chi_e > 0.0
    assert fluxes.D_e >= 0.0


def test_stiffness():
    params1 = GyrokineticsParams(
        R_L_Ti=6.0,
        R_L_Te=1.0,
        R_L_ne=1.0,
        q=1.5,
        s_hat=1.0,
        alpha_MHD=0.0,
        Te_Ti=1.0,
        Z_eff=1.5,
        nu_star=0.1,
        beta_e=0.01,
        epsilon=0.1,
    )
    params2 = GyrokineticsParams(
        R_L_Ti=12.0,
        R_L_Te=1.0,
        R_L_ne=1.0,
        q=1.5,
        s_hat=1.0,
        alpha_MHD=0.0,
        Te_Ti=1.0,
        Z_eff=1.5,
        nu_star=0.1,
        beta_e=0.01,
        epsilon=0.1,
    )
    spec1 = compute_spectrum(params1)
    fluxes1 = quasilinear_fluxes(params1, spec1)

    spec2 = compute_spectrum(params2)
    fluxes2 = quasilinear_fluxes(params2, spec2)

    assert fluxes2.chi_i > fluxes1.chi_i * 1.5


def test_transport_model_eval():
    model = GyrokineticTransportModel()
    rho = 0.5
    profiles = {
        "R0": 2.0,
        "a": 0.5,
        "B0": 5.0,
        "q": 1.5,
        "s_hat": 1.0,
        "Te": 5.0,
        "Ti": 5.0,
        "ne": 5.0,
        "dTe_dr": -50.0,
        "dTi_dr": -50.0,
        "dne_dr": -50.0,
    }
    chi_i, chi_e, D_e = model.evaluate(rho, profiles)
    assert chi_i >= 0.0
    assert chi_e >= 0.0
    assert D_e >= 0.0


def test_transport_model_eval_profile():
    model = GyrokineticTransportModel()
    rho = np.linspace(0, 1, 10)
    profiles = {
        "R0": 2.0,
        "a": 0.5,
        "B0": 5.0,
        "q": np.full(10, 1.5),
        "s_hat": np.full(10, 1.0),
        "Te": np.linspace(5.0, 0.1, 10),
        "Ti": np.linspace(5.0, 0.1, 10),
        "ne": np.linspace(5.0, 0.1, 10),
        "dTe_dr": np.full(10, -10.0),
        "dTi_dr": np.full(10, -10.0),
        "dne_dr": np.full(10, -10.0),
    }
    chi_i, chi_e, D_e = model.evaluate_profile(rho, profiles)
    assert len(chi_i) == 10
    assert len(chi_e) == 10
    assert len(D_e) == 10
    assert np.all(chi_i >= 0.0)
    assert chi_i[0] == 0.01  # boundary handling
