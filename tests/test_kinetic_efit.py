# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Kinetic EFIT Tests
from __future__ import annotations

import numpy as np

from scpn_fusion.control.realtime_efit import MagneticDiagnostics
from scpn_fusion.core.kinetic_efit import (
    FastIonPressure,
    KineticConstraints,
    KineticEFIT,
    mse_pitch_angle,
)


def mock_diagnostics() -> MagneticDiagnostics:
    return MagneticDiagnostics([(2.0, 1.0)], [(2.0, 1.0, "R")], rogowski_radius=3.0)


def mock_kin_constraints() -> KineticConstraints:
    return KineticConstraints(
        Te_points=[(6.2, 0.0, 10.0)],
        ne_points=[(6.2, 0.0, 5.0)],
        Ti_points=[(6.2, 0.0, 10.0)],
        mse_points=[(6.5, 0.0, 5.0)],
    )


def test_fast_ion_pressure():
    fi = FastIonPressure(E_fast_keV=100.0, n_fast_frac=0.1, anisotropy_sigma=0.2)
    rho = np.linspace(0, 1, 10)
    ne = np.ones(10) * 5.0

    p_perp = fi.p_perp(rho, ne)
    p_par = fi.p_par(rho, ne)
    p_iso = fi.p_isotropic_equivalent(rho, ne)

    assert np.all(p_perp > p_par)
    assert np.allclose((2 * p_perp + p_par) / 3.0, p_iso)


def test_mse_pitch_angle():
    pitch = mse_pitch_angle(B_R=0.0, B_Z=0.5, B_phi=5.0, v_beam=1e6, R=6.0)
    assert 5.0 < pitch < 6.0  # arctan(0.1) ~ 5.7 deg


def test_kinetic_efit_isotropic():
    diag = mock_diagnostics()
    kin = mock_kin_constraints()
    fi = FastIonPressure(100.0, 0.1, 0.0)

    R = np.linspace(4, 8, 33)
    Z = np.linspace(-3, 3, 33)
    kefit = KineticEFIT(diag, kin, fi, R, Z)

    res = kefit.reconstruct({})

    assert res.pressure_consistency == 0.1
    assert res.wall_time_ms < 200.0
    assert len(res.p_kinetic) == 50


def test_kinetic_efit_anisotropic():
    diag = mock_diagnostics()
    kin = mock_kin_constraints()
    fi = FastIonPressure(100.0, 0.1, 0.2)

    R = np.linspace(4, 8, 33)
    Z = np.linspace(-3, 3, 33)
    kefit = KineticEFIT(diag, kin, fi, R, Z)

    res = kefit.reconstruct({})

    assert res.pressure_consistency > 0.1  # slightly worse initially due to anisotropic terms
    assert np.all(res.sigma_anisotropy == 0.2)
    assert res.beta_fast > 0.0


def test_mse_constraint_q_profile():
    diag = mock_diagnostics()
    kin = mock_kin_constraints()
    fi = FastIonPressure(100.0, 0.0, 0.0)

    R = np.linspace(4, 8, 33)
    Z = np.linspace(-3, 3, 33)

    kefit_mse = KineticEFIT(diag, kin, fi, R, Z)

    kin_no_mse = mock_kin_constraints()
    kin_no_mse.mse_points = []
    kefit_no_mse = KineticEFIT(diag, kin_no_mse, fi, R, Z)

    res_mse = kefit_mse.reconstruct({})
    res_no_mse = kefit_no_mse.reconstruct({})

    # MSE should constrain q to be closer to 1.0 at axis
    assert res_mse.q_profile[0] < res_no_mse.q_profile[0]
