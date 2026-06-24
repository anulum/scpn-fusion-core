# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Kinetic EFIT Tests
from __future__ import annotations

import numpy as np
import pytest

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


def mock_measurements(diag: MagneticDiagnostics) -> dict[str, object]:
    return {
        "flux_loops": np.zeros(len(diag.flux_loops), dtype=float),
        "b_probes": np.zeros(len(diag.b_probes), dtype=float),
        "coil_currents": np.zeros(1, dtype=float),
        "Ip": 15.0e6,
    }


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


def test_mse_pitch_angle_depends_on_beam_geometry_for_nonzero_br():
    slow = mse_pitch_angle(B_R=1.0, B_Z=0.2, B_phi=5.0, v_beam=1e5, R=8.0)
    fast = mse_pitch_angle(B_R=1.0, B_Z=0.2, B_phi=5.0, v_beam=2e7, R=3.0)
    assert fast > slow


def test_mse_pitch_angle_rejects_invalid_beam_inputs():
    with pytest.raises(ValueError, match="v_beam"):
        mse_pitch_angle(B_R=0.0, B_Z=0.5, B_phi=5.0, v_beam=0.0, R=6.0)
    with pytest.raises(ValueError, match="R"):
        mse_pitch_angle(B_R=0.0, B_Z=0.5, B_phi=5.0, v_beam=1e6, R=0.0)


def test_kinetic_efit_isotropic():
    diag = mock_diagnostics()
    kin = mock_kin_constraints()
    fi = FastIonPressure(100.0, 0.1, 0.0)

    R = np.linspace(4, 8, 33)
    Z = np.linspace(-3, 3, 33)
    kefit = KineticEFIT(diag, kin, fi, R, Z)

    res = kefit.reconstruct(mock_measurements(diag))

    assert res.pressure_consistency == pytest.approx(0.1)
    assert np.isfinite(res.wall_time_ms)
    assert res.wall_time_ms > 0.0
    assert len(res.p_kinetic) == 50


def test_kinetic_efit_anisotropic():
    diag = mock_diagnostics()
    kin = mock_kin_constraints()
    fi = FastIonPressure(100.0, 0.1, 0.2)

    R = np.linspace(4, 8, 33)
    Z = np.linspace(-3, 3, 33)
    kefit = KineticEFIT(diag, kin, fi, R, Z)

    res = kefit.reconstruct(mock_measurements(diag))

    assert res.pressure_consistency > 0.1
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

    res_mse = kefit_mse.reconstruct(mock_measurements(diag))
    res_no_mse = kefit_no_mse.reconstruct(mock_measurements(diag))

    # MSE should constrain q to be closer to 1.0 at axis
    assert res_mse.q_profile[0] < res_no_mse.q_profile[0]


def test_mse_pitch_content_changes_q_profile_axis_level():
    diag = mock_diagnostics()
    kin_low_pitch = mock_kin_constraints()
    kin_high_pitch = mock_kin_constraints()
    kin_low_pitch.mse_points = [(6.5, 0.0, 2.0)]
    kin_high_pitch.mse_points = [(6.5, 0.0, 12.0)]
    fi = FastIonPressure(100.0, 0.0, 0.0)

    R = np.linspace(4, 8, 33)
    Z = np.linspace(-3, 3, 33)
    res_low = KineticEFIT(diag, kin_low_pitch, fi, R, Z).reconstruct(mock_measurements(diag))
    res_high = KineticEFIT(diag, kin_high_pitch, fi, R, Z).reconstruct(mock_measurements(diag))
    assert res_high.q_profile[0] < res_low.q_profile[0]


def test_fast_ion_pressure_rejects_invalid_anisotropy():
    with pytest.raises(ValueError, match="anisotropy_sigma"):
        FastIonPressure(E_fast_keV=100.0, n_fast_frac=0.1, anisotropy_sigma=3.0)


def test_ti_points_affect_kinetic_pressure_level():
    diag = mock_diagnostics()
    kin_cold = mock_kin_constraints()
    kin_hot = mock_kin_constraints()
    kin_cold.Ti_points = [(6.2, 0.0, 5.0)]
    kin_hot.Ti_points = [(6.2, 0.0, 20.0)]
    fi = FastIonPressure(100.0, 0.0, 0.0)

    R = np.linspace(4, 8, 33)
    Z = np.linspace(-3, 3, 33)
    res_cold = KineticEFIT(diag, kin_cold, fi, R, Z).reconstruct(mock_measurements(diag))
    res_hot = KineticEFIT(diag, kin_hot, fi, R, Z).reconstruct(mock_measurements(diag))

    assert float(np.mean(res_hot.p_kinetic)) > float(np.mean(res_cold.p_kinetic))


def test_fast_ion_pressure_constructor_validation() -> None:
    from scpn_fusion.core.kinetic_efit import FastIonPressure
    with pytest.raises(ValueError, match="E_fast_keV"):
        FastIonPressure(E_fast_keV=0.0, n_fast_frac=0.05)
    with pytest.raises(ValueError, match="n_fast_frac"):
        FastIonPressure(E_fast_keV=80.0, n_fast_frac=1.5)


def test_mse_pitch_angle_validates_inputs() -> None:
    from scpn_fusion.core.kinetic_efit import mse_pitch_angle
    with pytest.raises(ValueError, match="must be finite"):
        mse_pitch_angle(float("nan"), 0.1, 5.0, 1e6, 6.2)
    with pytest.raises(ValueError, match="v_beam"):
        mse_pitch_angle(0.1, 0.1, 5.0, 0.0, 6.2)


def test_kinetic_efit_reconstruct_interpolates_rich_constraints() -> None:
    diag = mock_diagnostics()
    fi = FastIonPressure(100.0, 0.1, 0.0)
    R = np.linspace(4, 8, 33)
    Z = np.linspace(-3, 3, 33)
    # >=2 distinct rho samples (exercise the interp branch) + a non-finite point (skip branch).
    kin = KineticConstraints(
        Te_points=[(5.0, 0.0, 8.0), (6.5, 0.0, 12.0), (np.nan, 0.0, 9.0)],
        ne_points=[(5.0, 0.0, 4.0), (6.5, 0.0, 6.0), (np.inf, 0.0, 5.0)],
        Ti_points=[(5.0, 0.0, 7.0), (6.5, 0.0, 11.0)],
        mse_points=[(5.0, 0.0, 4.0), (6.8, 0.0, 6.0), (np.nan, 0.0, 5.0)],
    )
    kefit = KineticEFIT(diag, kin, fi, R, Z)
    res = kefit.reconstruct(mock_measurements(diag))
    assert res is not None
    assert np.all(np.isfinite(res.p_kinetic))


def test_kinetic_efit_reconstruct_falls_back_on_all_nonfinite_points() -> None:
    diag = mock_diagnostics()
    fi = FastIonPressure(100.0, 0.1, 0.0)
    R = np.linspace(4, 8, 33)
    Z = np.linspace(-3, 3, 33)
    # Every constraint point is non-finite -> profiles fall back to defaults, q -> base.
    kin = KineticConstraints(
        Te_points=[(np.nan, 0.0, 8.0)],
        ne_points=[(np.nan, 0.0, 4.0)],
        Ti_points=[(np.nan, 0.0, 7.0)],
        mse_points=[(np.nan, 0.0, 4.0)],
    )
    kefit = KineticEFIT(diag, kin, fi, R, Z)
    res = kefit.reconstruct(mock_measurements(diag))
    assert res is not None
    assert np.all(np.isfinite(res.p_kinetic))
