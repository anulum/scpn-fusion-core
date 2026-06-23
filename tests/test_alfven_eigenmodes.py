# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Alfven Eigenmodes Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.alfven_eigenmodes import (
    AlfvenContinuum,
    AlfvenStabilityAnalysis,
    FastParticleDrive,
    TAEMode,
    bae_accumulation_frequency,
    rsae_frequency,
)


def test_continuum_gaps():
    rho = np.linspace(0, 1, 100)
    q = np.linspace(1.0, 3.0, 100)
    ne = np.ones(100) * 5.0  # 5e19

    cont = AlfvenContinuum(rho, q, ne, B0=5.3, R0=6.2)

    # Check v_A is reasonable (~1e7 m/s for ITER)
    v_a = cont.alfven_speed(0.5)
    assert 5e6 < v_a < 5e7

    # Find gaps for n=1
    gaps = cont.find_gaps(n=1)

    # For n=1, m=1 -> q_gap = 1.5. m=2 -> q_gap = 2.5
    # Since q goes 1 to 3, we should find at least two gaps
    assert len(gaps) >= 2

    q_gaps = [(2.0 * g.m_coupling + 1.0) / (2.0) for g in gaps]
    assert 1.5 in q_gaps
    assert 2.5 in q_gaps

    # Gap width check
    for g in gaps:
        assert g.omega_upper > g.omega_lower


def test_tae_frequency():
    v_A = 1e7
    q = 1.5
    R0 = 6.0
    tae = TAEMode(n=1, q_rational=q, v_A=v_A, R0=R0)

    # omega = v_A / (2 q R0) = 1e7 / (2 * 1.5 * 6) = 1e7 / 18 ~ 5.5e5 rad/s
    omega = tae.frequency()
    assert np.isclose(omega, 1e7 / 18.0)

    # f = omega / 2pi ~ 88 kHz
    f_khz = tae.frequency_kHz()
    assert 50.0 < f_khz < 150.0


def test_fast_particle_drive():
    # 3.5 MeV alphas
    drive = FastParticleDrive(E_fast_keV=3500.0, n_fast_frac=0.01, m_fast_amu=4.0)

    v_A = 1e7
    # Resonance peaking at v_fast ~ v_A / 3 or v_A
    F1 = drive.resonance_function(v_A, v_A)
    F3 = drive.resonance_function(v_A / 3.0, v_A)
    F_off = drive.resonance_function(v_A / 10.0, v_A)

    assert F_off < F1
    assert F_off < F3

    b_fast = drive.beta_fast(ne_20=1.0, B0=5.3)
    assert b_fast > 0.0


def test_stability_analysis():
    rho = np.linspace(0, 1, 100)
    q = np.linspace(1.0, 3.0, 100)
    ne = np.ones(100) * 5.0

    cont = AlfvenContinuum(rho, q, ne, B0=5.3, R0=6.2)
    # 3.5 MeV alphas with high fraction to trigger instability
    drive = FastParticleDrive(E_fast_keV=3500.0, n_fast_frac=0.05, m_fast_amu=4.0)

    analysis = AlfvenStabilityAnalysis(cont, drive)

    results = analysis.tae_stability(n_range=[1])
    assert len(results) > 0

    res = results[0]
    assert res.gamma_drive > 0.0
    assert res.gamma_damp > 0.0

    beta_c = analysis.critical_beta_fast(n=1)
    assert beta_c > 0.0


def test_rsae_frequency():
    freq1 = rsae_frequency(q_min=1.4, n=1, m=1, v_A=1e7, R0=6.0)
    freq2 = rsae_frequency(q_min=1.1, n=1, m=1, v_A=1e7, R0=6.0)

    # As q_min approaches m/n=1, frequency drops toward BAE gap
    assert freq2 < freq1


def test_bae_accumulation_frequency_uses_thermal_ion_physics():
    base = bae_accumulation_frequency(
        Ti_keV=10.0,
        Te_keV=10.0,
        m_i_amu=2.5,
        R0=6.2,
    )
    hotter = bae_accumulation_frequency(
        Ti_keV=40.0,
        Te_keV=40.0,
        m_i_amu=2.5,
        R0=6.2,
    )
    colder_electrons = bae_accumulation_frequency(
        Ti_keV=10.0,
        Te_keV=5.0,
        m_i_amu=2.5,
        R0=6.2,
    )

    assert base > 0.0
    assert np.isclose(hotter / base, 2.0, rtol=0.05)
    assert colder_electrons < base


def test_rsae_frequency_accepts_bae_physics_parameters():
    near_rational = rsae_frequency(
        q_min=1.01,
        n=1,
        m=1,
        v_A=1.0e7,
        R0=6.2,
        Ti_keV=12.0,
        Te_keV=8.0,
        m_i_amu=2.5,
    )
    bae = bae_accumulation_frequency(Ti_keV=12.0, Te_keV=8.0, m_i_amu=2.5, R0=6.2)
    far_from_rational = rsae_frequency(
        q_min=1.4,
        n=1,
        m=1,
        v_A=1.0e7,
        R0=6.2,
        Ti_keV=12.0,
        Te_keV=8.0,
        m_i_amu=2.5,
    )

    assert near_rational > bae
    assert near_rational < 1.25 * bae
    assert far_from_rational > near_rational


def test_alfven_inputs_reject_nonphysical_values():
    with np.testing.assert_raises(ValueError):
        bae_accumulation_frequency(Ti_keV=0.0, Te_keV=10.0, m_i_amu=2.5, R0=6.2)

    with np.testing.assert_raises(ValueError):
        rsae_frequency(q_min=-1.0, n=1, m=1, v_A=1.0e7, R0=6.2)


def test_fast_particle_drive_rejects_invalid_resonance_params() -> None:
    with np.testing.assert_raises(ValueError):
        FastParticleDrive(E_fast_keV=3500.0, n_fast_frac=0.01, main_resonance_width=0.0)
    with np.testing.assert_raises(ValueError):
        FastParticleDrive(E_fast_keV=3500.0, n_fast_frac=0.01, sideband_resonance_width=-0.1)
    with np.testing.assert_raises(ValueError):
        FastParticleDrive(E_fast_keV=3500.0, n_fast_frac=0.01, sideband_weight=-1.0)


def test_resonance_response_width_controls_off_resonant_suppression() -> None:
    broad = FastParticleDrive(
        E_fast_keV=3500.0,
        n_fast_frac=0.01,
        main_resonance_width=0.35,
        sideband_resonance_width=0.2,
    )
    narrow = FastParticleDrive(
        E_fast_keV=3500.0,
        n_fast_frac=0.01,
        main_resonance_width=0.1,
        sideband_resonance_width=0.05,
    )
    v_a = 1.0e7
    v_off = 0.75e7
    f_broad = broad.resonance_function(v_off, v_a)
    f_narrow = narrow.resonance_function(v_off, v_a)
    assert f_narrow < f_broad


def test_alpha_particle_loss_estimate_is_bounded_and_monotone() -> None:
    rho = np.linspace(0, 1, 64)
    q = np.linspace(1.0, 3.0, 64)
    ne = np.full(64, 5.0)
    cont = AlfvenContinuum(rho, q, ne, B0=5.3, R0=6.2)
    drive = FastParticleDrive(E_fast_keV=3500.0, n_fast_frac=0.02, m_fast_amu=4.0)
    analysis = AlfvenStabilityAnalysis(cont, drive)

    loss_small = analysis.alpha_particle_loss_estimate(gamma_net=100.0, tau_sd=0.5)
    loss_large = analysis.alpha_particle_loss_estimate(gamma_net=2000.0, tau_sd=0.5)
    assert 0.0 <= loss_small <= 1.0
    assert 0.0 <= loss_large <= 1.0
    assert loss_large > loss_small
    assert analysis.alpha_particle_loss_estimate(gamma_net=-1.0, tau_sd=0.5) == 0.0


def test_alpha_particle_loss_estimate_rejects_invalid_inputs() -> None:
    rho = np.linspace(0, 1, 16)
    q = np.linspace(1.0, 3.0, 16)
    ne = np.full(16, 5.0)
    cont = AlfvenContinuum(rho, q, ne, B0=5.3, R0=6.2)
    drive = FastParticleDrive(E_fast_keV=3500.0, n_fast_frac=0.02, m_fast_amu=4.0)
    analysis = AlfvenStabilityAnalysis(cont, drive)

    with np.testing.assert_raises(ValueError):
        analysis.alpha_particle_loss_estimate(gamma_net=np.nan, tau_sd=0.5)
    with np.testing.assert_raises(ValueError):
        analysis.alpha_particle_loss_estimate(gamma_net=1.0, tau_sd=0.0)


def test_tae_gap_width_increases_with_shaping_and_inverse_aspect_ratio() -> None:
    rho = np.linspace(0, 1, 100)
    q = np.linspace(1.0, 3.0, 100)
    ne = np.full(100, 5.0)
    base = AlfvenContinuum(rho, q, ne, B0=5.3, R0=6.2, a=1.5, kappa=1.3)
    shaped = AlfvenContinuum(rho, q, ne, B0=5.3, R0=6.2, a=2.2, kappa=2.0)
    g_base = next(g for g in base.find_gaps(n=1) if g.m_coupling == 1)
    g_shaped = next(g for g in shaped.find_gaps(n=1) if g.m_coupling == 1)
    w_base = g_base.omega_upper - g_base.omega_lower
    w_shaped = g_shaped.omega_upper - g_shaped.omega_lower
    assert w_shaped > w_base


def test_alfven_continuum_rejects_invalid_geometry_and_gap_scale() -> None:
    rho = np.linspace(0, 1, 10)
    q = np.linspace(1.0, 2.0, 10)
    ne = np.full(10, 5.0)
    with np.testing.assert_raises(ValueError):
        AlfvenContinuum(rho, q, ne, B0=5.3, R0=0.0)
    with np.testing.assert_raises(ValueError):
        AlfvenContinuum(rho, q, ne, B0=5.3, R0=6.2, gap_width_scale=0.0)


def test_critical_beta_fast_rejects_invalid_mode_number() -> None:
    rho = np.linspace(0, 1, 64)
    q = np.linspace(1.0, 3.0, 64)
    ne = np.full(64, 5.0)
    cont = AlfvenContinuum(rho, q, ne, B0=5.3, R0=6.2)
    drive = FastParticleDrive(E_fast_keV=3500.0, n_fast_frac=0.02, m_fast_amu=4.0)
    analysis = AlfvenStabilityAnalysis(cont, drive)
    with np.testing.assert_raises(ValueError):
        analysis.critical_beta_fast(n=0)


def test_critical_beta_fast_matches_mode_local_formula() -> None:
    rho = np.linspace(0, 1, 64)
    q = np.linspace(1.0, 3.0, 64)
    ne = np.full(64, 5.0)
    drive = FastParticleDrive(E_fast_keV=3500.0, n_fast_frac=0.02, m_fast_amu=4.0)
    analysis = AlfvenStabilityAnalysis(AlfvenContinuum(rho, q, ne, B0=5.3, R0=6.2), drive)
    beta_crit = analysis.critical_beta_fast(n=1)
    assert beta_crit > 0.0

    manual = float("inf")
    for gap in analysis.continuum.find_gaps(1):
        q_gap = (2.0 * gap.m_coupling + 1.0) / 2.0
        v_a = analysis.continuum.alfven_speed(gap.rho_location)
        resonance = drive.resonance_function(drive.v_fast, v_a)
        coeff = q_gap**2 * resonance
        if coeff > 0.0:
            manual = min(manual, 0.01 / coeff)
    assert beta_crit == pytest.approx(manual)


def test_alfven_continuum_rejects_nonpositive_geometry() -> None:
    rho = np.linspace(0.0, 1.0, 5)
    q = np.linspace(1.0, 3.0, 5)
    ne = np.ones(5)
    for kw, match in [
        ({"B0": 0.0, "R0": 6.2}, "B0"),
        ({"B0": 5.3, "R0": 0.0}, "R0"),
        ({"B0": 5.3, "R0": 6.2, "a": 0.0}, "a must be"),
        ({"B0": 5.3, "R0": 6.2, "kappa": 0.0}, "kappa"),
        ({"B0": 5.3, "R0": 6.2, "gap_width_scale": 0.0}, "gap_width_scale"),
    ]:
        with pytest.raises(ValueError, match=match):
            AlfvenContinuum(rho, q, ne, **kw)


def test_fast_particle_resonance_rejects_nonpositive_velocities() -> None:
    drive = FastParticleDrive(E_fast_keV=3500.0, n_fast_frac=0.01, m_fast_amu=4.0)
    with pytest.raises(ValueError, match="v_fast"):
        drive.resonance_function(0.0, 1.0e6)
    with pytest.raises(ValueError, match="v_A"):
        drive.resonance_function(1.0e6, 0.0)


def _alfven_analysis():
    rho = np.linspace(0.0, 1.0, 5)
    q = np.linspace(1.0, 3.0, 5)
    ne = np.ones(5)
    cont = AlfvenContinuum(rho, q, ne, B0=5.3, R0=6.2)
    drive = FastParticleDrive(E_fast_keV=3500.0, n_fast_frac=0.01, m_fast_amu=4.0)
    return cont, AlfvenStabilityAnalysis(cont, drive)


def test_alfven_find_gaps_and_critical_beta_reject_nonpositive_n() -> None:
    cont, analysis = _alfven_analysis()
    assert cont.continuum(1, 1).shape == cont.q.shape
    with pytest.raises(ValueError, match="positive integer"):
        analysis.critical_beta_fast(0)
    # a valid mode reaches the beta-crit accumulation path
    assert analysis.critical_beta_fast(1) >= 0.0


def test_alfven_alpha_loss_estimate_validation() -> None:
    _, analysis = _alfven_analysis()
    with pytest.raises(ValueError, match="tau_sd"):
        analysis.alpha_particle_loss_estimate(0.1, tau_sd=0.0)
    with pytest.raises(ValueError, match="nonlinear_saturation"):
        analysis.alpha_particle_loss_estimate(0.1, nonlinear_saturation=0.0)
    with pytest.raises(ValueError, match="transport_threshold"):
        analysis.alpha_particle_loss_estimate(0.1, transport_threshold=0.0)


def test_rsae_frequency_rejects_zero_toroidal_mode() -> None:
    import inspect
    sig = inspect.signature(rsae_frequency)
    args = {p: (0 if p == "n" else 1.5) for p in sig.parameters if sig.parameters[p].default is inspect._empty}
    with pytest.raises(ValueError, match="nonzero"):
        rsae_frequency(**args)


def test_alfven_critical_beta_infinite_without_rational_gaps() -> None:
    rho = np.linspace(0.0, 1.0, 5)
    q = np.full(5, 2.0)  # flat q: no rational gaps
    ne = np.ones(5)
    cont = AlfvenContinuum(rho, q, ne, B0=5.3, R0=6.2)
    drive = FastParticleDrive(E_fast_keV=3500.0, n_fast_frac=0.01, m_fast_amu=4.0)
    assert AlfvenStabilityAnalysis(cont, drive).critical_beta_fast(2) == float("inf")
