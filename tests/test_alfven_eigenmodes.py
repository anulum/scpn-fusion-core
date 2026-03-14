# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Alfven Eigenmodes Tests
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import numpy as np

from scpn_fusion.core.alfven_eigenmodes import (
    AlfvenContinuum,
    AlfvenStabilityAnalysis,
    FastParticleDrive,
    TAEMode,
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
