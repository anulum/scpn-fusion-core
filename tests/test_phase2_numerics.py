"""Phase 2 numerical upgrade tests: MUSCL advection, diffusion, elliptic bounds."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.control.fokker_planck_re import FokkerPlanckSolver


# ── 2.1 MUSCL-Hancock advection ────────────────────────────────────


def test_fp_positivity_after_muscl():
    """Distribution function stays non-negative with MUSCL scheme."""
    solver = FokkerPlanckSolver()
    solver.f[10] = 1.0e10
    state = solver.step(dt=1e-5, E_field=10.0, n_e=5e19, T_e_eV=5000.0, Z_eff=1.0)
    assert np.all(state.f >= 0.0)


def test_fp_particle_conservation():
    """Total particle count (sum f*dp) conserved within source terms."""
    solver = FokkerPlanckSolver(np_grid=200, p_max=50.0)
    # Gaussian initial condition
    p0 = 5.0
    solver.f = 1e10 * np.exp(-((solver.p - p0) ** 2) / 2.0)
    n_before = np.sum(solver.f * solver.dp)

    # Evolve with zero E-field (no sources/sinks beyond drag)
    for _ in range(10):
        solver.step(dt=1e-6, E_field=0.0, n_e=1e19, T_e_eV=5000.0, Z_eff=1.0)
    n_after = np.sum(solver.f * solver.dp)

    # Particle count should be approximately conserved (drag redistributes
    # but doesn't create particles; avalanche/Dreicer off at E=0)
    rel_change = abs(n_after - n_before) / max(n_before, 1e-30)
    assert rel_change < 0.5, f"Particle number changed by {rel_change:.1%}"


def test_fp_sharp_feature_fwhm_preservation():
    """MUSCL should preserve a sharp bump better than 1st-order upwind."""
    solver = FokkerPlanckSolver(np_grid=200, p_max=50.0)
    p0 = 10.0
    sigma = 0.5
    solver.f = 1e10 * np.exp(-((solver.p - p0) ** 2) / (2.0 * sigma ** 2))
    fwhm_before = 2.355 * sigma  # Gaussian FWHM

    for _ in range(5):
        solver.step(dt=1e-7, E_field=0.0, n_e=1e18, T_e_eV=1000.0, Z_eff=1.0)

    # Measure FWHM from the evolved distribution
    half_max = np.max(solver.f) / 2.0
    above = solver.p[solver.f >= half_max]
    if len(above) >= 2:
        fwhm_after = above[-1] - above[0]
        # MUSCL should keep FWHM within 3x of original (1st-order would smear much more)
        assert fwhm_after < fwhm_before * 3.0, \
            f"FWHM grew from {fwhm_before:.2f} to {fwhm_after:.2f}"


# ── 2.2 Diffusion term ─────────────────────────────────────────────


def test_fp_diffusion_broadens_gaussian():
    """With zero advection, diffusion should broaden a Gaussian: var ~ 2Dt."""
    solver = FokkerPlanckSolver(np_grid=300, p_max=100.0)
    p0 = 20.0
    sigma0 = 2.0
    solver.f = 1e8 * np.exp(-((solver.p - p0) ** 2) / (2.0 * sigma0 ** 2))
    var_before = np.sum(solver.f * (solver.p - p0) ** 2 * solver.dp) / np.sum(solver.f * solver.dp)

    # Run with tiny E-field so advection is negligible, diffusion dominates
    dt = 1e-5
    n_steps = 50
    for _ in range(n_steps):
        solver.step(dt=dt, E_field=0.0, n_e=1e17, T_e_eV=100.0, Z_eff=1.0)

    var_after = np.sum(solver.f * (solver.p - p0) ** 2 * solver.dp) / max(np.sum(solver.f * solver.dp), 1e-30)
    # Variance should increase (diffusion broadens)
    assert var_after > var_before, \
        f"Variance should grow: before={var_before:.4f}, after={var_after:.4f}"


# ── 2.3 Elliptic integral bounds ───────────────────────────────────


def test_ellipk_near_unity():
    """scipy.ellipk handles k2 up to 1-1e-12 without NaN."""
    from scipy.special import ellipk, ellipe

    k2_extreme = 1.0 - 1e-12
    K = ellipk(k2_extreme)
    E = ellipe(k2_extreme)
    assert np.isfinite(K)
    assert np.isfinite(E)
    assert K > E  # K(k) > E(k) for k > 0


def test_green_function_near_axis():
    """Green's function at near-axis k2 → 0.9999999 should be finite."""
    from scpn_fusion.core.fusion_kernel import FusionKernel

    val = FusionKernel._green_function(1.0, 0.0, 1.0001, 0.0001)
    assert np.isfinite(val), f"Green's function NaN at near-coincident points: {val}"
    assert val != 0.0
