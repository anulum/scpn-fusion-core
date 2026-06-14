# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Orbit Following Tests
"""Guiding-centre orbit and NRL fast-ion slowing-down physics tests.

The slowing-down assertions are checked against hand-computed NRL Plasma
Formulary (2019) values for an ITER-like D-T alpha; the drift assertion is
checked against the analytic grad-B drift speed mu|grad B|/(q B).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scpn_fusion.core.orbit_following import (
    ALPHA_MASS_AMU,
    ATOMIC_MASS_KG,
    ELEMENTARY_CHARGE_C,
    GuidingCenterOrbit,
    MonteCarloEnsemble,
    OrbitClassifier,
    SlowingDown,
    first_orbit_loss,
)


def mock_b_field(R, Z):
    """Toroidal field B0 R0 / R plus a small poloidal field for trapping."""
    B0 = 5.0
    R0 = 6.0
    B_phi = B0 * R0 / R
    B_R = -0.1 * Z
    B_Z = 0.1 * (R - R0)
    return B_R, B_Z, B_phi


def _b_magnitude(R, Z):
    B_R, B_Z, B_phi = mock_b_field(R, Z)
    return math.sqrt(B_R**2 + B_Z**2 + B_phi**2)


# ── Guiding-centre integrator ────────────────────────────────────────


def test_passing_orbit_keeps_sign_and_stays_in_vessel():
    orbit = GuidingCenterOrbit(4.0, 2, 3500.0, 0.0, 6.2, 0.0)
    for _ in range(10):
        orbit.step(mock_b_field, 1e-6)
    assert orbit.v_par > 0.0
    assert orbit.R > 0.0


def test_magnetic_moment_is_set_from_first_field_sample():
    orbit = GuidingCenterOrbit(4.0, 2, 3500.0, math.pi / 2 - 0.1, 6.2, 0.0)
    orbit.step(mock_b_field, 1e-6)
    assert orbit.mu > 0.0


def test_energy_is_conserved_in_static_field():
    """RK4 guiding-centre motion conserves kinetic energy to high accuracy."""
    orbit = GuidingCenterOrbit(4.0, 2, 3500.0, 0.6, 6.2, 0.0)
    orbit.step(mock_b_field, 1e-9)  # initialise mu

    def kinetic_energy() -> float:
        b = _b_magnitude(orbit.R, orbit.Z)
        v_perp_sq = 2.0 * orbit.mu * b / orbit.m
        return 0.5 * orbit.m * (orbit.v_par**2 + v_perp_sq)

    e0 = kinetic_energy()
    for _ in range(2000):
        orbit.step(mock_b_field, 1e-9)
    assert abs(kinetic_energy() - e0) / e0 < 1e-6


def test_grad_b_drift_matches_analytic_speed():
    """A purely perpendicular particle drifts at mu|grad B|/(q B) (grad-B drift).

    This pins the drift coefficient: the perpendicular contribution is
    v_perp^2/2 = mu B / m, so the bug of using mu B (which suppresses the
    grad-B drift by a factor m) is caught here.
    """
    orbit = GuidingCenterOrbit(4.0, 2, 3500.0, math.pi / 2, 6.5, 0.5)
    state = np.array([orbit.R, orbit.Z, orbit.phi, orbit.v_par])
    deriv = orbit._eom(state, mock_b_field)
    drift_speed = math.sqrt(deriv[0] ** 2 + deriv[1] ** 2 + (orbit.R * deriv[2]) ** 2)

    b = _b_magnitude(orbit.R, orbit.Z)
    eps = 1e-4
    grad_r = (_b_magnitude(orbit.R + eps, orbit.Z) - b) / eps
    grad_z = (_b_magnitude(orbit.R, orbit.Z + eps) - b) / eps
    grad_b = math.sqrt(grad_r**2 + grad_z**2)
    analytic = orbit.mu * grad_b / (orbit.Z_charge * b)
    assert drift_speed == pytest.approx(analytic, rel=1e-3)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"m_amu": 0.0, "Z": 2, "E_keV": 3500.0, "pitch_angle": 0.0, "R0_init": 6.2, "Z0_init": 0.0}, "m_amu"),
        ({"m_amu": 4.0, "Z": 0, "E_keV": 3500.0, "pitch_angle": 0.0, "R0_init": 6.2, "Z0_init": 0.0}, "Z"),
        ({"m_amu": 4.0, "Z": 2, "E_keV": -1.0, "pitch_angle": 0.0, "R0_init": 6.2, "Z0_init": 0.0}, "E_keV"),
        ({"m_amu": 4.0, "Z": 2, "E_keV": 3500.0, "pitch_angle": 4.0, "R0_init": 6.2, "Z0_init": 0.0}, "pitch_angle"),
        ({"m_amu": 4.0, "Z": 2, "E_keV": 3500.0, "pitch_angle": 0.0, "R0_init": 0.0, "Z0_init": 0.0}, "R0_init"),
    ],
)
def test_guiding_center_rejects_unphysical_inputs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        GuidingCenterOrbit(**kwargs)


def test_step_rejects_non_positive_dt():
    orbit = GuidingCenterOrbit(4.0, 2, 3500.0, 0.3, 6.2, 0.0)
    with pytest.raises(ValueError, match="dt"):
        orbit.step(mock_b_field, 0.0)


# ── Orbit classifier ─────────────────────────────────────────────────


def test_orbit_classifier_passing_trapped_lost():
    R = np.ones(10) * 6.0
    Z = np.zeros(10)
    v_par_pass = np.ones(10)
    v_par_trap = np.array([1.0, 0.5, -0.5, -1.0, -0.5, 0.5, 1.0, 1.0, 1.0, 1.0])
    assert OrbitClassifier.classify(R, Z, v_par_pass, 10.0, 5.0) == "passing"
    assert OrbitClassifier.classify(R, Z, v_par_trap, 10.0, 5.0) == "trapped"

    R_lost = np.array([6.0, 7.0, 11.0, 12.0])
    assert OrbitClassifier.classify(R_lost, Z[:4], np.ones(4), 10.0, 5.0) == "lost"


def test_orbit_classifier_rejects_empty_and_bad_walls():
    with pytest.raises(ValueError, match="non-empty"):
        OrbitClassifier.classify(np.array([]), np.array([]), np.array([]), 10.0, 5.0)
    with pytest.raises(ValueError, match="R_wall"):
        OrbitClassifier.classify(np.ones(3), np.zeros(3), np.ones(3), -1.0, 5.0)


# ── Monte-Carlo ensemble ─────────────────────────────────────────────


def test_mc_ensemble_is_deterministic_for_fixed_seed():
    ens_a = MonteCarloEnsemble(12, 3500.0, 6.2, 2.0, 5.3)
    ens_a.initialize(seed=7)
    ens_b = MonteCarloEnsemble(12, 3500.0, 6.2, 2.0, 5.3)
    ens_b.initialize(seed=7)
    res_a = ens_a.follow(mock_b_field, n_steps=40)
    res_b = ens_b.follow(mock_b_field, n_steps=40)
    assert (res_a.n_passing, res_a.n_trapped, res_a.n_lost) == (
        res_b.n_passing,
        res_b.n_trapped,
        res_b.n_lost,
    )
    assert res_a.n_passing + res_a.n_trapped + res_a.n_lost == 12


def test_mc_ensemble_follow_before_initialize_raises():
    ens = MonteCarloEnsemble(5, 3500.0, 6.2, 2.0, 5.3)
    with pytest.raises(ValueError, match="initialize"):
        ens.follow(mock_b_field)


def test_mc_ensemble_rejects_bad_construction():
    with pytest.raises(ValueError, match="n_particles"):
        MonteCarloEnsemble(0, 3500.0, 6.2, 2.0, 5.3)
    with pytest.raises(ValueError, match="a"):
        MonteCarloEnsemble(5, 3500.0, 6.2, -1.0, 5.3)


# ── First-orbit loss (poloidal-gyroradius scaling) ───────────────────


def test_first_orbit_loss_falls_with_plasma_current():
    """Larger plasma current -> smaller banana width -> smaller loss zone."""
    low_ip = first_orbit_loss(R0=6.2, a=2.0, Ip_MA=5.0)
    high_ip = first_orbit_loss(R0=6.2, a=2.0, Ip_MA=15.0)
    assert high_ip < low_ip


def test_first_orbit_loss_iter_vs_compact_device():
    iter_loss = first_orbit_loss(R0=6.2, a=2.0, Ip_MA=15.0)
    compact_loss = first_orbit_loss(R0=0.9, a=0.6, Ip_MA=1.0)
    assert 0.0 < iter_loss < 0.2
    assert compact_loss > iter_loss
    assert compact_loss == pytest.approx(1.0)


def test_first_orbit_loss_matches_poloidal_gyroradius_definition():
    r0, a, ip_ma = 6.2, 2.0, 15.0
    loss = first_orbit_loss(R0=r0, a=a, Ip_MA=ip_ma)
    mass = ALPHA_MASS_AMU * ATOMIC_MASS_KG
    v = math.sqrt(2.0 * 3520.0e3 * ELEMENTARY_CHARGE_C / mass)
    b_pol = 1.25663706212e-6 * ip_ma * 1e6 / (2.0 * math.pi * a)
    rho_pol = mass * v / (2 * ELEMENTARY_CHARGE_C * b_pol)
    assert loss == pytest.approx(rho_pol / a, rel=1e-9)


def test_first_orbit_loss_rejects_bad_inputs():
    with pytest.raises(ValueError, match="Ip_MA"):
        first_orbit_loss(R0=6.2, a=2.0, Ip_MA=0.0)


# ── NRL slowing-down physics ─────────────────────────────────────────


def test_coulomb_logarithm_matches_nrl_value():
    # ln Lambda = 24 - ln(sqrt(1e14)/2e4) = 24 - ln(500) = 17.79
    assert SlowingDown.coulomb_logarithm_ei(20.0, 1.0) == pytest.approx(17.785, abs=2e-3)


def test_critical_velocity_matches_nrl_and_is_density_independent():
    v_c = SlowingDown.critical_velocity(20.0, 2.5)
    assert v_c == pytest.approx(5.56e6, rel=1e-2)
    # The critical energy of a D-T alpha is ~0.6 MeV.
    e_c_kev = 0.5 * ALPHA_MASS_AMU * ATOMIC_MASS_KG * v_c**2 / (1e3 * ELEMENTARY_CHARGE_C)
    assert 500.0 < e_c_kev < 750.0
    # v_c scales as sqrt(T_e) and does not depend on density.
    assert SlowingDown.critical_velocity(80.0, 2.5) == pytest.approx(2.0 * v_c, rel=1e-9)


def test_electron_slowing_down_time_iter_value_and_scalings():
    tau = SlowingDown.electron_slowing_down_time(20.0, 1.0, 4.0, 2)
    # ITER-like D-T alpha electron drag time is of order 1 s.
    assert 0.5 < tau < 1.5
    # tau_se ~ T_e^{3/2} (at fixed Coulomb log).
    tau_hot = SlowingDown.electron_slowing_down_time(40.0, 1.0, 4.0, 2, coulomb_log=17.0)
    tau_ref = SlowingDown.electron_slowing_down_time(20.0, 1.0, 4.0, 2, coulomb_log=17.0)
    assert tau_hot / tau_ref == pytest.approx(2.0**1.5, rel=1e-9)
    # tau_se ~ 1 / n_e.
    tau_dense = SlowingDown.electron_slowing_down_time(20.0, 2.0, 4.0, 2, coulomb_log=17.0)
    assert tau_dense / tau_ref == pytest.approx(0.5, rel=1e-9)


def test_slowing_down_time_is_monotonic_and_bounded():
    v_c = SlowingDown.critical_velocity(20.0, 2.5)
    tau_se = SlowingDown.electron_slowing_down_time(20.0, 1.0, 4.0, 2)
    v0 = 1.3e7
    t_to_vc = SlowingDown.slowing_down_time(v0, v_c, v_c, tau_se)
    t_to_half_vc = SlowingDown.slowing_down_time(v0, 0.5 * v_c, v_c, tau_se)
    assert t_to_vc > 0.0
    assert t_to_half_vc > t_to_vc  # reaching a lower speed takes longer
    # No net time to "slow" to the same speed.
    assert SlowingDown.slowing_down_time(v0, v0, v_c, tau_se) == pytest.approx(0.0, abs=1e-12)


def test_slowing_down_time_rejects_speed_up():
    with pytest.raises(ValueError, match="v2 must not exceed v1"):
        SlowingDown.slowing_down_time(1.0e6, 2.0e6, 5.0e6, 1.0)


def test_heating_partition_crosses_over_at_critical_velocity():
    v_c = 5.0e6
    f_ion_fast, f_e_fast = SlowingDown.heating_partition(3.0 * v_c, v_c)
    f_ion_slow, f_e_slow = SlowingDown.heating_partition(0.3 * v_c, v_c)
    assert f_e_fast > f_ion_fast  # fast ion heats electrons
    assert f_ion_slow > f_e_slow  # slow ion heats bulk ions
    # Exactly at v_c the split is even.
    f_ion_crit, f_e_crit = SlowingDown.heating_partition(v_c, v_c)
    assert f_ion_crit == pytest.approx(0.5)
    assert f_e_crit == pytest.approx(0.5)


@pytest.mark.parametrize(
    "call",
    [
        lambda: SlowingDown.coulomb_logarithm_ei(-1.0, 1.0),
        lambda: SlowingDown.critical_velocity(0.0),
        lambda: SlowingDown.electron_slowing_down_time(20.0, 0.0),
        lambda: SlowingDown.heating_partition(1.0e6, 0.0),
    ],
)
def test_slowing_down_rejects_unphysical_inputs(call):
    with pytest.raises(ValueError):
        call()


# ── Additional contract guards (degenerate-field and per-parameter) ──


def test_eom_rejects_zero_magnitude_field():
    orbit = GuidingCenterOrbit(4.0, 2, 3500.0, 0.3, 6.2, 0.0)
    with pytest.raises(ValueError, match="B_field magnitude"):
        orbit._eom(
            np.array([6.2, 0.0, 0.0, orbit.v_par]),
            lambda R, Z: (0.0, 0.0, 0.0),
        )


def test_guiding_center_rejects_non_finite_z():
    with pytest.raises(ValueError, match="Z0_init"):
        GuidingCenterOrbit(4.0, 2, 3500.0, 0.0, 6.2, float("nan"))


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: MonteCarloEnsemble(5, -1.0, 6.2, 2.0, 5.3), "E_birth_keV"),
        (lambda: MonteCarloEnsemble(5, 3500.0, 6.2, 2.0, 0.0), "B0"),
    ],
)
def test_mc_ensemble_more_construction_guards(call, match):
    with pytest.raises(ValueError, match=match):
        call()


def test_mc_ensemble_follow_rejects_bad_steps_and_dt():
    ens = MonteCarloEnsemble(3, 3500.0, 6.2, 2.0, 5.3)
    ens.initialize(seed=1)
    with pytest.raises(ValueError, match="n_steps"):
        ens.follow(mock_b_field, n_steps=0)
    with pytest.raises(ValueError, match="dt"):
        ens.follow(mock_b_field, dt=0.0)


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: first_orbit_loss(R0=0.0, a=2.0, Ip_MA=15.0), "R0"),
        (lambda: first_orbit_loss(R0=6.2, a=0.0, Ip_MA=15.0), "a"),
        (lambda: first_orbit_loss(R0=6.2, a=2.0, Ip_MA=15.0, E_alpha_keV=0.0), "E_alpha_keV"),
        (lambda: first_orbit_loss(R0=6.2, a=2.0, Ip_MA=15.0, fast_ion_amu=0.0), "fast_ion_amu"),
        (lambda: first_orbit_loss(R0=6.2, a=2.0, Ip_MA=15.0, fast_ion_Z=0), "fast_ion_Z"),
    ],
)
def test_first_orbit_loss_more_guards(call, match):
    with pytest.raises(ValueError, match=match):
        call()


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: SlowingDown.critical_velocity(20.0, 0.0), "background_ion_amu"),
        (lambda: SlowingDown.electron_slowing_down_time(20.0, 1.0, 0.0, 2), "fast_ion_amu"),
        (lambda: SlowingDown.electron_slowing_down_time(20.0, 1.0, 4.0, 0), "fast_ion_Z"),
        (
            lambda: SlowingDown.electron_slowing_down_time(20.0, 1.0, 4.0, 2, coulomb_log=0.0),
            "coulomb_log",
        ),
        (lambda: SlowingDown.electron_slowing_down_time(0.0, 1.0, 4.0, 2), "Te_keV"),
        (lambda: SlowingDown.slowing_down_time(0.0, 1.0e6, 5.0e6, 1.0), "v1"),
        (lambda: SlowingDown.slowing_down_time(1.0e7, 1.0e6, 0.0, 1.0), "v_c"),
        (lambda: SlowingDown.heating_partition(-1.0, 5.0e6), "v must"),
    ],
)
def test_slowing_down_per_parameter_guards(call, match):
    with pytest.raises(ValueError, match=match):
        call()
