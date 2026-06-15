# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Runaway Electrons Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.runaway_electrons import (
    C_LIGHT,
    E_CHARGE,
    EPS_0,
    M_E,
    RunawayEvolution,
    RunawayMitigationAssessment,
    RunawayParams,
    avalanche_growth_rate,
    critical_field,
    dream_fluid_density_balance,
    dreicer_field,
    dreicer_generation_rate,
    hot_tail_seed,
)


def test_critical_and_dreicer_field_absolute_values() -> None:
    """Pin the Connor-Hastie critical field and the Dreicer/critical relation.

    The other tests use ``critical_field`` only relatively (E_par vs E_c), so a
    wrong prefactor would pass. ``E_c`` is ~0.08 V/m per 1e20 m^-3 and the
    Dreicer/critical ratio is exactly ``m_e c^2 / T_e``.
    """
    ne_20, ln_lambda, te_keV = 1.0, 15.0, 10.0
    e_c = critical_field(ne_20, ln_lambda)
    expected = (
        (ne_20 * 1e20) * E_CHARGE**3 * ln_lambda / (4.0 * np.pi * EPS_0**2 * M_E * C_LIGHT**2)
    )
    assert e_c == pytest.approx(expected, rel=1e-12)
    assert 0.05 < e_c < 0.15  # ITER-relevant magnitude
    assert critical_field(5.0, ln_lambda) == pytest.approx(5.0 * e_c, rel=1e-12)

    e_d = dreicer_field(ne_20, te_keV, ln_lambda)
    assert e_d / e_c == pytest.approx(M_E * C_LIGHT**2 / (te_keV * 1e3 * E_CHARGE), rel=1e-12)


def test_zero_epar():
    params = RunawayParams(ne_20=1.0, Te_keV=1.0, E_par=0.0, Z_eff=1.5, B0=5.3, R0=6.2)
    assert dreicer_generation_rate(params) == 0.0
    assert avalanche_growth_rate(params, n_RE=1e10) == 0.0


def test_subcritical_epar():
    E_c = critical_field(ne_20=1.0)
    # Set E_par below E_c
    params = RunawayParams(ne_20=1.0, Te_keV=1.0, E_par=E_c * 0.5, Z_eff=1.5, B0=5.3, R0=6.2)

    # No avalanche
    assert avalanche_growth_rate(params, n_RE=1e10) == 0.0

    # Dreicer should be negligible
    assert dreicer_generation_rate(params) < 1e-10


def test_dreicer_generation():
    params = RunawayParams(ne_20=1.0, Te_keV=1.0, E_par=0.0, Z_eff=1.5, B0=5.3, R0=6.2)
    # E_D is large for Te=1keV
    # Let's set E_par artificially high to get Dreicer
    E_D = params.ne_20 * 1e20 * 1.6e-19**3 * 15.0 / (4 * np.pi * 8.85e-12**2 * 1.6e-16)  # rough
    params.E_par = 100.0  # Massive field, e.g. TQ

    rate = dreicer_generation_rate(params)
    assert rate > 0.0


def test_avalanche_multiplication():
    params = RunawayParams(ne_20=1.0, Te_keV=0.01, E_par=10.0, Z_eff=1.5, B0=5.3, R0=6.2)
    # E_c for ne_20=1 is ~0.1 V/m. So E_par = 10 is 100x E_c.
    E_c = critical_field(1.0)
    assert params.E_par > E_c

    n_RE = 1e10
    rate = avalanche_growth_rate(params, n_RE)
    assert rate > 0.0

    # Should scale linearly with n_RE
    rate2 = avalanche_growth_rate(params, 2e10)
    assert np.isclose(rate2, 2.0 * rate)


def test_hot_tail():
    # Drop from 10 keV to 10 eV in 1 ms
    n_seed = hot_tail_seed(Te_pre_keV=10.0, Te_post_keV=0.01, ne_20=1.0, quench_time_ms=1.0)
    assert n_seed > 1e10  # Substantial seed


def test_hot_tail_seed_is_higher_for_faster_quench() -> None:
    fast = hot_tail_seed(Te_pre_keV=10.0, Te_post_keV=0.01, ne_20=1.0, quench_time_ms=0.2)
    slow = hot_tail_seed(Te_pre_keV=10.0, Te_post_keV=0.01, ne_20=1.0, quench_time_ms=5.0)
    assert fast > slow


def test_hot_tail_seed_rejects_invalid_domain() -> None:
    with np.testing.assert_raises(ValueError):
        hot_tail_seed(Te_pre_keV=0.0, Te_post_keV=0.01, ne_20=1.0, quench_time_ms=1.0)
    with np.testing.assert_raises(ValueError):
        hot_tail_seed(Te_pre_keV=10.0, Te_post_keV=0.01, ne_20=1.0, quench_time_ms=0.0)
    with np.testing.assert_raises(ValueError):
        hot_tail_seed(
            Te_pre_keV=10.0,
            Te_post_keV=0.01,
            ne_20=1.0,
            quench_time_ms=1.0,
            vc_vte_ref=0.0,
        )


def test_runaway_evolution():
    params = RunawayParams(ne_20=1.0, Te_keV=0.01, E_par=0.0, Z_eff=1.5, B0=5.3, R0=6.2)
    ev = RunawayEvolution(params)

    # Constant E_par
    t, n = ev.evolve(n_RE_0=1e10, E_par_profile=lambda t: 5.0, t_span=(0, 0.1), dt=0.01)

    assert len(t) == 11
    assert n[-1] > n[0]  # Should grow exponentially


def test_dream_fluid_balance_matches_density_equation_contract():
    params = RunawayParams(ne_20=1.0, Te_keV=0.04, E_par=8.0, Z_eff=2.0, B0=5.0, R0=6.0)
    n_re = 2.0e12

    balance = dream_fluid_density_balance(params, n_re, loss_time_s=0.2)

    assert balance.dreicer_source > 0.0
    assert balance.avalanche_source > 0.0
    assert balance.loss_source == pytest.approx(n_re / 0.2)
    assert balance.total_source == pytest.approx(
        balance.dreicer_source + balance.avalanche_source - balance.loss_source
    )
    assert balance.runaway_fraction == pytest.approx(n_re / 1.0e20)
    assert balance.growth_time_s == pytest.approx(n_re / balance.total_source)


def test_dream_fluid_balance_keeps_avalanche_threshold_contract():
    params = RunawayParams(
        ne_20=1.0, Te_keV=0.04, E_par=0.5 * critical_field(1.0), Z_eff=2.0, B0=5.0, R0=6.0
    )

    balance = dream_fluid_density_balance(params, 1.0e12)

    assert balance.avalanche_source == 0.0
    assert balance.total_source == pytest.approx(balance.dreicer_source)


def test_runaway_evolution_step_applies_density_cap():
    params = RunawayParams(ne_20=1.0, Te_keV=0.02, E_par=50.0, Z_eff=2.0, B0=5.0, R0=6.0)
    ev = RunawayEvolution(params)

    capped = ev.step(1.0, 9.0e13, 50.0, max_runaway_fraction=1.0e-6)

    assert capped == pytest.approx(1.0e14)


def test_dream_fluid_balance_rejects_invalid_contract_domain():
    params = RunawayParams(ne_20=1.0, Te_keV=0.04, E_par=8.0, Z_eff=2.0, B0=5.0, R0=6.0)

    with pytest.raises(ValueError, match="loss_time_s"):
        dream_fluid_density_balance(params, 1.0e12, loss_time_s=0.0)
    with pytest.raises(ValueError, match="max_runaway_fraction"):
        dream_fluid_density_balance(params, 1.0e12, max_runaway_fraction=1.5)
    with pytest.raises(ValueError, match="density cap"):
        dream_fluid_density_balance(params, 2.0e15, max_runaway_fraction=1.0e-6)


def test_runaway_evolution_step_rejects_invalid_domain() -> None:
    params = RunawayParams(ne_20=1.0, Te_keV=0.01, E_par=0.0, Z_eff=1.5, B0=5.3, R0=6.2)
    ev = RunawayEvolution(params)
    with np.testing.assert_raises(ValueError):
        ev.step(dt=0.0, n_RE=1e10, E_par=5.0)
    with np.testing.assert_raises(ValueError):
        ev.step(dt=1e-3, n_RE=-1.0, E_par=5.0)
    with np.testing.assert_raises(ValueError):
        ev.step(dt=1e-3, n_RE=1e10, E_par=float("nan"))


def test_runaway_evolution_evolve_rejects_invalid_domain() -> None:
    params = RunawayParams(ne_20=1.0, Te_keV=0.01, E_par=0.0, Z_eff=1.5, B0=5.3, R0=6.2)
    ev = RunawayEvolution(params)
    with np.testing.assert_raises(ValueError):
        ev.evolve(n_RE_0=1e10, E_par_profile=lambda _: 5.0, t_span=(0.1, 0.0), dt=0.01)
    with np.testing.assert_raises(ValueError):
        ev.evolve(n_RE_0=1e10, E_par_profile=lambda _: 5.0, t_span=(0.0, 0.1), dt=0.0)
    with np.testing.assert_raises(ValueError):
        ev.evolve(n_RE_0=-1.0, E_par_profile=lambda _: 5.0, t_span=(0.0, 0.1), dt=0.01)
    with np.testing.assert_raises(ValueError):
        ev.evolve(
            n_RE_0=1e10,
            E_par_profile=lambda _: float("inf"),
            t_span=(0.0, 0.1),
            dt=0.01,
        )


def test_mitigation_assessment():
    mit = RunawayMitigationAssessment()

    # Typical ITER E_par during disruption is ~10-50 V/m
    E_par = 20.0
    ne_req = mit.required_density_for_suppression(E_par, Z_eff=1.0)
    # Should be massive (> 100 * 10^20 m^-3)
    assert ne_req > 100.0

    E_max = mit.maximum_re_energy(B0=5.3, R0=6.2)
    assert 10.0 < E_max < 100.0

    load = mit.wall_heat_load(n_RE=1e16, E_max_MeV=E_max, A_wet=10.0)
    assert load > 0.0


def test_wall_heat_load_accepts_explicit_mean_energy() -> None:
    mit = RunawayMitigationAssessment()
    load = mit.wall_heat_load(
        n_RE=1.0e16,
        E_max_MeV=30.0,
        A_wet=10.0,
        volume=100.0,
        mean_energy_MeV=20.0,
    )
    expected = 1.0e16 * 100.0 * 20.0e6 * 1.602176634e-19 / 1.0e6 / 10.0
    assert load == pytest.approx(expected)


def test_wall_heat_load_rejects_invalid_energy_domain() -> None:
    mit = RunawayMitigationAssessment()
    with pytest.raises(ValueError, match="mean_energy_MeV"):
        mit.wall_heat_load(n_RE=1.0e16, E_max_MeV=30.0, A_wet=10.0, mean_energy_MeV=40.0)


def test_current_fraction_is_bounded_and_rejects_invalid_domain() -> None:
    params = RunawayParams(ne_20=1.0, Te_keV=0.01, E_par=0.0, Z_eff=1.5, B0=5.3, R0=6.2)
    ev = RunawayEvolution(params)
    assert 0.0 <= ev.current_fraction(n_RE=1e20, I_p_MA=15.0) <= 1.0
    assert ev.current_fraction(n_RE=1e30, I_p_MA=0.0) == 0.0
    assert ev.current_fraction(n_RE=1e30, I_p_MA=15.0) == 1.0
    with np.testing.assert_raises(ValueError):
        ev.current_fraction(n_RE=-1.0, I_p_MA=15.0)
    with np.testing.assert_raises(ValueError):
        ev.current_fraction(n_RE=1e20, I_p_MA=float("nan"))


def test_maximum_re_energy_scales_with_density_and_pitch() -> None:
    mit = RunawayMitigationAssessment()
    base = mit.maximum_re_energy(B0=5.3, R0=6.2, ne_20=1.0, Z_eff=1.5, pitch_angle_rad=0.35)
    high_density = mit.maximum_re_energy(B0=5.3, R0=6.2, ne_20=4.0, Z_eff=1.5, pitch_angle_rad=0.35)
    high_pitch = mit.maximum_re_energy(B0=5.3, R0=6.2, ne_20=1.0, Z_eff=1.5, pitch_angle_rad=0.7)
    assert high_density < base
    assert high_pitch < base


def test_maximum_re_energy_rejects_invalid_inputs() -> None:
    mit = RunawayMitigationAssessment()
    with np.testing.assert_raises(ValueError):
        mit.maximum_re_energy(B0=0.0, R0=6.2)
    with np.testing.assert_raises(ValueError):
        mit.maximum_re_energy(B0=5.3, R0=6.2, ne_20=0.0)
    with np.testing.assert_raises(ValueError):
        mit.maximum_re_energy(B0=5.3, R0=6.2, pitch_angle_rad=0.0)


def test_required_density_for_suppression_uses_zeff_scaling() -> None:
    mit = RunawayMitigationAssessment()
    n_low_zeff = mit.required_density_for_suppression(E_par=20.0, Z_eff=1.0)
    n_high_zeff = mit.required_density_for_suppression(E_par=20.0, Z_eff=4.0)
    assert n_high_zeff < n_low_zeff


def test_required_density_for_suppression_rejects_invalid_domain() -> None:
    mit = RunawayMitigationAssessment()
    with np.testing.assert_raises(ValueError):
        mit.required_density_for_suppression(E_par=20.0, Z_eff=0.0)
    with np.testing.assert_raises(ValueError):
        mit.required_density_for_suppression(E_par=20.0, Z_eff=2.0, coulomb_log=0.0)


def test_dreicer_field_exceeds_critical_field_by_thermal_energy_ratio() -> None:
    """At 10 keV the Dreicer-to-critical field ratio remains in the expected range."""
    from scpn_fusion.core.runaway_electrons import dreicer_field

    e_d = dreicer_field(ne_20=1.0, Te_keV=10.0)
    e_c = critical_field(ne_20=1.0)

    assert e_d > e_c > 0.0
    assert 40.0 < e_d / e_c < 60.0
