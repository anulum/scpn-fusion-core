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


def test_zero_epar() -> None:
    params = RunawayParams(ne_20=1.0, Te_keV=1.0, E_par=0.0, Z_eff=1.5, B0=5.3, R0=6.2)
    assert dreicer_generation_rate(params) == 0.0
    assert avalanche_growth_rate(params, n_RE=1e10) == 0.0


def test_subcritical_epar() -> None:
    E_c = critical_field(ne_20=1.0)
    # Set E_par below E_c
    params = RunawayParams(ne_20=1.0, Te_keV=1.0, E_par=E_c * 0.5, Z_eff=1.5, B0=5.3, R0=6.2)

    # No avalanche
    assert avalanche_growth_rate(params, n_RE=1e10) == 0.0

    # Dreicer should be negligible
    assert dreicer_generation_rate(params) < 1e-10


def test_dreicer_generation() -> None:
    params = RunawayParams(ne_20=1.0, Te_keV=1.0, E_par=0.0, Z_eff=1.5, B0=5.3, R0=6.2)
    # E_D is large for Te=1keV
    # Let's set E_par artificially high to get Dreicer
    E_D = params.ne_20 * 1e20 * 1.6e-19**3 * 15.0 / (4 * np.pi * 8.85e-12**2 * 1.6e-16)  # rough
    params.E_par = 100.0  # Massive field, e.g. TQ

    rate = dreicer_generation_rate(params)
    assert rate > 0.0


def test_avalanche_multiplication() -> None:
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


def test_hot_tail() -> None:
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


def test_runaway_evolution() -> None:
    params = RunawayParams(ne_20=1.0, Te_keV=0.01, E_par=0.0, Z_eff=1.5, B0=5.3, R0=6.2)
    ev = RunawayEvolution(params)

    # Constant E_par
    t, n = ev.evolve(n_RE_0=1e10, E_par_profile=lambda t: 5.0, t_span=(0, 0.1), dt=0.01)

    assert len(t) == 11
    assert n[-1] > n[0]  # Should grow exponentially


def test_dream_fluid_balance_matches_density_equation_contract() -> None:
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


def test_dream_fluid_balance_keeps_avalanche_threshold_contract() -> None:
    params = RunawayParams(
        ne_20=1.0, Te_keV=0.04, E_par=0.5 * critical_field(1.0), Z_eff=2.0, B0=5.0, R0=6.0
    )

    balance = dream_fluid_density_balance(params, 1.0e12)

    assert balance.avalanche_source == 0.0
    assert balance.total_source == pytest.approx(balance.dreicer_source)


def test_runaway_evolution_step_applies_density_cap() -> None:
    params = RunawayParams(ne_20=1.0, Te_keV=0.02, E_par=50.0, Z_eff=2.0, B0=5.0, R0=6.0)
    ev = RunawayEvolution(params)

    capped = ev.step(1.0, 9.0e13, 50.0, max_runaway_fraction=1.0e-6)

    assert capped == pytest.approx(1.0e14)


def test_dream_fluid_balance_rejects_invalid_contract_domain() -> None:
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


def test_mitigation_assessment() -> None:
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


def test_dreicer_generation_rate_zero_for_nonpositive_drive() -> None:
    params = RunawayParams(ne_20=1.0, Te_keV=1.0, E_par=0.0, Z_eff=1.5, B0=5.3, R0=6.2)
    assert dreicer_generation_rate(params) == 0.0


def test_hot_tail_seed_rejects_nonpositive_inputs() -> None:
    base = dict(Te_pre_keV=5.0, Te_post_keV=0.1, ne_20=1.0, quench_time_ms=1.0)
    for bad, match in [
        ({"Te_pre_keV": 0.0}, "Te_pre_keV"),
        ({"Te_post_keV": 0.0}, "Te_post_keV"),
        ({"ne_20": 0.0}, "ne_20"),
        ({"quench_time_ms": 0.0}, "quench_time_ms"),
        ({"vc_vte_ref": 0.0}, "vc_vte_ref"),
        ({"quench_exponent": 0.0}, "quench_exponent"),
    ]:
        with pytest.raises(ValueError, match=match):
            hot_tail_seed(**{**base, **bad})


def test_hot_tail_seed_zero_when_no_cooling() -> None:
    # Te_post >= Te_pre -> no hot-tail seed.
    assert hot_tail_seed(Te_pre_keV=5.0, Te_post_keV=5.0, ne_20=1.0, quench_time_ms=1.0) == 0.0


def test_runaway_evolution_step_validates_timestep() -> None:
    params = RunawayParams(ne_20=1.0, Te_keV=1.0, E_par=10.0, Z_eff=1.5, B0=5.3, R0=6.2)
    ev = RunawayEvolution(params)
    with pytest.raises(ValueError, match="dt must be finite"):
        ev.step(0.0, 1.0e15, 10.0)


def test_runaway_current_fraction_is_bounded() -> None:
    params = RunawayParams(ne_20=1.0, Te_keV=1.0, E_par=10.0, Z_eff=1.5, B0=5.3, R0=6.2)
    ev = RunawayEvolution(params)
    frac = ev.current_fraction(1.0e15, 15.0)
    assert frac >= 0.0


def test_runaway_mitigation_required_density_validation() -> None:
    mit = RunawayMitigationAssessment()
    # E_par <= 0 yields zero required density (no drive, no runaways).
    assert mit.required_density_for_suppression(0.0, 1.5) == 0.0
    with pytest.raises(ValueError, match="Z_eff"):
        mit.required_density_for_suppression(10.0, 0.0)
    with pytest.raises(ValueError, match="coulomb_log"):
        mit.required_density_for_suppression(10.0, 1.5, coulomb_log=0.0)


def test_dream_fluid_density_balance_validation() -> None:
    params = RunawayParams(ne_20=1.0, Te_keV=1.0, E_par=10.0, Z_eff=1.5, B0=5.3, R0=6.2)
    for kw, match in [
        ({"n_RE": -1.0}, "n_RE must be finite"),
        ({"n_RE": 1.0, "max_runaway_fraction": 0.0}, "max_runaway_fraction"),
        ({"n_RE": 1.0, "loss_time_s": 0.0}, "loss_time_s"),
        ({"n_RE": 1.0e25, "max_runaway_fraction": 1.0}, "exceeds the configured"),
    ]:
        with pytest.raises(ValueError, match=match):
            dream_fluid_density_balance(params, **kw)


def test_runaway_evolution_evolve_validates_span() -> None:
    params = RunawayParams(ne_20=1.0, Te_keV=1.0, E_par=10.0, Z_eff=1.5, B0=5.3, R0=6.2)
    ev = RunawayEvolution(params)
    efield = lambda t: 10.0  # noqa: E731
    with pytest.raises(ValueError, match="t_span must be a"):
        # deliberately wrong container type to exercise the runtime guard
        ev.evolve(0.0, efield, [0.0, 1.0], 0.1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dt must be finite"):
        ev.evolve(0.0, efield, (0.0, 1.0), 0.0)
    with pytest.raises(ValueError, match="t_end > t_start"):
        ev.evolve(0.0, efield, (1.0, 0.0), 0.1)
    with pytest.raises(ValueError, match="n_RE_0"):
        ev.evolve(-1.0, efield, (0.0, 1.0), 0.1)


def test_runaway_maximum_re_energy_validation() -> None:
    mit = RunawayMitigationAssessment()
    base = dict(B0=5.3, R0=6.2, ne_20=1.0, Z_eff=1.5, pitch_angle_rad=0.1)
    for bad, match in [
        ({"B0": 0.0}, "B0"),
        ({"R0": 0.0}, "R0"),
        ({"ne_20": 0.0}, "ne_20"),
        ({"Z_eff": 0.0}, "Z_eff"),
        ({"pitch_angle_rad": 0.0}, "pitch_angle"),
    ]:
        with pytest.raises(ValueError, match=match):
            mit.maximum_re_energy(**{**base, **bad})


def test_dream_balance_rejects_zero_electron_density() -> None:
    params = RunawayParams(ne_20=0.0, Te_keV=1.0, E_par=10.0, Z_eff=1.5, B0=5.3, R0=6.2)
    with pytest.raises(ValueError, match="electron density"):
        dream_fluid_density_balance(params, n_RE=1.0)


def test_wall_heat_load_validation() -> None:
    mit = RunawayMitigationAssessment()
    base = dict(n_RE=1.0e15, E_max_MeV=10.0, A_wet=1.0)
    for bad, match in [
        ({"n_RE": -1.0}, "n_RE"),
        ({"E_max_MeV": 0.0}, "E_max_MeV"),
        ({"volume": 0.0}, "volume"),
    ]:
        with pytest.raises(ValueError, match=match):
            mit.wall_heat_load(**{**base, **bad})


def test_wall_heat_load_infinite_for_zero_wetted_area() -> None:
    mit = RunawayMitigationAssessment()
    assert mit.wall_heat_load(n_RE=1.0e15, E_max_MeV=10.0, A_wet=0.0) == float("inf")


def test_coulomb_logarithm_thermal_matches_reference_form() -> None:
    """lnLambda_T = 14.9 - 0.5 ln(n_20) + ln(T_keV) (DREAM CoulombLogarithm.cpp)."""
    from scpn_fusion.core.runaway_electrons import coulomb_logarithm_thermal

    assert coulomb_logarithm_thermal(1.0, 1.0) == pytest.approx(14.9, rel=1e-12)
    assert coulomb_logarithm_thermal(0.5, 0.1) == pytest.approx(
        14.9 - 0.5 * np.log(0.5) + np.log(0.1), rel=1e-12
    )
    with pytest.raises(ValueError, match="ne_20"):
        coulomb_logarithm_thermal(0.0, 1.0)
    with pytest.raises(ValueError, match="Te_keV"):
        coulomb_logarithm_thermal(1.0, -1.0)


def test_coulomb_logarithm_relativistic_matches_reference_form() -> None:
    """lnLambda_c = 14.6 + 0.5 ln(T_eV/n_20) (Hesslow NF 59 084004, arXiv:1904.00602)."""
    from scpn_fusion.core.runaway_electrons import coulomb_logarithm_relativistic

    assert coulomb_logarithm_relativistic(1.0, 1.0) == pytest.approx(
        14.6 + 0.5 * np.log(1e3), rel=1e-12
    )
    assert coulomb_logarithm_relativistic(0.5, 0.1) == pytest.approx(
        14.6 + 0.5 * np.log(100.0 / 0.5), rel=1e-12
    )
    with pytest.raises(ValueError, match="ne_20"):
        coulomb_logarithm_relativistic(-1.0, 1.0)
    with pytest.raises(ValueError, match="Te_keV"):
        coulomb_logarithm_relativistic(1.0, 0.0)


def test_dreicer_rate_matches_really_executed_dream_reference() -> None:
    """The corrected Connor-Hastie rate reproduces the acquired DREAM value.

    Reference: validation/reference_data/dream/ (DREAM a08edc0d fluid
    same-case; gammaDreicer median 2.104e20 m^-3 s^-1). Measured post-fix
    ratio 0.969; asserted here as a regression band, not exact equality.
    """
    params = RunawayParams(ne_20=0.5, Te_keV=0.1, E_par=6.0, Z_eff=1.0, B0=5.0, R0=1.65, a=0.22)
    rate = dreicer_generation_rate(params)
    assert 0.85 * 2.104e20 <= rate <= 1.15 * 2.104e20


def test_avalanche_rate_matches_compact_rp_form_against_dream() -> None:
    """The RP compact avalanche rate sits in the measured band vs DREAM.

    Gamma_ava = e (E - E_c) / (m_e c lnLambda_c sqrt(5 + Z_eff)); DREAM's
    matched-formula GammaAva is 109.6 1/s at the committed same-case state,
    and the compact form is systematically lower (measured ratio 0.755).
    """
    params = RunawayParams(ne_20=0.5, Te_keV=0.1, E_par=6.0, Z_eff=1.0, B0=5.0, R0=1.65, a=0.22)
    exp_rate = avalanche_growth_rate(params, 1.0)
    assert 0.60 * 109.6 <= exp_rate <= 1.00 * 109.6


def test_legacy_fixed_coulomb_log_mode_is_still_available() -> None:
    """Passing coulomb_log forces the fixed-lnLambda legacy mode."""
    params = RunawayParams(ne_20=0.5, Te_keV=0.1, E_par=6.0, Z_eff=1.0, B0=5.0, R0=1.65, a=0.22)
    fixed = avalanche_growth_rate(params, 1.0, coulomb_log=15.0)
    computed = avalanche_growth_rate(params, 1.0)
    assert fixed != computed
    assert fixed > 0.0 and computed > 0.0
