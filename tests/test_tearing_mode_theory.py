# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Analytic validation for FKR matching and Rutherford saturation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from scpn_fusion import core as core_api
from scpn_fusion.core.tearing_mode_theory import (
    HarrisSheetTearingModel,
    fkr_constant_psi_growth_rate_per_s,
)
from scpn_fusion.core.stability_mhd import QProfile, mercier_stability
from scpn_fusion.core.stability_mhd_extended import rwm_stability


def _unstable_model() -> HarrisSheetTearingModel:
    """Return a Harris sheet in the positive-Delta-prime constant-psi regime."""
    return HarrisSheetTearingModel(
        sheet_half_width_m=0.1,
        wave_number_per_m=5.0,
        finite_width_coefficient=2.0,
    )


def test_core_api_exports_tearing_theory_contracts() -> None:
    """The public lazy API resolves both analytic tearing contracts."""
    assert core_api.HarrisSheetTearingModel is HarrisSheetTearingModel
    assert core_api.fkr_constant_psi_growth_rate_per_s is fkr_constant_psi_growth_rate_per_s


def test_harris_outer_matching_changes_sign_at_unit_ka() -> None:
    """The analytic outer solution is unstable only for ka below unity."""
    unstable = _unstable_model()
    marginal = HarrisSheetTearingModel(0.1, 10.0, 2.0)
    stable = HarrisSheetTearingModel(0.1, 15.0, 2.0)

    assert unstable.delta_prime_zero_per_m == pytest.approx(30.0)
    assert unstable.linearly_unstable is True
    assert marginal.delta_prime_zero_per_m == pytest.approx(0.0)
    assert marginal.linearly_unstable is False
    assert stable.delta_prime_zero_per_m < 0.0
    assert stable.fkr_growth_rate_per_s(alfven_time_s=1e-6, lundquist_number=1e6) == 0.0


def test_fkr_growth_obeys_matched_asymptotic_exponents() -> None:
    """Growth follows S^-3/5, k^2/5, and Delta-prime^4/5 exactly."""

    def growth(
        *,
        delta_prime: float = 10.0,
        wave_number: float = 5.0,
        alfven_time: float = 2e-6,
        lundquist: float = 1e6,
    ) -> float:
        return fkr_constant_psi_growth_rate_per_s(
            delta_prime_per_m=delta_prime,
            sheet_half_width_m=0.1,
            wave_number_per_m=wave_number,
            alfven_time_s=alfven_time,
            lundquist_number=lundquist,
        )

    base = growth()
    high_s = growth(lundquist=32e6)
    high_k = growth(wave_number=160.0)
    high_delta = growth(delta_prime=320.0)

    assert base > 0.0
    assert high_s / base == pytest.approx(32.0 ** (-3.0 / 5.0))
    assert high_k / base == pytest.approx(32.0 ** (2.0 / 5.0))
    assert high_delta / base == pytest.approx(32.0 ** (4.0 / 5.0))
    assert growth(alfven_time=4e-6) == pytest.approx(base / 2.0)


def test_rutherford_integrator_converges_to_declared_theory_width() -> None:
    """The numerical Rutherford lane reaches the finite-width Delta-prime root."""
    model = _unstable_model()
    saturation = model.rutherford_saturation_width_m
    diffusivity = 0.02
    dt = 0.002
    steps = 2000
    trace = model.integrate_rutherford_width(
        0.001,
        magnetic_diffusivity_m2_per_s=diffusivity,
        dt_s=dt,
        n_steps=steps,
    )
    analytic = model.analytic_rutherford_width_m(
        0.001,
        dt * steps,
        magnetic_diffusivity_m2_per_s=diffusivity,
    )

    assert saturation == pytest.approx(0.15)
    assert trace[-1] == pytest.approx(analytic, rel=2e-9)
    assert abs(trace[-1] - saturation) / saturation < 4e-4
    assert model.delta_prime_per_m(saturation) == pytest.approx(0.0, abs=1e-12)
    assert model.rutherford_width_rate_m_per_s(
        saturation, magnetic_diffusivity_m2_per_s=diffusivity
    ) == pytest.approx(0.0, abs=1e-12)


def test_stable_rutherford_branch_decays_to_zero() -> None:
    """A negative-Delta-prime sheet has no positive saturation island."""
    model = HarrisSheetTearingModel(0.1, 15.0, 2.0)

    assert model.rutherford_saturation_width_m == 0.0
    assert model.analytic_rutherford_width_m(0.01, 10.0, magnetic_diffusivity_m2_per_s=0.02) == 0.0
    trace = model.integrate_rutherford_width(
        0.001,
        magnetic_diffusivity_m2_per_s=0.02,
        dt_s=0.01,
        n_steps=10,
    )
    assert np.all(trace >= 0.0)
    assert trace[-1] == 0.0


def test_mercier_proxy_matches_declared_suydam_formula() -> None:
    """The cylindrical proxy returns D_M=s^2/4-alpha at every radius."""
    rho = np.linspace(0.0, 1.0, 6)
    q = 1.0 + rho
    shear = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    alpha = np.array([0.0, 0.005, 0.02, 0.04, 0.10, 0.30])
    profile = QProfile(
        rho=rho,
        q=q,
        shear=shear,
        alpha_mhd=alpha,
        q_min=1.0,
        q_min_rho=0.0,
        q_edge=2.0,
    )

    result = mercier_stability(profile)
    expected = shear**2 / 4.0 - alpha

    np.testing.assert_allclose(result.D_M, expected, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(result.stable, expected >= 0.0)
    assert result.first_unstable_rho == pytest.approx(1.0)


def test_rwm_proxy_matches_declared_wall_time_scaling() -> None:
    """The RWM proxy verifies magnitude, not only the growth-rate sign."""
    between_limits = rwm_stability(beta_N=3.0, g_nowall=2.8, g_wall=3.5)
    above_wall = rwm_stability(beta_N=3.6, g_nowall=2.8, g_wall=3.5)

    assert between_limits.mode_growth_rate == pytest.approx((3.0 - 2.8) / (3.5 - 3.0))
    assert above_wall.mode_growth_rate == pytest.approx((3.6 - 2.8) / 0.01)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sheet_half_width_m": 0.0}, "sheet_half_width_m"),
        ({"wave_number_per_m": float("inf")}, "wave_number_per_m"),
        ({"finite_width_coefficient": -1.0}, "finite_width_coefficient"),
    ],
)
def test_model_rejects_invalid_equilibrium_inputs(kwargs: dict[str, float], message: str) -> None:
    """Equilibrium lengths, wave numbers, and nonlinear slopes fail closed."""
    values = {
        "sheet_half_width_m": 0.1,
        "wave_number_per_m": 5.0,
        "finite_width_coefficient": 2.0,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        HarrisSheetTearingModel(**values)


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (lambda model: model.delta_prime_per_m(-1.0), "island_width_m"),
        (
            lambda model: model.fkr_growth_rate_per_s(alfven_time_s=0.0, lundquist_number=1e6),
            "alfven_time_s",
        ),
        (
            lambda model: model.fkr_growth_rate_per_s(alfven_time_s=1e-6, lundquist_number=0.0),
            "lundquist_number",
        ),
        (
            lambda model: fkr_constant_psi_growth_rate_per_s(
                delta_prime_per_m=float("nan"),
                sheet_half_width_m=model.sheet_half_width_m,
                wave_number_per_m=model.wave_number_per_m,
                alfven_time_s=1e-6,
                lundquist_number=1e6,
            ),
            "delta_prime_per_m",
        ),
        (
            lambda model: model.rutherford_width_rate_m_per_s(
                0.1, magnetic_diffusivity_m2_per_s=0.0
            ),
            "magnetic_diffusivity_m2_per_s",
        ),
        (
            lambda model: model.analytic_rutherford_width_m(
                -1.0, 1.0, magnetic_diffusivity_m2_per_s=0.1
            ),
            "initial_width_m",
        ),
        (
            lambda model: model.analytic_rutherford_width_m(
                0.1, -1.0, magnetic_diffusivity_m2_per_s=0.1
            ),
            "time_s",
        ),
        (
            lambda model: model.integrate_rutherford_width(
                -1.0, magnetic_diffusivity_m2_per_s=0.1, dt_s=0.1, n_steps=1
            ),
            "initial_width_m",
        ),
        (
            lambda model: model.integrate_rutherford_width(
                0.1, magnetic_diffusivity_m2_per_s=0.1, dt_s=0.0, n_steps=1
            ),
            "dt_s",
        ),
        (
            lambda model: model.integrate_rutherford_width(
                0.1, magnetic_diffusivity_m2_per_s=0.1, dt_s=0.1, n_steps=True
            ),
            "n_steps",
        ),
        (
            lambda model: model.integrate_rutherford_width(
                0.1, magnetic_diffusivity_m2_per_s=0.1, dt_s=0.1, n_steps=1.5
            ),
            "n_steps",
        ),
    ],
)
def test_runtime_contracts_reject_invalid_inputs(
    operation: Callable[[HarrisSheetTearingModel], object], message: str
) -> None:
    """Runtime inputs reject invalid widths, times, diffusivities, and step counts."""
    with pytest.raises(ValueError, match=message):
        operation(_unstable_model())
