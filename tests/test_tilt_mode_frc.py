# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FRC Tilt-Mode Tests
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from numpy.typing import NDArray
import pytest

from scpn_fusion.core import RigidRotorFRCInputs, solve_frc_equilibrium
from scpn_fusion.core.frc_rigid_rotor import ELEMENTARY_CHARGE_C, MU_0, FRCEquilibriumState
from scpn_fusion.core.pulsed_compression import (
    CoilGeometry,
    PulsedCompressionConfig,
    initial_pulsed_compression_state,
    run_pulsed_compression,
)
from scpn_fusion.core.tilt_mode_frc import (
    BELOVA_MHD_GROWTH_COEFFICIENT,
    FRCTiltModeThresholds,
    alfven_speed_m_s,
    belova_table1_acceptance_status,
    claim_boundary,
    frc_tilt_growth_rate,
    rigid_body_flr_regime,
    s_over_elongation,
    tilt_mode_report,
    tilt_mode_stable,
    tilt_mode_trajectory_from_pulsed_compression,
)


def _equilibrium() -> FRCEquilibriumState:
    t_i = 10_000.0
    t_e = 5_000.0
    b_ext = 5.0
    n0 = b_ext**2 / (2.0 * MU_0) / ((t_i + t_e) * ELEMENTARY_CHARGE_C)
    rho: NDArray[np.float64] = np.linspace(0.0, 0.4, 129, dtype=np.float64)
    return solve_frc_equilibrium(
        RigidRotorFRCInputs(
            n0=n0,
            T_i_eV=t_i,
            T_e_eV=t_e,
            theta_dot=0.0,
            R_s=0.2,
            B_ext=b_ext,
            delta=0.02,
        ),
        rho,
    )


def _compression_config(eq: FRCEquilibriumState) -> PulsedCompressionConfig:
    return PulsedCompressionConfig(
        equilibrium=eq,
        coil=CoilGeometry(
            N_turns=12,
            L_coil_m=0.5,
            R_coil_m=0.25,
            L_inductance_H=8.0e-6,
            R_resistance_ohm=0.02,
            bank_voltage_max_V=25_000.0,
        ),
        coil_current_t=lambda _t: 2.0e5,
        plasma_mass_kg=2.0e-6,
        ion_temperature_eV=10_000.0,
        electron_temperature_eV=5_000.0,
        plasma_length_m=1.0,
    )


def test_tilt_growth_matches_belova_alfven_time_scaling() -> None:
    eq = _equilibrium()
    elongation = 4.0

    growth = frc_tilt_growth_rate(eq, elongation)

    expected = (
        BELOVA_MHD_GROWTH_COEFFICIENT
        * alfven_speed_m_s(eq)
        / (elongation * eq.target_separatrix_radius_m)
    )
    assert growth == pytest.approx(expected, rel=1.0e-14)
    assert growth > 0.0


def test_tilt_growth_decreases_with_elongation() -> None:
    eq = _equilibrium()

    short = frc_tilt_growth_rate(eq, 2.0)
    long = frc_tilt_growth_rate(eq, 6.0)

    assert long == pytest.approx(short / 3.0, rel=1.0e-14)


def test_rigid_body_regime_uses_s_over_elongation_thresholds() -> None:
    eq = _equilibrium()
    thresholds = FRCTiltModeThresholds()
    high_elongation = eq.s_parameter / (0.5 * thresholds.diamagnetic_s_over_e)
    low_elongation = eq.s_parameter / (1.1 * thresholds.combined_flr_s_over_e)

    stable_regime, stable_threshold = rigid_body_flr_regime(eq, high_elongation)
    unstable_regime, unstable_threshold = rigid_body_flr_regime(eq, low_elongation)

    assert stable_regime == "diamagnetic_flr_threshold_passed"
    assert stable_threshold
    assert unstable_regime == "mhd_tilt_susceptible"
    assert not unstable_threshold
    assert s_over_elongation(eq, high_elongation) < s_over_elongation(eq, low_elongation)


def test_tilt_report_is_fail_closed_until_external_parity_exists() -> None:
    eq = _equilibrium()

    report = tilt_mode_report(eq, 4.0)

    assert report.growth_rate_s_inv > 0.0
    assert report.conservative_stable is False
    assert report.claim_status == "diagnostic_only_not_hybrid_eigenvalue_accepted"
    assert report.external_parity_status == "blocked_missing_public_digitised_reference"
    assert report.alfven_transit_time_s == pytest.approx(1.0 / (report.growth_rate_s_inv / 1.2))


def test_tilt_mode_stable_tuple_is_conservative() -> None:
    eq = _equilibrium()

    stable, growth = tilt_mode_stable(eq, 5.0)

    assert stable is False
    assert growth == pytest.approx(frc_tilt_growth_rate(eq, 5.0), rel=1.0e-14)


def test_tilt_trajectory_tracks_pulsed_compression_projection() -> None:
    eq = _equilibrium()
    config = _compression_config(eq)
    initial = initial_pulsed_compression_state(config)
    states = run_pulsed_compression(initial, config, dt_s=1.0e-8, n_steps=6)

    trajectory = tilt_mode_trajectory_from_pulsed_compression(states, eq, elongation=4.0)

    assert len(trajectory) == len(states)
    assert trajectory[0].report.s_parameter == pytest.approx(eq.s_parameter, rel=1.0e-14)
    expected_s = (
        eq.s_parameter
        * (states[1].R_s_m / states[0].R_s_m)
        * (abs(states[1].B_ext_T) / abs(states[0].B_ext_T))
        * np.sqrt(states[0].T_i_eV / states[1].T_i_eV)
    )
    assert trajectory[1].report.s_parameter == pytest.approx(expected_s, rel=1.0e-14)
    assert trajectory[-1].report.growth_rate_s_inv > 0.0
    assert trajectory[0].cumulative_growth_integral == pytest.approx(0.0)
    assert trajectory[1].cumulative_growth_integral == pytest.approx(
        trajectory[1].report.growth_rate_s_inv * (states[1].t_s - states[0].t_s),
        rel=1.0e-14,
    )
    assert trajectory[-1].cumulative_growth_integral >= trajectory[1].cumulative_growth_integral
    assert np.isfinite(trajectory[-1].perturbation_amplification)
    assert trajectory[-1].perturbation_amplification >= 1.0
    assert trajectory[-1].amplification_overflow_limited is False
    assert trajectory[-1].report.external_parity_status == (
        "blocked_missing_public_digitised_reference"
    )


def test_tilt_trajectory_growth_integral_limits_extreme_amplification() -> None:
    eq = _equilibrium()
    states = (
        SimpleNamespace(t_s=0.0, R_s_m=1.0e-6, T_i_eV=10_000.0, density_m3=1.0e6, B_ext_T=100.0),
        SimpleNamespace(t_s=1.0, R_s_m=1.0e-6, T_i_eV=10_000.0, density_m3=1.0e6, B_ext_T=100.0),
    )

    trajectory = tilt_mode_trajectory_from_pulsed_compression(states, eq, elongation=4.0)

    assert trajectory[-1].cumulative_growth_integral > np.log(np.finfo(np.float64).max)
    assert np.isfinite(trajectory[-1].perturbation_amplification)
    assert trajectory[-1].perturbation_amplification > 1.0e300
    assert trajectory[-1].amplification_overflow_limited is True


def test_external_acceptance_and_claim_boundary_are_explicit() -> None:
    status = belova_table1_acceptance_status()
    boundary = claim_boundary()

    assert status["case"] == "belova_2001_table1_tilt_stability"
    assert status["status"] == "blocked_missing_public_digitised_reference"
    assert "MHD" in boundary["accepted"]
    assert "hybrid eigenvalue" in boundary["not_accepted"]


def test_tilt_inputs_fail_closed() -> None:
    eq = _equilibrium()

    with pytest.raises(ValueError, match="positive"):
        frc_tilt_growth_rate(eq, 0.0)
    with pytest.raises(ValueError, match="positive integer"):
        frc_tilt_growth_rate(eq, 3.0, n_modes=0)
    with pytest.raises(ValueError, match="positive"):
        frc_tilt_growth_rate(eq, 3.0, mhd_coefficient=0.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        rigid_body_flr_regime(eq, 3.0, FRCTiltModeThresholds(2.0, 1.0, 3.0))
    with pytest.raises(ValueError, match="positive"):
        tilt_mode_stable(eq, 3.0, s_threshold=0.0)


def test_tilt_trajectory_inputs_fail_closed() -> None:
    eq = _equilibrium()
    config = _compression_config(eq)
    initial = initial_pulsed_compression_state(config)
    states = run_pulsed_compression(initial, config, dt_s=1.0e-8, n_steps=3)

    with pytest.raises(ValueError, match="at least one"):
        tilt_mode_trajectory_from_pulsed_compression((), eq, elongation=4.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        tilt_mode_trajectory_from_pulsed_compression((states[1], states[0]), eq, elongation=4.0)
