# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Faraday Recovery Tests
from __future__ import annotations

import math

import numpy as np
import pytest

from scpn_fusion.core import (
    FaradayRecoveryTrajectoryPoint,
    faraday_back_emf,
    faraday_back_emf_from_values,
    integrated_recovery_energy,
    magnetic_flux_wb,
)


def test_constant_radius_and_field_have_zero_back_emf() -> None:
    emf = faraday_back_emf(
        lambda _t: 0.20,
        lambda _t: 20.0,
        12,
        1.0e-6,
        dR_s_dt=lambda _t: 0.0,
        dB_ext_dt=lambda _t: 0.0,
    )

    assert emf == pytest.approx(0.0, abs=0.0)


def test_closed_form_constant_field_radial_expansion() -> None:
    emf = faraday_back_emf_from_values(
        separatrix_radius_m=0.20,
        b_ext_t=20.0,
        d_radius_dt_m_s=1.5e4,
        d_b_ext_dt_t_s=0.0,
        N_turns=8,
    )

    expected = -8.0 * math.pi * (2.0 * 20.0 * 0.20 * 1.5e4)
    assert emf == pytest.approx(expected, rel=1.0e-15)


def test_callable_finite_difference_matches_linear_history() -> None:
    radius_0 = 0.18
    speed = -2.0e4
    field_0 = 19.0
    field_rate = 3.0e6
    time_s = 2.5e-7

    emf = faraday_back_emf(
        lambda t: radius_0 + speed * t,
        lambda t: field_0 + field_rate * t,
        5,
        time_s,
        finite_difference_dt_s=1.0e-10,
    )
    expected = faraday_back_emf_from_values(
        radius_0 + speed * time_s,
        field_0 + field_rate * time_s,
        speed,
        field_rate,
        5,
    )

    assert emf == pytest.approx(expected, rel=1.0e-9)


def test_integrated_recovery_energy_matches_analytical_linear_radius_case() -> None:
    turns = 6
    resistance = 0.08
    b_ext = 20.0
    radius_0 = 0.15
    speed = 4.0e3
    duration = 1.0e-6
    times = np.linspace(0.0, duration, 257)
    trajectory = [
        FaradayRecoveryTrajectoryPoint(
            t_s=float(t), separatrix_radius_m=radius_0 + speed * t, b_ext_t=b_ext
        )
        for t in times
    ]

    report = integrated_recovery_energy(trajectory, turns, resistance)

    coefficient = turns * math.pi * 2.0 * b_ext * speed
    expected = coefficient * coefficient / resistance
    expected *= ((radius_0 + speed * duration) ** 3 - radius_0**3) / (3.0 * speed)
    assert report.recovered_energy_j == pytest.approx(expected, rel=2.0e-5)
    assert report.budget_claim_status == "blocked_missing_compression_work"
    assert report.energy_budget_passed is None


def test_integrated_recovery_energy_reports_budget_when_work_is_supplied() -> None:
    trajectory = [
        FaradayRecoveryTrajectoryPoint(
            t_s=0.0, separatrix_radius_m=0.20, b_ext_t=20.0, d_radius_dt_m_s=0.0, d_b_ext_dt_t_s=0.0
        ),
        FaradayRecoveryTrajectoryPoint(
            t_s=1.0e-6,
            separatrix_radius_m=0.20,
            b_ext_t=20.0,
            d_radius_dt_m_s=0.0,
            d_b_ext_dt_t_s=0.0,
        ),
    ]

    report = integrated_recovery_energy(trajectory, 4, 0.1, compression_work_j=1.0e-12)

    assert report.recovered_energy_j == pytest.approx(0.0, abs=0.0)
    assert report.energy_budget_passed is False
    assert report.budget_claim_status == "failed"


def test_faraday_recovery_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        faraday_back_emf_from_values(0.2, 20.0, 1.0, 0.0, 0)
    with pytest.raises(ValueError, match="positive"):
        magnetic_flux_wb(0.0, 20.0)
    with pytest.raises(ValueError, match="at least two"):
        integrated_recovery_energy([FaradayRecoveryTrajectoryPoint(0.0, 0.2, 20.0)], 2, 0.1)
    with pytest.raises(ValueError, match="strictly increasing"):
        integrated_recovery_energy(
            [
                FaradayRecoveryTrajectoryPoint(0.0, 0.2, 20.0),
                FaradayRecoveryTrajectoryPoint(0.0, 0.21, 20.0),
            ],
            2,
            0.1,
        )
    with pytest.raises(ValueError, match="all supplied or all omitted"):
        integrated_recovery_energy(
            [
                FaradayRecoveryTrajectoryPoint(0.0, 0.2, 20.0, d_radius_dt_m_s=1.0),
                FaradayRecoveryTrajectoryPoint(1.0e-6, 0.21, 20.0),
            ],
            2,
            0.1,
        )
