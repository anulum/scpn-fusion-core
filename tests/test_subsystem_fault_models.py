# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Subsystem Fault Model Tests
"""Tests for reduced-order REBCO, DEC, and structural fault screens."""

from __future__ import annotations

import pytest

from scpn_fusion.core.direct_energy_conversion import (
    DirectEnergyConversionChannel,
    DirectEnergyConversionFault,
    evaluate_direct_energy_conversion_fault,
)
from scpn_fusion.core.disruption_structural_response import (
    DisruptionLoad,
    StructuralMember,
    evaluate_disruption_structural_response,
)
from scpn_fusion.core.hts_quench import REBCOConductor, evaluate_rebco_quench


def test_default_rebco_quench_screen_passes_with_bounded_claim() -> None:
    report = evaluate_rebco_quench()

    assert report.passes_thresholds is True
    assert report.detection_voltage_v >= REBCOConductor().quench_detection_threshold_v
    assert report.peak_terminal_voltage_v <= REBCOConductor().max_terminal_voltage_v
    assert report.hotspot_temperature_k <= REBCOConductor().max_hotspot_temperature_k
    assert "not a certified magnet protection design" in report.claim_boundary


def test_rebco_quench_screen_fails_closed_for_insensitive_detector() -> None:
    report = evaluate_rebco_quench(
        conductor=REBCOConductor(quench_detection_threshold_v=0.08),
    )

    assert report.passes_thresholds is False
    assert "detection_voltage_below_threshold" in report.failure_reasons


def test_default_direct_energy_conversion_fault_passes_with_bounded_claim() -> None:
    report = evaluate_direct_energy_conversion_fault()

    assert report.passes_thresholds is True
    assert report.fail_closed_time_ms == pytest.approx(7.9)
    assert report.isolated_energy_mj <= DirectEnergyConversionChannel().max_unisolated_energy_mj
    assert "not a validated DEC subsystem" in report.claim_boundary


def test_direct_energy_conversion_fault_fails_closed_under_slow_isolation() -> None:
    report = evaluate_direct_energy_conversion_fault(
        channel=DirectEnergyConversionChannel(isolation_time_ms=80.0),
        fault=DirectEnergyConversionFault(load_rejection_fraction=1.0),
    )

    assert report.passes_thresholds is False
    assert "unisolated_energy" in report.failure_reasons


def test_default_structural_response_passes_with_bounded_claim() -> None:
    report = evaluate_disruption_structural_response()

    assert report.passes_thresholds is True
    assert report.stress_margin > 1.0
    assert report.strain_margin > 1.0
    assert report.displacement_mm < 10.0
    assert "not finite-element analysis" in report.claim_boundary


def test_structural_response_fails_closed_for_unbraced_thin_member() -> None:
    report = evaluate_disruption_structural_response(
        member=StructuralMember(
            support_span_m=2.4,
            wall_thickness_m=0.055,
            effective_width_m=1.0,
        ),
        load=DisruptionLoad(wall_force_mn_per_m=2.2, vertical_force_mn=18.0),
    )

    assert report.passes_thresholds is False
    assert {"stress_margin", "strain_margin", "displacement_limit"}.issubset(
        set(report.failure_reasons)
    )


def test_subsystem_models_reject_invalid_physical_inputs() -> None:
    with pytest.raises(ValueError, match="critical_current_a"):
        evaluate_rebco_quench(conductor=REBCOConductor(critical_current_a=10_000.0))

    with pytest.raises(ValueError, match="charged_particle_fraction"):
        evaluate_direct_energy_conversion_fault(
            channel=DirectEnergyConversionChannel(charged_particle_fraction=1.5)
        )

    with pytest.raises(ValueError, match="wall_thickness_m"):
        evaluate_disruption_structural_response(member=StructuralMember(wall_thickness_m=0.0))


def test_disruption_structural_response_rejects_non_finite_geometry() -> None:
    """A non-finite section dimension must fail the finiteness guard before the <= 0 check."""
    with pytest.raises(ValueError, match="radius_m must be finite"):
        evaluate_disruption_structural_response(member=StructuralMember(radius_m=float("inf")))


def test_disruption_structural_response_rejects_nan_load() -> None:
    """A NaN load magnitude must be caught by the finiteness guard on the load side."""
    with pytest.raises(ValueError, match="halo_current_ma must be finite"):
        evaluate_disruption_structural_response(load=DisruptionLoad(halo_current_ma=float("nan")))
