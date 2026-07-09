# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — REBCO/HTS Quench Screen Tests
"""Contract tests for the reduced-order REBCO/HTS quench protection screen.

The tests cover the public surface end to end: finite/positive/non-negative
input validation, the linear current-sharing-temperature screen and its two
ordering guards, the physical identities of the nominal quench evaluation, and
every fail-closed protection branch (detection voltage, terminal voltage,
hotspot temperature, current-sharing margin, critical-current margin, strain
proxy). Purely-arithmetic screening logic with no compute hot path and no
Rust/Julia/Go counterpart, so no polyglot parity surface is involved.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from scpn_fusion.core.hts_quench import (
    QuenchReport,
    QuenchScenario,
    REBCOConductor,
    current_sharing_temperature_k,
    evaluate_rebco_quench,
)


class TestInputValidation:
    """Reject non-finite, non-positive, and negative conductor/scenario fields."""

    def test_non_finite_field_rejected(self) -> None:
        """Route a non-finite conductor field through the finite guard."""
        with pytest.raises(ValueError, match="operating_current_a must be finite"):
            evaluate_rebco_quench(REBCOConductor(operating_current_a=math.nan))

    def test_non_positive_field_rejected(self) -> None:
        """Reject a zero value on a strictly-positive conductor field."""
        with pytest.raises(ValueError, match=r"operating_current_a must be > 0"):
            evaluate_rebco_quench(REBCOConductor(operating_current_a=0.0))

    def test_negative_non_negative_field_rejected(self) -> None:
        """Reject a negative value on a non-negative scenario field."""
        with pytest.raises(ValueError, match=r"normal_zone_velocity_m_s must be >= 0"):
            evaluate_rebco_quench(scenario=QuenchScenario(normal_zone_velocity_m_s=-1.0))


class TestCurrentSharingTemperature:
    """Exercise the linear current-sharing-temperature screen and its guards."""

    def test_nominal_value_between_operating_and_critical(self) -> None:
        """The screen returns a temperature strictly above the operating point."""
        conductor = REBCOConductor()
        tcs = current_sharing_temperature_k(conductor)
        assert conductor.operating_temperature_k < tcs < conductor.critical_temperature_k

    def test_requires_critical_temperature_above_operating(self) -> None:
        """Reject a critical temperature not above the operating temperature."""
        with pytest.raises(ValueError, match="critical_temperature_k must exceed"):
            current_sharing_temperature_k(
                REBCOConductor(operating_temperature_k=90.0, critical_temperature_k=90.0)
            )

    def test_requires_critical_current_above_operating(self) -> None:
        """Reject a critical current not above the operating current."""
        with pytest.raises(ValueError, match="critical_current_a must exceed"):
            current_sharing_temperature_k(
                REBCOConductor(operating_current_a=28_000.0, critical_current_a=28_000.0)
            )


class TestNominalEvaluation:
    """Verify the nominal conductor passes with consistent protection metrics."""

    def test_default_conductor_passes_all_thresholds(self) -> None:
        """The default conductor and scenario clear every protection gate."""
        report = evaluate_rebco_quench()
        assert isinstance(report, QuenchReport)
        assert report.passes_thresholds is True
        assert report.failure_reasons == ()
        assert report.status == "reduced_order_quench_screen"

    def test_detection_time_sums_delays(self) -> None:
        """Detection time is the sum of detection and protection-switch delays."""
        scenario = QuenchScenario()
        report = evaluate_rebco_quench(scenario=scenario)
        expected = scenario.detection_delay_s + scenario.protection_switch_delay_s
        assert report.detection_time_s == pytest.approx(expected)

    def test_dump_time_constant_and_terminal_voltage_identities(self) -> None:
        """Time constant is L/R and terminal voltage is the dump-resistor drop."""
        conductor = REBCOConductor()
        report = evaluate_rebco_quench(conductor=conductor)
        assert report.dump_time_constant_s == pytest.approx(
            conductor.inductance_h / conductor.dump_resistance_ohm
        )
        assert report.peak_terminal_voltage_v == pytest.approx(
            conductor.operating_current_a * conductor.dump_resistance_ohm
        )

    def test_current_after_one_second_decays(self) -> None:
        """Dump current after the clamped window is below the operating current."""
        report = evaluate_rebco_quench()
        assert 0.0 < report.current_after_1s_a < REBCOConductor().operating_current_a


class TestFailureModes:
    """Provoke each fail-closed protection branch."""

    def test_detection_voltage_below_threshold(self) -> None:
        """A raised detection threshold trips the detection-voltage gate."""
        report = evaluate_rebco_quench(
            REBCOConductor(quench_detection_threshold_v=1.0),
        )
        assert "detection_voltage_below_threshold" in report.failure_reasons
        assert report.passes_thresholds is False

    def test_terminal_voltage_limit_in_isolation(self) -> None:
        """A tightened terminal-voltage limit trips only that gate."""
        report = evaluate_rebco_quench(REBCOConductor(max_terminal_voltage_v=100.0))
        assert report.failure_reasons == ("terminal_voltage_limit",)

    def test_hotspot_temperature_limit_in_isolation(self) -> None:
        """A tightened hotspot limit trips only the hotspot-temperature gate."""
        report = evaluate_rebco_quench(REBCOConductor(max_hotspot_temperature_k=15.0))
        assert report.failure_reasons == ("hotspot_temperature_limit",)

    def test_critical_current_margin_in_isolation(self) -> None:
        """A narrowed critical current trips only the current-margin gate."""
        report = evaluate_rebco_quench(REBCOConductor(critical_current_a=19_000.0))
        assert report.failure_reasons == ("critical_current_margin",)

    def test_thermal_runaway_trips_hotspot_margin_and_strain(self) -> None:
        """No cooling with a strong quench trips the coupled thermal gates."""
        conductor = REBCOConductor(
            coolant_heat_transfer_w_m2_k=0.0,
            stabilizer_resistivity_ohm_m=2.0e-8,
            stabilizer_area_m2=4.0e-5,
        )
        scenario = QuenchScenario(
            initial_normal_zone_m=0.5,
            normal_zone_velocity_m_s=2.0,
            detection_delay_s=0.2,
            protection_switch_delay_s=0.05,
        )
        report = evaluate_rebco_quench(conductor, scenario)
        assert report.passes_thresholds is False
        assert "hotspot_temperature_limit" in report.failure_reasons
        assert "current_sharing_temperature_margin" in report.failure_reasons
        assert "strain_proxy_limit" in report.failure_reasons
        assert report.hotspot_temperature_k > conductor.max_hotspot_temperature_k


class TestReportSerialisation:
    """Verify the report contract is JSON-serialisable and immutable."""

    def test_to_dict_lists_failure_reasons(self) -> None:
        """to_dict converts the failure-reason tuple into a JSON list."""
        report = evaluate_rebco_quench(REBCOConductor(max_terminal_voltage_v=100.0))
        payload = report.to_dict()
        assert isinstance(payload, dict)
        assert payload["failure_reasons"] == ["terminal_voltage_limit"]
        assert "not a certified magnet" in payload["claim_boundary"]

    def test_report_is_frozen(self) -> None:
        """The report dataclass rejects post-construction mutation."""
        report = evaluate_rebco_quench()
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.status = "mutated"  # type: ignore[misc]  # deliberate frozen-mutation probe
