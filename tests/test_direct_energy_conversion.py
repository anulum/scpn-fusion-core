# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Direct-Energy-Conversion Fault Boundary Tests
"""Contract tests for the reduced-order direct-energy-conversion fault boundary.

The tests exercise the public evaluation surface end to end: finite/positive/
non-negative input validation, the physical identities of the nominal channel,
each independent fail-closed threshold branch in isolation, and the
JSON-serialisable report contract. Every threshold branch is provoked on its own
so that a regression in one gate cannot be masked by a second simultaneous
failure.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from scpn_fusion.core.direct_energy_conversion import (
    DirectEnergyConversionChannel,
    DirectEnergyConversionFault,
    DirectEnergyConversionReport,
    evaluate_direct_energy_conversion_fault,
)


class TestInputValidation:
    """Reject non-finite, non-positive, negative, and out-of-range inputs."""

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_non_finite_channel_field_rejected(self, bad: float) -> None:
        """Route a non-finite channel field through the finite guard."""
        with pytest.raises(ValueError, match="thermal_power_mw must be finite"):
            evaluate_direct_energy_conversion_fault(
                channel=DirectEnergyConversionChannel(thermal_power_mw=bad)
            )

    @pytest.mark.parametrize("bad", [0.0, -5.0])
    def test_non_positive_channel_field_rejected(self, bad: float) -> None:
        """Reject a zero or negative value on a strictly-positive channel field."""
        with pytest.raises(ValueError, match=r"thermal_power_mw must be > 0"):
            evaluate_direct_energy_conversion_fault(
                channel=DirectEnergyConversionChannel(thermal_power_mw=bad)
            )

    def test_negative_non_negative_channel_field_rejected(self) -> None:
        """Reject a negative value on a non-negative timing field."""
        with pytest.raises(ValueError, match=r"isolation_time_ms must be >= 0"):
            evaluate_direct_energy_conversion_fault(
                channel=DirectEnergyConversionChannel(isolation_time_ms=-1.0)
            )

    def test_negative_non_negative_fault_field_rejected(self) -> None:
        """Reject a negative value on a non-negative fault-latency field."""
        with pytest.raises(ValueError, match=r"control_latency_ms must be >= 0"):
            evaluate_direct_energy_conversion_fault(
                fault=DirectEnergyConversionFault(control_latency_ms=-0.1)
            )

    def test_charged_particle_fraction_above_unity_rejected(self) -> None:
        """Reject a charged-particle fraction that exceeds unity."""
        with pytest.raises(ValueError, match=r"charged_particle_fraction must be <= 1"):
            evaluate_direct_energy_conversion_fault(
                channel=DirectEnergyConversionChannel(charged_particle_fraction=1.5)
            )

    def test_nominal_efficiency_above_unity_rejected(self) -> None:
        """Reject a nominal efficiency that exceeds unity."""
        with pytest.raises(ValueError, match=r"nominal_efficiency must be <= 1"):
            evaluate_direct_energy_conversion_fault(
                channel=DirectEnergyConversionChannel(nominal_efficiency=1.5)
            )

    def test_efficiency_drop_fraction_above_unity_rejected(self) -> None:
        """Reject a fault efficiency-drop fraction that exceeds unity."""
        with pytest.raises(ValueError, match=r"efficiency_drop_fraction must be <= 1"):
            evaluate_direct_energy_conversion_fault(
                fault=DirectEnergyConversionFault(efficiency_drop_fraction=1.5)
            )

    def test_load_rejection_fraction_above_unity_rejected(self) -> None:
        """Reject a fault load-rejection fraction that exceeds unity."""
        with pytest.raises(ValueError, match=r"load_rejection_fraction must be <= 1"):
            evaluate_direct_energy_conversion_fault(
                fault=DirectEnergyConversionFault(load_rejection_fraction=1.5)
            )


class TestNominalEvaluation:
    """Verify the nominal channel passes with physically-consistent metrics."""

    def test_default_channel_passes_all_thresholds(self) -> None:
        """The default channel and fault clear every fail-closed threshold."""
        report = evaluate_direct_energy_conversion_fault()
        assert isinstance(report, DirectEnergyConversionReport)
        assert report.passes_thresholds is True
        assert report.failure_reasons == ()
        assert report.status == "reduced_order_dec_fault_boundary"

    def test_nominal_electric_power_identity(self) -> None:
        """Nominal electric power equals thermal × charged fraction × efficiency."""
        channel = DirectEnergyConversionChannel()
        report = evaluate_direct_energy_conversion_fault(channel=channel)
        expected = (
            channel.thermal_power_mw
            * channel.charged_particle_fraction
            * channel.nominal_efficiency
        )
        assert report.nominal_electric_power_mw == pytest.approx(expected)

    def test_fail_closed_time_sums_latencies(self) -> None:
        """Fail-closed time is the sum of detection, control, isolation, crowbar."""
        channel = DirectEnergyConversionChannel()
        fault = DirectEnergyConversionFault()
        report = evaluate_direct_energy_conversion_fault(channel=channel, fault=fault)
        expected = (
            fault.sensor_detection_latency_ms
            + fault.control_latency_ms
            + channel.isolation_time_ms
            + channel.crowbar_time_ms
        )
        assert report.fail_closed_time_ms == pytest.approx(expected)

    def test_degraded_efficiency_floor_clamp(self) -> None:
        """Degraded power uses the efficiency floor when the drop undercuts it."""
        # efficiency * (1 - drop) = 0.42 * 0.1 = 0.042 < floor 0.25 -> floor wins.
        channel = DirectEnergyConversionChannel()
        fault = DirectEnergyConversionFault(
            efficiency_drop_fraction=0.9,
            degraded_efficiency_floor=0.25,
        )
        report = evaluate_direct_energy_conversion_fault(channel=channel, fault=fault)
        expected = channel.thermal_power_mw * channel.charged_particle_fraction * 0.25
        assert report.degraded_electric_power_mw == pytest.approx(expected)

    def test_boundary_efficiency_of_unity_is_accepted(self) -> None:
        """An efficiency of exactly unity is inside the accepted range."""
        report = evaluate_direct_energy_conversion_fault(
            channel=DirectEnergyConversionChannel(nominal_efficiency=1.0)
        )
        assert report.nominal_electric_power_mw > 0.0


class TestFailureModes:
    """Provoke each fail-closed threshold branch in isolation."""

    def test_unisolated_energy_breach(self) -> None:
        """Slow isolation with full load rejection breaches the unisolated cap."""
        report = evaluate_direct_energy_conversion_fault(
            channel=DirectEnergyConversionChannel(isolation_time_ms=80.0),
            fault=DirectEnergyConversionFault(load_rejection_fraction=1.0),
        )
        assert report.passes_thresholds is False
        assert "unisolated_energy" in report.failure_reasons

    def test_bus_overvoltage_breach_in_isolation(self) -> None:
        """A tightened overvoltage limit trips only the bus-overvoltage gate."""
        report = evaluate_direct_energy_conversion_fault(
            channel=DirectEnergyConversionChannel(max_bus_overvoltage_fraction=1.0e-4),
        )
        assert report.bus_overvoltage_fraction > 1.0e-4
        assert report.failure_reasons == ("bus_overvoltage",)
        assert report.passes_thresholds is False

    def test_dump_power_breach_in_isolation(self) -> None:
        """A tightened dump-power limit trips only the dump-power gate."""
        report = evaluate_direct_energy_conversion_fault(
            channel=DirectEnergyConversionChannel(max_dump_power_mw=1.0e-3),
        )
        assert report.peak_dump_power_mw > 1.0e-3
        assert report.failure_reasons == ("dump_power",)
        assert report.passes_thresholds is False

    def test_degraded_power_floor_breach_in_isolation(self) -> None:
        """A zero efficiency floor with a total efficiency drop collapses power."""
        report = evaluate_direct_energy_conversion_fault(
            fault=DirectEnergyConversionFault(
                efficiency_drop_fraction=1.0,
                degraded_efficiency_floor=0.0,
            ),
        )
        assert report.degraded_electric_power_mw == pytest.approx(0.0)
        assert report.failure_reasons == ("degraded_power_floor",)
        assert report.passes_thresholds is False


class TestReportSerialisation:
    """Verify the report contract is immutable and JSON-serialisable."""

    def test_to_dict_lists_failure_reasons(self) -> None:
        """to_dict converts the failure-reason tuple into a JSON list."""
        report = evaluate_direct_energy_conversion_fault(
            channel=DirectEnergyConversionChannel(max_dump_power_mw=1.0e-3),
        )
        payload = report.to_dict()
        assert isinstance(payload, dict)
        assert isinstance(payload["failure_reasons"], list)
        assert payload["failure_reasons"] == ["dump_power"]
        assert payload["passes_thresholds"] is False

    def test_to_dict_round_trips_scalar_fields(self) -> None:
        """to_dict preserves the scalar metric fields of the report."""
        report = evaluate_direct_energy_conversion_fault()
        payload = report.to_dict()
        assert payload["nominal_electric_power_mw"] == report.nominal_electric_power_mw
        assert payload["fail_closed_time_ms"] == report.fail_closed_time_ms
        assert "not a validated DEC subsystem" in payload["claim_boundary"]

    def test_report_is_frozen(self) -> None:
        """The report dataclass rejects post-construction mutation."""
        report = evaluate_direct_energy_conversion_fault()
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.status = "mutated"  # type: ignore[misc]  # deliberate frozen-mutation probe
