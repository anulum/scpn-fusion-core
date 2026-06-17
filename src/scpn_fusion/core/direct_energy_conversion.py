# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Direct-Energy-Conversion Fault Boundary
"""Reduced-order direct-energy-conversion channel and fault-boundary model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


def _finite(name: str, value: float) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite.")
    return out


def _positive(name: str, value: float) -> float:
    out = _finite(name, value)
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0.")
    return out


def _non_negative(name: str, value: float) -> float:
    out = _finite(name, value)
    if out < 0.0:
        raise ValueError(f"{name} must be >= 0.")
    return out


@dataclass(frozen=True)
class DirectEnergyConversionChannel:
    """Nominal conversion-channel assumptions for reduced DEC screening."""

    thermal_power_mw: float = 120.0
    charged_particle_fraction: float = 0.18
    nominal_efficiency: float = 0.42
    bus_voltage_kv: float = 24.0
    bus_capacitance_f: float = 0.18
    dump_resistance_ohm: float = 4.0
    isolation_time_ms: float = 4.0
    crowbar_time_ms: float = 2.0
    max_bus_overvoltage_fraction: float = 0.18
    max_unisolated_energy_mj: float = 0.16
    max_dump_power_mw: float = 160.0


@dataclass(frozen=True)
class DirectEnergyConversionFault:
    """Fault scenario for one DEC reduced-order channel."""

    efficiency_drop_fraction: float = 0.65
    load_rejection_fraction: float = 0.55
    sensor_detection_latency_ms: float = 1.5
    control_latency_ms: float = 0.4
    degraded_efficiency_floor: float = 0.10


@dataclass(frozen=True)
class DirectEnergyConversionReport:
    """Reduced DEC fault-isolation metrics and gate status."""

    status: str
    nominal_electric_power_mw: float
    degraded_electric_power_mw: float
    isolated_energy_mj: float
    bus_overvoltage_fraction: float
    peak_dump_power_mw: float
    fail_closed_time_ms: float
    passes_thresholds: bool
    failure_reasons: tuple[str, ...]
    claim_boundary: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable report dictionary."""
        out = asdict(self)
        out["failure_reasons"] = list(self.failure_reasons)
        return out


def evaluate_direct_energy_conversion_fault(
    channel: DirectEnergyConversionChannel | None = None,
    fault: DirectEnergyConversionFault | None = None,
) -> DirectEnergyConversionReport:
    """Evaluate reduced DEC channel isolation and dump-load bounds."""
    c = channel or DirectEnergyConversionChannel()
    f = fault or DirectEnergyConversionFault()
    thermal_power = _positive("thermal_power_mw", c.thermal_power_mw)
    charged_fraction = _positive("charged_particle_fraction", c.charged_particle_fraction)
    if charged_fraction > 1.0:
        raise ValueError("charged_particle_fraction must be <= 1.")
    efficiency = _positive("nominal_efficiency", c.nominal_efficiency)
    if efficiency > 1.0:
        raise ValueError("nominal_efficiency must be <= 1.")
    bus_voltage = _positive("bus_voltage_kv", c.bus_voltage_kv)
    capacitance = _positive("bus_capacitance_f", c.bus_capacitance_f)
    dump_r = _positive("dump_resistance_ohm", c.dump_resistance_ohm)
    isolation_ms = _non_negative("isolation_time_ms", c.isolation_time_ms)
    crowbar_ms = _non_negative("crowbar_time_ms", c.crowbar_time_ms)
    max_overvoltage = _positive("max_bus_overvoltage_fraction", c.max_bus_overvoltage_fraction)
    max_unisolated = _positive("max_unisolated_energy_mj", c.max_unisolated_energy_mj)
    max_dump_power = _positive("max_dump_power_mw", c.max_dump_power_mw)

    efficiency_drop = _non_negative("efficiency_drop_fraction", f.efficiency_drop_fraction)
    if efficiency_drop > 1.0:
        raise ValueError("efficiency_drop_fraction must be <= 1.")
    load_rejection = _non_negative("load_rejection_fraction", f.load_rejection_fraction)
    if load_rejection > 1.0:
        raise ValueError("load_rejection_fraction must be <= 1.")
    detection_ms = _non_negative("sensor_detection_latency_ms", f.sensor_detection_latency_ms)
    control_ms = _non_negative("control_latency_ms", f.control_latency_ms)
    degraded_floor = _non_negative("degraded_efficiency_floor", f.degraded_efficiency_floor)

    nominal_power = thermal_power * charged_fraction * efficiency
    degraded_efficiency = max(efficiency * (1.0 - efficiency_drop), degraded_floor)
    degraded_power = thermal_power * charged_fraction * degraded_efficiency
    fail_closed_ms = detection_ms + control_ms + isolation_ms + crowbar_ms
    unisolated_power_mw = nominal_power * load_rejection
    isolated_energy_mj = unisolated_power_mw * fail_closed_ms * 1.0e-3
    stored_j = 0.5 * capacitance * (bus_voltage * 1.0e3) ** 2
    injected_j = isolated_energy_mj * 1.0e6
    bus_overvoltage = math.sqrt((stored_j + injected_j) / max(stored_j, 1.0e-12)) - 1.0
    peak_dump_power = (bus_voltage * 1.0e3) ** 2 / dump_r / 1.0e6

    failures: list[str] = []
    if isolated_energy_mj > max_unisolated:
        failures.append("unisolated_energy")
    if bus_overvoltage > max_overvoltage:
        failures.append("bus_overvoltage")
    if peak_dump_power > max_dump_power:
        failures.append("dump_power")
    if degraded_power <= 0.0:
        failures.append("degraded_power_floor")

    return DirectEnergyConversionReport(
        status="reduced_order_dec_fault_boundary",
        nominal_electric_power_mw=float(nominal_power),
        degraded_electric_power_mw=float(degraded_power),
        isolated_energy_mj=float(isolated_energy_mj),
        bus_overvoltage_fraction=float(bus_overvoltage),
        peak_dump_power_mw=float(peak_dump_power),
        fail_closed_time_ms=float(fail_closed_ms),
        passes_thresholds=not failures,
        failure_reasons=tuple(failures),
        claim_boundary=(
            "Reduced-order direct-energy-conversion fault boundary only; not a "
            "validated DEC subsystem, power-electronics design, or hardware interlock."
        ),
    )
