# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — REBCO/HTS Quench Dynamics
"""Reduced-order REBCO/HTS quench dynamics and protection metrics."""

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
class REBCOConductor:
    """Lumped HTS conductor and stabilizer properties for quench screening."""

    operating_temperature_k: float = 20.0
    critical_temperature_k: float = 90.0
    operating_current_a: float = 18_000.0
    critical_current_a: float = 28_000.0
    inductance_h: float = 0.18
    dump_resistance_ohm: float = 0.045
    stabilizer_resistivity_ohm_m: float = 2.0e-10
    stabilizer_area_m2: float = 1.6e-4
    conductor_density_kg_m3: float = 8_400.0
    conductor_heat_capacity_j_kg_k: float = 290.0
    wetted_perimeter_m: float = 0.055
    coolant_heat_transfer_w_m2_k: float = 650.0
    quench_detection_threshold_v: float = 0.0015
    max_terminal_voltage_v: float = 1_200.0
    max_hotspot_temperature_k: float = 120.0


@dataclass(frozen=True)
class QuenchScenario:
    """Fault scenario parameters for a single reduced-order HTS quench event."""

    initial_normal_zone_m: float = 0.08
    normal_zone_velocity_m_s: float = 0.45
    detection_delay_s: float = 0.035
    protection_switch_delay_s: float = 0.010
    simulation_duration_s: float = 1.2
    assumed_coolant_temperature_k: float = 18.0


@dataclass(frozen=True)
class QuenchReport:
    """Reduced-order quench metrics and gate status."""

    status: str
    current_sharing_margin_k: float
    current_margin_fraction: float
    detection_voltage_v: float
    detection_time_s: float
    dump_time_constant_s: float
    current_after_1s_a: float
    peak_terminal_voltage_v: float
    normal_zone_length_m: float
    joule_energy_j: float
    cooling_energy_j: float
    hotspot_temperature_k: float
    strain_proxy_percent: float
    passes_thresholds: bool
    failure_reasons: tuple[str, ...]
    claim_boundary: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable report dictionary."""
        out = asdict(self)
        out["failure_reasons"] = list(self.failure_reasons)
        return out


def current_sharing_temperature_k(conductor: REBCOConductor) -> float:
    """Estimate current-sharing temperature from a linear Ic(T) screen."""
    t_op = _positive("operating_temperature_k", conductor.operating_temperature_k)
    t_crit = _positive("critical_temperature_k", conductor.critical_temperature_k)
    i_op = _positive("operating_current_a", conductor.operating_current_a)
    i_crit = _positive("critical_current_a", conductor.critical_current_a)
    if t_crit <= t_op:
        raise ValueError("critical_temperature_k must exceed operating_temperature_k.")
    if i_crit <= i_op:
        raise ValueError("critical_current_a must exceed operating_current_a.")
    return float(t_op + (t_crit - t_op) * (1.0 - i_op / i_crit))


def evaluate_rebco_quench(
    conductor: REBCOConductor | None = None,
    scenario: QuenchScenario | None = None,
) -> QuenchReport:
    """Evaluate a bounded REBCO/HTS quench protection screen.

    The model is a lumped engineering screen. It is suitable for regression and
    claim-boundary tracking, not for magnet protection design certification.
    """
    c = conductor or REBCOConductor()
    s = scenario or QuenchScenario()
    i0 = _positive("operating_current_a", c.operating_current_a)
    inductance = _positive("inductance_h", c.inductance_h)
    dump_r = _positive("dump_resistance_ohm", c.dump_resistance_ohm)
    rho = _positive("stabilizer_resistivity_ohm_m", c.stabilizer_resistivity_ohm_m)
    area = _positive("stabilizer_area_m2", c.stabilizer_area_m2)
    density = _positive("conductor_density_kg_m3", c.conductor_density_kg_m3)
    cp = _positive("conductor_heat_capacity_j_kg_k", c.conductor_heat_capacity_j_kg_k)
    perimeter = _positive("wetted_perimeter_m", c.wetted_perimeter_m)
    htc = _non_negative("coolant_heat_transfer_w_m2_k", c.coolant_heat_transfer_w_m2_k)
    threshold_v = _positive("quench_detection_threshold_v", c.quench_detection_threshold_v)
    max_voltage = _positive("max_terminal_voltage_v", c.max_terminal_voltage_v)
    max_hotspot = _positive("max_hotspot_temperature_k", c.max_hotspot_temperature_k)

    initial_zone = _positive("initial_normal_zone_m", s.initial_normal_zone_m)
    velocity = _non_negative("normal_zone_velocity_m_s", s.normal_zone_velocity_m_s)
    detection_delay = _non_negative("detection_delay_s", s.detection_delay_s)
    switch_delay = _non_negative("protection_switch_delay_s", s.protection_switch_delay_s)
    duration = _positive("simulation_duration_s", s.simulation_duration_s)
    coolant_t = _positive("assumed_coolant_temperature_k", s.assumed_coolant_temperature_k)

    detection_time = detection_delay + switch_delay
    normal_length = initial_zone + velocity * detection_time
    normal_resistance = rho * normal_length / area
    detection_voltage = i0 * normal_resistance
    tau = inductance / dump_r
    current_after_1s = i0 * math.exp(-min(1.0, duration) / tau)
    peak_terminal_voltage = i0 * dump_r

    delay_energy = i0 * i0 * normal_resistance * detection_time
    remaining_energy = 0.5 * i0 * i0 * normal_resistance * tau
    joule_energy = delay_energy + remaining_energy
    exposed_area = perimeter * normal_length
    average_delta_t = max(c.operating_temperature_k - coolant_t, 0.0) + 18.0
    cooling_energy = htc * exposed_area * average_delta_t * duration
    heated_mass = density * area * normal_length
    net_energy = max(joule_energy - cooling_energy, 0.0)
    hotspot = c.operating_temperature_k + net_energy / max(heated_mass * cp, 1.0e-12)
    tcs = current_sharing_temperature_k(c)
    current_margin = (c.critical_current_a - i0) / c.critical_current_a
    strain_proxy = 0.16 + 0.0025 * max(hotspot - c.operating_temperature_k, 0.0)

    failures: list[str] = []
    if detection_voltage < threshold_v:
        failures.append("detection_voltage_below_threshold")
    if peak_terminal_voltage > max_voltage:
        failures.append("terminal_voltage_limit")
    if hotspot > max_hotspot:
        failures.append("hotspot_temperature_limit")
    if hotspot >= tcs:
        failures.append("current_sharing_temperature_margin")
    if current_margin < 0.20:
        failures.append("critical_current_margin")
    if strain_proxy > 0.45:
        failures.append("strain_proxy_limit")

    return QuenchReport(
        status="reduced_order_quench_screen",
        current_sharing_margin_k=float(tcs - c.operating_temperature_k),
        current_margin_fraction=float(current_margin),
        detection_voltage_v=float(detection_voltage),
        detection_time_s=float(detection_time),
        dump_time_constant_s=float(tau),
        current_after_1s_a=float(current_after_1s),
        peak_terminal_voltage_v=float(peak_terminal_voltage),
        normal_zone_length_m=float(normal_length),
        joule_energy_j=float(joule_energy),
        cooling_energy_j=float(cooling_energy),
        hotspot_temperature_k=float(hotspot),
        strain_proxy_percent=float(strain_proxy),
        passes_thresholds=not failures,
        failure_reasons=tuple(failures),
        claim_boundary=(
            "Reduced-order REBCO/HTS quench screen only; not a certified magnet "
            "protection design, hardware quench detector, or conductor qualification."
        ),
    )
