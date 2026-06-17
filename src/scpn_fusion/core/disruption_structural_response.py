# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Disruption Structural Response
"""Reduced-order structural shock and strain response for disruption loads."""

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


@dataclass(frozen=True)
class StructuralMember:
    """Simplified vessel/support structural section for disruption screening."""

    radius_m: float = 3.2
    support_span_m: float = 1.5
    wall_thickness_m: float = 0.18
    effective_width_m: float = 3.0
    youngs_modulus_pa: float = 190.0e9
    yield_strength_pa: float = 520.0e6
    allowable_strain: float = 0.0025
    dynamic_amplification: float = 1.35
    safety_factor: float = 1.5


@dataclass(frozen=True)
class DisruptionLoad:
    """Halo/VDE disruption load envelope for reduced structural response."""

    halo_current_ma: float = 2.4
    plasma_current_ma: float = 8.0
    vertical_force_mn: float = 18.0
    wall_force_mn_per_m: float = 2.2
    impulse_duration_ms: float = 8.0


@dataclass(frozen=True)
class StructuralResponseReport:
    """Stress, strain, displacement, and margin summary."""

    status: str
    peak_line_force_mn_per_m: float
    bending_stress_mpa: float
    hoop_stress_mpa: float
    equivalent_stress_mpa: float
    peak_strain: float
    displacement_mm: float
    stress_margin: float
    strain_margin: float
    passes_thresholds: bool
    failure_reasons: tuple[str, ...]
    claim_boundary: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable report dictionary."""
        out = asdict(self)
        out["failure_reasons"] = list(self.failure_reasons)
        return out


def evaluate_disruption_structural_response(
    member: StructuralMember | None = None,
    load: DisruptionLoad | None = None,
) -> StructuralResponseReport:
    """Evaluate a reduced structural response to VDE/halo disruption loads."""
    m = member or StructuralMember()
    l = load or DisruptionLoad()
    radius = _positive("radius_m", m.radius_m)
    span = _positive("support_span_m", m.support_span_m)
    thickness = _positive("wall_thickness_m", m.wall_thickness_m)
    width = _positive("effective_width_m", m.effective_width_m)
    young = _positive("youngs_modulus_pa", m.youngs_modulus_pa)
    yield_strength = _positive("yield_strength_pa", m.yield_strength_pa)
    allowable_strain = _positive("allowable_strain", m.allowable_strain)
    dyn = _positive("dynamic_amplification", m.dynamic_amplification)
    safety = _positive("safety_factor", m.safety_factor)

    halo_current = _positive("halo_current_ma", l.halo_current_ma)
    plasma_current = _positive("plasma_current_ma", l.plasma_current_ma)
    vertical_force = _positive("vertical_force_mn", l.vertical_force_mn)
    wall_force = _positive("wall_force_mn_per_m", l.wall_force_mn_per_m)
    impulse_ms = _positive("impulse_duration_ms", l.impulse_duration_ms)

    electromagnetic_scale = (halo_current * plasma_current) / max(2.4 * 8.0, 1.0e-12)
    line_force = wall_force * electromagnetic_scale * dyn
    distributed_n_m = line_force * 1.0e6
    vertical_line_n_m = vertical_force * 1.0e6 / max(2.0 * math.pi * radius, 1.0e-12)
    total_line_n_m = distributed_n_m + vertical_line_n_m
    section_modulus = width * thickness * thickness / 6.0
    inertia = width * thickness**3 / 12.0
    moment = total_line_n_m * span * span / 8.0
    bending_stress = moment / max(section_modulus, 1.0e-18)
    hoop_stress = distributed_n_m * radius / max(thickness, 1.0e-18)
    equivalent = math.sqrt(bending_stress**2 + hoop_stress**2 - bending_stress * hoop_stress)
    strain = equivalent / young
    displacement = 5.0 * total_line_n_m * span**4 / (384.0 * young * max(inertia, 1.0e-18))
    impulse_factor = math.sqrt(max(impulse_ms, 1.0) / 8.0)
    equivalent *= impulse_factor
    strain *= impulse_factor
    displacement *= impulse_factor

    allowable_stress = yield_strength / safety
    stress_margin = allowable_stress / max(equivalent, 1.0e-18)
    strain_margin = allowable_strain / max(strain, 1.0e-18)

    failures: list[str] = []
    if stress_margin < 1.0:
        failures.append("stress_margin")
    if strain_margin < 1.0:
        failures.append("strain_margin")
    if displacement > 0.010:
        failures.append("displacement_limit")

    return StructuralResponseReport(
        status="reduced_order_structural_shock_screen",
        peak_line_force_mn_per_m=float(line_force),
        bending_stress_mpa=float(bending_stress / 1.0e6),
        hoop_stress_mpa=float(hoop_stress / 1.0e6),
        equivalent_stress_mpa=float(equivalent / 1.0e6),
        peak_strain=float(strain),
        displacement_mm=float(displacement * 1.0e3),
        stress_margin=float(stress_margin),
        strain_margin=float(strain_margin),
        passes_thresholds=not failures,
        failure_reasons=tuple(failures),
        claim_boundary=(
            "Reduced-order structural shock screen only; not finite-element "
            "analysis, vessel certification, or component stress qualification."
        ),
    )
