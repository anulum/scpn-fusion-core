# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — MIF/FRC Tilt-Mode Diagnostics
"""Conservative n=1 FRC tilt-mode diagnostics for the MIF lane.

The accepted public contract is deliberately narrower than a full Belova
hybrid eigenvalue solver. It exposes the MHD Alfvén-time growth scaling, the
Steinhauer ``s`` parameter carried by the accepted FRC equilibrium, and
rigid-body FLR threshold diagnostics. Belova Table I and hybrid eigenvalue
parity remain fail-closed until a redistributable digitised reference exists.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frc_rigid_rotor import (
    ATOMIC_MASS_KG,
    DEUTERIUM_MASS_AMU,
    FRCEquilibriumState,
    MU_0,
    s_parameter as frc_s_parameter,
)

BELOVA_MHD_GROWTH_COEFFICIENT = 1.2
DIAMAGNETIC_S_OVER_E_THRESHOLD = 1.7
GYROVISCOUS_S_OVER_E_THRESHOLD = 2.2
COMBINED_FLR_S_OVER_E_THRESHOLD = 2.8


@dataclass(frozen=True)
class FRCTiltModeThresholds:
    """Rigid-body FLR threshold diagnostics from Ji-style FRC estimates."""

    diamagnetic_s_over_e: float = DIAMAGNETIC_S_OVER_E_THRESHOLD
    gyroviscous_s_over_e: float = GYROVISCOUS_S_OVER_E_THRESHOLD
    combined_flr_s_over_e: float = COMBINED_FLR_S_OVER_E_THRESHOLD


@dataclass(frozen=True)
class FRCTiltModeReport:
    """Conservative FRC tilt diagnostic report for a supplied equilibrium."""

    growth_rate_s_inv: float
    alfven_speed_m_s: float
    alfven_transit_time_s: float
    s_parameter: float
    elongation: float
    s_over_elongation: float
    rigid_body_regime: str
    rigid_body_threshold_passed: bool
    conservative_stable: bool
    claim_status: str
    external_parity_status: str


def alfven_speed_m_s(
    eq: FRCEquilibriumState,
    *,
    ion_mass_amu: float = DEUTERIUM_MASS_AMU,
) -> float:
    """Return ``V_A = B_ref / sqrt(mu0*rho_m)`` for the FRC peak-density state."""

    mass = _require_positive("ion_mass_amu", ion_mass_amu) * ATOMIC_MASS_KG
    density = _require_positive("eq.density_peak_m3", eq.density_peak_m3) * mass
    field = float(np.max(np.abs(eq.B_z)))
    if not np.isfinite(field) or field <= 0.0:
        raise ValueError("eq.B_z must contain a positive finite reference field")
    return float(field / np.sqrt(MU_0 * density))


def axial_half_length_m(eq: FRCEquilibriumState, elongation: float) -> float:
    """Return the prolate FRC axial half-length used for the Alfvén time."""

    radius = _require_positive("eq.target_separatrix_radius_m", eq.target_separatrix_radius_m)
    elongation_value = _require_positive("elongation", elongation)
    return float(radius * elongation_value)


def frc_tilt_growth_rate(
    eq: FRCEquilibriumState,
    elongation: float,
    *,
    n_modes: int = 32,
    mhd_coefficient: float = BELOVA_MHD_GROWTH_COEFFICIENT,
    ion_mass_amu: float = DEUTERIUM_MASS_AMU,
) -> float:
    """Return the Belova-normalised MHD tilt growth estimate ``C*V_A/Z_s``.

    ``n_modes`` is validated for API compatibility with the future eigenvalue
    solver surface, but this accepted diagnostic is the n=1 MHD scaling only.
    """

    if isinstance(n_modes, bool) or n_modes < 1:
        raise ValueError("n_modes must be a positive integer")
    coefficient = _require_positive("mhd_coefficient", mhd_coefficient)
    speed = alfven_speed_m_s(eq, ion_mass_amu=ion_mass_amu)
    length = axial_half_length_m(eq, elongation)
    return float(coefficient * speed / length)


def s_parameter(eq: FRCEquilibriumState, mass_amu: float = DEUTERIUM_MASS_AMU) -> float:
    """Return the Steinhauer ``s`` parameter carried by the FRC equilibrium."""

    return float(frc_s_parameter(eq, mass_amu=mass_amu))


def s_over_elongation(eq: FRCEquilibriumState, elongation: float) -> float:
    """Return the dimensionless ``s / E`` rigid-body stability diagnostic."""

    return float(s_parameter(eq) / _require_positive("elongation", elongation))


def rigid_body_flr_regime(
    eq: FRCEquilibriumState,
    elongation: float,
    thresholds: FRCTiltModeThresholds | None = None,
) -> tuple[str, bool]:
    """Classify the rigid-body FLR threshold regime without claiming parity."""

    ratio = s_over_elongation(eq, elongation)
    thresholds = FRCTiltModeThresholds() if thresholds is None else thresholds
    _validate_thresholds(thresholds)
    if ratio <= thresholds.diamagnetic_s_over_e:
        return "diamagnetic_flr_threshold_passed", True
    if ratio <= thresholds.gyroviscous_s_over_e:
        return "gyroviscous_flr_threshold_passed", True
    if ratio <= thresholds.combined_flr_s_over_e:
        return "combined_flr_threshold_passed", True
    return "mhd_tilt_susceptible", False


def tilt_mode_report(
    eq: FRCEquilibriumState,
    elongation: float,
    *,
    thresholds: FRCTiltModeThresholds | None = None,
    mhd_coefficient: float = BELOVA_MHD_GROWTH_COEFFICIENT,
    ion_mass_amu: float = DEUTERIUM_MASS_AMU,
) -> FRCTiltModeReport:
    """Return a fail-closed FRC n=1 tilt-mode diagnostic report."""

    growth = frc_tilt_growth_rate(
        eq,
        elongation,
        mhd_coefficient=mhd_coefficient,
        ion_mass_amu=ion_mass_amu,
    )
    speed = alfven_speed_m_s(eq, ion_mass_amu=ion_mass_amu)
    length = axial_half_length_m(eq, elongation)
    thresholds = FRCTiltModeThresholds() if thresholds is None else thresholds
    regime, threshold_passed = rigid_body_flr_regime(eq, elongation, thresholds)
    parity = belova_table1_acceptance_status()["status"]
    return FRCTiltModeReport(
        growth_rate_s_inv=growth,
        alfven_speed_m_s=speed,
        alfven_transit_time_s=float(length / speed),
        s_parameter=s_parameter(eq),
        elongation=float(elongation),
        s_over_elongation=s_over_elongation(eq, elongation),
        rigid_body_regime=regime,
        rigid_body_threshold_passed=threshold_passed,
        conservative_stable=False,
        claim_status="diagnostic_only_not_hybrid_eigenvalue_accepted",
        external_parity_status=parity,
    )


def tilt_mode_stable(
    eq: FRCEquilibriumState,
    elongation: float,
    s_threshold: float = 3.0,
) -> tuple[bool, float]:
    """Return conservative stability and MHD growth estimate for n=1 tilt.

    The stability boolean remains fail-closed while Belova hybrid eigenvalue
    and Table I reproduction evidence is unavailable. ``s_threshold`` is still
    validated and exposed because downstream safety guards use it as a
    diagnostic floor, not as an acceptance proof.
    """

    _require_positive("s_threshold", s_threshold)
    report = tilt_mode_report(eq, elongation)
    return (False, report.growth_rate_s_inv)


def belova_table1_acceptance_status() -> dict[str, str]:
    """Return the current fail-closed status for Belova Table I parity."""

    return {
        "case": "belova_2001_table1_tilt_stability",
        "status": "blocked_missing_public_digitised_reference",
        "required_artifact": (
            "digitised Belova Table I growth/stability data with provenance, "
            "checksum, and matching equilibrium-deck metadata"
        ),
    }


def claim_boundary() -> dict[str, str]:
    """Return the public claim boundary for this FUS-C.5 surface."""

    return {
        "accepted": "MHD Alfvén-time tilt-growth diagnostic with conservative fail-closed status",
        "not_accepted": "full Belova hybrid eigenvalue solver or Table I same-case parity",
    }


def _require_positive(name: str, value: float) -> float:
    checked = float(value)
    if not np.isfinite(checked):
        raise ValueError(f"{name} must be finite")
    if checked <= 0.0:
        raise ValueError(f"{name} must be positive")
    return checked


def _validate_thresholds(thresholds: FRCTiltModeThresholds) -> None:
    values: tuple[tuple[str, float], ...] = (
        ("diamagnetic_s_over_e", thresholds.diamagnetic_s_over_e),
        ("gyroviscous_s_over_e", thresholds.gyroviscous_s_over_e),
        ("combined_flr_s_over_e", thresholds.combined_flr_s_over_e),
    )
    checked = tuple(_require_positive(name, value) for name, value in values)
    if not (checked[0] < checked[1] < checked[2]):
        raise ValueError("rigid-body thresholds must be strictly increasing")
