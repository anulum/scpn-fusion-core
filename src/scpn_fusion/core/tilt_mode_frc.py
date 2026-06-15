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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

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
FLOAT64_LOG_MAX = float(np.log(np.finfo(np.float64).max))


class PulsedCompressionLike(Protocol):
    """Minimal FUS-C.6 state surface required by the FUS-C.5 adapter."""

    t_s: float
    R_s_m: float
    T_i_eV: float
    density_m3: float
    B_ext_T: float


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


@dataclass(frozen=True)
class FRCTiltModeTrajectoryPoint:
    """Tilt diagnostic projected onto one FUS-C.6 compression state."""

    t_s: float
    R_s_m: float
    T_i_eV: float
    density_m3: float
    B_reference_T: float
    report: FRCTiltModeReport
    cumulative_growth_integral: float
    perturbation_amplification: float
    amplification_overflow_limited: bool


def alfven_speed_m_s(
    eq: FRCEquilibriumState,
    *,
    ion_mass_amu: float = DEUTERIUM_MASS_AMU,
) -> float:
    """Return ``V_A = B_ref / sqrt(mu0*rho_m)`` for the FRC peak-density state."""

    field = float(np.max(np.abs(eq.B_z)))
    return _alfven_speed_from_values(
        b_reference_t=field,
        density_peak_m3=eq.density_peak_m3,
        ion_mass_amu=ion_mass_amu,
    )


def axial_half_length_m(eq: FRCEquilibriumState, elongation: float) -> float:
    """Return the prolate FRC axial half-length used for the Alfvén time."""

    return _axial_half_length_from_values(eq.target_separatrix_radius_m, elongation)


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
    return _rigid_body_flr_regime_from_ratio(ratio, thresholds)


def tilt_mode_trajectory_from_pulsed_compression(
    states: Sequence[PulsedCompressionLike],
    eq: FRCEquilibriumState,
    elongation: float,
    *,
    thresholds: FRCTiltModeThresholds | None = None,
    mhd_coefficient: float = BELOVA_MHD_GROWTH_COEFFICIENT,
    ion_mass_amu: float = DEUTERIUM_MASS_AMU,
) -> tuple[FRCTiltModeTrajectoryPoint, ...]:
    """Project the accepted FUS-C.5 diagnostic over a FUS-C.6 trajectory.

    The pulsed-compression state does not carry a full radial equilibrium at
    each time sample, so this adapter does not claim a Belova eigenvalue solve.
    It keeps the trajectory diagnostic self-consistent by projecting the
    Steinhauer ``s`` number with the self-similar gyroradius scaling
    ``s(t)=s0*(R/R0)*(B/B0)*sqrt(T_i0/T_i)`` and by recomputing the
    Alfvén-time growth rate from the instantaneous radius, density, and field.

    The cumulative growth integral is accumulated with the trapezoidal rule from
    the interval-endpoint growth rates, ``+= 0.5*(gamma(start)+gamma(end))*dt``,
    which is second-order accurate in the sampling interval for the time-varying
    compression drivers and reduces to a single ``gamma*dt`` term when the growth
    rate is constant across the interval.
    """

    if len(states) == 0:
        raise ValueError("states must contain at least one compression state")

    thresholds = FRCTiltModeThresholds() if thresholds is None else thresholds
    _validate_thresholds(thresholds)
    coefficient = _require_positive("mhd_coefficient", mhd_coefficient)
    mass_amu = _require_positive("ion_mass_amu", ion_mass_amu)
    _require_positive("elongation", elongation)

    first = states[0]
    reference_s = s_parameter(eq, mass_amu=mass_amu)
    reference_radius = _state_positive(first, "R_s_m")
    reference_field = _state_positive(first, "B_ext_T")
    reference_temperature = _state_positive(first, "T_i_eV")

    points: list[FRCTiltModeTrajectoryPoint] = []
    previous_t: float | None = None
    previous_growth_rate: float | None = None
    cumulative_growth_integral = 0.0
    for state in states:
        t_s = _state_finite(state, "t_s")
        if previous_t is not None and t_s <= previous_t:
            raise ValueError("compression-state times must be strictly increasing")
        radius = _state_positive(state, "R_s_m")
        temperature = _state_positive(state, "T_i_eV")
        density = _state_positive(state, "density_m3")
        field = _state_positive(state, "B_ext_T")
        projected_s = _compressed_s_parameter(
            reference_s=reference_s,
            reference_radius_m=reference_radius,
            reference_field_t=reference_field,
            reference_temperature_eV=reference_temperature,
            radius_m=radius,
            field_t=field,
            temperature_eV=temperature,
        )
        report = _tilt_mode_report_from_values(
            s_value=projected_s,
            b_reference_t=field,
            density_peak_m3=density,
            radius_m=radius,
            elongation=elongation,
            thresholds=thresholds,
            mhd_coefficient=coefficient,
            ion_mass_amu=mass_amu,
        )
        if previous_t is not None and previous_growth_rate is not None:
            # Trapezoidal cumulative growth exponent from the interval-endpoint
            # tilt growth rates: second-order accurate in dt for the time-varying
            # compression drivers, against a first-order endpoint rectangle, and
            # exact for a growth rate that varies linearly across the interval.
            cumulative_growth_integral += (
                0.5 * (previous_growth_rate + report.growth_rate_s_inv) * (t_s - previous_t)
            )
        amplification_overflow_limited = cumulative_growth_integral > FLOAT64_LOG_MAX
        perturbation_amplification = float(np.exp(min(cumulative_growth_integral, FLOAT64_LOG_MAX)))
        points.append(
            FRCTiltModeTrajectoryPoint(
                t_s=t_s,
                R_s_m=radius,
                T_i_eV=temperature,
                density_m3=density,
                B_reference_T=field,
                report=report,
                cumulative_growth_integral=float(cumulative_growth_integral),
                perturbation_amplification=perturbation_amplification,
                amplification_overflow_limited=bool(amplification_overflow_limited),
            )
        )
        previous_t = t_s
        previous_growth_rate = report.growth_rate_s_inv
    return tuple(points)


def _rigid_body_flr_regime_from_ratio(
    ratio: float,
    thresholds: FRCTiltModeThresholds,
) -> tuple[str, bool]:
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

    field = float(np.max(np.abs(eq.B_z)))
    thresholds = FRCTiltModeThresholds() if thresholds is None else thresholds
    return _tilt_mode_report_from_values(
        s_value=s_parameter(eq, mass_amu=ion_mass_amu),
        b_reference_t=field,
        density_peak_m3=eq.density_peak_m3,
        radius_m=eq.target_separatrix_radius_m,
        elongation=elongation,
        thresholds=thresholds,
        mhd_coefficient=mhd_coefficient,
        ion_mass_amu=ion_mass_amu,
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
    checked = _require_finite(name, value)
    if checked <= 0.0:
        raise ValueError(f"{name} must be positive")
    return checked


def _require_finite(name: str, value: float) -> float:
    checked = float(value)
    if not np.isfinite(checked):
        raise ValueError(f"{name} must be finite")
    return checked


def _state_positive(state: PulsedCompressionLike, attribute: str) -> float:
    return _require_positive(f"state.{attribute}", getattr(state, attribute))


def _state_finite(state: PulsedCompressionLike, attribute: str) -> float:
    return _require_finite(f"state.{attribute}", getattr(state, attribute))


def _alfven_speed_from_values(
    *,
    b_reference_t: float,
    density_peak_m3: float,
    ion_mass_amu: float,
) -> float:
    mass = _require_positive("ion_mass_amu", ion_mass_amu) * ATOMIC_MASS_KG
    density = _require_positive("density_peak_m3", density_peak_m3) * mass
    field = abs(_require_finite("b_reference_t", b_reference_t))
    if field <= 0.0:
        raise ValueError("b_reference_t must be positive")
    return float(field / np.sqrt(MU_0 * density))


def _axial_half_length_from_values(radius_m: float, elongation: float) -> float:
    return float(
        _require_positive("radius_m", radius_m) * _require_positive("elongation", elongation)
    )


def _compressed_s_parameter(
    *,
    reference_s: float,
    reference_radius_m: float,
    reference_field_t: float,
    reference_temperature_eV: float,
    radius_m: float,
    field_t: float,
    temperature_eV: float,
) -> float:
    return float(
        _require_positive("reference_s", reference_s)
        * (
            _require_positive("radius_m", radius_m)
            / _require_positive("reference_radius_m", reference_radius_m)
        )
        * (
            _require_positive("field_t", abs(field_t))
            / _require_positive("reference_field_t", abs(reference_field_t))
        )
        * np.sqrt(
            _require_positive("reference_temperature_eV", reference_temperature_eV)
            / _require_positive("temperature_eV", temperature_eV)
        )
    )


def _tilt_mode_report_from_values(
    *,
    s_value: float,
    b_reference_t: float,
    density_peak_m3: float,
    radius_m: float,
    elongation: float,
    thresholds: FRCTiltModeThresholds,
    mhd_coefficient: float,
    ion_mass_amu: float,
) -> FRCTiltModeReport:
    speed = _alfven_speed_from_values(
        b_reference_t=b_reference_t,
        density_peak_m3=density_peak_m3,
        ion_mass_amu=ion_mass_amu,
    )
    length = _axial_half_length_from_values(radius_m, elongation)
    coefficient = _require_positive("mhd_coefficient", mhd_coefficient)
    checked_s = _require_positive("s_value", s_value)
    ratio = checked_s / _require_positive("elongation", elongation)
    regime, threshold_passed = _rigid_body_flr_regime_from_ratio(ratio, thresholds)
    return FRCTiltModeReport(
        growth_rate_s_inv=float(coefficient * speed / length),
        alfven_speed_m_s=speed,
        alfven_transit_time_s=float(length / speed),
        s_parameter=checked_s,
        elongation=float(elongation),
        s_over_elongation=float(ratio),
        rigid_body_regime=regime,
        rigid_body_threshold_passed=threshold_passed,
        conservative_stable=False,
        claim_status="diagnostic_only_not_hybrid_eigenvalue_accepted",
        external_parity_status=belova_table1_acceptance_status()["status"],
    )


def _validate_thresholds(thresholds: FRCTiltModeThresholds) -> None:
    values: tuple[tuple[str, float], ...] = (
        ("diamagnetic_s_over_e", thresholds.diamagnetic_s_over_e),
        ("gyroviscous_s_over_e", thresholds.gyroviscous_s_over_e),
        ("combined_flr_s_over_e", thresholds.combined_flr_s_over_e),
    )
    checked = tuple(_require_positive(name, value) for name, value in values)
    if not (checked[0] < checked[1] < checked[2]):
        raise ValueError("rigid-body thresholds must be strictly increasing")
