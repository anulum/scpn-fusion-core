# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — FRC Faraday Recovery
"""Classical Faraday back-EMF and recovery-energy contract for MIF/FRC.

The implemented contract is the closed-form induction relation from the current
MIF work lane:

``EMF = -N_turns * pi * (R_s^2 dB_ext/dt + 2 B_ext R_s dR_s/dt)``.

It operates on supplied pulsed-compression trajectories. It does not generate
the unresolved FUS-C.6 Hall-MHD compression trajectory and does not fabricate a
Slough-style compression-work comparison when that trajectory/work artefact is
missing.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class FaradayRecoveryTrajectoryPoint:
    """Single supplied FRC recovery-trajectory sample."""

    t_s: float
    separatrix_radius_m: float
    b_ext_t: float
    d_radius_dt_m_s: float | None = None
    d_b_ext_dt_t_s: float | None = None


@dataclass(frozen=True)
class FaradayRecoverySample:
    """Derived Faraday recovery quantities at one trajectory sample."""

    t_s: float
    separatrix_radius_m: float
    b_ext_t: float
    d_radius_dt_m_s: float
    d_b_ext_dt_t_s: float
    magnetic_flux_wb: float
    back_emf_v: float
    load_current_a: float
    load_power_w: float


@dataclass(frozen=True)
class FaradayRecoveryReport:
    """Integrated recovery-energy result over a supplied trajectory."""

    samples: tuple[FaradayRecoverySample, ...]
    n_turns: int
    coil_resistance_ohm: float
    recovered_energy_j: float
    flux_initial_wb: float
    flux_final_wb: float
    max_abs_back_emf_v: float
    max_abs_load_current_a: float
    compression_work_j: float | None
    energy_budget_relative_error: float | None
    energy_budget_passed: bool | None
    budget_claim_status: str


def faraday_trajectory_from_pulsed_compression(
    states: Sequence[object],
) -> tuple[FaradayRecoveryTrajectoryPoint, ...]:
    """Return Faraday trajectory samples from a FUS-C.6 pulsed-compression run.

    The adapter accepts the public Python ``PulsedCompressionState`` attribute
    names and the Rust snake-case field names used by the equivalent native
    contract. It supplies the separatrix radial speed carried by the
    compression integrator and leaves ``dB_ext/dt`` to the recovery
    integrator's finite-difference path because the compression state stores
    field values, not an independent field-rate sidecar.
    """

    if len(states) < 2:
        raise ValueError("pulsed-compression trajectory must contain at least two states")
    samples = []
    for state in states:
        time_s = _state_attr(state, "t_s")
        radius_m = _state_attr(state, "R_s_m", "r_s_m")
        field_t = _state_attr(state, "B_ext_T", "b_ext_t")
        radial_speed = _state_attr(state, "dR_s_dt_m_s", "d_r_s_dt_m_s")
        samples.append(
            FaradayRecoveryTrajectoryPoint(
                t_s=_require_finite("state.t_s", time_s),
                separatrix_radius_m=_require_positive("state.R_s_m", radius_m),
                b_ext_t=_require_finite("state.B_ext_T", field_t),
                d_radius_dt_m_s=_require_finite("state.dR_s_dt_m_s", radial_speed),
                d_b_ext_dt_t_s=None,
            )
        )
    return tuple(samples)


def compression_work_from_pulsed_compression(states: Sequence[object]) -> float:
    """Return final compression work from a FUS-C.6 trajectory sidecar."""

    if len(states) < 2:
        raise ValueError("pulsed-compression trajectory must contain at least two states")
    return _require_positive(
        "compression_work_j",
        _state_attr(states[-1], "compression_work_J", "compression_work_j"),
    )


def _require_finite(name: str, value: float) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _require_positive(name: str, value: float) -> float:
    result = _require_finite(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _require_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def magnetic_flux_wb(separatrix_radius_m: float, b_ext_t: float) -> float:
    """Return linked magnetic flux ``B_ext*pi*R_s^2`` in weber per turn."""

    radius = _require_positive("separatrix_radius_m", separatrix_radius_m)
    b_ext = _require_finite("b_ext_t", b_ext_t)
    return float(b_ext * np.pi * radius * radius)


def faraday_back_emf_from_values(
    separatrix_radius_m: float,
    b_ext_t: float,
    d_radius_dt_m_s: float,
    d_b_ext_dt_t_s: float,
    N_turns: int,
) -> float:
    """Return closed-form recovery-coil back-EMF in volts."""

    turns = _require_positive_int("N_turns", N_turns)
    radius = _require_positive("separatrix_radius_m", separatrix_radius_m)
    b_ext = _require_finite("b_ext_t", b_ext_t)
    d_radius_dt = _require_finite("d_radius_dt_m_s", d_radius_dt_m_s)
    d_b_ext_dt = _require_finite("d_b_ext_dt_t_s", d_b_ext_dt_t_s)
    return float(
        -turns * np.pi * (radius * radius * d_b_ext_dt + 2.0 * b_ext * radius * d_radius_dt)
    )


def faraday_back_emf(
    R_s_t: Callable[[float], float],
    B_ext_t: Callable[[float], float],
    N_turns: int,
    t: float,
    *,
    dR_s_dt: Callable[[float], float] | None = None,
    dB_ext_dt: Callable[[float], float] | None = None,
    finite_difference_dt_s: float = 1.0e-9,
) -> float:
    """Return closed-form back-EMF from callable radius and field histories."""

    time_s = _require_finite("t", t)
    dt = _require_positive("finite_difference_dt_s", finite_difference_dt_s)
    radius = _require_positive("R_s_t(t)", R_s_t(time_s))
    b_ext = _require_finite("B_ext_t(t)", B_ext_t(time_s))
    if dR_s_dt is None:
        d_radius_dt = (R_s_t(time_s + dt) - R_s_t(time_s - dt)) / (2.0 * dt)
    else:
        d_radius_dt = dR_s_dt(time_s)
    if dB_ext_dt is None:
        d_b_ext_dt = (B_ext_t(time_s + dt) - B_ext_t(time_s - dt)) / (2.0 * dt)
    else:
        d_b_ext_dt = dB_ext_dt(time_s)
    return faraday_back_emf_from_values(radius, b_ext, d_radius_dt, d_b_ext_dt, N_turns)


def integrated_recovery_energy(
    trajectory: Sequence[FaradayRecoveryTrajectoryPoint | Mapping[str, Any] | object],
    N_turns: int,
    coil_resistance_ohm: float,
    *,
    compression_work_j: float | None = None,
    budget_tolerance: float = 0.01,
) -> FaradayRecoveryReport:
    """Integrate recoverable load energy over a supplied FRC trajectory.

    The load model is a resistive recovery coil, so instantaneous power is
    ``EMF^2 / R_load``. When a self-consistent compression-work value is not
    supplied by FUS-C.6, the energy-budget gate is reported as blocked instead
    of being inferred from synthetic data.
    """

    turns = _require_positive_int("N_turns", N_turns)
    resistance = _require_positive("coil_resistance_ohm", coil_resistance_ohm)
    tolerance = _require_positive("budget_tolerance", budget_tolerance)
    points = tuple(_coerce_point(point) for point in trajectory)
    if len(points) < 2:
        raise ValueError("trajectory must contain at least two samples")

    time_s = _array_from_points(points, "t_s")
    radius_m = _array_from_points(points, "separatrix_radius_m")
    b_ext_t = _array_from_points(points, "b_ext_t")
    if not np.all(np.diff(time_s) > 0.0):
        raise ValueError("trajectory time samples must be strictly increasing")
    if np.any(radius_m <= 0.0):
        raise ValueError("trajectory separatrix radii must be positive")

    d_radius_dt = _trajectory_derivative(points, "d_radius_dt_m_s", time_s, radius_m)
    d_b_ext_dt = _trajectory_derivative(points, "d_b_ext_dt_t_s", time_s, b_ext_t)

    samples = []
    for index, time_value in enumerate(time_s):
        flux = magnetic_flux_wb(radius_m[index], b_ext_t[index])
        emf = faraday_back_emf_from_values(
            radius_m[index],
            b_ext_t[index],
            d_radius_dt[index],
            d_b_ext_dt[index],
            turns,
        )
        current = emf / resistance
        power = emf * emf / resistance
        samples.append(
            FaradayRecoverySample(
                t_s=float(time_value),
                separatrix_radius_m=float(radius_m[index]),
                b_ext_t=float(b_ext_t[index]),
                d_radius_dt_m_s=float(d_radius_dt[index]),
                d_b_ext_dt_t_s=float(d_b_ext_dt[index]),
                magnetic_flux_wb=flux,
                back_emf_v=emf,
                load_current_a=float(current),
                load_power_w=float(power),
            )
        )

    power_w = np.asarray([sample.load_power_w for sample in samples], dtype=np.float64)
    recovered_energy_j = _trapezoid(time_s, power_w)
    if compression_work_j is None:
        energy_budget_relative_error = None
        energy_budget_passed = None
        budget_claim_status = "blocked_missing_compression_work"
    else:
        work = _require_positive("compression_work_j", compression_work_j)
        scale = max(abs(work), abs(recovered_energy_j), float(np.finfo(np.float64).eps))
        energy_budget_relative_error = abs(recovered_energy_j - work) / scale
        energy_budget_passed = energy_budget_relative_error <= tolerance
        budget_claim_status = "passed" if energy_budget_passed else "failed"

    return FaradayRecoveryReport(
        samples=tuple(samples),
        n_turns=turns,
        coil_resistance_ohm=resistance,
        recovered_energy_j=recovered_energy_j,
        flux_initial_wb=samples[0].magnetic_flux_wb,
        flux_final_wb=samples[-1].magnetic_flux_wb,
        max_abs_back_emf_v=float(np.max(np.abs([sample.back_emf_v for sample in samples]))),
        max_abs_load_current_a=float(np.max(np.abs([sample.load_current_a for sample in samples]))),
        compression_work_j=compression_work_j,
        energy_budget_relative_error=energy_budget_relative_error,
        energy_budget_passed=energy_budget_passed,
        budget_claim_status=budget_claim_status,
    )


def _coerce_point(
    point: FaradayRecoveryTrajectoryPoint | Mapping[str, Any] | object,
) -> FaradayRecoveryTrajectoryPoint:
    if isinstance(point, FaradayRecoveryTrajectoryPoint):
        return point
    if isinstance(point, Mapping):
        data = point
        return FaradayRecoveryTrajectoryPoint(
            t_s=_require_finite("t_s", data["t_s"]),
            separatrix_radius_m=_require_positive(
                "separatrix_radius_m",
                data["separatrix_radius_m"],
            ),
            b_ext_t=_require_finite("b_ext_t", data["b_ext_t"]),
            d_radius_dt_m_s=_optional_finite("d_radius_dt_m_s", data.get("d_radius_dt_m_s")),
            d_b_ext_dt_t_s=_optional_finite("d_b_ext_dt_t_s", data.get("d_b_ext_dt_t_s")),
        )
    raw = cast(Any, point)
    return FaradayRecoveryTrajectoryPoint(
        t_s=_require_finite("t_s", raw.t_s),
        separatrix_radius_m=_require_positive(
            "separatrix_radius_m",
            raw.separatrix_radius_m,
        ),
        b_ext_t=_require_finite("b_ext_t", raw.b_ext_t),
        d_radius_dt_m_s=_optional_finite(
            "d_radius_dt_m_s", getattr(point, "d_radius_dt_m_s", None)
        ),
        d_b_ext_dt_t_s=_optional_finite("d_b_ext_dt_t_s", getattr(point, "d_b_ext_dt_t_s", None)),
    )


def _state_attr(state: object, *names: str) -> float:
    raw = cast(Any, state)
    for name in names:
        if hasattr(raw, name):
            return cast(float, getattr(raw, name))
        if isinstance(raw, Mapping) and name in raw:
            return cast(float, raw[name])
    joined = ", ".join(names)
    raise ValueError(f"pulsed-compression state is missing required field: {joined}")


def _optional_finite(name: str, value: object | None) -> float | None:
    if value is None:
        return None
    return _require_finite(name, cast(float, value))


def _array_from_points(points: Sequence[FaradayRecoveryTrajectoryPoint], field: str) -> FloatArray:
    values = np.asarray([getattr(point, field) for point in points], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"trajectory {field} samples must be finite")
    return cast(FloatArray, values)


def _trajectory_derivative(
    points: Sequence[FaradayRecoveryTrajectoryPoint],
    field: str,
    time_s: FloatArray,
    values: FloatArray,
) -> FloatArray:
    supplied = [getattr(point, field) for point in points]
    have_supplied = [value is not None for value in supplied]
    if any(have_supplied) and not all(have_supplied):
        raise ValueError(f"trajectory {field} samples must be all supplied or all omitted")
    if all(have_supplied):
        result = np.asarray(cast(list[float], supplied), dtype=np.float64)
        if not np.all(np.isfinite(result)):
            raise ValueError(f"trajectory {field} samples must be finite")
        return cast(FloatArray, result)

    result = np.empty_like(values)
    if values.size == 2:
        slope = (values[1] - values[0]) / (time_s[1] - time_s[0])
        result[:] = slope
        return cast(FloatArray, result)
    result[0] = (values[1] - values[0]) / (time_s[1] - time_s[0])
    result[-1] = (values[-1] - values[-2]) / (time_s[-1] - time_s[-2])
    result[1:-1] = (values[2:] - values[:-2]) / (time_s[2:] - time_s[:-2])
    return cast(FloatArray, result)


def _trapezoid(x: FloatArray, y: FloatArray) -> float:
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1]), dtype=np.float64))
