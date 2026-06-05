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
class FaradayCompressionFluxBudget:
    """FUS-C.6 flux-budget sidecar consumed by the Faraday recovery report."""

    source_increment_checksum: float
    damping_decrement_checksum: float
    update_residual_abs_max: float
    budget_claim_status: str
    coupling_status: str


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
    coil_source_work_j: float | None
    source_energy_budget_relative_error: float | None
    source_energy_budget_passed: bool | None
    source_budget_claim_status: str
    compression_flux_budget: FaradayCompressionFluxBudget | None
    compression_flux_budget_passed: bool | None
    compression_flux_budget_claim_status: str


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


def compression_flux_budget_from_pulsed_compression(
    states: Sequence[object],
) -> FaradayCompressionFluxBudget:
    """Return aggregate FUS-C.6 flux-budget evidence for Faraday reporting."""

    if len(states) < 2:
        raise ValueError("pulsed-compression trajectory must contain at least two states")
    post_initial_states = states[1:]
    residual = max(
        _state_attr(state, "flux_update_residual_abs_max", "flux_state.update_residual_abs_max")
        for state in post_initial_states
    )
    source_checksum = sum(
        _state_attr(
            state,
            "flux_source_increment_checksum",
            "flux_state.source_increment_checksum",
        )
        for state in post_initial_states
    )
    damping_checksum = sum(
        _state_attr(
            state,
            "flux_damping_decrement_checksum",
            "flux_state.damping_decrement_checksum",
        )
        for state in post_initial_states
    )
    statuses = tuple(
        _state_str_attr(state, "flux_budget_claim_status", "flux_state.budget_claim_status")
        for state in post_initial_states
    )
    coupling_statuses = tuple(
        _state_str_attr(state, "flux_coupling_status", "flux_state.coupling_status")
        for state in post_initial_states
    )
    budget_claim_status = "passed" if all(status == "passed" for status in statuses) else "failed"
    coupling_status = (
        coupling_statuses[0]
        if all(status == coupling_statuses[0] for status in coupling_statuses)
        else "mixed_flux_coupling_status"
    )
    return FaradayCompressionFluxBudget(
        source_increment_checksum=_require_finite(
            "compression_flux_source_increment_checksum",
            source_checksum,
        ),
        damping_decrement_checksum=_require_finite(
            "compression_flux_damping_decrement_checksum",
            damping_checksum,
        ),
        update_residual_abs_max=_require_finite(
            "compression_flux_update_residual_abs_max",
            residual,
        ),
        budget_claim_status=budget_claim_status,
        coupling_status=coupling_status,
    )


def faraday_trajectory_from_voltage_driven_compression(
    result: object,
) -> tuple[FaradayRecoveryTrajectoryPoint, ...]:
    """Return Faraday trajectory samples from an FUS-C.6 voltage-driven result."""

    return faraday_trajectory_from_pulsed_compression(_sequence_attr(result, "compression"))


def compression_work_from_voltage_driven_compression(result: object) -> float:
    """Return final plasma compression work from an FUS-C.6 voltage-driven result."""

    return compression_work_from_pulsed_compression(_sequence_attr(result, "compression"))


def compression_flux_budget_from_voltage_driven_compression(
    result: object,
) -> FaradayCompressionFluxBudget:
    """Return aggregate FUS-C.6 flux-budget evidence from a voltage-driven result."""

    return compression_flux_budget_from_pulsed_compression(_sequence_attr(result, "compression"))


def coil_source_work_from_voltage_driven_compression(result: object) -> float:
    """Return final positive coil-source work from an FUS-C.6 voltage-driven result."""

    coil_circuit = _sequence_attr(result, "coil_circuit")
    if len(coil_circuit) < 2:
        raise ValueError("voltage-driven coil circuit must contain at least two samples")
    return _require_positive(
        "coil_source_work_j",
        _state_attr(coil_circuit[-1], "source_work_J", "source_work_j"),
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
    coil_source_work_j: float | None = None,
    compression_flux_budget: FaradayCompressionFluxBudget | None = None,
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
    (
        compression_work_checked,
        energy_budget_relative_error,
        energy_budget_passed,
        budget_claim_status,
    ) = _evaluate_budget(
        recovered_energy_j,
        compression_work_j,
        tolerance,
        work_name="compression_work_j",
        missing_status="blocked_missing_compression_work",
    )
    (
        coil_source_work_checked,
        source_energy_budget_relative_error,
        source_energy_budget_passed,
        source_budget_claim_status,
    ) = _evaluate_budget(
        recovered_energy_j,
        coil_source_work_j,
        tolerance,
        work_name="coil_source_work_j",
        missing_status="blocked_missing_coil_source_work",
    )
    (
        compression_flux_budget_checked,
        compression_flux_budget_passed,
        compression_flux_budget_claim_status,
    ) = _evaluate_compression_flux_budget(compression_flux_budget)

    return FaradayRecoveryReport(
        samples=tuple(samples),
        n_turns=turns,
        coil_resistance_ohm=resistance,
        recovered_energy_j=recovered_energy_j,
        flux_initial_wb=samples[0].magnetic_flux_wb,
        flux_final_wb=samples[-1].magnetic_flux_wb,
        max_abs_back_emf_v=float(np.max(np.abs([sample.back_emf_v for sample in samples]))),
        max_abs_load_current_a=float(np.max(np.abs([sample.load_current_a for sample in samples]))),
        compression_work_j=compression_work_checked,
        energy_budget_relative_error=energy_budget_relative_error,
        energy_budget_passed=energy_budget_passed,
        budget_claim_status=budget_claim_status,
        coil_source_work_j=coil_source_work_checked,
        source_energy_budget_relative_error=source_energy_budget_relative_error,
        source_energy_budget_passed=source_energy_budget_passed,
        source_budget_claim_status=source_budget_claim_status,
        compression_flux_budget=compression_flux_budget_checked,
        compression_flux_budget_passed=compression_flux_budget_passed,
        compression_flux_budget_claim_status=compression_flux_budget_claim_status,
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
        if "." in name:
            container_name, nested_name = name.split(".", 1)
            if hasattr(raw, container_name):
                container = getattr(raw, container_name)
                if hasattr(container, nested_name):
                    return cast(float, getattr(container, nested_name))
            if isinstance(raw, Mapping) and container_name in raw:
                container = raw[container_name]
                if isinstance(container, Mapping) and nested_name in container:
                    return cast(float, container[nested_name])
                if hasattr(container, nested_name):
                    return cast(float, getattr(container, nested_name))
        if hasattr(raw, name):
            return cast(float, getattr(raw, name))
        if isinstance(raw, Mapping) and name in raw:
            return cast(float, raw[name])
    joined = ", ".join(names)
    raise ValueError(f"pulsed-compression state is missing required field: {joined}")


def _state_str_attr(state: object, *names: str) -> str:
    raw = cast(Any, state)
    for name in names:
        if "." in name:
            container_name, nested_name = name.split(".", 1)
            if hasattr(raw, container_name):
                container = getattr(raw, container_name)
                if hasattr(container, nested_name):
                    value = getattr(container, nested_name)
                    if isinstance(value, str) and value:
                        return value
            if isinstance(raw, Mapping) and container_name in raw:
                container = raw[container_name]
                if isinstance(container, Mapping) and nested_name in container:
                    value = container[nested_name]
                    if isinstance(value, str) and value:
                        return value
                if hasattr(container, nested_name):
                    value = getattr(container, nested_name)
                    if isinstance(value, str) and value:
                        return value
        if hasattr(raw, name):
            value = getattr(raw, name)
            if isinstance(value, str) and value:
                return value
        if isinstance(raw, Mapping) and name in raw:
            value = raw[name]
            if isinstance(value, str) and value:
                return value
    joined = ", ".join(names)
    raise ValueError(f"pulsed-compression state is missing required field: {joined}")


def _sequence_attr(container: object, *names: str) -> Sequence[object]:
    raw = cast(Any, container)
    for name in names:
        if hasattr(raw, name):
            value = getattr(raw, name)
            if isinstance(value, Sequence) and not isinstance(value, str):
                return cast(Sequence[object], value)
            raise ValueError(f"{name} must be a sequence")
        if isinstance(raw, Mapping) and name in raw:
            value = raw[name]
            if isinstance(value, Sequence) and not isinstance(value, str):
                return cast(Sequence[object], value)
            raise ValueError(f"{name} must be a sequence")
    joined = ", ".join(names)
    raise ValueError(f"voltage-driven compression result is missing required field: {joined}")


def _evaluate_budget(
    recovered_energy_j: float,
    supplied_work_j: float | None,
    tolerance: float,
    *,
    work_name: str,
    missing_status: str,
) -> tuple[float | None, float | None, bool | None, str]:
    if supplied_work_j is None:
        return None, None, None, missing_status
    work = _require_positive(work_name, supplied_work_j)
    scale = max(abs(work), abs(recovered_energy_j), float(np.finfo(np.float64).eps))
    relative_error = abs(recovered_energy_j - work) / scale
    passed = relative_error <= tolerance
    return work, relative_error, passed, "passed" if passed else "failed"


def _evaluate_compression_flux_budget(
    budget: FaradayCompressionFluxBudget | None,
) -> tuple[FaradayCompressionFluxBudget | None, bool | None, str]:
    if budget is None:
        return None, None, "blocked_missing_compression_flux_budget"
    _require_finite("compression_flux_source_increment_checksum", budget.source_increment_checksum)
    _require_finite(
        "compression_flux_damping_decrement_checksum", budget.damping_decrement_checksum
    )
    _require_finite("compression_flux_update_residual_abs_max", budget.update_residual_abs_max)
    if not budget.budget_claim_status:
        raise ValueError("compression_flux_budget.budget_claim_status must be non-empty")
    if not budget.coupling_status:
        raise ValueError("compression_flux_budget.coupling_status must be non-empty")
    passed = budget.budget_claim_status == "passed"
    return budget, passed, budget.budget_claim_status


def _optional_finite(name: str, value: object | None) -> float | None:
    if value is None:
        return None
    return _require_finite(name, cast(float, value))


def _array_from_points(points: Sequence[FaradayRecoveryTrajectoryPoint], field: str) -> FloatArray:
    values = np.asarray([getattr(point, field) for point in points], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"trajectory {field} samples must be finite")
    return values


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
        return result

    result = np.empty_like(values)
    if values.size == 2:
        slope = (values[1] - values[0]) / (time_s[1] - time_s[0])
        result[:] = slope
        return result
    result[0] = (values[1] - values[0]) / (time_s[1] - time_s[0])
    result[-1] = (values[-1] - values[-2]) / (time_s[-1] - time_s[-2])
    result[1:-1] = (values[2:] - values[:-2]) / (time_s[2:] - time_s[:-2])
    return result


def _trapezoid(x: FloatArray, y: FloatArray) -> float:
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1]), dtype=np.float64))
