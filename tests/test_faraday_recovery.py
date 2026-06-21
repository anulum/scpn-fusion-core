# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Faraday Recovery Tests
"""Tests for Faraday back-EMF recovery energy and compression-sidecar evidence."""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from scpn_fusion.core import (
    CoilGeometry,
    FaradayRecoveryTrajectoryPoint,
    RigidRotorFRCInputs,
    coil_source_work_from_voltage_driven_compression,
    compression_flux_budget_from_pulsed_compression,
    compression_flux_budget_from_voltage_driven_compression,
    compression_trajectory_diagnostics_from_pulsed_compression,
    compression_trajectory_diagnostics_from_voltage_driven_compression,
    compression_work_from_pulsed_compression,
    compression_work_from_voltage_driven_compression,
    faraday_back_emf,
    faraday_back_emf_from_values,
    faraday_trajectory_from_pulsed_compression,
    faraday_trajectory_from_voltage_driven_compression,
    initial_pulsed_compression_state,
    integrated_recovery_energy,
    magnetic_flux_wb,
    run_pulsed_compression,
    run_voltage_driven_pulsed_compression,
    solve_frc_equilibrium,
)
from scpn_fusion.core.frc_rigid_rotor import ELEMENTARY_CHARGE_C, MU_0
from scpn_fusion.core.pulsed_compression import PulsedCompressionConfig


def test_constant_radius_and_field_have_zero_back_emf() -> None:
    """Constant radius and field produce zero Faraday back-EMF."""
    emf = faraday_back_emf(
        lambda _t: 0.20,
        lambda _t: 20.0,
        12,
        1.0e-6,
        dR_s_dt=lambda _t: 0.0,
        dB_ext_dt=lambda _t: 0.0,
    )

    assert emf == pytest.approx(0.0, abs=0.0)


def test_closed_form_constant_field_radial_expansion() -> None:
    """Recovery energy matches the closed form for constant-field radial expansion."""
    emf = faraday_back_emf_from_values(
        separatrix_radius_m=0.20,
        b_ext_t=20.0,
        d_radius_dt_m_s=1.5e4,
        d_b_ext_dt_t_s=0.0,
        N_turns=8,
    )

    expected = -8.0 * math.pi * (2.0 * 20.0 * 0.20 * 1.5e4)
    assert emf == pytest.approx(expected, rel=1.0e-15)


def test_callable_finite_difference_matches_linear_history() -> None:
    """Callable-history finite differencing matches a linear radius history."""
    radius_0 = 0.18
    speed = -2.0e4
    field_0 = 19.0
    field_rate = 3.0e6
    time_s = 2.5e-7

    emf = faraday_back_emf(
        lambda t: radius_0 + speed * t,
        lambda t: field_0 + field_rate * t,
        5,
        time_s,
        finite_difference_dt_s=1.0e-10,
    )
    expected = faraday_back_emf_from_values(
        radius_0 + speed * time_s,
        field_0 + field_rate * time_s,
        speed,
        field_rate,
        5,
    )

    assert emf == pytest.approx(expected, rel=1.0e-9)


def test_integrated_recovery_energy_matches_analytical_linear_radius_case() -> None:
    """Integrated recovery energy matches the analytical linear-radius case."""
    turns = 6
    resistance = 0.08
    b_ext = 20.0
    radius_0 = 0.15
    speed = 4.0e3
    duration = 1.0e-6
    times = np.linspace(0.0, duration, 257)
    trajectory = [
        FaradayRecoveryTrajectoryPoint(
            t_s=float(t), separatrix_radius_m=radius_0 + speed * t, b_ext_t=b_ext
        )
        for t in times
    ]

    report = integrated_recovery_energy(trajectory, turns, resistance)

    coefficient = turns * math.pi * 2.0 * b_ext * speed
    expected = coefficient * coefficient / resistance
    expected *= ((radius_0 + speed * duration) ** 3 - radius_0**3) / (3.0 * speed)
    assert report.recovered_energy_j == pytest.approx(expected, rel=2.0e-5)
    assert report.budget_claim_status == "blocked_missing_compression_work"
    assert report.energy_budget_passed is None
    assert report.source_budget_claim_status == "blocked_missing_coil_source_work"
    assert report.source_energy_budget_passed is None
    assert report.flux_derivative_closure_passed is True
    assert report.flux_derivative_residual_linf <= 2.0e-2
    assert len(report.flux_derivative_residual_wb_s) == len(trajectory)
    assert report.max_abs_flux_rate_field_term_wb_s == pytest.approx(0.0, abs=0.0)
    assert report.max_abs_flux_rate_radial_term_wb_s > 0.0
    assert report.max_abs_flux_rate_total_wb_s == pytest.approx(
        report.max_abs_flux_rate_radial_term_wb_s
    )
    assert report.samples[-1].flux_rate_total_wb_s == pytest.approx(
        -report.samples[-1].back_emf_v / turns
    )


def test_faraday_recovery_flags_inconsistent_derivative_sidecars() -> None:
    """Inconsistent derivative sidecars are flagged."""
    trajectory = [
        FaradayRecoveryTrajectoryPoint(
            t_s=float(t),
            separatrix_radius_m=0.2 + 1.0e3 * t,
            b_ext_t=20.0,
            d_radius_dt_m_s=0.0,
            d_b_ext_dt_t_s=0.0,
        )
        for t in np.linspace(0.0, 1.0e-6, 17)
    ]

    report = integrated_recovery_energy(trajectory, 8, 0.1)

    assert report.flux_derivative_closure_passed is False
    assert report.flux_derivative_residual_linf > 0.5
    assert report.compression_flux_budget_claim_status == "blocked_missing_compression_flux_budget"
    assert report.compression_flux_budget_passed is None
    assert (
        report.compression_trajectory_diagnostics_claim_status
        == "blocked_missing_compression_trajectory_diagnostics"
    )
    assert report.compression_trajectory_diagnostics_passed is None


def test_integrated_recovery_energy_reports_budget_when_work_is_supplied() -> None:
    """A supplied work term yields a recovery budget."""
    trajectory = [
        FaradayRecoveryTrajectoryPoint(
            t_s=0.0, separatrix_radius_m=0.20, b_ext_t=20.0, d_radius_dt_m_s=0.0, d_b_ext_dt_t_s=0.0
        ),
        FaradayRecoveryTrajectoryPoint(
            t_s=1.0e-6,
            separatrix_radius_m=0.20,
            b_ext_t=20.0,
            d_radius_dt_m_s=0.0,
            d_b_ext_dt_t_s=0.0,
        ),
    ]

    report = integrated_recovery_energy(trajectory, 4, 0.1, compression_work_j=1.0e-12)

    assert report.recovered_energy_j == pytest.approx(0.0, abs=0.0)
    assert report.energy_budget_passed is False
    assert report.budget_claim_status == "failed"
    assert report.source_budget_claim_status == "blocked_missing_coil_source_work"
    assert report.compression_flux_budget_claim_status == "blocked_missing_compression_flux_budget"


def test_faraday_recovery_evaluates_pulsed_compression_work_sidecar() -> None:
    """The pulsed-compression work sidecar is evaluated into the report."""
    b_ext = 5.0
    t_i = 10_000.0
    t_e = 5_000.0
    n0 = b_ext**2 / (2.0 * MU_0) / ((t_i + t_e) * ELEMENTARY_CHARGE_C)
    equilibrium = solve_frc_equilibrium(
        RigidRotorFRCInputs(
            n0=n0,
            T_i_eV=t_i,
            T_e_eV=t_e,
            theta_dot=0.0,
            R_s=0.20,
            B_ext=b_ext,
            delta=0.02,
        ),
        np.linspace(0.0, 0.4, 65, dtype=np.float64),
    )
    coil = CoilGeometry(
        N_turns=80,
        L_coil_m=1.0,
        R_coil_m=0.35,
        L_inductance_H=2.0e-6,
        R_resistance_ohm=0.02,
        bank_voltage_max_V=20_000.0,
    )
    config = PulsedCompressionConfig(
        equilibrium=equilibrium,
        coil=coil,
        coil_current_t=lambda _t: 5.0e5,
        plasma_mass_kg=2.0e-5,
        ion_temperature_eV=t_i,
        electron_temperature_eV=t_e,
        tau_psi_s=np.inf,
    )
    states = run_pulsed_compression(initial_pulsed_compression_state(config), config, 1.0e-9, 16)

    trajectory = faraday_trajectory_from_pulsed_compression(states)
    compression_work = compression_work_from_pulsed_compression(states)
    compression_flux_budget = compression_flux_budget_from_pulsed_compression(states)
    compression_trajectory_diagnostics = compression_trajectory_diagnostics_from_pulsed_compression(
        states,
        radius_floor_m=config.min_radius_m,
    )
    report = integrated_recovery_energy(
        trajectory,
        coil.N_turns,
        coil.R_resistance_ohm,
        compression_work_j=compression_work,
        compression_flux_budget=compression_flux_budget,
        compression_trajectory_diagnostics=compression_trajectory_diagnostics,
    )

    assert len(trajectory) == len(states)
    assert trajectory[-1].separatrix_radius_m == pytest.approx(states[-1].R_s_m)
    assert trajectory[-1].d_radius_dt_m_s == pytest.approx(states[-1].dR_s_dt_m_s)
    assert compression_work > 0.0
    assert report.compression_work_j == pytest.approx(compression_work)
    assert report.energy_budget_passed is not None
    assert report.energy_budget_relative_error is not None
    assert np.isfinite(report.energy_budget_relative_error)
    assert report.budget_claim_status in {"passed", "failed"}
    assert report.source_budget_claim_status == "blocked_missing_coil_source_work"
    assert report.source_energy_budget_passed is None
    assert compression_flux_budget.budget_claim_status == "passed"
    assert compression_flux_budget.update_residual_abs_max <= 1.0e-12
    assert report.compression_flux_budget == compression_flux_budget
    assert report.compression_flux_budget_passed is True
    assert report.compression_flux_budget_claim_status == "passed"
    assert report.compression_trajectory_diagnostics == compression_trajectory_diagnostics
    assert report.compression_trajectory_diagnostics_passed is True
    assert report.compression_trajectory_diagnostics_claim_status == "passed"
    assert compression_trajectory_diagnostics.compression_ratio > 1.0
    assert isinstance(report.flux_derivative_closure_passed, bool)
    assert np.isfinite(report.flux_derivative_residual_linf)
    assert np.isfinite(report.flux_derivative_residual_l2)
    assert report.max_abs_flux_rate_total_wb_s > 0.0


def test_faraday_recovery_evaluates_voltage_driven_source_sidecars() -> None:
    """The voltage-driven source sidecars are evaluated into the report."""
    b_ext = 5.0
    t_i = 10_000.0
    t_e = 5_000.0
    n0 = b_ext**2 / (2.0 * MU_0) / ((t_i + t_e) * ELEMENTARY_CHARGE_C)
    equilibrium = solve_frc_equilibrium(
        RigidRotorFRCInputs(
            n0=n0,
            T_i_eV=t_i,
            T_e_eV=t_e,
            theta_dot=0.0,
            R_s=0.20,
            B_ext=b_ext,
            delta=0.02,
        ),
        np.linspace(0.0, 0.4, 65, dtype=np.float64),
    )
    coil = CoilGeometry(
        N_turns=80,
        L_coil_m=1.0,
        R_coil_m=0.35,
        L_inductance_H=2.0e-6,
        R_resistance_ohm=0.02,
        bank_voltage_max_V=20_000.0,
    )
    config = PulsedCompressionConfig(
        equilibrium=equilibrium,
        coil=coil,
        coil_current_t=lambda _t: 1.0,
        plasma_mass_kg=2.0e-5,
        ion_temperature_eV=t_i,
        electron_temperature_eV=t_e,
        tau_psi_s=np.inf,
    )
    result = run_voltage_driven_pulsed_compression(
        config,
        lambda _t: 20_000.0,
        1.0e-9,
        32,
        initial_current_A=5.0e5,
    )

    trajectory = faraday_trajectory_from_voltage_driven_compression(result)
    compression_work = compression_work_from_voltage_driven_compression(result)
    compression_flux_budget = compression_flux_budget_from_voltage_driven_compression(result)
    compression_trajectory_diagnostics = (
        compression_trajectory_diagnostics_from_voltage_driven_compression(
            result,
            radius_floor_m=config.min_radius_m,
        )
    )
    source_work = coil_source_work_from_voltage_driven_compression(result)
    report = integrated_recovery_energy(
        trajectory,
        coil.N_turns,
        coil.R_resistance_ohm,
        compression_work_j=compression_work,
        coil_source_work_j=source_work,
        compression_flux_budget=compression_flux_budget,
        compression_trajectory_diagnostics=compression_trajectory_diagnostics,
    )

    assert len(trajectory) == len(result.compression)
    assert compression_work == pytest.approx(result.compression[-1].compression_work_J)
    assert source_work == pytest.approx(result.coil_circuit[-1].source_work_J)
    assert source_work > 0.0
    assert report.compression_work_j == pytest.approx(compression_work)
    assert report.coil_source_work_j == pytest.approx(source_work)
    assert report.energy_budget_relative_error is not None
    assert np.isfinite(report.energy_budget_relative_error)
    assert report.source_energy_budget_relative_error is not None
    assert np.isfinite(report.source_energy_budget_relative_error)
    assert report.budget_claim_status in {"passed", "failed"}
    assert report.source_budget_claim_status in {"passed", "failed"}
    assert report.compression_flux_budget_passed is True
    assert report.compression_flux_budget_claim_status == "passed"
    assert report.compression_trajectory_diagnostics_passed is True
    assert report.compression_trajectory_diagnostics_claim_status == "passed"
    assert compression_trajectory_diagnostics.compression_ratio > 1.0
    assert isinstance(report.flux_derivative_closure_passed, bool)
    assert np.isfinite(report.flux_derivative_residual_linf)
    assert np.isfinite(report.flux_derivative_residual_l2)
    assert report.max_abs_flux_rate_total_wb_s > 0.0


def test_faraday_recovery_propagates_failed_compression_flux_budget() -> None:
    """A failed compression flux budget propagates into the report."""
    b_ext = 5.0
    t_i = 10_000.0
    t_e = 5_000.0
    n0 = b_ext**2 / (2.0 * MU_0) / ((t_i + t_e) * ELEMENTARY_CHARGE_C)
    equilibrium = solve_frc_equilibrium(
        RigidRotorFRCInputs(
            n0=n0,
            T_i_eV=t_i,
            T_e_eV=t_e,
            theta_dot=0.0,
            R_s=0.20,
            B_ext=b_ext,
            delta=0.02,
        ),
        np.linspace(0.0, 0.4, 65, dtype=np.float64),
    )
    coil = CoilGeometry(
        N_turns=80,
        L_coil_m=1.0,
        R_coil_m=0.35,
        L_inductance_H=2.0e-6,
        R_resistance_ohm=0.02,
        bank_voltage_max_V=20_000.0,
    )
    config = PulsedCompressionConfig(
        equilibrium=equilibrium,
        coil=coil,
        coil_current_t=lambda _t: 5.0e5,
        plasma_mass_kg=2.0e-5,
        ion_temperature_eV=t_i,
        electron_temperature_eV=t_e,
        tau_psi_s=np.inf,
    )
    states = list(
        run_pulsed_compression(initial_pulsed_compression_state(config), config, 1.0e-9, 4)
    )
    states[-1] = replace(
        states[-1],
        flux_budget_claim_status="failed",
        flux_update_residual_abs_max=1.0e-3,
    )
    compression_flux_budget = compression_flux_budget_from_pulsed_compression(states)
    compression_trajectory_diagnostics = compression_trajectory_diagnostics_from_pulsed_compression(
        states,
        radius_floor_m=config.min_radius_m,
    )
    report = integrated_recovery_energy(
        faraday_trajectory_from_pulsed_compression(states),
        coil.N_turns,
        coil.R_resistance_ohm,
        compression_work_j=compression_work_from_pulsed_compression(states),
        compression_flux_budget=compression_flux_budget,
        compression_trajectory_diagnostics=compression_trajectory_diagnostics,
    )

    assert compression_flux_budget.budget_claim_status == "failed"
    assert report.compression_flux_budget_passed is False
    assert report.compression_flux_budget_claim_status == "failed"
    assert compression_trajectory_diagnostics.all_flux_budgets_passed is False
    assert report.compression_trajectory_diagnostics_passed is False
    assert report.compression_trajectory_diagnostics_claim_status == "failed"


def test_faraday_recovery_inputs_fail_closed() -> None:
    """Invalid Faraday recovery inputs fail closed with a ValueError."""
    with pytest.raises(ValueError, match="positive integer"):
        faraday_back_emf_from_values(0.2, 20.0, 1.0, 0.0, 0)
    with pytest.raises(ValueError, match="positive"):
        magnetic_flux_wb(0.0, 20.0)
    with pytest.raises(ValueError, match="at least two"):
        integrated_recovery_energy([FaradayRecoveryTrajectoryPoint(0.0, 0.2, 20.0)], 2, 0.1)
    with pytest.raises(ValueError, match="strictly increasing"):
        integrated_recovery_energy(
            [
                FaradayRecoveryTrajectoryPoint(0.0, 0.2, 20.0),
                FaradayRecoveryTrajectoryPoint(0.0, 0.21, 20.0),
            ],
            2,
            0.1,
        )
    with pytest.raises(ValueError, match="all supplied or all omitted"):
        integrated_recovery_energy(
            [
                FaradayRecoveryTrajectoryPoint(0.0, 0.2, 20.0, d_radius_dt_m_s=1.0),
                FaradayRecoveryTrajectoryPoint(1.0e-6, 0.21, 20.0),
            ],
            2,
            0.1,
        )
    with pytest.raises(ValueError, match="at least two states"):
        faraday_trajectory_from_pulsed_compression([object()])
    with pytest.raises(ValueError, match="voltage-driven compression result is missing"):
        faraday_trajectory_from_voltage_driven_compression(object())
    with pytest.raises(ValueError, match="at least two samples"):
        coil_source_work_from_voltage_driven_compression({"coil_circuit": []})


# --- branch / error-path coverage for compression-sidecar adapters ---

from types import SimpleNamespace  # noqa: E402

from scpn_fusion.core.faraday_recovery import (  # noqa: E402
    _coerce_point,
    _require_finite,
    _require_positive_int,
    _sequence_attr,
    _state_attr,
    _state_str_attr,
)


def test_compression_sidecars_reject_too_short_trajectory() -> None:
    """Compression-work and flux-budget adapters require at least two states."""
    one = [SimpleNamespace(compression_work_J=1.0)]
    with pytest.raises(ValueError, match="at least two states"):
        compression_work_from_pulsed_compression(one)
    with pytest.raises(ValueError, match="at least two states"):
        compression_flux_budget_from_pulsed_compression(one)


def test_require_finite_and_positive_int_reject_bad_values() -> None:
    """The scalar validators reject non-finite and non-positive-integer inputs."""
    with pytest.raises(ValueError, match="must be finite"):
        _require_finite("x", float("nan"))
    with pytest.raises(ValueError, match="positive integer"):
        _require_positive_int("n", 0)
    with pytest.raises(ValueError, match="positive integer"):
        _require_positive_int("n", 2.5)  # type: ignore[arg-type]


def test_coerce_point_accepts_mapping_and_object() -> None:
    """_coerce_point builds a trajectory point from a mapping or an attribute object."""
    mapping = {"t_s": 0.0, "separatrix_radius_m": 0.5, "b_ext_t": 1.0}
    from_map = _coerce_point(mapping)
    assert from_map.separatrix_radius_m == 0.5

    obj = SimpleNamespace(t_s=1.0, separatrix_radius_m=0.4, b_ext_t=2.0)
    from_obj = _coerce_point(obj)
    assert from_obj.b_ext_t == 2.0

    passthrough = _coerce_point(from_map)
    assert passthrough is from_map


def test_state_attr_resolves_object_mapping_and_nested_paths() -> None:
    """_state_attr reads flat/nested attributes from objects and mappings."""
    assert _state_attr({"compression_work_j": 3.0}, "compression_work_j") == 3.0
    nested_obj = SimpleNamespace(budget=SimpleNamespace(work=4.0))
    assert _state_attr(nested_obj, "budget.work") == 4.0
    nested_map = {"budget": {"work": 5.0}}
    assert _state_attr(nested_map, "budget.work") == 5.0
    nested_map_obj = {"budget": SimpleNamespace(work=6.0)}
    assert _state_attr(nested_map_obj, "budget.work") == 6.0
    with pytest.raises(ValueError, match="missing required field"):
        _state_attr(SimpleNamespace(), "absent")


def test_state_str_attr_resolves_strings_and_rejects_missing() -> None:
    """_state_str_attr reads non-empty string fields from objects/mappings/nests."""
    assert _state_str_attr({"label": "ok"}, "label") == "ok"
    assert _state_str_attr(SimpleNamespace(label="obj"), "label") == "obj"
    nested_obj = SimpleNamespace(meta=SimpleNamespace(name="deep"))
    assert _state_str_attr(nested_obj, "meta.name") == "deep"
    nested_map = {"meta": {"name": "dmap"}}
    assert _state_str_attr(nested_map, "meta.name") == "dmap"
    nested_map_obj = {"meta": SimpleNamespace(name="dmo")}
    assert _state_str_attr(nested_map_obj, "meta.name") == "dmo"
    with pytest.raises(ValueError, match="missing required field"):
        _state_str_attr(SimpleNamespace(), "absent")


def test_sequence_attr_resolves_sequences_and_rejects_non_sequences() -> None:
    """_sequence_attr returns sequence fields and rejects scalars / missing names."""
    assert list(_sequence_attr(SimpleNamespace(states=[1, 2]), "states")) == [1, 2]
    assert list(_sequence_attr({"states": (3, 4)}, "states")) == [3, 4]
    with pytest.raises(ValueError, match="must be a sequence"):
        _sequence_attr(SimpleNamespace(states=5), "states")
    with pytest.raises(ValueError, match="must be a sequence"):
        _sequence_attr({"states": 6}, "states")
    with pytest.raises(ValueError, match="missing required field"):
        _sequence_attr(SimpleNamespace(), "absent")


from scpn_fusion.core.faraday_recovery import (  # noqa: E402
    FaradayCompressionFluxBudget,
    FaradayCompressionTrajectoryDiagnostics,
    _array_from_points,
    _evaluate_compression_flux_budget,
    _optional_finite,
    _trajectory_derivative,
    _validate_compression_trajectory_diagnostics,
)


def _budget(claim: str = "passed", coupling: str = "coupled") -> FaradayCompressionFluxBudget:
    """Build a FaradayCompressionFluxBudget with finite checksums and given statuses."""
    return FaradayCompressionFluxBudget(
        source_increment_checksum=1.0,
        damping_decrement_checksum=1.0,
        update_residual_abs_max=0.0,
        budget_claim_status=claim,
        coupling_status=coupling,
    )


def _diagnostics(**overrides: object) -> FaradayCompressionTrajectoryDiagnostics:
    """Build a valid FaradayCompressionTrajectoryDiagnostics with optional overrides."""
    base: dict[str, object] = {
        "monotonic_time": True,
        "min_radius_m": 0.1,
        "max_abs_radial_acceleration_m_s2": 1.0,
        "radius_floor_contact_count": 0,
        "radial_turning_point_count": 0,
        "compression_ratio": 2.0,
        "all_flux_budgets_passed": True,
    }
    base.update(overrides)
    return FaradayCompressionTrajectoryDiagnostics(**base)  # type: ignore[arg-type]


def test_evaluate_compression_flux_budget_requires_non_empty_statuses() -> None:
    """Empty claim/coupling statuses are rejected by the flux-budget evaluator."""
    with pytest.raises(ValueError, match="budget_claim_status must be non-empty"):
        _evaluate_compression_flux_budget(_budget(claim=""))
    with pytest.raises(ValueError, match="coupling_status must be non-empty"):
        _evaluate_compression_flux_budget(_budget(coupling=""))


def test_validate_compression_trajectory_diagnostics_rejects_negative_counters() -> None:
    """Negative acceleration/contact/turning-point counters are rejected."""
    with pytest.raises(ValueError, match="must be non-negative"):
        _validate_compression_trajectory_diagnostics(
            _diagnostics(max_abs_radial_acceleration_m_s2=-1.0)
        )
    with pytest.raises(ValueError, match="radius_floor_contact_count must be non-negative"):
        _validate_compression_trajectory_diagnostics(_diagnostics(radius_floor_contact_count=-1))
    with pytest.raises(ValueError, match="radial_turning_point_count must be non-negative"):
        _validate_compression_trajectory_diagnostics(_diagnostics(radial_turning_point_count=-1))


def test_optional_finite_returns_value_when_present() -> None:
    """_optional_finite validates and returns a present finite value."""
    assert _optional_finite("x", 2.0) == 2.0
    assert _optional_finite("x", None) is None


def test_array_and_derivative_helpers_reject_non_finite_samples() -> None:
    """Non-finite trajectory samples are rejected by the array/derivative helpers."""
    nan_pt = FaradayRecoveryTrajectoryPoint(t_s=float("nan"), separatrix_radius_m=0.5, b_ext_t=1.0)
    with pytest.raises(ValueError, match="t_s samples must be finite"):
        _array_from_points([nan_pt], "t_s")

    supplied = [
        FaradayRecoveryTrajectoryPoint(
            t_s=0.0, separatrix_radius_m=0.5, b_ext_t=1.0, d_radius_dt_m_s=float("nan")
        ),
        FaradayRecoveryTrajectoryPoint(
            t_s=1.0, separatrix_radius_m=0.6, b_ext_t=1.0, d_radius_dt_m_s=float("nan")
        ),
    ]
    time_s = np.asarray([0.0, 1.0], dtype=np.float64)
    values = np.asarray([0.5, 0.6], dtype=np.float64)
    with pytest.raises(ValueError, match="d_radius_dt_m_s samples must be finite"):
        _trajectory_derivative(supplied, "d_radius_dt_m_s", time_s, values)


def test_integrated_recovery_energy_rejects_non_positive_radius() -> None:
    """A directly-supplied trajectory point with non-positive radius is rejected."""
    pts = [
        FaradayRecoveryTrajectoryPoint(t_s=0.0, separatrix_radius_m=0.5, b_ext_t=1.0),
        FaradayRecoveryTrajectoryPoint(t_s=1.0, separatrix_radius_m=-0.5, b_ext_t=1.0),
    ]
    with pytest.raises(ValueError, match="separatrix radii must be positive"):
        integrated_recovery_energy(pts, N_turns=1, coil_resistance_ohm=1.0)
