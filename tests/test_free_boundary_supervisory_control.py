# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Free-Boundary Supervisory Control Tests
"""Tests for free-boundary supervisory control runtime behavior."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.control.free_boundary_supervisory_control import (
    FreeBoundarySafetySupervisor,
    FreeBoundaryStateEstimator,
    FreeBoundaryTarget,
    estimate_free_boundary_safety_margins,
    extract_free_boundary_state,
    run_free_boundary_supervisory_simulation,
)
from scpn_fusion.control.fusion_sota_mpc import NeuralSurrogate


class _DivertedDummyKernel:
    """Deterministic diverted equilibrium response for controller tests."""

    def __init__(self, _config_file: str) -> None:
        self.cfg = {
            "physics": {"plasma_current_target": 7.0},
            "coils": [
                {"name": "PF1", "current": 0.0},
                {"name": "PF2", "current": 0.0},
                {"name": "PF3", "current": 0.0},
                {"name": "PF4", "current": 0.0},
            ],
        }
        self.R = np.linspace(5.75, 6.25, 41)
        self.Z = np.linspace(-0.45, 0.45, 41)
        self.NR = len(self.R)
        self.NZ = len(self.Z)
        self.dR = float(self.R[1] - self.R[0])
        self.dZ = float(self.Z[1] - self.Z[0])
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)
        self.Psi = np.zeros((self.NZ, self.NR), dtype=np.float64)
        self._x_point = (5.02, -3.48)
        self.solve_equilibrium()

    def solve_equilibrium(self) -> None:
        i = np.asarray([float(c["current"]) for c in self.cfg["coils"]], dtype=np.float64)
        ip = float(self.cfg["physics"]["plasma_current_target"])
        radial_drive = 0.95 * i[2] - 0.42 * i[1] + 0.16 * i[3]
        vertical_drive = 0.82 * i[3] - 0.68 * i[0] + 0.18 * i[2]
        divertor_drive_r = 0.74 * i[1] - 0.38 * i[0] + 0.12 * i[2]
        divertor_drive_z = 0.88 * i[3] - 0.52 * i[2] + 0.10 * i[0]

        current_shift = ip - 7.0
        center_r = 6.0 + 0.07 * np.tanh(radial_drive / 0.75) + 0.010 * current_shift
        center_z = 0.0 + 0.06 * np.tanh(vertical_drive / 0.75) - 0.006 * current_shift
        x_r = 5.02 + 0.05 * np.tanh(divertor_drive_r / 0.70) + 0.006 * current_shift
        x_z = -3.48 + 0.06 * np.tanh(divertor_drive_z / 0.72) - 0.010 * current_shift

        ir = int(np.argmin(np.abs(self.R - center_r)))
        iz = int(np.argmin(np.abs(self.Z - center_z)))
        self.Psi.fill(-1.0)
        self.Psi[iz, ir] = 1.0 + 0.005 * ip
        self._x_point = (float(x_r), float(x_z))

    def find_x_point(self, _psi: np.ndarray) -> tuple[tuple[float, float], float]:
        return self._x_point, 0.0


def test_extract_free_boundary_state_returns_expected_geometry_vector() -> None:
    kernel = _DivertedDummyKernel("dummy.json")
    state = extract_free_boundary_state(kernel)
    assert state.shape == (4,)
    assert np.all(np.isfinite(state))
    assert state[0] == pytest.approx(6.0, abs=0.05)
    assert state[1] == pytest.approx(0.0, abs=0.05)
    assert state[2] == pytest.approx(5.02, abs=0.06)
    assert state[3] == pytest.approx(-3.48, abs=0.06)


def test_state_estimator_tracks_persistent_sensor_bias() -> None:
    surrogate = NeuralSurrogate(n_coils=4, n_state=4, verbose=False)
    surrogate.B = np.eye(4, dtype=np.float64)
    estimator = FreeBoundaryStateEstimator(
        surrogate,
        measurement_gain=0.6,
        bias_gain=0.12,
        bias_decay=0.99,
    )
    estimator.reset(np.array([6.0, 0.0, 5.02, -3.48], dtype=np.float64))

    applied_action = np.zeros(4, dtype=np.float64)
    bias = np.array([0.02, -0.015, 0.03, -0.02], dtype=np.float64)
    for _ in range(40):
        estimate = estimator.step(estimator.corrected_state + bias, applied_action)

    assert np.linalg.norm(estimate.bias_hat) > 0.02
    assert np.all(np.isfinite(estimate.corrected_state))


def test_state_estimator_tracks_persistent_actuator_bias() -> None:
    surrogate = NeuralSurrogate(n_coils=4, n_state=4, verbose=False)
    surrogate.B = np.eye(4, dtype=np.float64)
    estimator = FreeBoundaryStateEstimator(
        surrogate,
        measurement_gain=0.55,
        bias_gain=0.05,
        bias_decay=0.99,
        actuator_bias_gain=0.10,
        actuator_bias_decay=0.995,
        max_actuator_bias=0.25,
    )
    plant_state = np.array([6.0, 0.0, 5.02, -3.48], dtype=np.float64)
    estimator.reset(plant_state)

    commanded_action = np.array([0.06, -0.05, 0.04, -0.03], dtype=np.float64)
    actuator_bias = np.array([0.08, -0.07, 0.05, -0.04], dtype=np.float64)
    for _ in range(45):
        plant_state = surrogate.predict(plant_state, commanded_action + actuator_bias)
        estimate = estimator.step(
            plant_state,
            commanded_action,
            measured_actuator_action=commanded_action + actuator_bias,
        )

    assert np.linalg.norm(estimate.actuator_bias_hat) > 0.03
    assert np.all(np.isfinite(estimate.actuator_bias_hat))
    assert np.linalg.norm(estimate.actuator_bias_hat - actuator_bias) < 0.12


def test_estimate_free_boundary_safety_margins_is_directional() -> None:
    safe = estimate_free_boundary_safety_margins(
        corrected_state=np.array([6.0, 0.0, 5.02, -3.48], dtype=np.float64),
        target_state=FreeBoundaryTarget(6.0, 0.0, 5.02, -3.48).as_vector(),
        bias_hat=np.zeros(4, dtype=np.float64),
        coil_currents=np.array([0.05, -0.04, 0.03, -0.02], dtype=np.float64),
        target_ip_ma=7.2,
        q95_floor=4.0,
        beta_n_ceiling=2.2,
        disruption_risk_ceiling=0.80,
    )
    stressed = estimate_free_boundary_safety_margins(
        corrected_state=np.array([6.08, 0.06, 5.10, -3.36], dtype=np.float64),
        target_state=FreeBoundaryTarget(6.0, 0.0, 5.02, -3.48).as_vector(),
        bias_hat=np.array([0.03, 0.02, 0.02, 0.01], dtype=np.float64),
        coil_currents=np.array([1.2, -1.1, 0.9, -0.8], dtype=np.float64),
        target_ip_ma=9.5,
        q95_floor=4.0,
        beta_n_ceiling=2.2,
        disruption_risk_ceiling=0.80,
        risk_signal_history=[0.42, 0.47, 0.53],
    )
    assert safe.q95 > stressed.q95
    assert safe.beta_n < stressed.beta_n
    assert safe.disruption_risk < stressed.disruption_risk
    assert stressed.q95_margin < safe.q95_margin
    assert stressed.beta_margin < safe.beta_margin
    assert stressed.risk_margin < safe.risk_margin


def test_safety_supervisor_clips_action_and_backs_off_current_target() -> None:
    supervisor = FreeBoundarySafetySupervisor(
        coil_current_limits=(-1.0, 1.0),
        coil_delta_limit=0.3,
        total_action_l1_limit=0.8,
        ip_bounds_ma=(7.0, 9.0),
        severe_axis_error_m=0.05,
        severe_xpoint_error_m=0.07,
        severe_bias_norm_m=0.04,
        ip_backoff_ma=0.2,
        axis_r_bounds_m=(5.95, 6.05),
        axis_z_bounds_m=(-0.06, 0.06),
        xpoint_r_bounds_m=(4.99, 5.05),
        xpoint_z_bounds_m=(-3.52, -3.44),
    )
    result = supervisor.filter_action(
        np.array([1.0, -1.0, 0.8, -0.8], dtype=np.float64),
        corrected_state=np.array([6.15, 0.08, 5.18, -3.34], dtype=np.float64),
        target_state=FreeBoundaryTarget(6.0, 0.0, 5.02, -3.48).as_vector(),
        bias_hat=np.array([0.03, 0.02, 0.02, 0.01], dtype=np.float64),
        coil_currents=np.array([0.9, -0.9, 0.85, -0.85], dtype=np.float64),
        target_ip_ma=8.0,
        predicted_next_state=np.array([6.10, 0.07, 5.10, -3.38], dtype=np.float64),
        safety_margins=estimate_free_boundary_safety_margins(
            corrected_state=np.array([6.15, 0.08, 5.18, -3.34], dtype=np.float64),
            target_state=FreeBoundaryTarget(6.0, 0.0, 5.02, -3.48).as_vector(),
            bias_hat=np.array([0.03, 0.02, 0.02, 0.01], dtype=np.float64),
            coil_currents=np.array([0.9, -0.9, 0.85, -0.85], dtype=np.float64),
            target_ip_ma=8.0,
            q95_floor=4.2,
            beta_n_ceiling=2.0,
            disruption_risk_ceiling=0.70,
            risk_signal_history=[0.45, 0.50, 0.58],
        ),
    )

    assert np.max(np.abs(result.action)) <= 0.3 + 1e-12
    assert np.max(np.abs(result.predicted_currents)) <= 1.0 + 1e-12
    assert result.intervention_active is True
    assert result.failsafe_active is True
    assert result.fallback_mode_active is True
    assert result.invariant_violation_active is True
    assert result.physics_guard_active is True
    assert result.q95_guard_active is True
    assert result.beta_guard_active is True
    assert result.risk_guard_active is False
    assert result.alert_level == 3
    assert result.requested_alert_level == 3
    assert result.alert_mode == "fallback"
    assert result.alert_transition_active is True
    assert result.recovery_transition_active is False
    assert result.risk_score > 1.35
    assert result.q95_margin < 0.0
    assert result.beta_margin < 0.0
    assert result.risk_margin > 0.0
    assert result.safe_target_ip_ma < 8.0
    assert "failsafe_trip" in result.reasons
    assert "fallback_latched" in result.reasons
    assert "q95_guard" in result.reasons
    assert result.reasons


def test_safety_supervisor_alert_policy_recovers_one_level_at_a_time() -> None:
    supervisor = FreeBoundarySafetySupervisor(
        coil_current_limits=(-1.0, 1.0),
        coil_delta_limit=0.3,
        total_action_l1_limit=0.8,
        ip_bounds_ma=(7.0, 9.0),
        alert_recovery_hold_steps=2,
    )
    target_state = FreeBoundaryTarget(6.0, 0.0, 5.02, -3.48).as_vector()
    corrected_state = np.array([6.04, 0.02, 5.07, -3.43], dtype=np.float64)
    bias_hat = np.array([0.01, -0.008, 0.01, -0.008], dtype=np.float64)
    coil_currents = np.array([0.45, -0.38, 0.31, -0.27], dtype=np.float64)
    proposed_action = np.array([0.10, -0.09, 0.08, -0.07], dtype=np.float64)

    stressed = estimate_free_boundary_safety_margins(
        corrected_state=corrected_state,
        target_state=target_state,
        bias_hat=bias_hat,
        coil_currents=coil_currents,
        target_ip_ma=8.3,
        q95_floor=4.32,
        beta_n_ceiling=1.95,
        disruption_risk_ceiling=0.26,
        risk_signal_history=[0.42, 0.49, 0.55],
    )
    fallback = supervisor.filter_action(
        proposed_action,
        corrected_state=corrected_state,
        target_state=target_state,
        bias_hat=bias_hat,
        coil_currents=coil_currents,
        target_ip_ma=8.3,
        predicted_next_state=corrected_state,
        safety_margins=stressed,
    )
    assert fallback.alert_level == 3
    assert fallback.fallback_mode_active is True

    recovery = estimate_free_boundary_safety_margins(
        corrected_state=np.array([6.01, 0.01, 5.04, -3.46], dtype=np.float64),
        target_state=target_state,
        bias_hat=np.zeros(4, dtype=np.float64),
        coil_currents=np.array([0.10, -0.08, 0.07, -0.05], dtype=np.float64),
        target_ip_ma=7.2,
        q95_floor=4.0,
        beta_n_ceiling=2.25,
        disruption_risk_ceiling=0.26,
        risk_signal_history=[0.08, 0.09, 0.10],
    )
    step1 = supervisor.filter_action(
        proposed_action,
        corrected_state=np.array([6.01, 0.01, 5.04, -3.46], dtype=np.float64),
        target_state=target_state,
        bias_hat=np.zeros(4, dtype=np.float64),
        coil_currents=np.array([0.10, -0.08, 0.07, -0.05], dtype=np.float64),
        target_ip_ma=7.2,
        predicted_next_state=np.array([6.01, 0.01, 5.04, -3.46], dtype=np.float64),
        safety_margins=recovery,
    )
    step2 = supervisor.filter_action(
        proposed_action,
        corrected_state=np.array([6.01, 0.01, 5.04, -3.46], dtype=np.float64),
        target_state=target_state,
        bias_hat=np.zeros(4, dtype=np.float64),
        coil_currents=np.array([0.10, -0.08, 0.07, -0.05], dtype=np.float64),
        target_ip_ma=7.2,
        predicted_next_state=np.array([6.01, 0.01, 5.04, -3.46], dtype=np.float64),
        safety_margins=recovery,
    )
    step3 = supervisor.filter_action(
        proposed_action,
        corrected_state=np.array([6.01, 0.01, 5.04, -3.46], dtype=np.float64),
        target_state=target_state,
        bias_hat=np.zeros(4, dtype=np.float64),
        coil_currents=np.array([0.10, -0.08, 0.07, -0.05], dtype=np.float64),
        target_ip_ma=7.2,
        predicted_next_state=np.array([6.01, 0.01, 5.04, -3.46], dtype=np.float64),
        safety_margins=recovery,
    )
    step4 = supervisor.filter_action(
        proposed_action,
        corrected_state=np.array([6.01, 0.01, 5.04, -3.46], dtype=np.float64),
        target_state=target_state,
        bias_hat=np.zeros(4, dtype=np.float64),
        coil_currents=np.array([0.10, -0.08, 0.07, -0.05], dtype=np.float64),
        target_ip_ma=7.2,
        predicted_next_state=np.array([6.01, 0.01, 5.04, -3.46], dtype=np.float64),
        safety_margins=recovery,
    )

    assert step1.alert_level == 3
    assert step2.alert_level == 2
    assert step2.recovery_transition_active is True
    assert step2.alert_mode == "guarded"
    assert step3.alert_level == 2
    assert step4.alert_level == 1
    assert step4.recovery_transition_active is True
    assert step4.alert_mode == "warning"


def test_safety_supervisor_enters_degraded_mode_on_dropout_faults() -> None:
    supervisor = FreeBoundarySafetySupervisor(
        coil_current_limits=(-1.0, 1.0),
        coil_delta_limit=0.3,
        total_action_l1_limit=0.8,
        ip_bounds_ma=(7.0, 9.0),
    )
    result = supervisor.filter_action(
        np.array([0.18, -0.16, 0.14, -0.12], dtype=np.float64),
        corrected_state=np.array([6.01, 0.01, 5.04, -3.46], dtype=np.float64),
        target_state=FreeBoundaryTarget(6.0, 0.0, 5.02, -3.48).as_vector(),
        bias_hat=np.zeros(4, dtype=np.float64),
        coil_currents=np.array([0.10, -0.08, 0.07, -0.05], dtype=np.float64),
        target_ip_ma=7.4,
        predicted_next_state=np.array([6.01, 0.01, 5.04, -3.46], dtype=np.float64),
        safety_margins=estimate_free_boundary_safety_margins(
            corrected_state=np.array([6.01, 0.01, 5.04, -3.46], dtype=np.float64),
            target_state=FreeBoundaryTarget(6.0, 0.0, 5.02, -3.48).as_vector(),
            bias_hat=np.zeros(4, dtype=np.float64),
            coil_currents=np.array([0.10, -0.08, 0.07, -0.05], dtype=np.float64),
            target_ip_ma=7.4,
            q95_floor=3.0,
            beta_n_ceiling=3.0,
            disruption_risk_ceiling=0.95,
        ),
        diagnostic_dropout_active=True,
        actuator_dropout_active=True,
    )

    assert result.degraded_mode_active is True
    assert result.diagnostic_dropout_active is True
    assert result.actuator_dropout_active is True
    assert result.alert_level == 3
    assert result.alert_mode == "fallback"
    assert "diagnostic_dropout_guard" in result.reasons
    assert "actuator_loss_guard" in result.reasons


def test_run_free_boundary_supervisory_simulation_is_finite_and_constrained() -> None:
    summary = run_free_boundary_supervisory_simulation(
        config_file="dummy.json",
        shot_length=64,
        target=FreeBoundaryTarget(6.0, 0.0, 5.02, -3.48),
        disturbance_start_step=16,
        disturbance_per_step_ma=0.08,
        coil_kick_step=24,
        coil_kick_vector=(0.20, -0.18, 0.16, -0.14),
        sensor_bias_step=30,
        sensor_bias_vector=(0.016, -0.014, 0.022, -0.019),
        measurement_noise_std=0.0012,
        current_target_bounds=(7.0, 10.0),
        coil_current_limits=(-1.5, 1.5),
        coil_delta_limit=0.35,
        save_plot=False,
        verbose=False,
        rng_seed=42,
        kernel_factory=_DivertedDummyKernel,
    )
    assert summary["config_path"] == "dummy.json"
    assert summary["steps"] == 64
    assert summary["plot_saved"] is False
    assert summary["plot_error"] is None
    assert np.isfinite(summary["runtime_seconds"])
    assert summary["max_abs_action"] <= 0.35 + 1e-9
    assert summary["max_abs_coil_current"] <= 1.5 + 1e-9
    assert summary["supervisor_intervention_count"] >= 1
    assert summary["p95_axis_error_m"] <= 0.10
    assert summary["p95_xpoint_error_m"] <= 0.13
    assert summary["stabilization_rate"] >= 0.80
    assert summary["mean_estimation_error_m"] <= 0.05


def test_run_free_boundary_supervisory_simulation_is_deterministic() -> None:
    kwargs = dict(
        config_file="dummy.json",
        shot_length=56,
        target=FreeBoundaryTarget(6.0, 0.0, 5.02, -3.48),
        disturbance_start_step=14,
        disturbance_per_step_ma=0.07,
        coil_kick_step=22,
        coil_kick_vector=(0.18, -0.16, 0.15, -0.12),
        sensor_bias_step=28,
        sensor_bias_vector=(0.014, -0.012, 0.020, -0.017),
        measurement_noise_std=0.0010,
        current_target_bounds=(7.0, 9.5),
        coil_current_limits=(-1.4, 1.4),
        coil_delta_limit=0.32,
        save_plot=False,
        verbose=False,
        rng_seed=11,
        kernel_factory=_DivertedDummyKernel,
    )
    a = run_free_boundary_supervisory_simulation(**kwargs)
    b = run_free_boundary_supervisory_simulation(**kwargs)
    for key in (
        "final_target_ip_ma",
        "final_r_axis",
        "final_z_axis",
        "final_xpoint_r",
        "final_xpoint_z",
        "p95_axis_error_m",
        "p95_xpoint_error_m",
        "stabilization_rate",
        "mean_estimation_error_m",
        "supervisor_intervention_count",
        "saturation_event_count",
        "max_abs_action",
        "max_abs_coil_current",
    ):
        assert a[key] == pytest.approx(b[key], rel=0.0, abs=0.0)


def test_run_free_boundary_supervisory_simulation_can_return_replay_trace() -> None:
    summary = run_free_boundary_supervisory_simulation(
        config_file="dummy.json",
        shot_length=24,
        target=FreeBoundaryTarget(6.0, 0.0, 5.02, -3.48),
        save_plot=False,
        verbose=False,
        return_trace=True,
        actuator_bias_step=12,
        actuator_bias_vector=(0.04, -0.03, 0.02, -0.01),
        kernel_factory=_DivertedDummyKernel,
    )
    trace = summary["trace"]
    assert isinstance(summary["replay_signature"], str)
    assert len(summary["replay_signature"]) == 16
    assert summary["failsafe_trip_count"] >= 0
    assert summary["degraded_mode_count"] >= 0
    assert summary["diagnostic_dropout_count"] >= 0
    assert summary["actuator_dropout_count"] >= 0
    assert summary["fallback_mode_count"] >= 0
    assert summary["invariant_violation_count"] >= 0
    assert summary["physics_guard_count"] >= 0
    assert summary["q95_guard_count"] >= 0
    assert summary["beta_guard_count"] >= 0
    assert summary["risk_guard_count"] >= 0
    assert summary["warning_mode_count"] >= 0
    assert summary["guarded_mode_count"] >= 0
    assert summary["peak_alert_level"] >= 0
    assert summary["final_alert_level"] >= 0
    assert summary["alert_transition_count"] >= 0
    assert summary["recovery_transition_count"] >= 0
    assert summary["max_risk_score"] >= 0.0
    assert summary["min_q95"] >= 0.0
    assert summary["max_beta_n"] >= 0.0
    assert summary["max_disruption_risk"] >= 0.0
    assert summary["min_target_ip_ma"] >= 0.0
    assert summary["max_target_ip_ma"] >= summary["min_target_ip_ma"]
    assert summary["mean_actuator_bias_estimation_error"] >= 0.0
    assert summary["max_uncertainty_norm"] >= 0.0
    assert summary["max_action_l1"] >= 0.0
    assert len(trace["time_s"]) == 24
    assert len(trace["true_states"]) == 24
    assert len(trace["estimated_states"]) == 24
    assert len(trace["actions"]) == 24
    assert len(trace["applied_actions"]) == 24
    assert len(trace["supervisor_interventions"]) == 24
    assert len(trace["saturation_events"]) == 24
    assert len(trace["failsafe_trips"]) == 24
    assert len(trace["degraded_mode"]) == 24
    assert len(trace["diagnostic_dropout"]) == 24
    assert len(trace["actuator_dropout"]) == 24
    assert len(trace["fallback_mode"]) == 24
    assert len(trace["invariant_violations"]) == 24
    assert len(trace["physics_guard"]) == 24
    assert len(trace["q95_guard"]) == 24
    assert len(trace["beta_guard"]) == 24
    assert len(trace["risk_guard"]) == 24
    assert len(trace["q95"]) == 24
    assert len(trace["beta_n"]) == 24
    assert len(trace["disruption_risk"]) == 24
    assert len(trace["q95_margin"]) == 24
    assert len(trace["beta_margin"]) == 24
    assert len(trace["risk_margin"]) == 24
    assert len(trace["alert_level"]) == 24
    assert len(trace["requested_alert_level"]) == 24
    assert len(trace["alert_transitions"]) == 24
    assert len(trace["recovery_transitions"]) == 24
    assert len(trace["risk_score"]) == 24
    assert len(trace["target_ip_ma"]) == 24
    assert len(trace["actuator_bias_hat"]) == 24
    assert len(trace["actuator_bias_true"]) == 24
    assert len(trace["uncertainty_norm"]) == 24


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"shot_length": 0}, "shot_length"),
        ({"control_dt_s": 0.0}, "control_dt_s"),
        ({"measurement_noise_std": -1.0}, "measurement_noise_std"),
        ({"current_target_bounds": (8.0, 8.0)}, "current_target_bounds"),
        ({"coil_current_limits": (1.0, -1.0)}, "coil_current_limits"),
        ({"actuator_bias_step": -1}, "actuator_bias_step"),
        ({"coil_kick_vector": (0.0, 0.0)}, "coil_kick_vector"),
        ({"sensor_bias_vector": (0.0, 0.0, 0.0)}, "sensor_bias_vector"),
        ({"actuator_bias_vector": (0.0, 0.0, 0.0)}, "actuator_bias_vector"),
    ],
)
def test_run_free_boundary_supervisory_simulation_rejects_invalid_inputs(
    kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        run_free_boundary_supervisory_simulation(
            config_file="dummy.json",
            save_plot=False,
            verbose=False,
            kernel_factory=_DivertedDummyKernel,
            **kwargs,
        )
