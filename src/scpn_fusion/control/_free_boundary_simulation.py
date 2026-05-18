# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core - Free-Boundary Supervisory Simulation
"""Deterministic supervisory free-boundary control simulation entry point."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from scpn_fusion.control.fusion_sota_mpc import NeuralSurrogate
from scpn_fusion.control._free_boundary_control_geometry import (
    FreeBoundarySupervisoryController,
    extract_free_boundary_state,
)
from scpn_fusion.control._free_boundary_estimator import FreeBoundaryStateEstimator
from scpn_fusion.control._free_boundary_plotting import plot_free_boundary_control
from scpn_fusion.control._free_boundary_simulation_arrays import (
    build_free_boundary_history_arrays,
)
from scpn_fusion.control._free_boundary_safety_supervisor import FreeBoundarySafetySupervisor
from scpn_fusion.control._free_boundary_simulation_summary import (
    build_free_boundary_simulation_summary,
)
from scpn_fusion.control._free_boundary_supervisory_types import (
    DEFAULT_TARGET_VECTOR,
    FloatArray,
    FreeBoundaryTarget,
    _normalize_bounds,
    _normalize_mask,
    _normalize_vector,
    _require_nonnegative_int,
    _require_positive_finite,
    _require_positive_int,
    estimate_free_boundary_safety_margins,
)

try:
    from scpn_fusion.core._rust_compat import FusionKernel
except ImportError:
    from scpn_fusion.core.fusion_kernel import FusionKernel


logger = logging.getLogger(__name__)


def run_free_boundary_supervisory_simulation(
    config_file: Optional[str] = None,
    *,
    shot_length: int = 72,
    target: Optional[FreeBoundaryTarget] = None,
    control_dt_s: float = 0.05,
    disturbance_start_step: int = 18,
    disturbance_stop_step: Optional[int] = None,
    disturbance_per_step_ma: float = 0.08,
    disturbance_recovery_step: Optional[int] = None,
    disturbance_recovery_per_step_ma: float = 0.0,
    diagnostic_dropout_step: Optional[int] = None,
    diagnostic_dropout_duration_steps: int = 0,
    diagnostic_dropout_mask: Optional[np.ndarray | tuple[float, ...] | list[float]] = None,
    coil_kick_step: int = 24,
    coil_kick_vector: Optional[np.ndarray | tuple[float, ...] | list[float]] = None,
    sensor_bias_step: int = 30,
    sensor_bias_vector: Optional[np.ndarray | tuple[float, ...] | list[float]] = None,
    sensor_bias_clear_step: Optional[int] = None,
    actuator_bias_step: Optional[int] = None,
    actuator_bias_vector: Optional[np.ndarray | tuple[float, ...] | list[float]] = None,
    actuator_bias_clear_step: Optional[int] = None,
    actuator_dropout_step: Optional[int] = None,
    actuator_dropout_duration_steps: int = 0,
    actuator_dropout_mask: Optional[np.ndarray | tuple[float, ...] | list[float]] = None,
    measurement_noise_std: float = 0.0015,
    current_target_bounds: tuple[float, float] = (7.0, 10.0),
    coil_current_limits: tuple[float, float] = (-1.5, 1.5),
    coil_delta_limit: float = 0.35,
    supervisor_total_action_l1_limit: float = 4.0,
    supervisor_axis_r_bounds_m: tuple[float, float] = (5.88, 6.12),
    supervisor_axis_z_bounds_m: tuple[float, float] = (-0.12, 0.12),
    supervisor_xpoint_r_bounds_m: tuple[float, float] = (4.92, 5.10),
    supervisor_xpoint_z_bounds_m: tuple[float, float] = (-3.60, -3.36),
    supervisor_q95_floor: float = 3.5,
    supervisor_beta_n_ceiling: float = 2.6,
    supervisor_disruption_risk_ceiling: float = 0.90,
    supervisor_warning_risk_score_threshold: float = 0.88,
    supervisor_guarded_risk_score_threshold: float = 1.05,
    supervisor_alert_recovery_hold_steps: int = 6,
    supervisor_fallback_hold_steps: int = 4,
    supervisor_fallback_action_scale: float = 0.45,
    estimator_measurement_gain: float = 0.55,
    estimator_bias_gain: float = 0.08,
    estimator_bias_decay: float = 0.985,
    estimator_actuator_bias_gain: float = 0.06,
    estimator_actuator_bias_decay: float = 0.992,
    estimator_max_actuator_bias: float = 0.35,
    return_trace: bool = False,
    save_plot: bool = False,
    output_path: str = "Free_Boundary_Supervisory_Control.png",
    verbose: bool = False,
    rng_seed: int = 42,
    kernel_factory: Callable[[str], Any] = FusionKernel,
) -> dict[str, Any]:
    """Run a closed-loop free-boundary shot with estimator, controller, and safety layer."""

    steps = _require_positive_int("shot_length", shot_length)
    dt_s = _require_positive_finite("control_dt_s", control_dt_s)
    disturbance_start = _require_nonnegative_int("disturbance_start_step", disturbance_start_step)
    disturbance_stop = (
        _require_nonnegative_int("disturbance_stop_step", disturbance_stop_step)
        if disturbance_stop_step is not None
        else steps + 1
    )
    disturbance_recovery_at = (
        _require_nonnegative_int("disturbance_recovery_step", disturbance_recovery_step)
        if disturbance_recovery_step is not None
        else steps + 1
    )
    diagnostic_dropout_at = (
        _require_nonnegative_int("diagnostic_dropout_step", diagnostic_dropout_step)
        if diagnostic_dropout_step is not None
        else steps + 1
    )
    diagnostic_dropout_duration = _require_nonnegative_int(
        "diagnostic_dropout_duration_steps",
        diagnostic_dropout_duration_steps,
    )
    coil_kick_at = _require_nonnegative_int("coil_kick_step", coil_kick_step)
    bias_step = _require_nonnegative_int("sensor_bias_step", sensor_bias_step)
    sensor_bias_clear_at = (
        _require_nonnegative_int("sensor_bias_clear_step", sensor_bias_clear_step)
        if sensor_bias_clear_step is not None
        else steps + 1
    )
    actuator_bias_at = (
        _require_nonnegative_int("actuator_bias_step", actuator_bias_step)
        if actuator_bias_step is not None
        else steps + 1
    )
    actuator_bias_clear_at = (
        _require_nonnegative_int("actuator_bias_clear_step", actuator_bias_clear_step)
        if actuator_bias_clear_step is not None
        else steps + 1
    )
    actuator_dropout_at = (
        _require_nonnegative_int("actuator_dropout_step", actuator_dropout_step)
        if actuator_dropout_step is not None
        else steps + 1
    )
    actuator_dropout_duration = _require_nonnegative_int(
        "actuator_dropout_duration_steps",
        actuator_dropout_duration_steps,
    )
    measurement_noise_std = float(measurement_noise_std)
    if not np.isfinite(measurement_noise_std) or measurement_noise_std < 0.0:
        raise ValueError("measurement_noise_std must be finite and >= 0.")
    disturbance_recovery_rate = float(disturbance_recovery_per_step_ma)
    if not np.isfinite(disturbance_recovery_rate):
        raise ValueError("disturbance_recovery_per_step_ma must be finite.")
    lo_ip, hi_ip = _normalize_bounds(current_target_bounds, "current_target_bounds")
    _normalize_bounds(coil_current_limits, "coil_current_limits")
    _require_positive_finite("coil_delta_limit", coil_delta_limit)

    if config_file is None:
        repo_root = Path(__file__).resolve().parents[3]
        config_file = str(repo_root / "iter_config.json")

    rng = np.random.default_rng(int(rng_seed))
    kernel = kernel_factory(str(config_file))
    kernel.solve_equilibrium()

    target_obj = (
        target if target is not None else FreeBoundaryTarget(*DEFAULT_TARGET_VECTOR.tolist())
    )
    surrogate = NeuralSurrogate(
        n_coils=len(kernel.cfg["coils"]),
        n_state=4,
        verbose=verbose,
    )
    surrogate.train_on_kernel(kernel, perturbation=0.25)

    estimator = FreeBoundaryStateEstimator(
        surrogate,
        measurement_gain=estimator_measurement_gain,
        bias_gain=estimator_bias_gain,
        bias_decay=estimator_bias_decay,
        actuator_bias_gain=estimator_actuator_bias_gain,
        actuator_bias_decay=estimator_actuator_bias_decay,
        max_actuator_bias=estimator_max_actuator_bias,
    )
    controller = FreeBoundarySupervisoryController(surrogate, target_obj)
    supervisor = FreeBoundarySafetySupervisor(
        coil_current_limits=coil_current_limits,
        coil_delta_limit=coil_delta_limit,
        total_action_l1_limit=supervisor_total_action_l1_limit,
        ip_bounds_ma=current_target_bounds,
        axis_r_bounds_m=supervisor_axis_r_bounds_m,
        axis_z_bounds_m=supervisor_axis_z_bounds_m,
        xpoint_r_bounds_m=supervisor_xpoint_r_bounds_m,
        xpoint_z_bounds_m=supervisor_xpoint_z_bounds_m,
        q95_floor=supervisor_q95_floor,
        beta_n_ceiling=supervisor_beta_n_ceiling,
        disruption_risk_ceiling=supervisor_disruption_risk_ceiling,
        warning_risk_score_threshold=supervisor_warning_risk_score_threshold,
        guarded_risk_score_threshold=supervisor_guarded_risk_score_threshold,
        alert_recovery_hold_steps=supervisor_alert_recovery_hold_steps,
        fallback_hold_steps=supervisor_fallback_hold_steps,
        fallback_action_scale=supervisor_fallback_action_scale,
    )
    supervisor.reset()

    initial_state = extract_free_boundary_state(kernel)
    estimator.reset(initial_state)

    n_coils = len(kernel.cfg["coils"])
    kick = _normalize_vector(
        coil_kick_vector,
        length=n_coils,
        default=0.0,
        name="coil_kick_vector",
    )
    diagnostic_dropout_mask_arr = _normalize_mask(
        diagnostic_dropout_mask,
        length=4,
        name="diagnostic_dropout_mask",
    )
    sensor_bias = np.zeros(4, dtype=np.float64)
    bias_step_vector = _normalize_vector(
        sensor_bias_vector,
        length=4,
        default=0.0,
        name="sensor_bias_vector",
    )
    actuator_bias = np.zeros(n_coils, dtype=np.float64)
    actuator_bias_step_vector = _normalize_vector(
        actuator_bias_vector,
        length=n_coils,
        default=0.0,
        name="actuator_bias_vector",
    )
    actuator_dropout_mask_arr = _normalize_mask(
        actuator_dropout_mask,
        length=n_coils,
        name="actuator_dropout_mask",
    )

    physics_cfg = kernel.cfg.setdefault("physics", {})
    target_ip_ma = float(np.clip(physics_cfg.get("plasma_current_target", lo_ip), lo_ip, hi_ip))
    physics_cfg["plasma_current_target"] = target_ip_ma

    target_vec = target_obj.as_vector()
    last_action = np.zeros(n_coils, dtype=np.float64)
    last_applied_action = np.zeros(n_coils, dtype=np.float64)

    times: list[float] = []
    measured_states: list[FloatArray] = []
    estimated_states: list[FloatArray] = []
    true_states: list[FloatArray] = []
    bias_norm_hist: list[float] = []
    uncertainty_hist: list[float] = []
    axis_error_hist: list[float] = []
    xpoint_error_hist: list[float] = []
    action_hist: list[FloatArray] = []
    applied_action_hist: list[FloatArray] = []
    actuator_bias_hat_hist: list[FloatArray] = []
    actuator_bias_true_hist: list[FloatArray] = []
    intervention_hist: list[int] = []
    saturation_hist: list[int] = []
    failsafe_hist: list[int] = []
    degraded_hist: list[int] = []
    diagnostic_dropout_hist: list[int] = []
    actuator_dropout_hist: list[int] = []
    fallback_hist: list[int] = []
    invariant_hist: list[int] = []
    physics_guard_hist: list[int] = []
    q95_guard_hist: list[int] = []
    beta_guard_hist: list[int] = []
    risk_guard_hist: list[int] = []
    q95_hist: list[float] = []
    beta_n_hist: list[float] = []
    disruption_risk_hist: list[float] = []
    q95_margin_hist: list[float] = []
    beta_margin_hist: list[float] = []
    risk_margin_hist: list[float] = []
    alert_level_hist: list[int] = []
    requested_alert_level_hist: list[int] = []
    alert_transition_hist: list[int] = []
    recovery_transition_hist: list[int] = []
    risk_score_hist: list[float] = []
    risk_signal_hist: list[float] = []
    target_ip_hist: list[float] = []
    last_valid_measurement = initial_state.copy()

    t0 = time.perf_counter()
    for step in range(steps):
        if disturbance_start <= step < disturbance_stop:
            target_ip_ma = float(
                np.clip(target_ip_ma + float(disturbance_per_step_ma), lo_ip, hi_ip)
            )
        if step >= disturbance_recovery_at and disturbance_recovery_rate != 0.0:
            target_ip_ma = float(np.clip(target_ip_ma + disturbance_recovery_rate, lo_ip, hi_ip))
        diagnostic_dropout_active = bool(
            diagnostic_dropout_duration > 0
            and np.any(diagnostic_dropout_mask_arr)
            and diagnostic_dropout_at <= step < diagnostic_dropout_at + diagnostic_dropout_duration
        )
        actuator_dropout_active = bool(
            actuator_dropout_duration > 0
            and np.any(actuator_dropout_mask_arr)
            and actuator_dropout_at <= step < actuator_dropout_at + actuator_dropout_duration
        )
        if step == coil_kick_at and np.any(kick != 0.0):
            for idx, delta in enumerate(kick):
                kernel.cfg["coils"][idx]["current"] = float(
                    kernel.cfg["coils"][idx].get("current", 0.0)
                ) + float(delta)
        if step == bias_step:
            sensor_bias = sensor_bias + bias_step_vector
        if step == sensor_bias_clear_at:
            sensor_bias = np.zeros_like(sensor_bias)
        if step == actuator_bias_at:
            actuator_bias = actuator_bias + actuator_bias_step_vector
        if step == actuator_bias_clear_at:
            actuator_bias = np.zeros_like(actuator_bias)

        physics_cfg["plasma_current_target"] = target_ip_ma
        true_state = extract_free_boundary_state(kernel)
        measurement = true_state + sensor_bias
        if measurement_noise_std > 0.0:
            measurement = measurement + rng.normal(0.0, measurement_noise_std, size=4)
        if diagnostic_dropout_active:
            measurement = np.where(diagnostic_dropout_mask_arr, last_valid_measurement, measurement)
        else:
            last_valid_measurement = np.asarray(measurement, dtype=np.float64).copy()

        estimate = estimator.step(
            measurement,
            last_action,
            measured_actuator_action=last_applied_action,
        )
        coil_currents = np.asarray(
            [float(coil.get("current", 0.0)) for coil in kernel.cfg["coils"]],
            dtype=np.float64,
        )
        proposed_action = controller.propose_action(estimate)
        safety_margins = estimate_free_boundary_safety_margins(
            corrected_state=estimate.corrected_state,
            target_state=target_vec,
            bias_hat=estimate.bias_hat,
            coil_currents=coil_currents,
            target_ip_ma=target_ip_ma,
            q95_floor=supervisor.q95_floor,
            beta_n_ceiling=supervisor.beta_n_ceiling,
            disruption_risk_ceiling=supervisor.disruption_risk_ceiling,
            risk_signal_history=(
                np.asarray(risk_signal_hist[-31:], dtype=np.float64) if risk_signal_hist else None
            ),
        )
        risk_signal_hist.append(float(safety_margins.signal_scalar))
        predictive_action = proposed_action + estimate.actuator_bias_hat
        if actuator_dropout_active:
            predictive_action = np.where(actuator_dropout_mask_arr, 0.0, predictive_action)
        predicted_next_state = surrogate.predict(
            estimate.corrected_state,
            predictive_action,
        )
        filtered = supervisor.filter_action(
            proposed_action,
            corrected_state=estimate.corrected_state,
            target_state=target_vec,
            bias_hat=estimate.bias_hat,
            coil_currents=coil_currents,
            target_ip_ma=target_ip_ma,
            predicted_next_state=predicted_next_state,
            safety_margins=safety_margins,
            diagnostic_dropout_active=diagnostic_dropout_active,
            actuator_dropout_active=actuator_dropout_active,
        )

        effective_action = np.asarray(filtered.action, dtype=np.float64)
        if actuator_dropout_active:
            effective_action = np.where(actuator_dropout_mask_arr, 0.0, effective_action)
        applied_currents = np.clip(
            coil_currents + effective_action + actuator_bias, *coil_current_limits
        )
        applied_action = applied_currents - coil_currents
        for idx, current in enumerate(applied_currents):
            kernel.cfg["coils"][idx]["current"] = float(current)
        target_ip_ma = float(filtered.safe_target_ip_ma)
        physics_cfg["plasma_current_target"] = target_ip_ma
        kernel.solve_equilibrium()

        post_state = extract_free_boundary_state(kernel)
        axis_error = float(np.linalg.norm(post_state[:2] - target_vec[:2]))
        xpoint_error = float(np.linalg.norm(post_state[2:] - target_vec[2:]))

        times.append(float(step * dt_s))
        measured_states.append(np.asarray(measurement, dtype=np.float64))
        estimated_states.append(np.asarray(estimate.corrected_state, dtype=np.float64))
        true_states.append(post_state)
        bias_norm_hist.append(float(np.linalg.norm(estimate.bias_hat)))
        uncertainty_hist.append(float(estimate.uncertainty_norm))
        axis_error_hist.append(axis_error)
        xpoint_error_hist.append(xpoint_error)
        action_hist.append(np.asarray(filtered.action, dtype=np.float64))
        applied_action_hist.append(np.asarray(applied_action, dtype=np.float64))
        actuator_bias_hat_hist.append(np.asarray(estimate.actuator_bias_hat, dtype=np.float64))
        actuator_bias_true_hist.append(np.asarray(actuator_bias, dtype=np.float64))
        intervention_hist.append(int(filtered.intervention_active))
        saturation_hist.append(int(filtered.saturation_active))
        failsafe_hist.append(int(filtered.failsafe_active))
        degraded_hist.append(int(filtered.degraded_mode_active))
        diagnostic_dropout_hist.append(int(filtered.diagnostic_dropout_active))
        actuator_dropout_hist.append(int(filtered.actuator_dropout_active))
        fallback_hist.append(int(filtered.fallback_mode_active))
        invariant_hist.append(int(filtered.invariant_violation_active))
        physics_guard_hist.append(int(filtered.physics_guard_active))
        q95_guard_hist.append(int(filtered.q95_guard_active))
        beta_guard_hist.append(int(filtered.beta_guard_active))
        risk_guard_hist.append(int(filtered.risk_guard_active))
        q95_hist.append(float(filtered.q95))
        beta_n_hist.append(float(filtered.beta_n))
        disruption_risk_hist.append(float(filtered.disruption_risk))
        q95_margin_hist.append(float(filtered.q95_margin))
        beta_margin_hist.append(float(filtered.beta_margin))
        risk_margin_hist.append(float(filtered.risk_margin))
        alert_level_hist.append(int(filtered.alert_level))
        requested_alert_level_hist.append(int(filtered.requested_alert_level))
        alert_transition_hist.append(int(filtered.alert_transition_active))
        recovery_transition_hist.append(int(filtered.recovery_transition_active))
        risk_score_hist.append(float(filtered.risk_score))
        target_ip_hist.append(float(target_ip_ma))
        last_action = np.asarray(filtered.action, dtype=np.float64)
        last_applied_action = np.asarray(applied_action, dtype=np.float64)

        if verbose and step % 10 == 0:
            logger.info(
                "Step %d: axis_err=%.4f xpoint_err=%.4f bias_norm=%.4f interventions=%d",
                step,
                axis_error,
                xpoint_error,
                bias_norm_hist[-1],
                intervention_hist[-1],
            )

    runtime_s = float(time.perf_counter() - t0)
    arrays = build_free_boundary_history_arrays(locals(), steps=steps)

    plot_saved = False
    plot_error: Optional[str] = None
    if save_plot:
        plot_saved, plot_error = plot_free_boundary_control(
            time_axis=arrays["times_arr"],
            states=arrays["true_arr"],
            target=target_obj,
            actions=arrays["action_arr"],
            output_path=output_path,
        )

    return build_free_boundary_simulation_summary(
        config_file=str(config_file),
        steps=int(steps),
        runtime_s=runtime_s,
        target_ip_ma=target_ip_ma,
        kernel=kernel,
        plot_saved=plot_saved,
        plot_error=plot_error,
        return_trace=return_trace,
        **arrays,
    )
