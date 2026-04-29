# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Free-Boundary Tracking Control
"""Closed-loop free-boundary tracking without a surrogate plant.

The controller re-identifies the local coil-to-objective response directly
from repeated :class:`scpn_fusion.core.fusion_kernel.FusionKernel` solves and
applies bounded least-squares coil corrections. When configured, supervisor
rejection can ramp the coil set toward explicit safe fallback currents instead
of freezing at the previous command, and an objective-space disturbance
observer can accumulate persistent residuals without introducing a reduced-order
plant model. Deterministic objective-space sensor bias and drift can also be
applied and compensated through configuration so hidden plant performance stays
visible during calibration-fault stress tests. Fixed-step measurement latency
can be injected in the same objective space, and an extrapolating current-state
estimator can compensate that latency without replacing the full kernel.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np

from scpn_fusion.control._free_boundary_tracking_config import (
    _FreeBoundaryTrackingConfigMixin,
)
from scpn_fusion.control._free_boundary_tracking_control import (
    _FreeBoundaryTrackingControlMixin,
)
from scpn_fusion.control._free_boundary_tracking_runtime import (
    _FreeBoundaryTrackingRuntimeMixin,
)
from scpn_fusion.control._free_boundary_tracking_shot import _FreeBoundaryTrackingShotMixin
from scpn_fusion.control._free_boundary_tracking_types import FloatArray
from scpn_fusion.control.state_estimator import ExtendedKalmanFilter
from scpn_fusion.core.fusion_kernel import CoilSet, FusionKernel


class FreeBoundaryTrackingController(
    _FreeBoundaryTrackingShotMixin,
    _FreeBoundaryTrackingControlMixin,
    _FreeBoundaryTrackingRuntimeMixin,
    _FreeBoundaryTrackingConfigMixin,
):
    """Direct free-boundary controller using local coil-response identification.

    This path keeps the full Grad-Shafranov kernel in the loop instead of
    replacing it with a reduced-order plant.

    Examples
    --------
    >>> from scpn_fusion.control.free_boundary_tracking import run_free_boundary_tracking
    >>> summary = run_free_boundary_tracking(
    ...     "iter_config.json",
    ...     shot_steps=3,
    ...     gain=0.8,
    ...     verbose=False,
    ... )
    >>> summary["boundary_variant"]
    'free_boundary'
    """

    def __init__(
        self,
        config_file: str,
        *,
        kernel_factory: Callable[[str], Any] = FusionKernel,
        verbose: bool = True,
        identification_perturbation: float = 0.25,
        correction_limit: float = 2.0,
        response_regularization: float = 1e-3,
        response_refresh_steps: int = 1,
        solve_max_outer_iter: int = 10,
        solve_tol: float = 1e-4,
        objective_tolerances: dict[str, float] | None = None,
        control_dt_s: float | None = None,
        coil_actuator_tau_s: float | None = None,
        coil_slew_limits: float | list[float] | None = None,
        supervisor_limits: dict[str, float] | None = None,
        hold_steps_after_reject: int | None = None,
        state_estimator: ExtendedKalmanFilter | None = None,
    ) -> None:
        self.kernel = kernel_factory(config_file)
        self.state_estimator = state_estimator
        self.verbose = bool(verbose)
        self.identification_perturbation = float(identification_perturbation)
        if (
            not np.isfinite(self.identification_perturbation)
            or self.identification_perturbation <= 0.0
        ):
            raise ValueError("identification_perturbation must be finite and > 0.")
        self.correction_limit = float(correction_limit)
        if not np.isfinite(self.correction_limit) or self.correction_limit <= 0.0:
            raise ValueError("correction_limit must be finite and > 0.")
        self.response_regularization = float(response_regularization)
        if not np.isfinite(self.response_regularization) or self.response_regularization < 0.0:
            raise ValueError("response_regularization must be finite and >= 0.")
        self.response_refresh_steps = int(response_refresh_steps)
        if self.response_refresh_steps < 1:
            raise ValueError("response_refresh_steps must be >= 1.")
        self.solve_max_outer_iter = int(solve_max_outer_iter)
        if self.solve_max_outer_iter < 1:
            raise ValueError("solve_max_outer_iter must be >= 1.")
        self.solve_tol = float(solve_tol)
        if not np.isfinite(self.solve_tol) or self.solve_tol <= 0.0:
            raise ValueError("solve_tol must be finite and > 0.")

        build_coilset = getattr(self.kernel, "build_coilset_from_config", None)
        if not callable(build_coilset):
            raise TypeError(
                "kernel must define build_coilset_from_config() for free-boundary tracking."
            )
        self.coils = build_coilset()
        self.n_coils = int(len(self.coils.positions))
        if self.n_coils < 1:
            raise ValueError("free-boundary tracking requires at least one external coil.")

        if self.coils.current_limits is None:
            self.coil_current_limits = np.full(self.n_coils, np.inf, dtype=np.float64)
        else:
            self.coil_current_limits = np.asarray(
                self.coils.current_limits, dtype=np.float64
            ).reshape(-1)
            if self.coil_current_limits.shape != (self.n_coils,):
                raise ValueError("CoilSet.current_limits must match the number of coils.")
            if np.any(~np.isfinite(self.coil_current_limits)) or np.any(
                self.coil_current_limits <= 0.0
            ):
                raise ValueError("CoilSet.current_limits must be finite and > 0.")

        tracking_cfg = self.kernel.cfg.get("free_boundary_tracking", {})
        self.control_dt_s = self._resolve_positive_float(
            tracking_cfg.get("control_dt_s"),
            control_dt_s,
            default=0.05,
            name="control_dt_s",
        )
        self.coil_actuator_tau_s = self._resolve_positive_float(
            tracking_cfg.get("coil_actuator_tau_s"),
            coil_actuator_tau_s,
            default=1e-6,
            name="coil_actuator_tau_s",
        )
        self.coil_slew_limits = self._resolve_coil_slew_limits(
            tracking_cfg.get("coil_slew_limits"),
            coil_slew_limits,
        )
        self.hold_steps_after_reject = self._resolve_nonnegative_int(
            tracking_cfg.get("hold_steps_after_reject"),
            hold_steps_after_reject,
            default=0,
            name="hold_steps_after_reject",
        )
        self.supervisor_limits = self._resolve_supervisor_limits(
            tracking_cfg.get("supervisor_limits"),
            supervisor_limits,
        )
        self.observer_gain = self._resolve_nonnegative_float(
            tracking_cfg.get("observer_gain"),
            default=0.0,
            name="free_boundary_tracking.observer_gain",
        )
        self.observer_forgetting = self._resolve_fraction(
            tracking_cfg.get("observer_forgetting"),
            default=0.0,
            name="free_boundary_tracking.observer_forgetting",
        )
        self.observer_max_abs = self._resolve_nonnegative_float(
            tracking_cfg.get("observer_max_abs"),
            default=np.inf,
            name="free_boundary_tracking.observer_max_abs",
        )
        self.fallback_currents = self._resolve_fallback_currents(
            tracking_cfg.get("fallback_currents"),
        )
        self._hold_steps_remaining = 0
        self._coil_actuators = self._build_coil_actuators()

        self.objective_tolerances = self._resolve_objective_tolerances(
            self.kernel.cfg.get("free_boundary", {}).get("objective_tolerances"),
            objective_tolerances,
        )
        self.target_vector, self.objective_blocks = self._build_target_vector()
        if self.target_vector.size < 1:
            raise ValueError("free-boundary tracking requires explicit target values.")
        self.measurement_bias_vector = self._resolve_measurement_vector(
            tracking_cfg.get("measurement_bias"),
            name="free_boundary_tracking.measurement_bias",
        )
        self.measurement_drift_per_step = self._resolve_measurement_vector(
            tracking_cfg.get("measurement_drift_per_step"),
            name="free_boundary_tracking.measurement_drift_per_step",
        )
        self.measurement_correction_bias = self._resolve_measurement_vector(
            tracking_cfg.get("measurement_correction_bias"),
            name="free_boundary_tracking.measurement_correction_bias",
        )
        self.measurement_correction_drift_per_step = self._resolve_measurement_vector(
            tracking_cfg.get("measurement_correction_drift_per_step"),
            name="free_boundary_tracking.measurement_correction_drift_per_step",
        )
        self.measurement_latency_steps = self._resolve_nonnegative_int(
            tracking_cfg.get("measurement_latency_steps"),
            None,
            default=0,
            name="free_boundary_tracking.measurement_latency_steps",
        )
        self.latency_compensation_gain = self._resolve_fraction(
            tracking_cfg.get("latency_compensation_gain"),
            default=0.0,
            name="free_boundary_tracking.latency_compensation_gain",
        )
        self.latency_rate_max_abs = self._resolve_nonnegative_float(
            tracking_cfg.get("latency_rate_max_abs"),
            default=np.inf,
            name="free_boundary_tracking.latency_rate_max_abs",
        )
        self.measurement_drift_state = np.zeros_like(self.target_vector, dtype=np.float64)
        self.measurement_correction_drift_state = np.zeros_like(
            self.target_vector, dtype=np.float64
        )
        self.control_objective_weights = self._build_control_objective_weights()
        self.objective_bias_estimate = np.zeros_like(self.target_vector, dtype=np.float64)
        self.objective_rate_estimate = np.zeros_like(self.target_vector, dtype=np.float64)
        self._measurement_latency_buffer: deque[FloatArray] = deque()
        self._last_delayed_measurement: FloatArray | None = None

        self.response_matrix = np.zeros((self.target_vector.size, self.n_coils), dtype=np.float64)
        self.response_rank = 0
        self.response_condition_number = float("inf")
        self.response_max_singular_value = 0.0
        self.response_degenerate = True
        self.last_coil_penalties = np.ones(self.n_coils, dtype=np.float64)
        self.history: dict[str, list[Any]] = {
            "t": [],
            "tracking_error_norm": [],
            "true_tracking_error_norm": [],
            "control_error_norm": [],
            "true_control_error_norm": [],
            "shape_rms": [],
            "true_shape_rms": [],
            "shape_max_abs": [],
            "true_shape_max_abs": [],
            "x_point_position_error": [],
            "true_x_point_position_error": [],
            "x_point_flux_error": [],
            "true_x_point_flux_error": [],
            "divertor_rms": [],
            "true_divertor_rms": [],
            "divertor_max_abs": [],
            "true_divertor_max_abs": [],
            "max_abs_delta_i": [],
            "max_abs_coil_current": [],
            "max_abs_actuator_lag": [],
            "max_abs_measurement_offset": [],
            "mean_abs_measurement_offset": [],
            "measurement_error_norm": [],
            "delayed_observation_error_norm": [],
            "estimated_observation_error_norm": [],
            "accepted_gain": [],
            "objective_converged": [],
            "max_abs_objective_bias_estimate": [],
            "mean_abs_objective_bias_estimate": [],
            "max_abs_objective_rate_estimate": [],
            "response_rank": [],
            "response_condition_number": [],
            "response_max_singular_value": [],
            "response_degenerate": [],
            "active_control_rows": [],
            "max_coil_penalty": [],
            "supervisor_intervened": [],
            "supervisor_safe": [],
            "supervisor_hold_steps_remaining": [],
            "fallback_active": [],
            "tolerance_regression_blocked": [],
        }



def run_free_boundary_tracking(
    config_file: str | None = None,
    *,
    shot_steps: int = 10,
    gain: float = 1.0,
    verbose: bool = True,
    kernel_factory: Callable[[str], Any] = FusionKernel,
    objective_tolerances: dict[str, float] | None = None,
    control_dt_s: float | None = None,
    coil_actuator_tau_s: float | None = None,
    coil_slew_limits: float | list[float] | None = None,
    supervisor_limits: dict[str, float] | None = None,
    hold_steps_after_reject: int | None = None,
    disturbance_callback: Callable[[Any, CoilSet, int], None] | None = None,
    stop_on_convergence: bool = False,
) -> dict[str, Any]:
    """Run deterministic free-boundary tracking over the configured objectives.

    Examples
    --------
    >>> summary = run_free_boundary_tracking("iter_config.json", shot_steps=2, gain=0.7, verbose=False)
    >>> bool(summary["objective_convergence_active"])
    True
    """
    if config_file is None:
        repo_root = Path(__file__).resolve().parents[3]
        config_file = str(repo_root / "iter_config.json")

    controller = FreeBoundaryTrackingController(
        str(config_file),
        kernel_factory=kernel_factory,
        verbose=verbose,
        objective_tolerances=objective_tolerances,
        control_dt_s=control_dt_s,
        coil_actuator_tau_s=coil_actuator_tau_s,
        coil_slew_limits=coil_slew_limits,
        supervisor_limits=supervisor_limits,
        hold_steps_after_reject=hold_steps_after_reject,
    )
    summary = controller.run_tracking_shot(
        shot_steps=shot_steps,
        gain=gain,
        disturbance_callback=disturbance_callback,
        stop_on_convergence=stop_on_convergence,
    )
    summary["config_path"] = str(config_file)
    return summary


__all__ = ["FreeBoundaryTrackingController", "run_free_boundary_tracking"]
