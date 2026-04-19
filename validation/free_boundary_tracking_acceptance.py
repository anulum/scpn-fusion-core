# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Free-Boundary Tracking Acceptance Campaign
"""Deterministic real-kernel acceptance campaign for free-boundary control."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np

from scpn_fusion.control.free_boundary_tracking import run_free_boundary_tracking
from scpn_fusion.core.fusion_kernel import CoilSet, FusionKernel

ROOT = Path(__file__).resolve().parents[1]

NOMINAL_THRESHOLDS = {
    "max_final_tracking_error_norm": 0.02,
    "max_shape_rms": 0.015,
    "require_objective_converged": True,
}
KICK_THRESHOLDS = {
    "max_final_tracking_error_norm": 0.02,
    "max_shape_rms": 0.015,
    "max_abs_coil_current": 5.0e4,
    "require_objective_converged": True,
}
MEASUREMENT_THRESHOLDS = {
    "min_measured_true_gap": 0.02,
    "min_measurement_offset": 0.04,
    "max_true_tracking_error_norm": 0.02,
}
CORRECTED_THRESHOLDS = {
    "max_measured_true_gap": 1.0e-10,
    "max_measurement_offset": 1.0e-10,
    "max_final_tracking_error_norm": 0.02,
    "max_shape_rms": 0.015,
    "require_objective_converged": True,
}
LATENCY_THRESHOLDS = {
    "min_delayed_observation_error_norm": 0.005,
    "max_true_tracking_error_norm": 0.02,
}
LATENCY_CORRECTED_THRESHOLDS = {
    "max_estimated_observation_error_norm": 0.01,
    "max_final_true_tracking_error_norm": 0.02,
    "require_objective_converged": True,
}
TOPOLOGY_THRESHOLDS = {
    "max_final_tracking_error_norm": 0.02,
    "max_shape_rms": 0.015,
    "max_x_point_position_error": 0.1,
    "max_x_point_flux_error": 0.01,
    "max_divertor_rms": 0.01,
    "max_divertor_max_abs": 0.01,
    "require_objective_converged": True,
}
TOPOLOGY_MEASUREMENT_THRESHOLDS = {
    "min_x_point_position_gap": 0.1,
    "min_x_point_flux_gap": 0.03,
    "min_divertor_rms_gap": 0.03,
    "min_divertor_max_abs_gap": 0.04,
    "min_measurement_offset": 0.08,
    "max_true_tracking_error_norm": 0.02,
    "max_true_x_point_position_error": TOPOLOGY_THRESHOLDS["max_x_point_position_error"],
    "max_true_x_point_flux_error": TOPOLOGY_THRESHOLDS["max_x_point_flux_error"],
    "max_true_divertor_rms": TOPOLOGY_THRESHOLDS["max_divertor_rms"],
    "max_true_divertor_max_abs": TOPOLOGY_THRESHOLDS["max_divertor_max_abs"],
    "require_objective_converged": False,
}
TOPOLOGY_LATENCY_MEASUREMENT_THRESHOLDS = {
    **TOPOLOGY_MEASUREMENT_THRESHOLDS,
    "min_divertor_max_abs_gap": 0.039,
}
TOPOLOGY_CORRECTED_THRESHOLDS = {
    "max_x_point_position_gap": 1.0e-10,
    "max_x_point_flux_gap": 1.0e-10,
    "max_divertor_rms_gap": 1.0e-10,
    "max_divertor_max_abs_gap": 1.0e-10,
    "max_measurement_offset": 1.0e-10,
    "max_final_tracking_error_norm": TOPOLOGY_THRESHOLDS["max_final_tracking_error_norm"],
    "max_x_point_position_error": TOPOLOGY_THRESHOLDS["max_x_point_position_error"],
    "max_x_point_flux_error": TOPOLOGY_THRESHOLDS["max_x_point_flux_error"],
    "max_divertor_rms": TOPOLOGY_THRESHOLDS["max_divertor_rms"],
    "max_divertor_max_abs": TOPOLOGY_THRESHOLDS["max_divertor_max_abs"],
    "require_objective_converged": True,
}
TOPOLOGY_SUPERVISOR_LATENCY_CORRECTED_THRESHOLDS = {
    **TOPOLOGY_CORRECTED_THRESHOLDS,
    "max_x_point_flux_gap": 1.0e-7,
    "max_divertor_rms_gap": 1.0e-7,
    "max_divertor_max_abs_gap": 1.0e-7,
    "max_estimated_observation_error_norm": 0.03,
    "max_final_true_tracking_error_norm": 0.02,
}
SUPERVISOR_FALLBACK_THRESHOLDS = {
    "min_supervisor_intervention_count": 1,
    "min_fallback_active_steps": 1,
    "max_abs_actuator_lag": 2.0,
    "min_lag_reduction_factor": 100.0,
    "max_final_tracking_error_norm": 0.02,
    "max_x_point_position_error": TOPOLOGY_THRESHOLDS["max_x_point_position_error"],
    "max_x_point_flux_error": TOPOLOGY_THRESHOLDS["max_x_point_flux_error"],
    "max_divertor_rms": TOPOLOGY_THRESHOLDS["max_divertor_rms"],
    "max_divertor_max_abs": TOPOLOGY_THRESHOLDS["max_divertor_max_abs"],
    "require_supervisor_active": True,
    "require_supervisor_safe": True,
    "require_objective_converged": True,
}
MEASUREMENT_SWEEP_SCALES = (0.0, 0.5, 1.0, 1.5)
LATENCY_STEP_SWEEP = (0, 1, 2, 3)
ACTUATOR_SLEW_LIMIT_SWEEP = (1.0e3, 1.0e2, 1.0e1, 1.0, 0.1)
COIL_KICK_SCALE_SWEEP = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
TOPOLOGY_KICK_SCALE_SWEEP = (1.0, 2.0, 4.0)
TOPOLOGY_DIVERTOR_STRIKE_POINTS = (
    (3.2, -2.0),
    (4.8, -2.0),
)


def _base_tracking_config() -> dict[str, Any]:
    return {
        "reactor_name": "Free-Boundary-Acceptance",
        "grid_resolution": [12, 12],
        "dimensions": {"R_min": 2.0, "R_max": 6.0, "Z_min": -3.0, "Z_max": 3.0},
        "physics": {"plasma_current_target": 1.0, "vacuum_permeability": 1.0},
        "coils": [
            {"name": "PF1", "r": 3.0, "z": 4.0, "current": 2.0},
            {"name": "PF2", "r": 5.0, "z": -4.0, "current": -1.0},
        ],
        "solver": {
            "max_iterations": 10,
            "convergence_threshold": 1e-3,
            "relaxation_factor": 0.15,
            "solver_method": "sor",
            "boundary_variant": "free_boundary",
        },
        "free_boundary": {
            "current_limits": [5.0e4, 5.0e4],
            "target_flux_points": [[3.5, 0.0], [4.0, 0.5]],
            "objective_tolerances": {"shape_rms": 0.25, "shape_max_abs": 0.35},
        },
    }


def _build_tracking_template(tmp_path: Path) -> dict[str, Any]:
    cfg = _base_tracking_config()
    template_path = tmp_path / "template.json"
    template_path.write_text(json.dumps(cfg), encoding="utf-8")
    kernel = FusionKernel(template_path)
    coils = kernel.build_coilset_from_config()
    kernel.solve_free_boundary(
        coils,
        max_outer_iter=2,
        tol=1.0e-2,
        optimize_shape=False,
    )
    flux_targets = kernel._sample_flux_at_points(coils.target_flux_points)
    cfg["free_boundary"]["target_flux_values"] = [float(v) for v in flux_targets]
    return cfg


def _build_topology_tracking_template(
    tmp_path: Path, *, template_cfg: dict[str, Any]
) -> dict[str, Any]:
    cfg = deepcopy(template_cfg)
    topology_path = tmp_path / "topology_template.json"
    topology_path.write_text(json.dumps(cfg), encoding="utf-8")
    kernel = FusionKernel(topology_path)
    coils = kernel.build_coilset_from_config()
    kernel.solve_free_boundary(
        coils,
        max_outer_iter=2,
        tol=1.0e-2,
        optimize_shape=False,
    )
    x_pos, _ = kernel.find_x_point(kernel.Psi)
    x_target = np.asarray(x_pos, dtype=np.float64).reshape(2)
    divertor_points = np.asarray(TOPOLOGY_DIVERTOR_STRIKE_POINTS, dtype=np.float64)
    objective_tolerances = deepcopy(cfg["free_boundary"].get("objective_tolerances", {}))
    objective_tolerances.update(
        {
            "x_point_position": TOPOLOGY_THRESHOLDS["max_x_point_position_error"],
            "x_point_flux": TOPOLOGY_THRESHOLDS["max_x_point_flux_error"],
            "divertor_rms": TOPOLOGY_THRESHOLDS["max_divertor_rms"],
            "divertor_max_abs": TOPOLOGY_THRESHOLDS["max_divertor_max_abs"],
        }
    )
    cfg["free_boundary"].update(
        {
            "x_point_target": [float(x_target[0]), float(x_target[1])],
            "x_point_flux_target": float(
                kernel._interp_psi(float(x_target[0]), float(x_target[1]))
            ),
            "divertor_strike_points": divertor_points.tolist(),
            "divertor_flux_values": [
                float(v) for v in kernel._sample_flux_at_points(divertor_points)
            ],
            "objective_tolerances": objective_tolerances,
        }
    )
    return cfg


def _write_tracking_config(
    path: Path,
    *,
    template_cfg: dict[str, Any],
    tracking_cfg: dict[str, Any] | None = None,
) -> Path:
    cfg = deepcopy(template_cfg)
    if tracking_cfg is not None:
        cfg["free_boundary_tracking"] = tracking_cfg
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def _merge_tracking_cfg(*parts: dict[str, Any] | None) -> dict[str, Any] | None:
    merged: dict[str, Any] = {}
    for part in parts:
        if part:
            merged.update(part)
    return merged or None


def _make_coil_kick_disturbance(scale: float = 1.0) -> Any:
    kick = np.array([2000.0, -1500.0], dtype=np.float64) * float(scale)
    limits = np.array([5.0e4, 5.0e4], dtype=np.float64)

    def disturbance(kernel: FusionKernel, coils: CoilSet, step: int) -> None:
        del kernel
        if step != 1:
            return
        coils.currents = np.clip(
            np.asarray(coils.currents, dtype=np.float64) + kick, -limits, limits
        )

    return disturbance


def _run_nominal(config_path: Path) -> dict[str, Any]:
    return run_free_boundary_tracking(
        config_file=str(config_path),
        shot_steps=4,
        gain=0.6,
        verbose=False,
        kernel_factory=FusionKernel,
        stop_on_convergence=False,
    )


def _run_kick(config_path: Path) -> dict[str, Any]:
    return run_free_boundary_tracking(
        config_file=str(config_path),
        shot_steps=4,
        gain=0.6,
        verbose=False,
        kernel_factory=FusionKernel,
        disturbance_callback=_make_coil_kick_disturbance(),
        stop_on_convergence=False,
    )


def _run_topology_kick(config_path: Path) -> dict[str, Any]:
    return run_free_boundary_tracking(
        config_file=str(config_path),
        shot_steps=4,
        gain=0.6,
        verbose=False,
        kernel_factory=FusionKernel,
        disturbance_callback=_make_coil_kick_disturbance(2.0),
        stop_on_convergence=False,
    )


def _run_measurement_fault(config_path: Path) -> dict[str, Any]:
    return run_free_boundary_tracking(
        config_file=str(config_path),
        shot_steps=4,
        gain=0.6,
        verbose=False,
        kernel_factory=FusionKernel,
        stop_on_convergence=False,
    )


def _run_latency_fault(config_path: Path) -> dict[str, Any]:
    return run_free_boundary_tracking(
        config_file=str(config_path),
        shot_steps=4,
        gain=0.6,
        verbose=False,
        kernel_factory=FusionKernel,
        disturbance_callback=_make_coil_kick_disturbance(32.0),
        stop_on_convergence=False,
    )


def _run_actuator_limited_kick(config_path: Path, *, coil_slew_limits: float) -> dict[str, Any]:
    return run_free_boundary_tracking(
        config_file=str(config_path),
        shot_steps=4,
        gain=8.0,
        verbose=False,
        kernel_factory=FusionKernel,
        disturbance_callback=_make_coil_kick_disturbance(),
        control_dt_s=0.1,
        coil_actuator_tau_s=0.05,
        coil_slew_limits=coil_slew_limits,
        stop_on_convergence=False,
    )


def _run_supervisor_fallback_kick(config_path: Path) -> dict[str, Any]:
    return run_free_boundary_tracking(
        config_file=str(config_path),
        shot_steps=4,
        gain=8.0,
        verbose=False,
        kernel_factory=FusionKernel,
        disturbance_callback=_make_coil_kick_disturbance(8.0),
        control_dt_s=0.1,
        coil_actuator_tau_s=0.05,
        coil_slew_limits=0.05,
        stop_on_convergence=False,
    )


def _measurement_tracking_cfg(scale: float, *, corrected: bool) -> dict[str, Any] | None:
    scale_value = float(scale)
    if scale_value == 0.0 and not corrected:
        return None
    tracking_cfg: dict[str, Any] = {
        "measurement_bias": {"shape_flux": [0.03 * scale_value, -0.02 * scale_value]},
        "measurement_drift_per_step": {"shape_flux": [0.004 * scale_value, -0.003 * scale_value]},
    }
    if corrected:
        tracking_cfg["measurement_correction_bias"] = tracking_cfg["measurement_bias"]
        tracking_cfg["measurement_correction_drift_per_step"] = tracking_cfg[
            "measurement_drift_per_step"
        ]
    return tracking_cfg


def _latency_tracking_cfg(latency_steps: int, *, corrected: bool) -> dict[str, Any] | None:
    steps = int(latency_steps)
    if steps == 0 and not corrected:
        return None
    tracking_cfg: dict[str, Any] = {
        "measurement_latency_steps": steps,
    }
    if corrected and steps > 0:
        tracking_cfg["latency_compensation_gain"] = 1.0
        tracking_cfg["latency_rate_max_abs"] = 0.5
    return tracking_cfg


def _topology_measurement_tracking_cfg(scale: float = 1.0, *, corrected: bool) -> dict[str, Any]:
    scale_value = float(scale)
    tracking_cfg: dict[str, Any] = {
        "measurement_bias": {
            "x_point_position": [0.06 * scale_value, -0.05 * scale_value],
            "x_point_flux": 0.025 * scale_value,
            "divertor_flux": [0.03 * scale_value, -0.02 * scale_value],
        },
        "measurement_drift_per_step": {
            "x_point_position": [0.01 * scale_value, -0.008 * scale_value],
            "x_point_flux": 0.004 * scale_value,
            "divertor_flux": [0.005 * scale_value, -0.003 * scale_value],
        },
    }
    if corrected:
        tracking_cfg["measurement_correction_bias"] = tracking_cfg["measurement_bias"]
        tracking_cfg["measurement_correction_drift_per_step"] = tracking_cfg[
            "measurement_drift_per_step"
        ]
    return tracking_cfg


def _evaluate_nominal(summary: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "final_tracking_error_norm": bool(
            float(summary["final_tracking_error_norm"])
            <= NOMINAL_THRESHOLDS["max_final_tracking_error_norm"]
        ),
        "shape_rms": bool(float(summary["shape_rms"]) <= NOMINAL_THRESHOLDS["max_shape_rms"]),
        "objective_converged": bool(
            summary["objective_converged"] is NOMINAL_THRESHOLDS["require_objective_converged"]
        ),
    }
    return {
        "thresholds": NOMINAL_THRESHOLDS.copy(),
        "checks": checks,
        "passes_thresholds": all(checks.values()),
    }


def _evaluate_kick(summary: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "final_tracking_error_norm": bool(
            float(summary["final_tracking_error_norm"])
            <= KICK_THRESHOLDS["max_final_tracking_error_norm"]
        ),
        "shape_rms": bool(float(summary["shape_rms"]) <= KICK_THRESHOLDS["max_shape_rms"]),
        "max_abs_coil_current": bool(
            float(summary["max_abs_coil_current"]) <= KICK_THRESHOLDS["max_abs_coil_current"]
        ),
        "objective_converged": bool(
            summary["objective_converged"] is KICK_THRESHOLDS["require_objective_converged"]
        ),
    }
    return {
        "thresholds": KICK_THRESHOLDS.copy(),
        "checks": checks,
        "passes_thresholds": all(checks.values()),
    }


def _evaluate_measurement_fault(summary: dict[str, Any]) -> dict[str, Any]:
    measured_true_gap = abs(
        float(summary["final_tracking_error_norm"])
        - float(summary["final_true_tracking_error_norm"])
    )
    checks = {
        "measured_true_gap": bool(
            measured_true_gap >= MEASUREMENT_THRESHOLDS["min_measured_true_gap"]
        ),
        "measurement_offset": bool(
            float(summary["max_abs_measurement_offset"])
            >= MEASUREMENT_THRESHOLDS["min_measurement_offset"]
        ),
        "true_tracking_error_norm": bool(
            float(summary["final_true_tracking_error_norm"])
            <= MEASUREMENT_THRESHOLDS["max_true_tracking_error_norm"]
        ),
    }
    return {
        "thresholds": MEASUREMENT_THRESHOLDS.copy(),
        "checks": checks,
        "passes_thresholds": all(checks.values()),
        "measured_true_gap": float(measured_true_gap),
    }


def _evaluate_corrected(summary: dict[str, Any]) -> dict[str, Any]:
    measured_true_gap = abs(
        float(summary["final_tracking_error_norm"])
        - float(summary["final_true_tracking_error_norm"])
    )
    checks = {
        "measured_true_gap": bool(
            measured_true_gap <= CORRECTED_THRESHOLDS["max_measured_true_gap"]
        ),
        "measurement_offset": bool(
            float(summary["max_abs_measurement_offset"])
            <= CORRECTED_THRESHOLDS["max_measurement_offset"]
        ),
        "final_tracking_error_norm": bool(
            float(summary["final_tracking_error_norm"])
            <= CORRECTED_THRESHOLDS["max_final_tracking_error_norm"]
        ),
        "shape_rms": bool(float(summary["shape_rms"]) <= CORRECTED_THRESHOLDS["max_shape_rms"]),
        "objective_converged": bool(
            summary["objective_converged"] is CORRECTED_THRESHOLDS["require_objective_converged"]
        ),
    }
    return {
        "thresholds": CORRECTED_THRESHOLDS.copy(),
        "checks": checks,
        "passes_thresholds": all(checks.values()),
        "measured_true_gap": float(measured_true_gap),
    }


def _evaluate_latency_fault(summary: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "delayed_observation_error_norm": bool(
            float(summary["max_delayed_observation_error_norm"])
            >= LATENCY_THRESHOLDS["min_delayed_observation_error_norm"]
        ),
        "true_tracking_error_norm": bool(
            float(summary["final_true_tracking_error_norm"])
            <= LATENCY_THRESHOLDS["max_true_tracking_error_norm"]
        ),
    }
    return {
        "thresholds": LATENCY_THRESHOLDS.copy(),
        "checks": checks,
        "passes_thresholds": all(checks.values()),
        "delayed_observation_error_norm": float(summary["max_delayed_observation_error_norm"]),
        "estimated_observation_error_norm": float(summary["max_estimated_observation_error_norm"]),
    }


def _evaluate_latency_corrected(
    summary: dict[str, Any],
    *,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    threshold_set = LATENCY_CORRECTED_THRESHOLDS if thresholds is None else thresholds
    checks = {
        "estimated_observation_error_norm": bool(
            float(summary["max_estimated_observation_error_norm"])
            <= threshold_set["max_estimated_observation_error_norm"]
        ),
        "final_true_tracking_error_norm": bool(
            float(summary["final_true_tracking_error_norm"])
            <= threshold_set["max_final_true_tracking_error_norm"]
        ),
        "objective_converged": bool(
            summary["objective_converged"] is threshold_set["require_objective_converged"]
        ),
    }
    return {
        "thresholds": dict(threshold_set),
        "checks": checks,
        "passes_thresholds": all(checks.values()),
        "delayed_observation_error_norm": float(summary["max_delayed_observation_error_norm"]),
        "estimated_observation_error_norm": float(summary["max_estimated_observation_error_norm"]),
    }


def _evaluate_topology(summary: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "final_tracking_error_norm": bool(
            float(summary["final_tracking_error_norm"])
            <= TOPOLOGY_THRESHOLDS["max_final_tracking_error_norm"]
        ),
        "shape_rms": bool(float(summary["shape_rms"]) <= TOPOLOGY_THRESHOLDS["max_shape_rms"]),
        "x_point_position_error": bool(
            summary["x_point_position_error"] is not None
            and np.isfinite(float(summary["x_point_position_error"]))
            and float(summary["x_point_position_error"])
            <= TOPOLOGY_THRESHOLDS["max_x_point_position_error"]
        ),
        "x_point_flux_error": bool(
            summary["x_point_flux_error"] is not None
            and np.isfinite(float(summary["x_point_flux_error"]))
            and float(summary["x_point_flux_error"])
            <= TOPOLOGY_THRESHOLDS["max_x_point_flux_error"]
        ),
        "divertor_rms": bool(
            summary["divertor_rms"] is not None
            and np.isfinite(float(summary["divertor_rms"]))
            and float(summary["divertor_rms"]) <= TOPOLOGY_THRESHOLDS["max_divertor_rms"]
        ),
        "divertor_max_abs": bool(
            summary["divertor_max_abs"] is not None
            and np.isfinite(float(summary["divertor_max_abs"]))
            and float(summary["divertor_max_abs"]) <= TOPOLOGY_THRESHOLDS["max_divertor_max_abs"]
        ),
        "objective_converged": bool(
            summary["objective_converged"] is TOPOLOGY_THRESHOLDS["require_objective_converged"]
        ),
    }
    return {
        "thresholds": TOPOLOGY_THRESHOLDS.copy(),
        "checks": checks,
        "passes_thresholds": all(checks.values()),
    }


def _evaluate_topology_measurement_fault(
    summary: dict[str, Any],
    *,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    threshold_set = TOPOLOGY_MEASUREMENT_THRESHOLDS if thresholds is None else thresholds
    x_point_position_gap = abs(
        float(summary["x_point_position_error"]) - float(summary["true_x_point_position_error"])
    )
    x_point_flux_gap = abs(
        float(summary["x_point_flux_error"]) - float(summary["true_x_point_flux_error"])
    )
    divertor_rms_gap = abs(float(summary["divertor_rms"]) - float(summary["true_divertor_rms"]))
    divertor_max_abs_gap = abs(
        float(summary["divertor_max_abs"]) - float(summary["true_divertor_max_abs"])
    )
    checks = {
        "x_point_position_gap": bool(
            x_point_position_gap >= threshold_set["min_x_point_position_gap"]
        ),
        "x_point_flux_gap": bool(x_point_flux_gap >= threshold_set["min_x_point_flux_gap"]),
        "divertor_rms_gap": bool(divertor_rms_gap >= threshold_set["min_divertor_rms_gap"]),
        "divertor_max_abs_gap": bool(
            divertor_max_abs_gap >= threshold_set["min_divertor_max_abs_gap"]
        ),
        "measurement_offset": bool(
            float(summary["max_abs_measurement_offset"]) >= threshold_set["min_measurement_offset"]
        ),
        "true_tracking_error_norm": bool(
            float(summary["final_true_tracking_error_norm"])
            <= threshold_set["max_true_tracking_error_norm"]
        ),
        "true_x_point_position_error": bool(
            float(summary["true_x_point_position_error"])
            <= threshold_set["max_true_x_point_position_error"]
        ),
        "true_x_point_flux_error": bool(
            float(summary["true_x_point_flux_error"])
            <= threshold_set["max_true_x_point_flux_error"]
        ),
        "true_divertor_rms": bool(
            float(summary["true_divertor_rms"]) <= threshold_set["max_true_divertor_rms"]
        ),
        "true_divertor_max_abs": bool(
            float(summary["true_divertor_max_abs"]) <= threshold_set["max_true_divertor_max_abs"]
        ),
        "objective_converged": bool(
            summary["objective_converged"] is threshold_set["require_objective_converged"]
        ),
    }
    return {
        "thresholds": dict(threshold_set),
        "checks": checks,
        "passes_thresholds": all(checks.values()),
        "x_point_position_gap": x_point_position_gap,
        "x_point_flux_gap": x_point_flux_gap,
        "divertor_rms_gap": divertor_rms_gap,
        "divertor_max_abs_gap": divertor_max_abs_gap,
    }


def _evaluate_topology_corrected(
    summary: dict[str, Any],
    *,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    threshold_set = TOPOLOGY_CORRECTED_THRESHOLDS if thresholds is None else thresholds
    x_point_position_gap = abs(
        float(summary["x_point_position_error"]) - float(summary["true_x_point_position_error"])
    )
    x_point_flux_gap = abs(
        float(summary["x_point_flux_error"]) - float(summary["true_x_point_flux_error"])
    )
    divertor_rms_gap = abs(float(summary["divertor_rms"]) - float(summary["true_divertor_rms"]))
    divertor_max_abs_gap = abs(
        float(summary["divertor_max_abs"]) - float(summary["true_divertor_max_abs"])
    )
    checks = {
        "x_point_position_gap": bool(
            x_point_position_gap <= threshold_set["max_x_point_position_gap"]
        ),
        "x_point_flux_gap": bool(x_point_flux_gap <= threshold_set["max_x_point_flux_gap"]),
        "divertor_rms_gap": bool(divertor_rms_gap <= threshold_set["max_divertor_rms_gap"]),
        "divertor_max_abs_gap": bool(
            divertor_max_abs_gap <= threshold_set["max_divertor_max_abs_gap"]
        ),
        "measurement_offset": bool(
            float(summary["max_abs_measurement_offset"]) <= threshold_set["max_measurement_offset"]
        ),
        "final_tracking_error_norm": bool(
            float(summary["final_tracking_error_norm"])
            <= threshold_set["max_final_tracking_error_norm"]
        ),
        "x_point_position_error": bool(
            float(summary["x_point_position_error"]) <= threshold_set["max_x_point_position_error"]
        ),
        "x_point_flux_error": bool(
            float(summary["x_point_flux_error"]) <= threshold_set["max_x_point_flux_error"]
        ),
        "divertor_rms": bool(float(summary["divertor_rms"]) <= threshold_set["max_divertor_rms"]),
        "divertor_max_abs": bool(
            float(summary["divertor_max_abs"]) <= threshold_set["max_divertor_max_abs"]
        ),
        "objective_converged": bool(
            summary["objective_converged"] is threshold_set["require_objective_converged"]
        ),
    }
    return {
        "thresholds": dict(threshold_set),
        "checks": checks,
        "passes_thresholds": all(checks.values()),
        "x_point_position_gap": x_point_position_gap,
        "x_point_flux_gap": x_point_flux_gap,
        "divertor_rms_gap": divertor_rms_gap,
        "divertor_max_abs_gap": divertor_max_abs_gap,
    }


def _evaluate_supervisor_only(summary: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "supervisor_intervention_count": bool(
            int(summary["supervisor_intervention_count"])
            >= SUPERVISOR_FALLBACK_THRESHOLDS["min_supervisor_intervention_count"]
        ),
        "fallback_active_steps": bool(
            int(summary["fallback_active_steps"])
            >= SUPERVISOR_FALLBACK_THRESHOLDS["min_fallback_active_steps"]
        ),
        "max_abs_actuator_lag": bool(
            float(summary["max_abs_actuator_lag"])
            <= SUPERVISOR_FALLBACK_THRESHOLDS["max_abs_actuator_lag"]
        ),
        "supervisor_active": bool(
            summary["supervisor_active"]
            is SUPERVISOR_FALLBACK_THRESHOLDS["require_supervisor_active"]
        ),
        "supervisor_safe": bool(
            summary["supervisor_safe"] is SUPERVISOR_FALLBACK_THRESHOLDS["require_supervisor_safe"]
        ),
    }
    return {
        "thresholds": {
            "min_supervisor_intervention_count": SUPERVISOR_FALLBACK_THRESHOLDS[
                "min_supervisor_intervention_count"
            ],
            "min_fallback_active_steps": SUPERVISOR_FALLBACK_THRESHOLDS[
                "min_fallback_active_steps"
            ],
            "max_abs_actuator_lag": SUPERVISOR_FALLBACK_THRESHOLDS["max_abs_actuator_lag"],
            "require_supervisor_active": SUPERVISOR_FALLBACK_THRESHOLDS[
                "require_supervisor_active"
            ],
            "require_supervisor_safe": SUPERVISOR_FALLBACK_THRESHOLDS["require_supervisor_safe"],
        },
        "checks": checks,
        "passes_thresholds": all(checks.values()),
    }


def _combine_evaluations(*evaluations: dict[str, Any]) -> dict[str, Any]:
    thresholds: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    extras: dict[str, Any] = {}
    for evaluation in evaluations:
        thresholds.update(dict(evaluation["thresholds"]))
        checks.update(dict(evaluation["checks"]))
        for key, value in evaluation.items():
            if key not in {"thresholds", "checks", "passes_thresholds"}:
                extras[key] = value
    return {
        "thresholds": thresholds,
        "checks": checks,
        "passes_thresholds": all(checks.values()),
        **extras,
    }


def _evaluate_supervisor_fallback(
    summary: dict[str, Any],
    *,
    unsupervised_reference: dict[str, Any],
) -> dict[str, Any]:
    safe_lag = float(summary["max_abs_actuator_lag"])
    reference_lag = float(unsupervised_reference["max_abs_actuator_lag"])
    lag_reduction_factor = float(reference_lag / max(safe_lag, 1.0e-12))
    checks = {
        "supervisor_intervention_count": bool(
            int(summary["supervisor_intervention_count"])
            >= SUPERVISOR_FALLBACK_THRESHOLDS["min_supervisor_intervention_count"]
        ),
        "fallback_active_steps": bool(
            int(summary["fallback_active_steps"])
            >= SUPERVISOR_FALLBACK_THRESHOLDS["min_fallback_active_steps"]
        ),
        "max_abs_actuator_lag": bool(
            safe_lag <= SUPERVISOR_FALLBACK_THRESHOLDS["max_abs_actuator_lag"]
        ),
        "lag_reduction_factor": bool(
            lag_reduction_factor >= SUPERVISOR_FALLBACK_THRESHOLDS["min_lag_reduction_factor"]
        ),
        "final_tracking_error_norm": bool(
            float(summary["final_tracking_error_norm"])
            <= SUPERVISOR_FALLBACK_THRESHOLDS["max_final_tracking_error_norm"]
        ),
        "x_point_position_error": bool(
            summary["x_point_position_error"] is not None
            and np.isfinite(float(summary["x_point_position_error"]))
            and float(summary["x_point_position_error"])
            <= SUPERVISOR_FALLBACK_THRESHOLDS["max_x_point_position_error"]
        ),
        "x_point_flux_error": bool(
            summary["x_point_flux_error"] is not None
            and np.isfinite(float(summary["x_point_flux_error"]))
            and float(summary["x_point_flux_error"])
            <= SUPERVISOR_FALLBACK_THRESHOLDS["max_x_point_flux_error"]
        ),
        "divertor_rms": bool(
            summary["divertor_rms"] is not None
            and np.isfinite(float(summary["divertor_rms"]))
            and float(summary["divertor_rms"]) <= SUPERVISOR_FALLBACK_THRESHOLDS["max_divertor_rms"]
        ),
        "divertor_max_abs": bool(
            summary["divertor_max_abs"] is not None
            and np.isfinite(float(summary["divertor_max_abs"]))
            and float(summary["divertor_max_abs"])
            <= SUPERVISOR_FALLBACK_THRESHOLDS["max_divertor_max_abs"]
        ),
        "supervisor_active": bool(
            summary["supervisor_active"]
            is SUPERVISOR_FALLBACK_THRESHOLDS["require_supervisor_active"]
        ),
        "supervisor_safe": bool(
            summary["supervisor_safe"] is SUPERVISOR_FALLBACK_THRESHOLDS["require_supervisor_safe"]
        ),
        "objective_converged": bool(
            summary["objective_converged"]
            is SUPERVISOR_FALLBACK_THRESHOLDS["require_objective_converged"]
        ),
    }
    return {
        "thresholds": SUPERVISOR_FALLBACK_THRESHOLDS.copy(),
        "checks": checks,
        "passes_thresholds": all(checks.values()),
        "lag_reduction_factor": lag_reduction_factor,
    }


def _is_monotone_non_decreasing(values: list[float], *, atol: float = 1.0e-12) -> bool:
    return all(float(b) + atol >= float(a) for a, b in zip(values[:-1], values[1:]))


def _is_monotone_non_increasing(values: list[float], *, atol: float = 1.0e-12) -> bool:
    return all(float(b) <= float(a) + atol for a, b in zip(values[:-1], values[1:]))


def _run_measurement_sweep(tmp_path: Path, *, template_cfg: dict[str, Any]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for scale in MEASUREMENT_SWEEP_SCALES:
        cfg = _write_tracking_config(
            tmp_path / f"measurement_sweep_{scale:.1f}.json",
            template_cfg=template_cfg,
            tracking_cfg=_measurement_tracking_cfg(scale, corrected=False),
        )
        summary = _run_measurement_fault(cfg)
        measured_true_gap = abs(
            float(summary["final_tracking_error_norm"])
            - float(summary["final_true_tracking_error_norm"])
        )
        entries.append(
            {
                "scale": float(scale),
                "final_tracking_error_norm": float(summary["final_tracking_error_norm"]),
                "final_true_tracking_error_norm": float(summary["final_true_tracking_error_norm"]),
                "measured_true_gap": float(measured_true_gap),
                "max_abs_measurement_offset": float(summary["max_abs_measurement_offset"]),
                "shape_rms": float(summary["shape_rms"]),
                "true_shape_rms": float(summary["true_shape_rms"]),
            }
        )
    measured_true_gaps = [float(entry["measured_true_gap"]) for entry in entries]
    measurement_offsets = [float(entry["max_abs_measurement_offset"]) for entry in entries]
    checks = {
        "measured_true_gap_monotone": _is_monotone_non_decreasing(measured_true_gaps),
        "measurement_offset_monotone": _is_monotone_non_decreasing(measurement_offsets),
    }
    return {
        "entries": entries,
        "checks": checks,
        "passes_thresholds": all(checks.values()),
    }


def _run_corrected_measurement_sweep(
    tmp_path: Path, *, template_cfg: dict[str, Any]
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for scale in MEASUREMENT_SWEEP_SCALES:
        cfg = _write_tracking_config(
            tmp_path / f"measurement_corrected_sweep_{scale:.1f}.json",
            template_cfg=template_cfg,
            tracking_cfg=_measurement_tracking_cfg(scale, corrected=True),
        )
        summary = _run_measurement_fault(cfg)
        measured_true_gap = abs(
            float(summary["final_tracking_error_norm"])
            - float(summary["final_true_tracking_error_norm"])
        )
        entries.append(
            {
                "scale": float(scale),
                "final_tracking_error_norm": float(summary["final_tracking_error_norm"]),
                "final_true_tracking_error_norm": float(summary["final_true_tracking_error_norm"]),
                "measured_true_gap": float(measured_true_gap),
                "max_abs_measurement_offset": float(summary["max_abs_measurement_offset"]),
                "shape_rms": float(summary["shape_rms"]),
                "true_shape_rms": float(summary["true_shape_rms"]),
            }
        )
    measured_true_gaps = [float(entry["measured_true_gap"]) for entry in entries]
    measurement_offsets = [float(entry["max_abs_measurement_offset"]) for entry in entries]
    final_tracking_error = [float(entry["final_tracking_error_norm"]) for entry in entries]
    checks = {
        "max_measured_true_gap": bool(
            max(measured_true_gaps) <= CORRECTED_THRESHOLDS["max_measured_true_gap"]
        ),
        "max_measurement_offset": bool(
            max(measurement_offsets) <= CORRECTED_THRESHOLDS["max_measurement_offset"]
        ),
        "tracking_error_constant": bool(
            max(final_tracking_error) - min(final_tracking_error)
            <= CORRECTED_THRESHOLDS["max_measured_true_gap"]
        ),
    }
    return {
        "entries": entries,
        "checks": checks,
        "passes_thresholds": all(checks.values()),
    }


def _run_latency_step_sweep(tmp_path: Path, *, template_cfg: dict[str, Any]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for latency_steps in LATENCY_STEP_SWEEP:
        cfg = _write_tracking_config(
            tmp_path / f"latency_sweep_{latency_steps}.json",
            template_cfg=template_cfg,
            tracking_cfg=_latency_tracking_cfg(latency_steps, corrected=False),
        )
        summary = _run_latency_fault(cfg)
        entries.append(
            {
                "latency_steps": int(latency_steps),
                "delayed_observation_error_norm": float(
                    summary["max_delayed_observation_error_norm"]
                ),
                "estimated_observation_error_norm": float(
                    summary["max_estimated_observation_error_norm"]
                ),
                "final_true_tracking_error_norm": float(summary["final_true_tracking_error_norm"]),
                "objective_converged": bool(summary["objective_converged"]),
            }
        )
    delayed_error = [float(entry["delayed_observation_error_norm"]) for entry in entries]
    final_true_tracking_error = [
        float(entry["final_true_tracking_error_norm"]) for entry in entries
    ]
    checks = {
        "delayed_observation_error_monotone": _is_monotone_non_decreasing(
            delayed_error, atol=1.0e-6
        ),
        "delayed_observation_error_active": bool(
            max(delayed_error) >= LATENCY_THRESHOLDS["min_delayed_observation_error_norm"]
        ),
        "final_true_tracking_error_bounded": all(
            value <= LATENCY_THRESHOLDS["max_true_tracking_error_norm"]
            for value in final_true_tracking_error
        ),
        "objective_converged_all": all(bool(entry["objective_converged"]) for entry in entries),
    }
    return {
        "entries": entries,
        "checks": checks,
        "passes_thresholds": all(checks.values()),
    }


def _run_corrected_latency_step_sweep(
    tmp_path: Path, *, template_cfg: dict[str, Any]
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for latency_steps in LATENCY_STEP_SWEEP:
        cfg = _write_tracking_config(
            tmp_path / f"latency_corrected_sweep_{latency_steps}.json",
            template_cfg=template_cfg,
            tracking_cfg=_latency_tracking_cfg(latency_steps, corrected=True),
        )
        summary = _run_latency_fault(cfg)
        entries.append(
            {
                "latency_steps": int(latency_steps),
                "delayed_observation_error_norm": float(
                    summary["max_delayed_observation_error_norm"]
                ),
                "estimated_observation_error_norm": float(
                    summary["max_estimated_observation_error_norm"]
                ),
                "final_true_tracking_error_norm": float(summary["final_true_tracking_error_norm"]),
                "objective_converged": bool(summary["objective_converged"]),
            }
        )
    estimated_error = [float(entry["estimated_observation_error_norm"]) for entry in entries]
    final_true_tracking_error = [
        float(entry["final_true_tracking_error_norm"]) for entry in entries
    ]
    checks = {
        "estimated_observation_error_bounded": all(
            value <= LATENCY_CORRECTED_THRESHOLDS["max_estimated_observation_error_norm"]
            for value in estimated_error
        ),
        "final_true_tracking_error_bounded": all(
            value <= LATENCY_CORRECTED_THRESHOLDS["max_final_true_tracking_error_norm"]
            for value in final_true_tracking_error
        ),
        "objective_converged_all": all(bool(entry["objective_converged"]) for entry in entries),
    }
    return {
        "entries": entries,
        "checks": checks,
        "passes_thresholds": all(checks.values()),
    }


def _run_actuator_slew_sweep(tmp_path: Path, *, template_cfg: dict[str, Any]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for slew_limit in ACTUATOR_SLEW_LIMIT_SWEEP:
        cfg = _write_tracking_config(
            tmp_path / f"actuator_slew_{slew_limit}.json",
            template_cfg=template_cfg,
        )
        summary = _run_actuator_limited_kick(cfg, coil_slew_limits=float(slew_limit))
        entries.append(
            {
                "coil_slew_limit": float(slew_limit),
                "max_abs_actuator_lag": float(summary["max_abs_actuator_lag"]),
                "mean_abs_actuator_lag": float(summary["mean_abs_actuator_lag"]),
                "max_abs_coil_current": float(summary["max_abs_coil_current"]),
                "final_tracking_error_norm": float(summary["final_tracking_error_norm"]),
            }
        )
    max_lag = [float(entry["max_abs_actuator_lag"]) for entry in entries]
    mean_lag = [float(entry["mean_abs_actuator_lag"]) for entry in entries]
    max_coil_current = [float(entry["max_abs_coil_current"]) for entry in entries]
    checks = {
        "max_abs_actuator_lag_monotone": _is_monotone_non_decreasing(max_lag),
        "mean_abs_actuator_lag_monotone": _is_monotone_non_decreasing(mean_lag),
        "max_abs_coil_current_monotone": _is_monotone_non_increasing(max_coil_current),
    }
    return {
        "entries": entries,
        "checks": checks,
        "passes_thresholds": all(checks.values()),
    }


def _run_coil_kick_scale_sweep(tmp_path: Path, *, template_cfg: dict[str, Any]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for scale in COIL_KICK_SCALE_SWEEP:
        cfg = _write_tracking_config(
            tmp_path / f"coil_kick_scale_{scale:.1f}.json",
            template_cfg=template_cfg,
        )
        summary = run_free_boundary_tracking(
            config_file=str(cfg),
            shot_steps=4,
            gain=0.6,
            verbose=False,
            kernel_factory=FusionKernel,
            disturbance_callback=_make_coil_kick_disturbance(scale),
            stop_on_convergence=False,
        )
        entries.append(
            {
                "kick_scale": float(scale),
                "final_tracking_error_norm": float(summary["final_tracking_error_norm"]),
                "mean_tracking_error_norm": float(summary["mean_tracking_error_norm"]),
                "shape_rms": float(summary["shape_rms"]),
                "max_abs_coil_current": float(summary["max_abs_coil_current"]),
                "objective_converged": bool(summary["objective_converged"]),
            }
        )
    max_coil_current = [float(entry["max_abs_coil_current"]) for entry in entries]
    final_tracking_error = [float(entry["final_tracking_error_norm"]) for entry in entries]
    objective_converged = [bool(entry["objective_converged"]) for entry in entries]
    checks = {
        "max_abs_coil_current_monotone": _is_monotone_non_decreasing(max_coil_current),
        "objective_converged_all": all(objective_converged),
        "final_tracking_error_bounded": max(final_tracking_error)
        <= NOMINAL_THRESHOLDS["max_final_tracking_error_norm"],
    }
    return {
        "entries": entries,
        "checks": checks,
        "passes_thresholds": all(checks.values()),
    }


def _run_topology_kick_scale_sweep(
    tmp_path: Path, *, topology_template_cfg: dict[str, Any]
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for scale in TOPOLOGY_KICK_SCALE_SWEEP:
        cfg = _write_tracking_config(
            tmp_path / f"topology_kick_scale_{scale:.1f}.json",
            template_cfg=topology_template_cfg,
        )
        summary = run_free_boundary_tracking(
            config_file=str(cfg),
            shot_steps=4,
            gain=0.6,
            verbose=False,
            kernel_factory=FusionKernel,
            disturbance_callback=_make_coil_kick_disturbance(scale),
            stop_on_convergence=False,
        )
        entries.append(
            {
                "kick_scale": float(scale),
                "final_tracking_error_norm": float(summary["final_tracking_error_norm"]),
                "x_point_position_error": float(summary["x_point_position_error"]),
                "x_point_flux_error": float(summary["x_point_flux_error"]),
                "divertor_rms": float(summary["divertor_rms"]),
                "divertor_max_abs": float(summary["divertor_max_abs"]),
                "objective_converged": bool(summary["objective_converged"]),
                "max_abs_coil_current": float(summary["max_abs_coil_current"]),
            }
        )
    max_coil_current = [float(entry["max_abs_coil_current"]) for entry in entries]
    x_point_flux_error = [float(entry["x_point_flux_error"]) for entry in entries]
    divertor_rms = [float(entry["divertor_rms"]) for entry in entries]
    divertor_max_abs = [float(entry["divertor_max_abs"]) for entry in entries]
    objective_converged = [bool(entry["objective_converged"]) for entry in entries]
    checks = {
        "max_abs_coil_current_monotone": _is_monotone_non_decreasing(max_coil_current),
        "x_point_flux_error_monotone": _is_monotone_non_decreasing(x_point_flux_error),
        "divertor_rms_monotone": _is_monotone_non_decreasing(divertor_rms),
        "divertor_max_abs_monotone": _is_monotone_non_decreasing(divertor_max_abs),
        "objective_converged_all": all(objective_converged),
        "topology_errors_bounded": bool(
            max(float(entry["final_tracking_error_norm"]) for entry in entries)
            <= TOPOLOGY_THRESHOLDS["max_final_tracking_error_norm"]
            and max(float(entry["x_point_position_error"]) for entry in entries)
            <= TOPOLOGY_THRESHOLDS["max_x_point_position_error"]
            and max(float(entry["x_point_flux_error"]) for entry in entries)
            <= TOPOLOGY_THRESHOLDS["max_x_point_flux_error"]
            and max(float(entry["divertor_rms"]) for entry in entries)
            <= TOPOLOGY_THRESHOLDS["max_divertor_rms"]
            and max(float(entry["divertor_max_abs"]) for entry in entries)
            <= TOPOLOGY_THRESHOLDS["max_divertor_max_abs"]
        ),
    }
    return {
        "entries": entries,
        "checks": checks,
        "passes_thresholds": all(checks.values()),
    }


def _run_topology_actuator_slew_sweep(
    tmp_path: Path, *, topology_template_cfg: dict[str, Any]
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for slew_limit in ACTUATOR_SLEW_LIMIT_SWEEP:
        cfg = _write_tracking_config(
            tmp_path / f"topology_actuator_slew_{slew_limit}.json",
            template_cfg=topology_template_cfg,
        )
        summary = run_free_boundary_tracking(
            config_file=str(cfg),
            shot_steps=4,
            gain=8.0,
            verbose=False,
            kernel_factory=FusionKernel,
            disturbance_callback=_make_coil_kick_disturbance(2.0),
            control_dt_s=0.1,
            coil_actuator_tau_s=0.05,
            coil_slew_limits=float(slew_limit),
            stop_on_convergence=False,
        )
        entries.append(
            {
                "coil_slew_limit": float(slew_limit),
                "max_abs_actuator_lag": float(summary["max_abs_actuator_lag"]),
                "x_point_flux_error": float(summary["x_point_flux_error"]),
                "divertor_rms": float(summary["divertor_rms"]),
                "divertor_max_abs": float(summary["divertor_max_abs"]),
                "objective_converged": bool(summary["objective_converged"]),
                "max_abs_coil_current": float(summary["max_abs_coil_current"]),
                "final_tracking_error_norm": float(summary["final_tracking_error_norm"]),
            }
        )
    max_abs_actuator_lag = [float(entry["max_abs_actuator_lag"]) for entry in entries]
    x_point_flux_error = [float(entry["x_point_flux_error"]) for entry in entries]
    divertor_rms = [float(entry["divertor_rms"]) for entry in entries]
    divertor_max_abs = [float(entry["divertor_max_abs"]) for entry in entries]
    max_abs_coil_current = [float(entry["max_abs_coil_current"]) for entry in entries]
    objective_converged = [bool(entry["objective_converged"]) for entry in entries]
    checks = {
        "max_abs_actuator_lag_monotone": _is_monotone_non_decreasing(max_abs_actuator_lag),
        "x_point_flux_error_monotone": _is_monotone_non_decreasing(x_point_flux_error),
        "divertor_rms_monotone": _is_monotone_non_decreasing(divertor_rms),
        "divertor_max_abs_monotone": _is_monotone_non_decreasing(divertor_max_abs),
        "max_abs_coil_current_monotone": _is_monotone_non_increasing(max_abs_coil_current),
        "objective_converged_all": all(objective_converged),
        "topology_errors_bounded": bool(
            max(float(entry["final_tracking_error_norm"]) for entry in entries)
            <= TOPOLOGY_THRESHOLDS["max_final_tracking_error_norm"]
            and max(float(entry["x_point_flux_error"]) for entry in entries)
            <= TOPOLOGY_THRESHOLDS["max_x_point_flux_error"]
            and max(float(entry["divertor_rms"]) for entry in entries)
            <= TOPOLOGY_THRESHOLDS["max_divertor_rms"]
            and max(float(entry["divertor_max_abs"]) for entry in entries)
            <= TOPOLOGY_THRESHOLDS["max_divertor_max_abs"]
        ),
    }
    return {
        "entries": entries,
        "checks": checks,
        "passes_thresholds": all(checks.values()),
    }


def _run_topology_measurement_sweep(
    tmp_path: Path, *, topology_template_cfg: dict[str, Any]
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for scale in MEASUREMENT_SWEEP_SCALES:
        cfg = _write_tracking_config(
            tmp_path / f"topology_measurement_scale_{scale:.1f}.json",
            template_cfg=topology_template_cfg,
            tracking_cfg=(
                None if scale == 0.0 else _topology_measurement_tracking_cfg(scale, corrected=False)
            ),
        )
        summary = _run_measurement_fault(cfg)
        entries.append(
            {
                "scale": float(scale),
                "x_point_position_gap": abs(
                    float(summary["x_point_position_error"])
                    - float(summary["true_x_point_position_error"])
                ),
                "x_point_flux_gap": abs(
                    float(summary["x_point_flux_error"]) - float(summary["true_x_point_flux_error"])
                ),
                "divertor_rms_gap": abs(
                    float(summary["divertor_rms"]) - float(summary["true_divertor_rms"])
                ),
                "divertor_max_abs_gap": abs(
                    float(summary["divertor_max_abs"]) - float(summary["true_divertor_max_abs"])
                ),
                "max_abs_measurement_offset": float(summary["max_abs_measurement_offset"]),
                "objective_converged": bool(summary["objective_converged"]),
            }
        )
    x_point_position_gap = [float(entry["x_point_position_gap"]) for entry in entries]
    x_point_flux_gap = [float(entry["x_point_flux_gap"]) for entry in entries]
    divertor_rms_gap = [float(entry["divertor_rms_gap"]) for entry in entries]
    divertor_max_abs_gap = [float(entry["divertor_max_abs_gap"]) for entry in entries]
    measurement_offset = [float(entry["max_abs_measurement_offset"]) for entry in entries]
    checks = {
        "x_point_position_gap_monotone": _is_monotone_non_decreasing(x_point_position_gap),
        "x_point_flux_gap_monotone": _is_monotone_non_decreasing(x_point_flux_gap),
        "divertor_rms_gap_monotone": _is_monotone_non_decreasing(divertor_rms_gap),
        "divertor_max_abs_gap_monotone": _is_monotone_non_decreasing(divertor_max_abs_gap),
        "measurement_offset_monotone": _is_monotone_non_decreasing(measurement_offset),
    }
    return {
        "entries": entries,
        "checks": checks,
        "passes_thresholds": all(checks.values()),
    }


def _run_topology_corrected_measurement_sweep(
    tmp_path: Path, *, topology_template_cfg: dict[str, Any]
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for scale in MEASUREMENT_SWEEP_SCALES:
        cfg = _write_tracking_config(
            tmp_path / f"topology_measurement_corrected_scale_{scale:.1f}.json",
            template_cfg=topology_template_cfg,
            tracking_cfg=_topology_measurement_tracking_cfg(scale, corrected=True),
        )
        summary = _run_measurement_fault(cfg)
        entries.append(
            {
                "scale": float(scale),
                "x_point_position_gap": abs(
                    float(summary["x_point_position_error"])
                    - float(summary["true_x_point_position_error"])
                ),
                "x_point_flux_gap": abs(
                    float(summary["x_point_flux_error"]) - float(summary["true_x_point_flux_error"])
                ),
                "divertor_rms_gap": abs(
                    float(summary["divertor_rms"]) - float(summary["true_divertor_rms"])
                ),
                "divertor_max_abs_gap": abs(
                    float(summary["divertor_max_abs"]) - float(summary["true_divertor_max_abs"])
                ),
                "max_abs_measurement_offset": float(summary["max_abs_measurement_offset"]),
                "objective_converged": bool(summary["objective_converged"]),
            }
        )
    checks = {
        "max_x_point_position_gap": bool(
            max(float(entry["x_point_position_gap"]) for entry in entries)
            <= TOPOLOGY_CORRECTED_THRESHOLDS["max_x_point_position_gap"]
        ),
        "max_x_point_flux_gap": bool(
            max(float(entry["x_point_flux_gap"]) for entry in entries)
            <= TOPOLOGY_CORRECTED_THRESHOLDS["max_x_point_flux_gap"]
        ),
        "max_divertor_rms_gap": bool(
            max(float(entry["divertor_rms_gap"]) for entry in entries)
            <= TOPOLOGY_CORRECTED_THRESHOLDS["max_divertor_rms_gap"]
        ),
        "max_divertor_max_abs_gap": bool(
            max(float(entry["divertor_max_abs_gap"]) for entry in entries)
            <= TOPOLOGY_CORRECTED_THRESHOLDS["max_divertor_max_abs_gap"]
        ),
        "max_measurement_offset": bool(
            max(float(entry["max_abs_measurement_offset"]) for entry in entries)
            <= TOPOLOGY_CORRECTED_THRESHOLDS["max_measurement_offset"]
        ),
        "objective_converged_all": all(bool(entry["objective_converged"]) for entry in entries),
    }
    return {
        "entries": entries,
        "checks": checks,
        "passes_thresholds": all(checks.values()),
    }


def _run_topology_supervisor_measurement_sweep(
    tmp_path: Path, *, topology_template_cfg: dict[str, Any]
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for scale in MEASUREMENT_SWEEP_SCALES:
        tracking_cfg: dict[str, Any] = {
            "fallback_currents": [0.0, 0.0],
            "supervisor_limits": {"max_abs_actuator_lag": 2.0},
            "hold_steps_after_reject": 2,
        }
        if scale != 0.0:
            tracking_cfg.update(_topology_measurement_tracking_cfg(scale, corrected=False))
        cfg = _write_tracking_config(
            tmp_path / f"topology_supervisor_measurement_scale_{scale:.1f}.json",
            template_cfg=topology_template_cfg,
            tracking_cfg=tracking_cfg,
        )
        summary = _run_supervisor_fallback_kick(cfg)
        entries.append(
            {
                "scale": float(scale),
                "x_point_position_gap": abs(
                    float(summary["x_point_position_error"])
                    - float(summary["true_x_point_position_error"])
                ),
                "x_point_flux_gap": abs(
                    float(summary["x_point_flux_error"]) - float(summary["true_x_point_flux_error"])
                ),
                "divertor_rms_gap": abs(
                    float(summary["divertor_rms"]) - float(summary["true_divertor_rms"])
                ),
                "divertor_max_abs_gap": abs(
                    float(summary["divertor_max_abs"]) - float(summary["true_divertor_max_abs"])
                ),
                "max_abs_measurement_offset": float(summary["max_abs_measurement_offset"]),
                "objective_converged": bool(summary["objective_converged"]),
                "supervisor_active": bool(summary["supervisor_active"]),
                "supervisor_safe": bool(summary["supervisor_safe"]),
                "supervisor_intervention_count": int(summary["supervisor_intervention_count"]),
                "fallback_active_steps": int(summary["fallback_active_steps"]),
                "max_abs_actuator_lag": float(summary["max_abs_actuator_lag"]),
            }
        )
    checks = {
        "x_point_position_gap_monotone": _is_monotone_non_decreasing(
            [float(entry["x_point_position_gap"]) for entry in entries]
        ),
        "x_point_flux_gap_monotone": _is_monotone_non_decreasing(
            [float(entry["x_point_flux_gap"]) for entry in entries]
        ),
        "divertor_rms_gap_monotone": _is_monotone_non_decreasing(
            [float(entry["divertor_rms_gap"]) for entry in entries]
        ),
        "divertor_max_abs_gap_monotone": _is_monotone_non_decreasing(
            [float(entry["divertor_max_abs_gap"]) for entry in entries]
        ),
        "measurement_offset_monotone": _is_monotone_non_decreasing(
            [float(entry["max_abs_measurement_offset"]) for entry in entries]
        ),
        "supervisor_active_all": all(bool(entry["supervisor_active"]) for entry in entries),
        "supervisor_safe_all": all(bool(entry["supervisor_safe"]) for entry in entries),
        "intervention_all": all(
            int(entry["supervisor_intervention_count"]) >= 1 for entry in entries
        ),
        "fallback_all": all(int(entry["fallback_active_steps"]) >= 1 for entry in entries),
        "lag_bounded_all": all(
            float(entry["max_abs_actuator_lag"])
            <= SUPERVISOR_FALLBACK_THRESHOLDS["max_abs_actuator_lag"]
            for entry in entries
        ),
    }
    return {
        "entries": entries,
        "checks": checks,
        "passes_thresholds": all(checks.values()),
    }


def _run_topology_supervisor_corrected_measurement_sweep(
    tmp_path: Path, *, topology_template_cfg: dict[str, Any]
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for scale in MEASUREMENT_SWEEP_SCALES:
        cfg = _write_tracking_config(
            tmp_path / f"topology_supervisor_measurement_corrected_scale_{scale:.1f}.json",
            template_cfg=topology_template_cfg,
            tracking_cfg={
                "fallback_currents": [0.0, 0.0],
                "supervisor_limits": {"max_abs_actuator_lag": 2.0},
                "hold_steps_after_reject": 2,
                **_topology_measurement_tracking_cfg(scale, corrected=True),
            },
        )
        summary = _run_supervisor_fallback_kick(cfg)
        entries.append(
            {
                "scale": float(scale),
                "x_point_position_gap": abs(
                    float(summary["x_point_position_error"])
                    - float(summary["true_x_point_position_error"])
                ),
                "x_point_flux_gap": abs(
                    float(summary["x_point_flux_error"]) - float(summary["true_x_point_flux_error"])
                ),
                "divertor_rms_gap": abs(
                    float(summary["divertor_rms"]) - float(summary["true_divertor_rms"])
                ),
                "divertor_max_abs_gap": abs(
                    float(summary["divertor_max_abs"]) - float(summary["true_divertor_max_abs"])
                ),
                "max_abs_measurement_offset": float(summary["max_abs_measurement_offset"]),
                "objective_converged": bool(summary["objective_converged"]),
                "supervisor_active": bool(summary["supervisor_active"]),
                "supervisor_safe": bool(summary["supervisor_safe"]),
                "supervisor_intervention_count": int(summary["supervisor_intervention_count"]),
                "fallback_active_steps": int(summary["fallback_active_steps"]),
                "max_abs_actuator_lag": float(summary["max_abs_actuator_lag"]),
            }
        )
    checks = {
        "max_x_point_position_gap": bool(
            max(float(entry["x_point_position_gap"]) for entry in entries)
            <= TOPOLOGY_CORRECTED_THRESHOLDS["max_x_point_position_gap"]
        ),
        "max_x_point_flux_gap": bool(
            max(float(entry["x_point_flux_gap"]) for entry in entries)
            <= TOPOLOGY_CORRECTED_THRESHOLDS["max_x_point_flux_gap"]
        ),
        "max_divertor_rms_gap": bool(
            max(float(entry["divertor_rms_gap"]) for entry in entries)
            <= TOPOLOGY_CORRECTED_THRESHOLDS["max_divertor_rms_gap"]
        ),
        "max_divertor_max_abs_gap": bool(
            max(float(entry["divertor_max_abs_gap"]) for entry in entries)
            <= TOPOLOGY_CORRECTED_THRESHOLDS["max_divertor_max_abs_gap"]
        ),
        "max_measurement_offset": bool(
            max(float(entry["max_abs_measurement_offset"]) for entry in entries)
            <= TOPOLOGY_CORRECTED_THRESHOLDS["max_measurement_offset"]
        ),
        "objective_converged_all": all(bool(entry["objective_converged"]) for entry in entries),
        "supervisor_active_all": all(bool(entry["supervisor_active"]) for entry in entries),
        "supervisor_safe_all": all(bool(entry["supervisor_safe"]) for entry in entries),
        "intervention_all": all(
            int(entry["supervisor_intervention_count"]) >= 1 for entry in entries
        ),
        "fallback_all": all(int(entry["fallback_active_steps"]) >= 1 for entry in entries),
        "lag_bounded_all": all(
            float(entry["max_abs_actuator_lag"])
            <= SUPERVISOR_FALLBACK_THRESHOLDS["max_abs_actuator_lag"]
            for entry in entries
        ),
    }
    return {
        "entries": entries,
        "checks": checks,
        "passes_thresholds": all(checks.values()),
    }


def run_campaign() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="scpn_free_boundary_acceptance_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        template_cfg = _build_tracking_template(tmp_path)

        nominal_cfg = _write_tracking_config(
            tmp_path / "nominal.json",
            template_cfg=template_cfg,
        )
        nominal = _run_nominal(nominal_cfg)

        kick_cfg = _write_tracking_config(
            tmp_path / "kick.json",
            template_cfg=template_cfg,
        )
        kick = _run_kick(kick_cfg)

        measurement_cfg = _write_tracking_config(
            tmp_path / "measurement.json",
            template_cfg=template_cfg,
            tracking_cfg={
                "measurement_bias": {"shape_flux": [0.03, -0.02]},
                "measurement_drift_per_step": {"shape_flux": [0.004, -0.003]},
            },
        )
        measurement_fault = _run_measurement_fault(measurement_cfg)

        corrected_cfg = _write_tracking_config(
            tmp_path / "measurement_corrected.json",
            template_cfg=template_cfg,
            tracking_cfg={
                "measurement_bias": {"shape_flux": [0.03, -0.02]},
                "measurement_drift_per_step": {"shape_flux": [0.004, -0.003]},
                "measurement_correction_bias": {"shape_flux": [0.03, -0.02]},
                "measurement_correction_drift_per_step": {"shape_flux": [0.004, -0.003]},
            },
        )
        corrected = _run_measurement_fault(corrected_cfg)

        latency_cfg = _write_tracking_config(
            tmp_path / "latency.json",
            template_cfg=template_cfg,
            tracking_cfg=_latency_tracking_cfg(2, corrected=False),
        )
        latency_fault = _run_latency_fault(latency_cfg)

        latency_corrected_cfg = _write_tracking_config(
            tmp_path / "latency_corrected.json",
            template_cfg=template_cfg,
            tracking_cfg=_latency_tracking_cfg(2, corrected=True),
        )
        latency_corrected = _run_latency_fault(latency_corrected_cfg)

        topology_template_cfg = _build_topology_tracking_template(
            tmp_path, template_cfg=template_cfg
        )
        topology_cfg = _write_tracking_config(
            tmp_path / "topology_kick.json",
            template_cfg=topology_template_cfg,
        )
        topology_kick = _run_topology_kick(topology_cfg)

        topology_supervisor_measurement_fault_cfg = _write_tracking_config(
            tmp_path / "topology_supervisor_measurement_fault.json",
            template_cfg=topology_template_cfg,
            tracking_cfg={
                "fallback_currents": [0.0, 0.0],
                "supervisor_limits": {"max_abs_actuator_lag": 2.0},
                "hold_steps_after_reject": 2,
                **_topology_measurement_tracking_cfg(corrected=False),
            },
        )
        topology_supervisor_measurement_fault = _run_supervisor_fallback_kick(
            topology_supervisor_measurement_fault_cfg
        )

        topology_supervisor_measurement_corrected_cfg = _write_tracking_config(
            tmp_path / "topology_supervisor_measurement_corrected.json",
            template_cfg=topology_template_cfg,
            tracking_cfg={
                "fallback_currents": [0.0, 0.0],
                "supervisor_limits": {"max_abs_actuator_lag": 2.0},
                "hold_steps_after_reject": 2,
                **_topology_measurement_tracking_cfg(corrected=True),
            },
        )
        topology_supervisor_measurement_corrected = _run_supervisor_fallback_kick(
            topology_supervisor_measurement_corrected_cfg
        )

        topology_supervisor_measurement_latency_cfg = _write_tracking_config(
            tmp_path / "topology_supervisor_measurement_latency.json",
            template_cfg=topology_template_cfg,
            tracking_cfg=_merge_tracking_cfg(
                {
                    "fallback_currents": [0.0, 0.0],
                    "supervisor_limits": {"max_abs_actuator_lag": 2.0},
                    "hold_steps_after_reject": 2,
                },
                _topology_measurement_tracking_cfg(corrected=False),
                _latency_tracking_cfg(2, corrected=False),
            ),
        )
        topology_supervisor_measurement_latency_fault = _run_supervisor_fallback_kick(
            topology_supervisor_measurement_latency_cfg
        )

        topology_supervisor_measurement_latency_corrected_cfg = _write_tracking_config(
            tmp_path / "topology_supervisor_measurement_latency_corrected.json",
            template_cfg=topology_template_cfg,
            tracking_cfg=_merge_tracking_cfg(
                {
                    "fallback_currents": [0.0, 0.0],
                    "supervisor_limits": {"max_abs_actuator_lag": 2.0},
                    "hold_steps_after_reject": 2,
                },
                _topology_measurement_tracking_cfg(corrected=True),
                _latency_tracking_cfg(2, corrected=True),
            ),
        )
        topology_supervisor_measurement_latency_corrected = _run_supervisor_fallback_kick(
            topology_supervisor_measurement_latency_corrected_cfg
        )

        topology_combined_measurement_cfg = _write_tracking_config(
            tmp_path / "topology_combined_measurement.json",
            template_cfg=topology_template_cfg,
            tracking_cfg=_topology_measurement_tracking_cfg(corrected=False),
        )
        topology_combined_measurement_fault = _run_topology_kick(topology_combined_measurement_cfg)

        topology_combined_corrected_cfg = _write_tracking_config(
            tmp_path / "topology_combined_measurement_corrected.json",
            template_cfg=topology_template_cfg,
            tracking_cfg=_topology_measurement_tracking_cfg(corrected=True),
        )
        topology_combined_measurement_corrected = _run_topology_kick(
            topology_combined_corrected_cfg
        )

        topology_measurement_cfg = _write_tracking_config(
            tmp_path / "topology_measurement.json",
            template_cfg=topology_template_cfg,
            tracking_cfg=_topology_measurement_tracking_cfg(corrected=False),
        )
        topology_measurement_fault = _run_measurement_fault(topology_measurement_cfg)

        topology_corrected_cfg = _write_tracking_config(
            tmp_path / "topology_measurement_corrected.json",
            template_cfg=topology_template_cfg,
            tracking_cfg=_topology_measurement_tracking_cfg(corrected=True),
        )
        topology_measurement_corrected = _run_measurement_fault(topology_corrected_cfg)

        topology_measurement_latency_cfg = _write_tracking_config(
            tmp_path / "topology_measurement_latency.json",
            template_cfg=topology_template_cfg,
            tracking_cfg=_merge_tracking_cfg(
                _topology_measurement_tracking_cfg(corrected=False),
                _latency_tracking_cfg(2, corrected=False),
            ),
        )
        topology_measurement_latency_fault = _run_measurement_fault(
            topology_measurement_latency_cfg
        )

        topology_measurement_latency_corrected_cfg = _write_tracking_config(
            tmp_path / "topology_measurement_latency_corrected.json",
            template_cfg=topology_template_cfg,
            tracking_cfg=_merge_tracking_cfg(
                _topology_measurement_tracking_cfg(corrected=True),
                _latency_tracking_cfg(2, corrected=True),
            ),
        )
        topology_measurement_latency_corrected = _run_measurement_fault(
            topology_measurement_latency_corrected_cfg
        )

        unsupervised_safety_cfg = _write_tracking_config(
            tmp_path / "unsupervised_safety_reference.json",
            template_cfg=topology_template_cfg,
        )
        unsupervised_safety_reference = _run_supervisor_fallback_kick(unsupervised_safety_cfg)

        supervisor_fallback_cfg = _write_tracking_config(
            tmp_path / "supervisor_fallback.json",
            template_cfg=topology_template_cfg,
            tracking_cfg={
                "fallback_currents": [0.0, 0.0],
                "supervisor_limits": {"max_abs_actuator_lag": 2.0},
                "hold_steps_after_reject": 2,
            },
        )
        supervisor_fallback = _run_supervisor_fallback_kick(supervisor_fallback_cfg)

        measurement_sweep = _run_measurement_sweep(tmp_path, template_cfg=template_cfg)
        corrected_measurement_sweep = _run_corrected_measurement_sweep(
            tmp_path, template_cfg=template_cfg
        )
        latency_step_sweep = _run_latency_step_sweep(tmp_path, template_cfg=template_cfg)
        corrected_latency_step_sweep = _run_corrected_latency_step_sweep(
            tmp_path, template_cfg=template_cfg
        )
        actuator_slew_sweep = _run_actuator_slew_sweep(tmp_path, template_cfg=template_cfg)
        coil_kick_scale_sweep = _run_coil_kick_scale_sweep(tmp_path, template_cfg=template_cfg)
        topology_kick_scale_sweep = _run_topology_kick_scale_sweep(
            tmp_path,
            topology_template_cfg=topology_template_cfg,
        )
        topology_actuator_slew_sweep = _run_topology_actuator_slew_sweep(
            tmp_path,
            topology_template_cfg=topology_template_cfg,
        )
        topology_measurement_sweep = _run_topology_measurement_sweep(
            tmp_path,
            topology_template_cfg=topology_template_cfg,
        )
        topology_corrected_measurement_sweep = _run_topology_corrected_measurement_sweep(
            tmp_path,
            topology_template_cfg=topology_template_cfg,
        )
        topology_supervisor_measurement_sweep = _run_topology_supervisor_measurement_sweep(
            tmp_path,
            topology_template_cfg=topology_template_cfg,
        )
        topology_supervisor_corrected_measurement_sweep = (
            _run_topology_supervisor_corrected_measurement_sweep(
                tmp_path,
                topology_template_cfg=topology_template_cfg,
            )
        )

    scenarios = {
        "nominal": {
            "summary": nominal,
            **_evaluate_nominal(nominal),
        },
        "coil_kick": {
            "summary": kick,
            **_evaluate_kick(kick),
        },
        "measurement_fault_uncorrected": {
            "summary": measurement_fault,
            **_evaluate_measurement_fault(measurement_fault),
        },
        "measurement_fault_corrected": {
            "summary": corrected,
            **_evaluate_corrected(corrected),
        },
        "measurement_latency_uncorrected": {
            "summary": latency_fault,
            **_evaluate_latency_fault(latency_fault),
        },
        "measurement_latency_corrected": {
            "summary": latency_corrected,
            **_evaluate_latency_corrected(latency_corrected),
        },
        "x_point_divertor_kick": {
            "summary": topology_kick,
            **_evaluate_topology(topology_kick),
        },
        "x_point_divertor_supervisor_measurement_fault_uncorrected": {
            "summary": topology_supervisor_measurement_fault,
            **_combine_evaluations(
                _evaluate_topology_measurement_fault(topology_supervisor_measurement_fault),
                _evaluate_supervisor_only(topology_supervisor_measurement_fault),
            ),
        },
        "x_point_divertor_supervisor_measurement_fault_corrected": {
            "summary": topology_supervisor_measurement_corrected,
            **_combine_evaluations(
                _evaluate_topology_corrected(topology_supervisor_measurement_corrected),
                _evaluate_supervisor_only(topology_supervisor_measurement_corrected),
            ),
        },
        "x_point_divertor_supervisor_measurement_latency_uncorrected": {
            "summary": topology_supervisor_measurement_latency_fault,
            **_combine_evaluations(
                _evaluate_topology_measurement_fault(
                    topology_supervisor_measurement_latency_fault,
                    thresholds=TOPOLOGY_LATENCY_MEASUREMENT_THRESHOLDS,
                ),
                _evaluate_supervisor_only(topology_supervisor_measurement_latency_fault),
                _evaluate_latency_fault(topology_supervisor_measurement_latency_fault),
            ),
        },
        "x_point_divertor_supervisor_measurement_latency_corrected": {
            "summary": topology_supervisor_measurement_latency_corrected,
            **_combine_evaluations(
                _evaluate_topology_corrected(
                    topology_supervisor_measurement_latency_corrected,
                    thresholds=TOPOLOGY_SUPERVISOR_LATENCY_CORRECTED_THRESHOLDS,
                ),
                _evaluate_supervisor_only(topology_supervisor_measurement_latency_corrected),
                _evaluate_latency_corrected(
                    topology_supervisor_measurement_latency_corrected,
                    thresholds=TOPOLOGY_SUPERVISOR_LATENCY_CORRECTED_THRESHOLDS,
                ),
            ),
        },
        "x_point_divertor_combined_fault_uncorrected": {
            "summary": topology_combined_measurement_fault,
            **_evaluate_topology_measurement_fault(topology_combined_measurement_fault),
        },
        "x_point_divertor_combined_fault_corrected": {
            "summary": topology_combined_measurement_corrected,
            **_evaluate_topology_corrected(topology_combined_measurement_corrected),
        },
        "x_point_divertor_measurement_fault_uncorrected": {
            "summary": topology_measurement_fault,
            **_evaluate_topology_measurement_fault(topology_measurement_fault),
        },
        "x_point_divertor_measurement_fault_corrected": {
            "summary": topology_measurement_corrected,
            **_evaluate_topology_corrected(topology_measurement_corrected),
        },
        "x_point_divertor_measurement_latency_uncorrected": {
            "summary": topology_measurement_latency_fault,
            **_combine_evaluations(
                _evaluate_topology_measurement_fault(
                    topology_measurement_latency_fault,
                    thresholds=TOPOLOGY_LATENCY_MEASUREMENT_THRESHOLDS,
                ),
                _evaluate_latency_fault(topology_measurement_latency_fault),
            ),
        },
        "x_point_divertor_measurement_latency_corrected": {
            "summary": topology_measurement_latency_corrected,
            **_combine_evaluations(
                _evaluate_topology_corrected(topology_measurement_latency_corrected),
                _evaluate_latency_corrected(topology_measurement_latency_corrected),
            ),
        },
        "supervisor_fallback_kick": {
            "summary": supervisor_fallback,
            "unsupervised_reference": unsupervised_safety_reference,
            **_evaluate_supervisor_fallback(
                supervisor_fallback,
                unsupervised_reference=unsupervised_safety_reference,
            ),
        },
    }
    sweeps = {
        "measurement_fault_scale": measurement_sweep,
        "measurement_fault_corrected_scale": corrected_measurement_sweep,
        "measurement_latency_steps": latency_step_sweep,
        "measurement_latency_corrected_steps": corrected_latency_step_sweep,
        "actuator_slew_limit": actuator_slew_sweep,
        "coil_kick_scale": coil_kick_scale_sweep,
        "topology_kick_scale": topology_kick_scale_sweep,
        "topology_actuator_slew_limit": topology_actuator_slew_sweep,
        "topology_measurement_fault_scale": topology_measurement_sweep,
        "topology_measurement_corrected_scale": topology_corrected_measurement_sweep,
        "topology_supervisor_measurement_fault_scale": topology_supervisor_measurement_sweep,
        "topology_supervisor_measurement_corrected_scale": topology_supervisor_corrected_measurement_sweep,
    }
    passes_thresholds = all(
        bool(entry["passes_thresholds"]) for entry in scenarios.values()
    ) and all(bool(entry["passes_thresholds"]) for entry in sweeps.values())
    return {
        "benchmark": "free_boundary_tracking_acceptance",
        "steps_per_scenario": 4,
        "scenarios": scenarios,
        "sweeps": sweeps,
        "passes_thresholds": passes_thresholds,
    }


def generate_report() -> dict[str, Any]:
    t0 = time.perf_counter()
    campaign = run_campaign()
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": float(time.perf_counter() - t0),
        "free_boundary_tracking_acceptance": campaign,
    }


def render_markdown(report: dict[str, Any]) -> str:
    campaign = report["free_boundary_tracking_acceptance"]
    lines = [
        "# Free-Boundary Tracking Acceptance",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Runtime: `{report['runtime_seconds']:.3f} s`",
        f"- Steps per scenario: `{campaign['steps_per_scenario']}`",
        f"- Pass: `{'YES' if campaign['passes_thresholds'] else 'NO'}`",
        "",
        "## Scenarios",
        "",
    ]
    for name, data in campaign["scenarios"].items():
        summary = data["summary"]
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Pass: `{'YES' if data['passes_thresholds'] else 'NO'}`",
                f"- Final tracking error: `{summary['final_tracking_error_norm']:.6e}`",
                f"- Final true tracking error: `{summary['final_true_tracking_error_norm']:.6e}`",
                f"- Shape RMS: `{summary['shape_rms']:.6e}`",
                f"- Max coil current: `{summary['max_abs_coil_current']:.6e}`",
                f"- Max measurement offset: `{summary['max_abs_measurement_offset']:.6e}`",
                f"- Supervisor interventions: `{summary['supervisor_intervention_count']}`",
                f"- Fallback active steps: `{summary['fallback_active_steps']}`",
                "",
            ]
        )
        if "lag_reduction_factor" in data:
            lines.extend(
                [
                    f"- Lag reduction factor vs unsupervised: `{data['lag_reduction_factor']:.2f}`",
                    "",
                ]
            )
    lines.extend(
        [
            "## Sweeps",
            "",
            "### measurement_fault_scale",
            "",
        ]
    )
    for entry in campaign["sweeps"]["measurement_fault_scale"]["entries"]:
        lines.append(
            "- "
            f"scale `{entry['scale']:.1f}`: gap `{entry['measured_true_gap']:.6e}`, "
            f"offset `{entry['max_abs_measurement_offset']:.6e}`"
        )
    lines.extend(
        [
            "",
            "### actuator_slew_limit",
            "",
        ]
    )
    for entry in campaign["sweeps"]["actuator_slew_limit"]["entries"]:
        lines.append(
            "- "
            f"slew `{entry['coil_slew_limit']:.3e}`: max lag `{entry['max_abs_actuator_lag']:.6e}`, "
            f"mean lag `{entry['mean_abs_actuator_lag']:.6e}`"
        )
    lines.extend(
        [
            "",
            "### measurement_fault_corrected_scale",
            "",
        ]
    )
    for entry in campaign["sweeps"]["measurement_fault_corrected_scale"]["entries"]:
        lines.append(
            "- "
            f"scale `{entry['scale']:.1f}`: gap `{entry['measured_true_gap']:.6e}`, "
            f"offset `{entry['max_abs_measurement_offset']:.6e}`"
        )
    lines.extend(
        [
            "",
            "### measurement_latency_steps",
            "",
        ]
    )
    for entry in campaign["sweeps"]["measurement_latency_steps"]["entries"]:
        lines.append(
            "- "
            f"steps `{entry['latency_steps']}`: delayed err `{entry['delayed_observation_error_norm']:.6e}`, "
            f"estimated err `{entry['estimated_observation_error_norm']:.6e}`"
        )
    lines.extend(
        [
            "",
            "### measurement_latency_corrected_steps",
            "",
        ]
    )
    for entry in campaign["sweeps"]["measurement_latency_corrected_steps"]["entries"]:
        lines.append(
            "- "
            f"steps `{entry['latency_steps']}`: delayed err `{entry['delayed_observation_error_norm']:.6e}`, "
            f"estimated err `{entry['estimated_observation_error_norm']:.6e}`"
        )
    lines.extend(
        [
            "",
            "### coil_kick_scale",
            "",
        ]
    )
    for entry in campaign["sweeps"]["coil_kick_scale"]["entries"]:
        lines.append(
            "- "
            f"scale `{entry['kick_scale']:.1f}`: max coil `{entry['max_abs_coil_current']:.6e}`, "
            f"final err `{entry['final_tracking_error_norm']:.6e}`"
        )
    lines.extend(
        [
            "",
            "### topology_kick_scale",
            "",
        ]
    )
    for entry in campaign["sweeps"]["topology_kick_scale"]["entries"]:
        lines.append(
            "- "
            f"scale `{entry['kick_scale']:.1f}`: x-flux err `{entry['x_point_flux_error']:.6e}`, "
            f"divertor rms `{entry['divertor_rms']:.6e}`, max coil `{entry['max_abs_coil_current']:.6e}`"
        )
    lines.extend(
        [
            "",
            "### topology_actuator_slew_limit",
            "",
        ]
    )
    for entry in campaign["sweeps"]["topology_actuator_slew_limit"]["entries"]:
        lines.append(
            "- "
            f"slew `{entry['coil_slew_limit']:.3e}`: max lag `{entry['max_abs_actuator_lag']:.6e}`, "
            f"x-flux err `{entry['x_point_flux_error']:.6e}`, divertor rms `{entry['divertor_rms']:.6e}`"
        )
    lines.extend(
        [
            "",
            "### topology_measurement_fault_scale",
            "",
        ]
    )
    for entry in campaign["sweeps"]["topology_measurement_fault_scale"]["entries"]:
        lines.append(
            "- "
            f"scale `{entry['scale']:.1f}`: x-pos gap `{entry['x_point_position_gap']:.6e}`, "
            f"x-flux gap `{entry['x_point_flux_gap']:.6e}`, divertor rms gap `{entry['divertor_rms_gap']:.6e}`"
        )
    lines.extend(
        [
            "",
            "### topology_measurement_corrected_scale",
            "",
        ]
    )
    for entry in campaign["sweeps"]["topology_measurement_corrected_scale"]["entries"]:
        lines.append(
            "- "
            f"scale `{entry['scale']:.1f}`: x-pos gap `{entry['x_point_position_gap']:.6e}`, "
            f"x-flux gap `{entry['x_point_flux_gap']:.6e}`, divertor rms gap `{entry['divertor_rms_gap']:.6e}`"
        )
    lines.extend(
        [
            "",
            "### topology_supervisor_measurement_fault_scale",
            "",
        ]
    )
    for entry in campaign["sweeps"]["topology_supervisor_measurement_fault_scale"]["entries"]:
        lines.append(
            "- "
            f"scale `{entry['scale']:.1f}`: x-pos gap `{entry['x_point_position_gap']:.6e}`, "
            f"x-flux gap `{entry['x_point_flux_gap']:.6e}`, lag `{entry['max_abs_actuator_lag']:.6e}`"
        )
    lines.extend(
        [
            "",
            "### topology_supervisor_measurement_corrected_scale",
            "",
        ]
    )
    for entry in campaign["sweeps"]["topology_supervisor_measurement_corrected_scale"]["entries"]:
        lines.append(
            "- "
            f"scale `{entry['scale']:.1f}`: x-pos gap `{entry['x_point_position_gap']:.6e}`, "
            f"x-flux gap `{entry['x_point_flux_gap']:.6e}`, lag `{entry['max_abs_actuator_lag']:.6e}`"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run free-boundary tracking acceptance campaign.")
    parser.add_argument(
        "--json-out",
        default=str(ROOT / "validation" / "reports" / "free_boundary_tracking_acceptance.json"),
        help="Path to write JSON report.",
    )
    parser.add_argument(
        "--md-out",
        default=str(ROOT / "validation" / "reports" / "free_boundary_tracking_acceptance.md"),
        help="Path to write Markdown report.",
    )
    args = parser.parse_args(argv)

    report = generate_report()
    json_path = Path(args.json_out)
    md_path = Path(args.md_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
