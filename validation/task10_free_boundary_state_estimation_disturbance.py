# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Task 10 Free-Boundary State Estimation And Disturbance Rejection
"""Task 10: free-boundary state estimation and disturbance rejection acceptance gate."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scpn_fusion.control.free_boundary_supervisory_control import (
    FreeBoundaryTarget,
    run_free_boundary_supervisory_simulation,
)


class _AcceptanceKernel:
    """Deterministic diverted free-boundary plant used for task10 acceptance."""

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


def _require_int(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    out = int(value)
    if out < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return out


def _require_finite(name: str, value: Any, minimum: float | None = None) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite.")
    if minimum is not None and out < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return out


def _recovery_steps(
    trace: dict[str, list[float] | list[list[float]]],
    *,
    start_index: int,
    axis_threshold_m: float = 0.05,
    xpoint_threshold_m: float = 0.06,
    hold_steps: int = 6,
) -> int:
    axis = [float(v) for v in trace["axis_error_m"]]
    xpoint = [float(v) for v in trace["xpoint_error_m"]]
    n = len(axis)
    for i in range(start_index, max(start_index, n - hold_steps + 1)):
        stable = True
        for j in range(i, i + hold_steps):
            if axis[j] > axis_threshold_m or xpoint[j] > xpoint_threshold_m:
                stable = False
                break
        if stable:
            return i - start_index
    return max(0, n - start_index)


def _run_nominal(seed: int, shot_length: int, control_dt_s: float) -> dict[str, Any]:
    return run_free_boundary_supervisory_simulation(
        config_file="task10_dummy.json",
        shot_length=shot_length,
        target=FreeBoundaryTarget(6.0, 0.0, 5.02, -3.48),
        control_dt_s=control_dt_s,
        disturbance_start_step=16,
        disturbance_per_step_ma=0.08,
        coil_kick_step=24,
        coil_kick_vector=(0.20, -0.18, 0.16, -0.14),
        sensor_bias_step=34,
        sensor_bias_vector=(0.018, -0.015, 0.024, -0.020),
        actuator_bias_step=None,
        measurement_noise_std=0.0015,
        current_target_bounds=(7.0, 10.0),
        coil_current_limits=(-1.5, 1.5),
        coil_delta_limit=0.35,
        supervisor_q95_floor=3.0,
        supervisor_beta_n_ceiling=3.0,
        supervisor_disruption_risk_ceiling=0.95,
        supervisor_warning_risk_score_threshold=10.0,
        supervisor_guarded_risk_score_threshold=20.0,
        estimator_measurement_gain=0.60,
        estimator_bias_gain=0.09,
        estimator_bias_decay=0.986,
        estimator_actuator_bias_gain=0.18,
        estimator_actuator_bias_decay=0.997,
        estimator_max_actuator_bias=0.35,
        return_trace=True,
        save_plot=False,
        verbose=False,
        rng_seed=seed,
        kernel_factory=_AcceptanceKernel,
    )


def _run_faulted(seed: int, shot_length: int, control_dt_s: float) -> dict[str, Any]:
    return run_free_boundary_supervisory_simulation(
        config_file="task10_dummy.json",
        shot_length=shot_length,
        target=FreeBoundaryTarget(6.0, 0.0, 5.02, -3.48),
        control_dt_s=control_dt_s,
        disturbance_start_step=14,
        disturbance_per_step_ma=0.10,
        coil_kick_step=22,
        coil_kick_vector=(0.30, -0.27, 0.22, -0.19),
        sensor_bias_step=30,
        sensor_bias_vector=(0.032, -0.028, 0.040, -0.034),
        actuator_bias_step=38,
        actuator_bias_vector=(0.08, -0.07, 0.06, -0.05),
        measurement_noise_std=0.0018,
        current_target_bounds=(7.0, 10.0),
        coil_current_limits=(-1.5, 1.5),
        coil_delta_limit=0.35,
        supervisor_q95_floor=3.0,
        supervisor_beta_n_ceiling=3.0,
        supervisor_disruption_risk_ceiling=0.95,
        supervisor_warning_risk_score_threshold=10.0,
        supervisor_guarded_risk_score_threshold=20.0,
        estimator_measurement_gain=0.60,
        estimator_bias_gain=0.09,
        estimator_bias_decay=0.986,
        estimator_actuator_bias_gain=0.18,
        estimator_actuator_bias_decay=0.997,
        estimator_max_actuator_bias=0.35,
        return_trace=True,
        save_plot=False,
        verbose=False,
        rng_seed=seed,
        kernel_factory=_AcceptanceKernel,
    )


def run_campaign(
    *,
    seed: int = 42,
    shot_length: int = 84,
    control_dt_s: float = 0.05,
) -> dict[str, Any]:
    seed_i = _require_int("seed", seed, 0)
    steps = _require_int("shot_length", shot_length, 48)
    dt_s = _require_finite("control_dt_s", control_dt_s, 1e-4)

    nominal = _run_nominal(seed_i, steps, dt_s)
    faulted = _run_faulted(seed_i + 35, steps, dt_s)

    nominal_trace = nominal["trace"]
    faulted_trace = faulted["trace"]
    sensor_bias_recovery_steps = _recovery_steps(faulted_trace, start_index=30)
    actuator_bias_recovery_steps = _recovery_steps(faulted_trace, start_index=38)
    axis_tracking_degradation_ratio = float(
        faulted["p95_axis_error_m"] / max(nominal["p95_axis_error_m"], 1e-9)
    )
    xpoint_tracking_degradation_ratio = float(
        faulted["p95_xpoint_error_m"] / max(nominal["p95_xpoint_error_m"], 1e-9)
    )

    thresholds = {
        "max_fault_mean_estimation_error_m": 0.04,
        "max_fault_mean_actuator_bias_estimation_error": 0.02,
        "max_fault_final_actuator_bias_estimation_error": 0.01,
        "max_fault_max_uncertainty_norm": 0.08,
        "max_sensor_bias_recovery_steps": 12,
        "max_actuator_bias_recovery_steps": 4,
        "min_fault_stabilization_rate": 0.95,
        "max_axis_tracking_degradation_ratio": 1.70,
        "max_xpoint_tracking_degradation_ratio": 1.20,
        "max_fault_p95_axis_error_m": 0.055,
        "max_fault_p95_xpoint_error_m": 0.045,
    }

    failure_reasons: list[str] = []
    if faulted["mean_estimation_error_m"] > thresholds["max_fault_mean_estimation_error_m"]:
        failure_reasons.append("fault_mean_estimation_error_m")
    if (
        faulted["mean_actuator_bias_estimation_error"]
        > thresholds["max_fault_mean_actuator_bias_estimation_error"]
    ):
        failure_reasons.append("fault_mean_actuator_bias_estimation_error")
    if (
        faulted["final_actuator_bias_estimation_error"]
        > thresholds["max_fault_final_actuator_bias_estimation_error"]
    ):
        failure_reasons.append("fault_final_actuator_bias_estimation_error")
    if faulted["max_uncertainty_norm"] > thresholds["max_fault_max_uncertainty_norm"]:
        failure_reasons.append("fault_max_uncertainty_norm")
    if sensor_bias_recovery_steps > thresholds["max_sensor_bias_recovery_steps"]:
        failure_reasons.append("sensor_bias_recovery_steps")
    if actuator_bias_recovery_steps > thresholds["max_actuator_bias_recovery_steps"]:
        failure_reasons.append("actuator_bias_recovery_steps")
    if faulted["stabilization_rate"] < thresholds["min_fault_stabilization_rate"]:
        failure_reasons.append("fault_stabilization_rate")
    if axis_tracking_degradation_ratio > thresholds["max_axis_tracking_degradation_ratio"]:
        failure_reasons.append("axis_tracking_degradation_ratio")
    if xpoint_tracking_degradation_ratio > thresholds["max_xpoint_tracking_degradation_ratio"]:
        failure_reasons.append("xpoint_tracking_degradation_ratio")
    if faulted["p95_axis_error_m"] > thresholds["max_fault_p95_axis_error_m"]:
        failure_reasons.append("fault_p95_axis_error_m")
    if faulted["p95_xpoint_error_m"] > thresholds["max_fault_p95_xpoint_error_m"]:
        failure_reasons.append("fault_p95_xpoint_error_m")

    return {
        "seed": seed_i,
        "task10_free_boundary_state_estimation_disturbance": {
            "nominal": {
                "summary": {key: value for key, value in nominal.items() if key != "trace"},
            },
            "faulted": {
                "summary": {key: value for key, value in faulted.items() if key != "trace"},
                "sensor_bias_recovery_steps": sensor_bias_recovery_steps,
                "actuator_bias_recovery_steps": actuator_bias_recovery_steps,
                "axis_tracking_degradation_ratio": axis_tracking_degradation_ratio,
                "xpoint_tracking_degradation_ratio": xpoint_tracking_degradation_ratio,
            },
            "thresholds": thresholds,
            "failure_reasons": failure_reasons,
            "passes_thresholds": bool(len(failure_reasons) == 0),
        },
    }


def generate_report(**kwargs: Any) -> dict[str, Any]:
    out = run_campaign(**kwargs)
    out["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    return out


def render_markdown(report: dict[str, Any]) -> str:
    g = report["task10_free_boundary_state_estimation_disturbance"]
    nominal = g["nominal"]["summary"]
    faulted = g["faulted"]
    faulted_summary = faulted["summary"]
    th = g["thresholds"]
    lines = [
        "# Task 10 Free-Boundary State Estimation And Disturbance Rejection",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Overall pass: `{'YES' if g['passes_thresholds'] else 'NO'}`",
        "",
        "## Nominal Baseline",
        "",
        f"- Nominal P95 axis error: `{nominal['p95_axis_error_m']:.4f} m`",
        f"- Nominal P95 X-point error: `{nominal['p95_xpoint_error_m']:.4f} m`",
        f"- Nominal stabilization rate: `{nominal['stabilization_rate']:.3f}`",
        "",
        "## Observer Performance",
        "",
        f"- Faulted mean state-estimation error: `{faulted_summary['mean_estimation_error_m']:.4f} m` (threshold `<= {th['max_fault_mean_estimation_error_m']:.3f}`)",
        f"- Faulted mean actuator-bias estimation error: `{faulted_summary['mean_actuator_bias_estimation_error']:.4f}` (threshold `<= {th['max_fault_mean_actuator_bias_estimation_error']:.3f}`)",
        f"- Faulted final actuator-bias estimation error: `{faulted_summary['final_actuator_bias_estimation_error']:.4f}` (threshold `<= {th['max_fault_final_actuator_bias_estimation_error']:.3f}`)",
        f"- Faulted max uncertainty norm: `{faulted_summary['max_uncertainty_norm']:.4f}` (threshold `<= {th['max_fault_max_uncertainty_norm']:.3f}`)",
        "",
        "## Disturbance Rejection",
        "",
        f"- Sensor-bias recovery steps: `{faulted['sensor_bias_recovery_steps']}` (threshold `<= {th['max_sensor_bias_recovery_steps']}`)",
        f"- Actuator-bias recovery steps: `{faulted['actuator_bias_recovery_steps']}` (threshold `<= {th['max_actuator_bias_recovery_steps']}`)",
        f"- Axis tracking degradation ratio: `{faulted['axis_tracking_degradation_ratio']:.3f}` (threshold `<= {th['max_axis_tracking_degradation_ratio']:.2f}`)",
        f"- X-point tracking degradation ratio: `{faulted['xpoint_tracking_degradation_ratio']:.3f}` (threshold `<= {th['max_xpoint_tracking_degradation_ratio']:.2f}`)",
        f"- Faulted stabilization rate: `{faulted_summary['stabilization_rate']:.3f}` (threshold `>= {th['min_fault_stabilization_rate']:.2f}`)",
        f"- Faulted P95 axis error: `{faulted_summary['p95_axis_error_m']:.4f} m` (threshold `<= {th['max_fault_p95_axis_error_m']:.3f}`)",
        f"- Faulted P95 X-point error: `{faulted_summary['p95_xpoint_error_m']:.4f} m` (threshold `<= {th['max_fault_p95_xpoint_error_m']:.3f}`)",
        "",
    ]
    if g["failure_reasons"]:
        lines.append(f"- Failure reasons: `{', '.join(g['failure_reasons'])}`")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shot-length", type=int, default=84)
    parser.add_argument("--control-dt-s", type=float, default=0.05)
    parser.add_argument(
        "--output-json",
        default=str(
            ROOT
            / "validation"
            / "reports"
            / "task10_free_boundary_state_estimation_disturbance.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        default=str(
            ROOT / "validation" / "reports" / "task10_free_boundary_state_estimation_disturbance.md"
        ),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    report = generate_report(
        seed=args.seed,
        shot_length=args.shot_length,
        control_dt_s=args.control_dt_s,
    )
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    g = report["task10_free_boundary_state_estimation_disturbance"]
    print("Task 10 free-boundary state-estimation/disturbance validation complete.")
    print(
        "Summary -> "
        f"sensor_recovery_steps={g['faulted']['sensor_bias_recovery_steps']}, "
        f"actuator_recovery_steps={g['faulted']['actuator_bias_recovery_steps']}, "
        f"mean_actuator_bias_error={g['faulted']['summary']['mean_actuator_bias_estimation_error']:.4f}, "
        f"passes_thresholds={g['passes_thresholds']}"
    )
    if args.strict and not g["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
