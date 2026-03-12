# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Task 13 Free-Boundary Disruption Policy Recovery
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Task 13: disruption-aware policy escalation and recovery for free-boundary control."""

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
    """Deterministic diverted free-boundary plant used for task13 acceptance."""

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


def run_campaign(
    *,
    seed: int = 42,
    shot_length: int = 104,
    control_dt_s: float = 0.05,
) -> dict[str, Any]:
    seed_i = _require_int("seed", seed, 0)
    steps = _require_int("shot_length", shot_length, 72)
    dt_s = _require_finite("control_dt_s", control_dt_s, 1e-4)

    summary = run_free_boundary_supervisory_simulation(
        config_file="task13_dummy.json",
        shot_length=steps,
        target=FreeBoundaryTarget(6.0, 0.0, 5.02, -3.48),
        control_dt_s=dt_s,
        disturbance_start_step=10,
        disturbance_stop_step=30,
        disturbance_per_step_ma=0.095,
        disturbance_recovery_step=50,
        disturbance_recovery_per_step_ma=-0.095,
        coil_kick_step=16,
        coil_kick_vector=(0.28, -0.25, 0.22, -0.18),
        sensor_bias_step=20,
        sensor_bias_vector=(0.040, -0.035, 0.048, -0.041),
        sensor_bias_clear_step=56,
        actuator_bias_step=24,
        actuator_bias_vector=(0.075, -0.060, 0.050, -0.040),
        actuator_bias_clear_step=60,
        measurement_noise_std=0.0018,
        current_target_bounds=(7.0, 10.0),
        coil_current_limits=(-1.5, 1.5),
        coil_delta_limit=0.35,
        supervisor_total_action_l1_limit=0.78,
        supervisor_axis_r_bounds_m=(5.93, 6.07),
        supervisor_axis_z_bounds_m=(-0.10, 0.10),
        supervisor_xpoint_r_bounds_m=(4.96, 5.08),
        supervisor_xpoint_z_bounds_m=(-3.54, -3.42),
        supervisor_q95_floor=4.08,
        supervisor_beta_n_ceiling=2.18,
        supervisor_disruption_risk_ceiling=0.23,
        supervisor_alert_recovery_hold_steps=3,
        supervisor_fallback_hold_steps=4,
        supervisor_fallback_action_scale=0.45,
        estimator_measurement_gain=0.60,
        estimator_bias_gain=0.09,
        estimator_bias_decay=0.986,
        estimator_actuator_bias_gain=0.18,
        estimator_actuator_bias_decay=0.997,
        estimator_max_actuator_bias=0.35,
        return_trace=True,
        save_plot=False,
        verbose=False,
        rng_seed=seed_i,
        kernel_factory=_AcceptanceKernel,
    )
    trace = summary["trace"]
    late_window = slice(max(steps - 12, 0), steps)
    late_axis = np.asarray(trace["axis_error_m"][late_window], dtype=np.float64)
    late_xpoint = np.asarray(trace["xpoint_error_m"][late_window], dtype=np.float64)
    late_risk = np.asarray(trace["disruption_risk"][late_window], dtype=np.float64)
    late_alert = np.asarray(trace["alert_level"][late_window], dtype=np.int64)

    recovery_metrics = {
        "late_mean_disruption_risk": float(np.mean(late_risk)) if late_risk.size else 0.0,
        "late_p95_axis_error_m": float(np.percentile(late_axis, 95)) if late_axis.size else 0.0,
        "late_p95_xpoint_error_m": float(np.percentile(late_xpoint, 95)) if late_xpoint.size else 0.0,
        "late_max_alert_level": int(np.max(late_alert)) if late_alert.size else 0,
    }

    thresholds = {
        "max_p95_axis_error_m": 0.064,
        "max_p95_xpoint_error_m": 0.058,
        "min_stabilization_rate": 0.97,
        "max_abs_action": 0.35,
        "max_action_l1": 0.78,
        "max_abs_coil_current": 1.50,
        "min_warning_mode_count": 6,
        "min_guarded_mode_count": 5,
        "min_fallback_mode_count": 8,
        "required_peak_alert_level": 3,
        "min_alert_transition_count": 4,
        "min_recovery_transition_count": 2,
        "max_final_alert_level": 1,
        "max_late_mean_disruption_risk": 0.21,
        "max_late_p95_axis_error_m": 0.055,
        "max_late_p95_xpoint_error_m": 0.055,
        "max_late_max_alert_level": 1,
        "max_invariant_violation_count": 0,
        "max_failsafe_trip_count": 0,
        "min_max_disruption_risk": 0.235,
    }

    failure_reasons: list[str] = []
    if summary["p95_axis_error_m"] > thresholds["max_p95_axis_error_m"]:
        failure_reasons.append("p95_axis_error_m")
    if summary["p95_xpoint_error_m"] > thresholds["max_p95_xpoint_error_m"]:
        failure_reasons.append("p95_xpoint_error_m")
    if summary["stabilization_rate"] < thresholds["min_stabilization_rate"]:
        failure_reasons.append("stabilization_rate")
    if summary["max_abs_action"] > thresholds["max_abs_action"] + 1e-9:
        failure_reasons.append("max_abs_action")
    if summary["max_action_l1"] > thresholds["max_action_l1"] + 1e-9:
        failure_reasons.append("max_action_l1")
    if summary["max_abs_coil_current"] > thresholds["max_abs_coil_current"] + 1e-9:
        failure_reasons.append("max_abs_coil_current")
    if summary["warning_mode_count"] < thresholds["min_warning_mode_count"]:
        failure_reasons.append("warning_mode_count")
    if summary["guarded_mode_count"] < thresholds["min_guarded_mode_count"]:
        failure_reasons.append("guarded_mode_count")
    if summary["fallback_mode_count"] < thresholds["min_fallback_mode_count"]:
        failure_reasons.append("fallback_mode_count")
    if summary["peak_alert_level"] != thresholds["required_peak_alert_level"]:
        failure_reasons.append("peak_alert_level")
    if summary["alert_transition_count"] < thresholds["min_alert_transition_count"]:
        failure_reasons.append("alert_transition_count")
    if summary["recovery_transition_count"] < thresholds["min_recovery_transition_count"]:
        failure_reasons.append("recovery_transition_count")
    if summary["final_alert_level"] > thresholds["max_final_alert_level"]:
        failure_reasons.append("final_alert_level")
    if recovery_metrics["late_mean_disruption_risk"] > thresholds["max_late_mean_disruption_risk"]:
        failure_reasons.append("late_mean_disruption_risk")
    if recovery_metrics["late_p95_axis_error_m"] > thresholds["max_late_p95_axis_error_m"]:
        failure_reasons.append("late_p95_axis_error_m")
    if recovery_metrics["late_p95_xpoint_error_m"] > thresholds["max_late_p95_xpoint_error_m"]:
        failure_reasons.append("late_p95_xpoint_error_m")
    if recovery_metrics["late_max_alert_level"] > thresholds["max_late_max_alert_level"]:
        failure_reasons.append("late_max_alert_level")
    if summary["invariant_violation_count"] > thresholds["max_invariant_violation_count"]:
        failure_reasons.append("invariant_violation_count")
    if summary["failsafe_trip_count"] > thresholds["max_failsafe_trip_count"]:
        failure_reasons.append("failsafe_trip_count")
    if summary["max_disruption_risk"] < thresholds["min_max_disruption_risk"]:
        failure_reasons.append("max_disruption_risk")

    return {
        "seed": seed_i,
        "task13_free_boundary_disruption_policy_recovery": {
            "scenario": {
                "control_dt_s": float(dt_s),
                "shot_length": int(steps),
                "policy_thresholds": {
                    "q95_floor": 4.08,
                    "beta_n_ceiling": 2.18,
                    "disruption_risk_ceiling": 0.23,
                    "alert_recovery_hold_steps": 3,
                },
            },
            "policy_closed_loop": {key: value for key, value in summary.items() if key != "trace"},
            "recovery_window": recovery_metrics,
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
    g = report["task13_free_boundary_disruption_policy_recovery"]
    c = g["policy_closed_loop"]
    r = g["recovery_window"]
    s = g["scenario"]["policy_thresholds"]
    th = g["thresholds"]
    lines = [
        "# Task 13 Free-Boundary Disruption Policy Recovery",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Overall pass: `{'YES' if g['passes_thresholds'] else 'NO'}`",
        "",
        "## Alert Regimes",
        "",
        f"- q95 floor: `{s['q95_floor']:.2f}`",
        f"- beta_N ceiling: `{s['beta_n_ceiling']:.2f}`",
        f"- disruption-risk ceiling: `{s['disruption_risk_ceiling']:.2f}`",
        f"- warning mode count: `{c['warning_mode_count']}` (threshold `>= {th['min_warning_mode_count']}`)",
        f"- guarded mode count: `{c['guarded_mode_count']}` (threshold `>= {th['min_guarded_mode_count']}`)",
        f"- fallback mode count: `{c['fallback_mode_count']}` (threshold `>= {th['min_fallback_mode_count']}`)",
        f"- peak alert level: `{c['peak_alert_level']}` (required `{th['required_peak_alert_level']}`)",
        f"- final alert level: `{c['final_alert_level']}` (threshold `<= {th['max_final_alert_level']}`)",
        f"- alert transitions: `{c['alert_transition_count']}` (threshold `>= {th['min_alert_transition_count']}`)",
        f"- recovery transitions: `{c['recovery_transition_count']}` (threshold `>= {th['min_recovery_transition_count']}`)",
        "",
        "## Recovery Window",
        "",
        f"- Late mean disruption risk: `{r['late_mean_disruption_risk']:.4f}` (threshold `<= {th['max_late_mean_disruption_risk']:.2f}`)",
        f"- Late P95 axis error: `{r['late_p95_axis_error_m']:.4f} m` (threshold `<= {th['max_late_p95_axis_error_m']:.3f}`)",
        f"- Late P95 X-point error: `{r['late_p95_xpoint_error_m']:.4f} m` (threshold `<= {th['max_late_p95_xpoint_error_m']:.3f}`)",
        f"- Late max alert level: `{r['late_max_alert_level']}` (threshold `<= {th['max_late_max_alert_level']}`)",
        "",
        "## Closed-Loop Outcome",
        "",
        f"- P95 axis error: `{c['p95_axis_error_m']:.4f} m` (threshold `<= {th['max_p95_axis_error_m']:.3f}`)",
        f"- P95 X-point error: `{c['p95_xpoint_error_m']:.4f} m` (threshold `<= {th['max_p95_xpoint_error_m']:.3f}`)",
        f"- Stabilization rate: `{c['stabilization_rate']:.3f}` (threshold `>= {th['min_stabilization_rate']:.2f}`)",
        f"- Max disruption risk: `{c['max_disruption_risk']:.4f}` (threshold `>= {th['min_max_disruption_risk']:.2f}`)",
        f"- Invariant violation count: `{c['invariant_violation_count']}` (threshold `<= {th['max_invariant_violation_count']}`)",
        f"- Failsafe trip count: `{c['failsafe_trip_count']}` (threshold `<= {th['max_failsafe_trip_count']}`)",
        "",
    ]
    if g["failure_reasons"]:
        lines.append(f"- Failure reasons: `{', '.join(g['failure_reasons'])}`")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shot-length", type=int, default=104)
    parser.add_argument("--control-dt-s", type=float, default=0.05)
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / "task13_free_boundary_disruption_policy_recovery.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "task13_free_boundary_disruption_policy_recovery.md"),
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

    g = report["task13_free_boundary_disruption_policy_recovery"]
    c = g["policy_closed_loop"]
    r = g["recovery_window"]
    print("Task 13 free-boundary disruption-policy recovery validation complete.")
    print(
        "Summary -> "
        f"peak_alert_level={c['peak_alert_level']}, "
        f"recovery_transitions={c['recovery_transition_count']}, "
        f"late_mean_disruption_risk={r['late_mean_disruption_risk']:.4f}, "
        f"passes_thresholds={g['passes_thresholds']}"
    )
    if args.strict and not g["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
