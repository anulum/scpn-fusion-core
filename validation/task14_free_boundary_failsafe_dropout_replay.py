# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Task 14 Free-Boundary Fail-Safe Dropout Replay
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Task 14: fail-safe degradation under actuator loss and diagnostic dropout."""

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
    """Deterministic diverted free-boundary plant used for task14 acceptance."""

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


def _run_faulted(seed: int, shot_length: int, control_dt_s: float) -> dict[str, Any]:
    return run_free_boundary_supervisory_simulation(
        config_file="task14_dummy.json",
        shot_length=shot_length,
        target=FreeBoundaryTarget(6.0, 0.0, 5.02, -3.48),
        control_dt_s=control_dt_s,
        disturbance_start_step=12,
        disturbance_stop_step=34,
        disturbance_per_step_ma=0.10,
        disturbance_recovery_step=42,
        disturbance_recovery_per_step_ma=-0.09,
        diagnostic_dropout_step=28,
        diagnostic_dropout_duration_steps=16,
        diagnostic_dropout_mask=(1.0, 1.0, 0.0, 0.0),
        coil_kick_step=20,
        coil_kick_vector=(0.24, -0.22, 0.18, -0.16),
        sensor_bias_step=24,
        sensor_bias_vector=(0.020, -0.018, 0.026, -0.022),
        sensor_bias_clear_step=58,
        actuator_bias_step=26,
        actuator_bias_vector=(0.040, -0.032, 0.028, -0.024),
        actuator_bias_clear_step=60,
        actuator_dropout_step=34,
        actuator_dropout_duration_steps=18,
        actuator_dropout_mask=(0.0, 1.0, 0.0, 1.0),
        measurement_noise_std=0.0016,
        current_target_bounds=(7.0, 10.0),
        coil_current_limits=(-1.5, 1.5),
        coil_delta_limit=0.35,
        supervisor_total_action_l1_limit=0.78,
        supervisor_q95_floor=3.0,
        supervisor_beta_n_ceiling=3.0,
        supervisor_disruption_risk_ceiling=0.95,
        supervisor_warning_risk_score_threshold=10.0,
        supervisor_guarded_risk_score_threshold=20.0,
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
        rng_seed=seed,
        kernel_factory=_AcceptanceKernel,
    )


def run_campaign(
    *,
    seed: int = 42,
    shot_length: int = 96,
    control_dt_s: float = 0.05,
) -> dict[str, Any]:
    seed_i = _require_int("seed", seed, 0)
    steps = _require_int("shot_length", shot_length, 64)
    dt_s = _require_finite("control_dt_s", control_dt_s, 1e-4)

    a = _run_faulted(seed_i, steps, dt_s)
    b = _run_faulted(seed_i, steps, dt_s)
    trace_a = a["trace"]
    trace_b = b["trace"]
    replay_deterministic = bool(
        a["replay_signature"] == b["replay_signature"]
        and trace_a["true_states"] == trace_b["true_states"]
        and trace_a["measured_states"] == trace_b["measured_states"]
        and trace_a["estimated_states"] == trace_b["estimated_states"]
        and trace_a["actions"] == trace_b["actions"]
        and trace_a["applied_actions"] == trace_b["applied_actions"]
        and trace_a["diagnostic_dropout"] == trace_b["diagnostic_dropout"]
        and trace_a["actuator_dropout"] == trace_b["actuator_dropout"]
        and trace_a["degraded_mode"] == trace_b["degraded_mode"]
    )

    late_window = slice(max(steps - 16, 0), steps)
    late_axis = np.asarray(trace_a["axis_error_m"][late_window], dtype=np.float64)
    late_xpoint = np.asarray(trace_a["xpoint_error_m"][late_window], dtype=np.float64)
    late_alert = np.asarray(trace_a["alert_level"][late_window], dtype=np.int64)
    late_degraded = np.asarray(trace_a["degraded_mode"][late_window], dtype=np.int64)

    recovery_window = {
        "late_p95_axis_error_m": float(np.percentile(late_axis, 95)) if late_axis.size else 0.0,
        "late_p95_xpoint_error_m": float(np.percentile(late_xpoint, 95)) if late_xpoint.size else 0.0,
        "late_max_alert_level": int(np.max(late_alert)) if late_alert.size else 0,
        "late_degraded_mode_count": int(np.sum(late_degraded)),
    }

    thresholds = {
        "require_replay_determinism": True,
        "max_p95_axis_error_m": 0.085,
        "max_p95_xpoint_error_m": 0.090,
        "min_stabilization_rate": 0.95,
        "max_abs_action": 0.35,
        "max_action_l1": 0.78,
        "max_abs_coil_current": 1.50,
        "min_diagnostic_dropout_count": 16,
        "min_actuator_dropout_count": 18,
        "min_degraded_mode_count": 24,
        "min_fallback_mode_count": 12,
        "min_recovery_transition_count": 2,
        "max_failsafe_trip_count": 0,
        "max_final_alert_level": 1,
        "max_late_p95_axis_error_m": 0.055,
        "max_late_p95_xpoint_error_m": 0.060,
        "max_late_alert_level": 1,
        "max_late_degraded_mode_count": 0,
    }

    failure_reasons: list[str] = []
    if thresholds["require_replay_determinism"] and not replay_deterministic:
        failure_reasons.append("replay_determinism")
    if a["p95_axis_error_m"] > thresholds["max_p95_axis_error_m"]:
        failure_reasons.append("p95_axis_error_m")
    if a["p95_xpoint_error_m"] > thresholds["max_p95_xpoint_error_m"]:
        failure_reasons.append("p95_xpoint_error_m")
    if a["stabilization_rate"] < thresholds["min_stabilization_rate"]:
        failure_reasons.append("stabilization_rate")
    if a["max_abs_action"] > thresholds["max_abs_action"] + 1e-9:
        failure_reasons.append("max_abs_action")
    if a["max_action_l1"] > thresholds["max_action_l1"] + 1e-9:
        failure_reasons.append("max_action_l1")
    if a["max_abs_coil_current"] > thresholds["max_abs_coil_current"] + 1e-9:
        failure_reasons.append("max_abs_coil_current")
    if a["diagnostic_dropout_count"] < thresholds["min_diagnostic_dropout_count"]:
        failure_reasons.append("diagnostic_dropout_count")
    if a["actuator_dropout_count"] < thresholds["min_actuator_dropout_count"]:
        failure_reasons.append("actuator_dropout_count")
    if a["degraded_mode_count"] < thresholds["min_degraded_mode_count"]:
        failure_reasons.append("degraded_mode_count")
    if a["fallback_mode_count"] < thresholds["min_fallback_mode_count"]:
        failure_reasons.append("fallback_mode_count")
    if a["recovery_transition_count"] < thresholds["min_recovery_transition_count"]:
        failure_reasons.append("recovery_transition_count")
    if a["failsafe_trip_count"] > thresholds["max_failsafe_trip_count"]:
        failure_reasons.append("failsafe_trip_count")
    if a["final_alert_level"] > thresholds["max_final_alert_level"]:
        failure_reasons.append("final_alert_level")
    if recovery_window["late_p95_axis_error_m"] > thresholds["max_late_p95_axis_error_m"]:
        failure_reasons.append("late_p95_axis_error_m")
    if recovery_window["late_p95_xpoint_error_m"] > thresholds["max_late_p95_xpoint_error_m"]:
        failure_reasons.append("late_p95_xpoint_error_m")
    if recovery_window["late_max_alert_level"] > thresholds["max_late_alert_level"]:
        failure_reasons.append("late_max_alert_level")
    if recovery_window["late_degraded_mode_count"] > thresholds["max_late_degraded_mode_count"]:
        failure_reasons.append("late_degraded_mode_count")

    return {
        "seed": seed_i,
        "task14_free_boundary_failsafe_dropout_replay": {
            "faulted_replay": {
                "summary": {key: value for key, value in a.items() if key != "trace"},
                "replay_deterministic": replay_deterministic,
            },
            "recovery_window": recovery_window,
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
    g = report["task14_free_boundary_failsafe_dropout_replay"]
    c = g["faulted_replay"]["summary"]
    r = g["recovery_window"]
    th = g["thresholds"]
    lines = [
        "# Task 14 Free-Boundary Fail-Safe Dropout Replay",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Overall pass: `{'YES' if g['passes_thresholds'] else 'NO'}`",
        "",
        "## Fault Envelope",
        "",
        f"- Replay deterministic: `{'YES' if g['faulted_replay']['replay_deterministic'] else 'NO'}`",
        f"- Diagnostic dropout count: `{c['diagnostic_dropout_count']}` (threshold `>= {th['min_diagnostic_dropout_count']}`)",
        f"- Actuator dropout count: `{c['actuator_dropout_count']}` (threshold `>= {th['min_actuator_dropout_count']}`)",
        f"- Degraded mode count: `{c['degraded_mode_count']}` (threshold `>= {th['min_degraded_mode_count']}`)",
        f"- Fallback mode count: `{c['fallback_mode_count']}` (threshold `>= {th['min_fallback_mode_count']}`)",
        "",
        "## Fail-Safe Degradation",
        "",
        f"- P95 axis error: `{c['p95_axis_error_m']:.4f} m` (threshold `<= {th['max_p95_axis_error_m']:.3f}`)",
        f"- P95 X-point error: `{c['p95_xpoint_error_m']:.4f} m` (threshold `<= {th['max_p95_xpoint_error_m']:.3f}`)",
        f"- Stabilization rate: `{c['stabilization_rate']:.3f}` (threshold `>= {th['min_stabilization_rate']:.2f}`)",
        f"- Failsafe trip count: `{c['failsafe_trip_count']}` (threshold `<= {th['max_failsafe_trip_count']}`)",
        f"- Recovery transitions: `{c['recovery_transition_count']}` (threshold `>= {th['min_recovery_transition_count']}`)",
        "",
        "## Recovery Window",
        "",
        f"- Final alert level: `{c['final_alert_level']}` (threshold `<= {th['max_final_alert_level']}`)",
        f"- Late P95 axis error: `{r['late_p95_axis_error_m']:.4f} m` (threshold `<= {th['max_late_p95_axis_error_m']:.3f}`)",
        f"- Late P95 X-point error: `{r['late_p95_xpoint_error_m']:.4f} m` (threshold `<= {th['max_late_p95_xpoint_error_m']:.3f}`)",
        f"- Late max alert level: `{r['late_max_alert_level']}` (threshold `<= {th['max_late_alert_level']}`)",
        f"- Late degraded mode count: `{r['late_degraded_mode_count']}` (threshold `<= {th['max_late_degraded_mode_count']}`)",
        "",
    ]
    if g["failure_reasons"]:
        lines.append(f"- Failure reasons: `{', '.join(g['failure_reasons'])}`")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shot-length", type=int, default=96)
    parser.add_argument("--control-dt-s", type=float, default=0.05)
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / "task14_free_boundary_failsafe_dropout_replay.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "task14_free_boundary_failsafe_dropout_replay.md"),
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

    g = report["task14_free_boundary_failsafe_dropout_replay"]
    c = g["faulted_replay"]["summary"]
    r = g["recovery_window"]
    print("Task 14 free-boundary fail-safe dropout replay validation complete.")
    print(
        "Summary -> "
        f"replay_deterministic={g['faulted_replay']['replay_deterministic']}, "
        f"degraded_mode_count={c['degraded_mode_count']}, "
        f"late_max_alert_level={r['late_max_alert_level']}, "
        f"passes_thresholds={g['passes_thresholds']}"
    )
    if args.strict and not g["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
