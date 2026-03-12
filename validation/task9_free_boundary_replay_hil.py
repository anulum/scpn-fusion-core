# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Task 9 Free-Boundary Replay And HIL Gate
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Task 9: deterministic replay, watchdog trip, and HIL compatibility for free-boundary control."""

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
from scpn_fusion.control.hil_harness import run_hil_benchmark_detailed


class _AcceptanceKernel:
    """Deterministic diverted free-boundary plant used for replay/HIL validation."""

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


def _run_nominal(seed: int, shot_length: int, control_dt_s: float) -> dict[str, Any]:
    return run_free_boundary_supervisory_simulation(
        config_file="task9_dummy.json",
        shot_length=shot_length,
        target=FreeBoundaryTarget(6.0, 0.0, 5.02, -3.48),
        control_dt_s=control_dt_s,
        disturbance_start_step=18,
        disturbance_per_step_ma=0.08,
        coil_kick_step=26,
        coil_kick_vector=(0.20, -0.18, 0.16, -0.14),
        sensor_bias_step=32,
        sensor_bias_vector=(0.016, -0.014, 0.022, -0.019),
        measurement_noise_std=0.0012,
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
        return_trace=True,
        save_plot=False,
        verbose=False,
        rng_seed=seed,
        kernel_factory=_AcceptanceKernel,
    )


def _run_faulted(seed: int, shot_length: int, control_dt_s: float) -> dict[str, Any]:
    return run_free_boundary_supervisory_simulation(
        config_file="task9_dummy.json",
        shot_length=shot_length,
        target=FreeBoundaryTarget(6.0, 0.0, 5.02, -3.48),
        control_dt_s=control_dt_s,
        disturbance_start_step=12,
        disturbance_per_step_ma=0.18,
        coil_kick_step=18,
        coil_kick_vector=(0.45, -0.40, 0.36, -0.32),
        sensor_bias_step=22,
        sensor_bias_vector=(0.12, -0.10, 0.15, -0.13),
        measurement_noise_std=0.0025,
        current_target_bounds=(7.0, 10.0),
        coil_current_limits=(-1.5, 1.5),
        coil_delta_limit=0.35,
        supervisor_q95_floor=3.0,
        supervisor_beta_n_ceiling=3.0,
        supervisor_disruption_risk_ceiling=0.95,
        supervisor_warning_risk_score_threshold=10.0,
        supervisor_guarded_risk_score_threshold=20.0,
        estimator_measurement_gain=0.60,
        estimator_bias_gain=0.10,
        estimator_bias_decay=0.986,
        return_trace=True,
        save_plot=False,
        verbose=False,
        rng_seed=seed,
        kernel_factory=_AcceptanceKernel,
    )


def run_campaign(
    *,
    seed: int = 42,
    shot_length: int = 72,
    control_dt_s: float = 0.05,
    hil_steps: int = 512,
) -> dict[str, Any]:
    seed_i = _require_int("seed", seed, 0)
    steps = _require_int("shot_length", shot_length, 24)
    dt_s = _require_finite("control_dt_s", control_dt_s, 1e-4)
    hil_n = _require_int("hil_steps", hil_steps, 64)

    thresholds = {
        "require_replay_determinism": True,
        "require_watchdog_trip": True,
        "max_hil_p95_us": 1000.0,
        "max_fault_max_abs_action": 0.35,
        "max_fault_max_abs_coil_current": 1.5,
        "max_fault_p95_axis_error_m": 0.16,
        "max_fault_p95_xpoint_error_m": 0.18,
        "min_fault_failsafe_trip_count": 1,
        "min_fault_max_risk_score": 1.35,
    }

    nominal_a = _run_nominal(seed_i, steps, dt_s)
    nominal_b = _run_nominal(seed_i, steps, dt_s)
    faulted = _run_faulted(seed_i + 101, steps, dt_s)

    trace_a = nominal_a["trace"]
    trace_b = nominal_b["trace"]
    replay_deterministic = bool(
        nominal_a["replay_signature"] == nominal_b["replay_signature"]
        and trace_a["true_states"] == trace_b["true_states"]
        and trace_a["measured_states"] == trace_b["measured_states"]
        and trace_a["estimated_states"] == trace_b["estimated_states"]
        and trace_a["actions"] == trace_b["actions"]
        and trace_a["supervisor_interventions"] == trace_b["supervisor_interventions"]
    )

    watchdog_trip = bool(
        faulted["failsafe_trip_count"] >= thresholds["min_fault_failsafe_trip_count"]
        and faulted["supervisor_intervention_count"] > nominal_a["supervisor_intervention_count"]
        and faulted["max_risk_score"] >= thresholds["min_fault_max_risk_score"]
    )

    hil = run_hil_benchmark_detailed(
        n_steps=hil_n,
        rng_seed=seed_i + 500,
        state_dim=8,
        control_dim=4,
    )

    failure_reasons: list[str] = []
    if thresholds["require_replay_determinism"] and not replay_deterministic:
        failure_reasons.append("replay_determinism")
    if thresholds["require_watchdog_trip"] and not watchdog_trip:
        failure_reasons.append("watchdog_trip")
    if hil["p95_us"] > thresholds["max_hil_p95_us"]:
        failure_reasons.append("hil_p95_us")
    if faulted["max_abs_action"] > thresholds["max_fault_max_abs_action"] + 1e-9:
        failure_reasons.append("fault_max_abs_action")
    if faulted["max_abs_coil_current"] > thresholds["max_fault_max_abs_coil_current"] + 1e-9:
        failure_reasons.append("fault_max_abs_coil_current")
    if faulted["p95_axis_error_m"] > thresholds["max_fault_p95_axis_error_m"]:
        failure_reasons.append("fault_p95_axis_error_m")
    if faulted["p95_xpoint_error_m"] > thresholds["max_fault_p95_xpoint_error_m"]:
        failure_reasons.append("fault_p95_xpoint_error_m")

    return {
        "seed": seed_i,
        "task9_free_boundary_replay_hil": {
            "nominal_replay": {
                "summary": {key: value for key, value in nominal_a.items() if key != "trace"},
                "replay_deterministic": replay_deterministic,
            },
            "faulted_watchdog": {
                "summary": {key: value for key, value in faulted.items() if key != "trace"},
                "watchdog_trip": watchdog_trip,
            },
            "hil_compatibility": hil,
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
    g = report["task9_free_boundary_replay_hil"]
    nominal = g["nominal_replay"]
    faulted = g["faulted_watchdog"]
    hil = g["hil_compatibility"]
    th = g["thresholds"]
    lines = [
        "# Task 9 Free-Boundary Replay And HIL Gate",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Overall pass: `{'YES' if g['passes_thresholds'] else 'NO'}`",
        "",
        "## Replay Determinism",
        "",
        f"- Replay deterministic: `{nominal['replay_deterministic']}`",
        f"- Replay signature: `{nominal['summary']['replay_signature']}`",
        f"- Nominal P95 axis error: `{nominal['summary']['p95_axis_error_m']:.4f} m`",
        f"- Nominal P95 X-point error: `{nominal['summary']['p95_xpoint_error_m']:.4f} m`",
        "",
        "## Watchdog And Fail-Safe",
        "",
        f"- Watchdog trip observed: `{faulted['watchdog_trip']}`",
        f"- Faulted supervisor interventions: `{faulted['summary']['supervisor_intervention_count']}`",
        f"- Faulted failsafe trip count: `{faulted['summary']['failsafe_trip_count']}` (threshold `>= {th['min_fault_failsafe_trip_count']}`)",
        f"- Faulted max risk score: `{faulted['summary']['max_risk_score']:.3f}` (threshold `>= {th['min_fault_max_risk_score']:.2f}`)",
        f"- Faulted final target Ip: `{faulted['summary']['final_target_ip_ma']:.3f} MA`",
        f"- Faulted max |action|: `{faulted['summary']['max_abs_action']:.3f}` (threshold `<= {th['max_fault_max_abs_action']:.2f}`)",
        f"- Faulted max |coil current|: `{faulted['summary']['max_abs_coil_current']:.3f}` (threshold `<= {th['max_fault_max_abs_coil_current']:.2f}`)",
        "",
        "## HIL Compatibility",
        "",
        f"- HIL P95 latency: `{hil['p95_us']:.2f} us` (threshold `<= {th['max_hil_p95_us']:.0f} us`)",
        f"- State-estimation P95: `{hil['stage_breakdown']['state_estimation_p95_us']:.2f} us`",
        f"- Controller-step P95: `{hil['stage_breakdown']['controller_step_p95_us']:.2f} us`",
        f"- Actuator-command P95: `{hil['stage_breakdown']['actuator_command_p95_us']:.2f} us`",
        "",
    ]
    if g["failure_reasons"]:
        lines.append(f"- Failure reasons: `{', '.join(g['failure_reasons'])}`")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shot-length", type=int, default=72)
    parser.add_argument("--control-dt-s", type=float, default=0.05)
    parser.add_argument("--hil-steps", type=int, default=512)
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / "task9_free_boundary_replay_hil.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "task9_free_boundary_replay_hil.md"),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    report = generate_report(
        seed=args.seed,
        shot_length=args.shot_length,
        control_dt_s=args.control_dt_s,
        hil_steps=args.hil_steps,
    )
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    g = report["task9_free_boundary_replay_hil"]
    print("Task 9 free-boundary replay/HIL validation complete.")
    print(
        "Summary -> "
        f"replay_deterministic={g['nominal_replay']['replay_deterministic']}, "
        f"watchdog_trip={g['faulted_watchdog']['watchdog_trip']}, "
        f"hil_p95_us={g['hil_compatibility']['p95_us']:.2f}, "
        f"passes_thresholds={g['passes_thresholds']}"
    )
    if args.strict and not g["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
