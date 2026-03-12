# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Task 8 Free-Boundary Supervisory Control
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Task 8: free-boundary closed-loop control with estimator and safety supervisor."""

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
    """Deterministic diverted free-boundary plant with coupled axis/X-point dynamics."""

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
    shot_length: int = 72,
    control_dt_s: float = 0.05,
) -> dict[str, Any]:
    seed_i = _require_int("seed", seed, 0)
    steps = _require_int("shot_length", shot_length, 24)
    dt_s = _require_finite("control_dt_s", control_dt_s, 1e-4)

    target = FreeBoundaryTarget(
        r_axis_m=6.0,
        z_axis_m=0.0,
        x_point_r_m=5.02,
        x_point_z_m=-3.48,
    )
    summary = run_free_boundary_supervisory_simulation(
        config_file="task8_dummy.json",
        shot_length=steps,
        target=target,
        control_dt_s=dt_s,
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
        save_plot=False,
        verbose=False,
        rng_seed=seed_i,
        kernel_factory=_AcceptanceKernel,
    )

    thresholds = {
        "max_p95_axis_error_m": 0.08,
        "max_p95_xpoint_error_m": 0.11,
        "min_stabilization_rate": 0.88,
        "max_abs_action": 0.35,
        "max_abs_coil_current": 1.5,
        "min_supervisor_interventions": 1,
        "max_estimation_error_m": 0.035,
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
    if summary["max_abs_coil_current"] > thresholds["max_abs_coil_current"] + 1e-9:
        failure_reasons.append("max_abs_coil_current")
    if summary["supervisor_intervention_count"] < thresholds["min_supervisor_interventions"]:
        failure_reasons.append("supervisor_intervention_count")
    if summary["mean_estimation_error_m"] > thresholds["max_estimation_error_m"]:
        failure_reasons.append("mean_estimation_error_m")

    return {
        "seed": seed_i,
        "task8_free_boundary_supervisory_control": {
            "scenario": {
                "control_dt_s": float(dt_s),
                "shot_length": int(steps),
                "target": {
                    "r_axis_m": float(target.r_axis_m),
                    "z_axis_m": float(target.z_axis_m),
                    "x_point_r_m": float(target.x_point_r_m),
                    "x_point_z_m": float(target.x_point_z_m),
                },
                "disturbances": {
                    "current_ramp_start_step": 18,
                    "current_ramp_per_step_ma": 0.08,
                    "coil_kick_step": 26,
                    "sensor_bias_step": 32,
                },
            },
            "closed_loop": summary,
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
    g = report["task8_free_boundary_supervisory_control"]
    s = g["scenario"]
    c = g["closed_loop"]
    th = g["thresholds"]
    lines = [
        "# Task 8 Free-Boundary Supervisory Control",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Overall pass: `{'YES' if g['passes_thresholds'] else 'NO'}`",
        "",
        "## Scenario",
        "",
        f"- Shot length: `{s['shot_length']}`",
        f"- Control dt: `{s['control_dt_s']:.3f} s`",
        f"- Current ramp start: `{s['disturbances']['current_ramp_start_step']}`",
        f"- Coil kick step: `{s['disturbances']['coil_kick_step']}`",
        f"- Sensor bias step: `{s['disturbances']['sensor_bias_step']}`",
        "",
        "## Closed-Loop Acceptance",
        "",
        f"- P95 axis error: `{c['p95_axis_error_m']:.4f} m` (threshold `<= {th['max_p95_axis_error_m']:.2f} m`)",
        f"- P95 X-point error: `{c['p95_xpoint_error_m']:.4f} m` (threshold `<= {th['max_p95_xpoint_error_m']:.2f} m`)",
        f"- Stabilization rate: `{c['stabilization_rate']:.3f}` (threshold `>= {th['min_stabilization_rate']:.2f}`)",
        f"- Mean estimation error: `{c['mean_estimation_error_m']:.5f} m` (threshold `<= {th['max_estimation_error_m']:.3f} m`)",
        f"- Max |action|: `{c['max_abs_action']:.3f}` (threshold `<= {th['max_abs_action']:.2f}`)",
        f"- Max |coil current|: `{c['max_abs_coil_current']:.3f}` (threshold `<= {th['max_abs_coil_current']:.2f}`)",
        "",
        "## Safety Supervisor",
        "",
        f"- Supervisor interventions: `{c['supervisor_intervention_count']}` (threshold `>= {th['min_supervisor_interventions']}`)",
        f"- Saturation events: `{c['saturation_event_count']}`",
        f"- Max bias norm: `{c['max_bias_norm_m']:.4f} m`",
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
    parser.add_argument(
        "--output-json",
        default=str(
            ROOT / "validation" / "reports" / "task8_free_boundary_supervisory_control.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "task8_free_boundary_supervisory_control.md"),
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

    g = report["task8_free_boundary_supervisory_control"]
    c = g["closed_loop"]
    print("Task 8 free-boundary supervisory control validation complete.")
    print(
        "Summary -> "
        f"p95_axis_error_m={c['p95_axis_error_m']:.4f}, "
        f"p95_xpoint_error_m={c['p95_xpoint_error_m']:.4f}, "
        f"stabilization_rate={c['stabilization_rate']:.3f}, "
        f"passes_thresholds={g['passes_thresholds']}"
    )
    if args.strict and not g["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
