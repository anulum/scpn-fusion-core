# CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
"""Standalone runner: PID baseline controller on all 6 benchmark scenarios.

Usage:
    python validation/run_pid_baseline.py

Results are saved to validation/results/pid_baseline.json and a summary
table is printed to stdout.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Ensure imports work regardless of how the script is invoked.
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from scpn_fusion.control.vertical_stability import VerticalStabilityPlant, PlantConfig
from scpn_fusion.control.pid_baseline import PIDController, PIDConfig, tune_ziegler_nichols
from validation.control_benchmark import SCENARIOS, Scenario, compute_metrics


# ---------------------------------------------------------------------------
# Closed-loop runner
# ---------------------------------------------------------------------------

def _run_scenario(
    scenario: Scenario,
    pid: PIDController,
) -> dict:
    """Run a single scenario and return raw trajectories + wall time."""
    # Build plant from scenario config
    sc = scenario.plant_config
    pcfg = PlantConfig(
        gamma=sc.gamma,
        gain=sc.gain,
        dt=sc.dt,
        noise_std=sc.noise_std,
        u_max=sc.u_max,
        z_max=sc.z_max,
        sensor_noise_std=sc.sensor_noise_std,
        sensor_delay_steps=sc.sensor_delay_steps,
    )
    plant = VerticalStabilityPlant(pcfg)
    plant.reset(z0=scenario.z0, dz0=scenario.dz0)
    pid.reset()

    n_steps = int(round(scenario.duration_s / sc.dt))
    z_traj = np.zeros(n_steps)
    u_traj = np.zeros(n_steps)

    t0 = time.perf_counter_ns()
    for k in range(n_steps):
        # Apply disturbance (if any) -- modelled as extra force on the plant
        # For "ramp" we add a linearly increasing external current perturbation.
        u_ctrl = pid.prev_output

        z_meas, dz_meas = plant.step(u_ctrl)
        u = pid.compute(z_meas, dz_meas)

        z_traj[k] = plant.z
        u_traj[k] = u
    elapsed_ns = time.perf_counter_ns() - t0
    wall_time_us_per_step = (elapsed_ns / 1000.0) / max(n_steps, 1)

    metrics = compute_metrics(z_traj, u_traj, dt=sc.dt, z_max=sc.z_max)
    metrics["wall_time_us_per_step"] = wall_time_us_per_step
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 72)
    print("  PID Baseline Controller -- Benchmark Suite")
    print("=" * 72)
    print()

    all_results: list[dict] = []

    for scenario in SCENARIOS:
        # Use default PID gains (DIII-D-like reference gains)
        pid_cfg = PIDConfig(
            dt=scenario.plant_config.dt,
            u_max=scenario.plant_config.u_max,
        )
        pid = PIDController(pid_cfg)

        metrics = _run_scenario(scenario, pid)
        result = {
            "controller": "pid_baseline",
            "scenario": scenario.name,
            **metrics,
        }
        all_results.append(result)

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    hdr = (
        f"{'Scenario':<22s}  "
        f"{'Settle[ms]':>10s}  "
        f"{'Overshoot[mm]':>14s}  "
        f"{'SS err[mm]':>10s}  "
        f"{'RMS u':>8s}  "
        f"{'Peak u':>8s}  "
        f"{'Disrupt':>7s}  "
        f"{'us/step':>8s}"
    )
    print(hdr)
    print("-" * len(hdr))

    for r in all_results:
        print(
            f"{r['scenario']:<22s}  "
            f"{r['settling_time_ms']:10.3f}  "
            f"{r['max_overshoot_mm']:14.3f}  "
            f"{r['steady_state_error_mm']:10.4f}  "
            f"{r['rms_control_effort']:8.3f}  "
            f"{r['peak_control_effort']:8.3f}  "
            f"{'YES' if r['disrupted'] else 'no':>7s}  "
            f"{r['wall_time_us_per_step']:8.2f}"
        )
    print()

    # ------------------------------------------------------------------
    # Save to JSON
    # ------------------------------------------------------------------
    results_dir = _REPO / "validation" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "pid_baseline.json"

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "controller": "pid_baseline",
        "pid_config": {
            "kp": PIDConfig.kp,
            "ki": PIDConfig.ki,
            "kd": PIDConfig.kd,
            "anti_windup": PIDConfig.anti_windup,
        },
        "scenarios": all_results,
    }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Results saved to {out_path}")

    # ------------------------------------------------------------------
    # Quick pass/fail summary
    # ------------------------------------------------------------------
    n_disrupted = sum(1 for r in all_results if r["disrupted"])
    if n_disrupted > 0:
        print(f"\nWARNING: {n_disrupted}/{len(all_results)} scenarios resulted in disruption.")
    else:
        print(f"\nAll {len(all_results)} scenarios completed without disruption.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
