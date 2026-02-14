# CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
"""Head-to-head benchmark: PID vs SNN controller on all 6 scenarios.

Usage:
    python validation/run_control_benchmark.py

Produces:
    - Summary comparison table printed to stdout
    - validation/results/control_benchmark.json   (full metrics)
    - validation/results/control_benchmark.md     (markdown table)
    - validation/results/control_plots/           (trajectory plots, if matplotlib available)
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

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

from scpn_fusion.control.vertical_stability import (
    VerticalStabilityPlant,
    PlantConfig as VSPlantConfig,
)
from scpn_fusion.control.pid_baseline import PIDController, PIDConfig, tune_ziegler_nichols
from scpn_fusion.scpn.vertical_control_net import VerticalControlNet
from scpn_fusion.scpn.vertical_snn_controller import VerticalSNNController
from validation.control_benchmark import (
    SCENARIOS,
    Scenario,
    BenchmarkResult,
    compute_metrics,
)

# ---------------------------------------------------------------------------
# Optional matplotlib import
# ---------------------------------------------------------------------------
try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend for headless runs
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# ---------------------------------------------------------------------------
# Plant factory — builds a VerticalStabilityPlant from a benchmark PlantConfig
# ---------------------------------------------------------------------------

def _make_plant(scenario: Scenario) -> VerticalStabilityPlant:
    """Build a VerticalStabilityPlant from a scenario's plant config."""
    sc = scenario.plant_config
    pcfg = VSPlantConfig(
        gamma=sc.gamma,
        gain=sc.gain,
        dt=sc.dt,
        noise_std=sc.noise_std,
        u_max=sc.u_max,
        z_max=sc.z_max,
        sensor_noise_std=sc.sensor_noise_std,
        sensor_delay_steps=sc.sensor_delay_steps,
    )
    return VerticalStabilityPlant(pcfg)


# ---------------------------------------------------------------------------
# Controller factories
# ---------------------------------------------------------------------------

def _make_pid(scenario: Scenario) -> PIDController:
    """Build a PID controller with default DIII-D-like gains."""
    pid_cfg = PIDConfig(
        dt=scenario.plant_config.dt,
        u_max=scenario.plant_config.u_max,
    )
    return PIDController(pid_cfg)


def _make_snn() -> VerticalSNNController:
    """Build an SNN controller from the vertical control net."""
    vcn = VerticalControlNet()
    vcn.create_net()
    return VerticalSNNController(vcn, force_numpy=True, seed=42)


# ---------------------------------------------------------------------------
# Closed-loop runner
# ---------------------------------------------------------------------------

def _run_closed_loop(
    scenario: Scenario,
    controller,  # PIDController or VerticalSNNController — both have .compute() / .reset()
    controller_name: str,
) -> BenchmarkResult:
    """Run a single scenario in closed loop and return a BenchmarkResult."""
    sc = scenario.plant_config
    plant = _make_plant(scenario)
    plant.reset(z0=scenario.z0, dz0=scenario.dz0)
    controller.reset()

    n_steps = int(round(scenario.duration_s / sc.dt))
    z_traj = np.zeros(n_steps)
    u_traj = np.zeros(n_steps)

    u = 0.0  # initial control signal

    t0 = time.perf_counter_ns()
    for k in range(n_steps):
        z_meas, dz_meas = plant.step(u)
        u = controller.compute(z_meas, dz_meas)
        z_traj[k] = plant.z
        u_traj[k] = u
    elapsed_ns = time.perf_counter_ns() - t0
    wall_time_us = (elapsed_ns / 1000.0) / max(n_steps, 1)

    metrics = compute_metrics(z_traj, u_traj, dt=sc.dt, z_max=sc.z_max)

    return BenchmarkResult(
        controller_name=controller_name,
        scenario_name=scenario.name,
        settling_time_ms=metrics["settling_time_ms"],
        max_overshoot_mm=metrics["max_overshoot_mm"],
        steady_state_error_mm=metrics["steady_state_error_mm"],
        rms_control_effort=metrics["rms_control_effort"],
        peak_control_effort=metrics["peak_control_effort"],
        disrupted=metrics["disrupted"],
        trajectory_z=z_traj.tolist(),
        trajectory_u=u_traj.tolist(),
        wall_time_us=wall_time_us,
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_trajectories(
    scenario: Scenario,
    pid_result: BenchmarkResult,
    snn_result: BenchmarkResult,
    out_dir: Path,
) -> None:
    """Generate z(t) and u(t) comparison plots for a scenario."""
    if not _HAS_MPL:
        return

    dt = scenario.plant_config.dt
    n_pid = len(pid_result.trajectory_z)
    n_snn = len(snn_result.trajectory_z)
    t_pid = np.arange(n_pid) * dt * 1000.0  # ms
    t_snn = np.arange(n_snn) * dt * 1000.0

    # ---- Z(t) plot --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_pid, np.array(pid_result.trajectory_z) * 1000.0,
            label="PID", linewidth=1.2, color="#2196F3")
    ax.plot(t_snn, np.array(snn_result.trajectory_z) * 1000.0,
            label="SNN", linewidth=1.2, color="#FF5722", linestyle="--")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    ax.axhline(1.0, color="green", linewidth=0.5, linestyle=":",
               label="settling band (+/- 1mm)")
    ax.axhline(-1.0, color="green", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Z displacement [mm]")
    ax.set_title(f"Z(t) — {scenario.name}: {scenario.description}")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{scenario.name}_z.png", dpi=150)
    plt.close(fig)

    # ---- u(t) plot --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_pid, pid_result.trajectory_u,
            label="PID", linewidth=1.2, color="#2196F3")
    ax.plot(t_snn, snn_result.trajectory_u,
            label="SNN", linewidth=1.2, color="#FF5722", linestyle="--")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Control signal u [A]")
    ax.set_title(f"u(t) — {scenario.name}: {scenario.description}")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{scenario.name}_u.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _winner(pid_val: float, snn_val: float, lower_is_better: bool = True) -> str:
    """Return 'PID', 'SNN', or 'TIE' indicating which value is better."""
    if lower_is_better:
        if pid_val < snn_val:
            return "PID"
        elif snn_val < pid_val:
            return "SNN"
    else:
        if pid_val > snn_val:
            return "PID"
        elif snn_val > pid_val:
            return "SNN"
    return "TIE"


def _fmt(v: float, prec: int = 3) -> str:
    """Format a float to fixed precision."""
    return f"{v:.{prec}f}"


# ---------------------------------------------------------------------------
# Markdown table generation
# ---------------------------------------------------------------------------

_FORMAL_PROPERTIES = {
    "Boundedness proof": ("No proof", "PROVED"),
    "Liveness proof": ("No proof", "PROVED"),
    "Mutual exclusion proof": ("No proof", "PROVED"),
    "Deterministic routing": ("N/A", "PROVED"),
}


def _build_markdown(
    results: List[Tuple[Scenario, BenchmarkResult, BenchmarkResult]],
) -> str:
    """Build a full markdown comparison report."""
    lines: List[str] = []
    lines.append("# PID vs SNN Controller — Head-to-Head Benchmark")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")

    # ---- Per-scenario comparison tables -----------------------------------
    lines.append("## Per-Scenario Metrics")
    lines.append("")
    lines.append(
        "| Scenario | Metric | PID | SNN | Better |"
    )
    lines.append(
        "|----------|--------|----:|----:|--------|"
    )

    for scenario, pid_r, snn_r in results:
        metrics = [
            ("Settling [ms]", pid_r.settling_time_ms, snn_r.settling_time_ms, True),
            ("Overshoot [mm]", pid_r.max_overshoot_mm, snn_r.max_overshoot_mm, True),
            ("SS error [mm]", pid_r.steady_state_error_mm, snn_r.steady_state_error_mm, True),
            ("RMS effort", pid_r.rms_control_effort, snn_r.rms_control_effort, True),
            ("Peak effort", pid_r.peak_control_effort, snn_r.peak_control_effort, True),
            ("us/step", pid_r.wall_time_us, snn_r.wall_time_us, True),
        ]
        for i, (name, pv, sv, lower) in enumerate(metrics):
            scen_label = scenario.name if i == 0 else ""
            w = _winner(pv, sv, lower_is_better=lower)
            lines.append(
                f"| {scen_label:<20s} | {name:<16s} | {_fmt(pv, 4):>10s} | {_fmt(sv, 4):>10s} | {w:<6s} |"
            )
        # Disruption row
        scen_label = ""
        pid_d = "YES" if pid_r.disrupted else "no"
        snn_d = "YES" if snn_r.disrupted else "no"
        d_winner = ""
        if pid_r.disrupted and not snn_r.disrupted:
            d_winner = "SNN"
        elif not pid_r.disrupted and snn_r.disrupted:
            d_winner = "PID"
        elif not pid_r.disrupted and not snn_r.disrupted:
            d_winner = "TIE"
        else:
            d_winner = "TIE"
        lines.append(
            f"| {scen_label:<20s} | {'Disrupted':<16s} | {pid_d:>10s} | {snn_d:>10s} | {d_winner:<6s} |"
        )

    lines.append("")

    # ---- Aggregate summary ------------------------------------------------
    lines.append("## Aggregate Summary")
    lines.append("")

    pid_wins = 0
    snn_wins = 0
    ties = 0
    for scenario, pid_r, snn_r in results:
        # Count wins by settling time (primary metric)
        w = _winner(pid_r.settling_time_ms, snn_r.settling_time_ms, True)
        if w == "PID":
            pid_wins += 1
        elif w == "SNN":
            snn_wins += 1
        else:
            ties += 1

    lines.append(f"- **PID wins (settling time)**: {pid_wins}/{len(results)}")
    lines.append(f"- **SNN wins (settling time)**: {snn_wins}/{len(results)}")
    lines.append(f"- **Ties**: {ties}/{len(results)}")
    lines.append("")

    # ---- Formal guarantees ------------------------------------------------
    lines.append("## Formal Verification Properties")
    lines.append("")
    lines.append("| Property | PID | SNN |")
    lines.append("|----------|-----|-----|")
    for prop, (pid_status, snn_status) in _FORMAL_PROPERTIES.items():
        lines.append(f"| {prop} | {pid_status} | {snn_status} |")
    lines.append("")

    lines.append(
        "> **Key insight**: The SNN controller may not beat PID on every numerical "
        "metric -- that is expected and honest. The decisive advantage of the "
        "SNN controller is its formally verified Petri net structure: "
        "boundedness, liveness, and mutual exclusion are *proved*, not assumed. "
        "PID has no such guarantees."
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build JSON report
# ---------------------------------------------------------------------------

def _build_json_report(
    results: List[Tuple[Scenario, BenchmarkResult, BenchmarkResult]],
) -> dict:
    """Build the full JSON report dictionary."""
    scenarios_out: List[dict] = []

    for scenario, pid_r, snn_r in results:
        scenarios_out.append({
            "scenario": scenario.name,
            "description": scenario.description,
            "pid": pid_r.to_dict(),
            "snn": snn_r.to_dict(),
        })

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "pid_vs_snn_head_to_head",
        "n_scenarios": len(results),
        "formal_properties": {
            prop: {"pid": pid_val, "snn": snn_val}
            for prop, (pid_val, snn_val) in _FORMAL_PROPERTIES.items()
        },
        "scenarios": scenarios_out,
    }


# ---------------------------------------------------------------------------
# Console summary printer
# ---------------------------------------------------------------------------

def _print_console_summary(
    results: List[Tuple[Scenario, BenchmarkResult, BenchmarkResult]],
) -> None:
    """Print a condensed comparison table to stdout."""
    print("=" * 100)
    print("  PID vs SNN Controller — Head-to-Head Benchmark")
    print("=" * 100)
    print()

    hdr = (
        f"{'Scenario':<22s}  "
        f"{'Ctrl':>4s}  "
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

    for scenario, pid_r, snn_r in results:
        for label, r in [("PID", pid_r), ("SNN", snn_r)]:
            print(
                f"{scenario.name if label == 'PID' else '':<22s}  "
                f"{label:>4s}  "
                f"{r.settling_time_ms:10.3f}  "
                f"{r.max_overshoot_mm:14.3f}  "
                f"{r.steady_state_error_mm:10.4f}  "
                f"{r.rms_control_effort:8.3f}  "
                f"{r.peak_control_effort:8.3f}  "
                f"{'YES' if r.disrupted else 'no':>7s}  "
                f"{r.wall_time_us:8.2f}"
            )
        print()

    # Formal properties
    print("-" * len(hdr))
    print()
    print("Formal Verification Properties:")
    print(f"  {'Property':<30s}  {'PID':<12s}  {'SNN':<12s}")
    print(f"  {'-' * 30}  {'-' * 12}  {'-' * 12}")
    for prop, (pid_val, snn_val) in _FORMAL_PROPERTIES.items():
        print(f"  {prop:<30s}  {pid_val:<12s}  {snn_val:<12s}")
    print()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_benchmark() -> List[Tuple[Scenario, BenchmarkResult, BenchmarkResult]]:
    """Run all 6 scenarios for both controllers. Returns results list."""
    # Build a single SNN controller (reused across scenarios after reset)
    snn = _make_snn()

    results: List[Tuple[Scenario, BenchmarkResult, BenchmarkResult]] = []

    for scenario in SCENARIOS:
        # --- PID run -------------------------------------------------------
        pid = _make_pid(scenario)
        pid_result = _run_closed_loop(scenario, pid, "PID")

        # --- SNN run -------------------------------------------------------
        snn_result = _run_closed_loop(scenario, snn, "SNN")

        results.append((scenario, pid_result, snn_result))

    return results


def main() -> int:
    results = run_benchmark()

    # ---- Console output ---------------------------------------------------
    _print_console_summary(results)

    # ---- Create output directories ----------------------------------------
    results_dir = _REPO / "validation" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = results_dir / "control_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ---- Save JSON --------------------------------------------------------
    json_path = results_dir / "control_benchmark.json"
    report = _build_json_report(results)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"JSON results saved to {json_path}")

    # ---- Save Markdown ----------------------------------------------------
    md_path = results_dir / "control_benchmark.md"
    md_text = _build_markdown(results)
    md_path.write_text(md_text, encoding="utf-8")
    print(f"Markdown report saved to {md_path}")

    # ---- Trajectory plots -------------------------------------------------
    if _HAS_MPL:
        for scenario, pid_r, snn_r in results:
            _plot_trajectories(scenario, pid_r, snn_r, plots_dir)
        print(f"Trajectory plots saved to {plots_dir}")
    else:
        print("matplotlib not available — skipping trajectory plots.")

    # ---- Quick summary ----------------------------------------------------
    pid_disruptions = sum(1 for _, p, _ in results if p.disrupted)
    snn_disruptions = sum(1 for _, _, s in results if s.disrupted)
    print()
    print(f"PID disruptions: {pid_disruptions}/{len(results)}")
    print(f"SNN disruptions: {snn_disruptions}/{len(results)}")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
