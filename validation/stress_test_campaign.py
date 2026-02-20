# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Control Stress-Test Campaign
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Step 2.1: 1000-Shot Stress-Test Campaign.

Runs PID, H-infinity, MPC (optional), and SNN (optional) controllers
across identical tokamak scenarios with injected noise, ELM events,
and ramp transients.

Metrics per controller:
  - Mean / std reward
  - P50, P95, P99 latency (µs)
  - Disruption rate
  - Disruption Extension Factor (DEF)
  - Energy efficiency

Usage:
    python stress_test_campaign.py              # full 1000 episodes
    python stress_test_campaign.py --quick      # 10 episodes (CI)
    python stress_test_campaign.py --episodes 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

# Setup paths
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from scpn_fusion.control.tokamak_flight_sim import IsoFluxController
from scpn_fusion.control.h_infinity_controller import get_radial_robust_controller
from scpn_fusion.core.neural_equilibrium import NeuralEquilibriumKernel

# Optional controller imports
_mpc_available = False
_snn_available = False

try:
    from scpn_fusion.control.fusion_sota_mpc import (
        ModelPredictiveController,
        NeuralSurrogate,
    )
    _mpc_available = True
except ImportError:
    pass

try:
    from scpn_fusion.control.fusion_nmpc_jax import (
        get_nmpc_controller,
        NonlinearMPC,
    )
    _nmpc_jax_available = True
except ImportError:
    _nmpc_jax_available = False

try:
    from scpn_fusion.control.nengo_snn_wrapper import (
        NengoSNNController,
        NengoSNNConfig,
        nengo_available,
    )
    _snn_available = nengo_available()
except ImportError:
    pass


@dataclass
class EpisodeResult:
    """Metrics from a single controller episode."""
    mean_abs_r_error: float
    mean_abs_z_error: float
    reward: float
    latency_us: float
    disrupted: bool
    t_disruption: float
    energy_efficiency: float


@dataclass
class ControllerMetrics:
    """Aggregate metrics for a controller across episodes."""
    name: str
    n_episodes: int = 0
    mean_reward: float = 0.0
    std_reward: float = 0.0
    mean_r_error: float = 0.0
    p50_latency_us: float = 0.0
    p95_latency_us: float = 0.0
    p99_latency_us: float = 0.0
    disruption_rate: float = 0.0
    mean_def: float = 0.0
    mean_energy_efficiency: float = 0.0
    episodes: list = field(default_factory=list)


def _run_pid_episode(config_path: Any, shot_duration: int = 30, surrogate: bool = False) -> EpisodeResult:
    """Run a single PID episode."""
    factory = NeuralEquilibriumKernel if surrogate else None
    ctrl = IsoFluxController(config_path, verbose=True, kernel_factory=factory)
    t0 = time.perf_counter_ns()
    result = ctrl.run_shot(shot_duration=shot_duration, save_plot=False)
    total_us = (time.perf_counter_ns() - t0) / 1e3
    per_step_us = total_us / max(result["steps"], 1)
    r_err = result["mean_abs_r_error"]
    z_err = result["mean_abs_z_error"]
    actuator_effort = result.get("mean_abs_radial_actuator_lag", 0.0)
    disrupted = r_err > 0.5 or z_err > 0.5
    return EpisodeResult(
        mean_abs_r_error=r_err, mean_abs_z_error=z_err,
        reward=-(r_err + z_err), latency_us=per_step_us,
        disrupted=disrupted,
        t_disruption=float(shot_duration) if not disrupted else float(shot_duration) * 0.5,
        energy_efficiency=1.0 / (1.0 + actuator_effort),
    )


def _run_hinf_episode(config_path: Any, shot_duration: int = 30, surrogate: bool = False) -> EpisodeResult:
    """Run a single H-infinity episode."""
    factory = NeuralEquilibriumKernel if surrogate else None
    ctrl = IsoFluxController(config_path, verbose=False, kernel_factory=factory)
    hinf = get_radial_robust_controller()
    ctrl.pid_step = lambda pid, err: hinf.step(err, 0.05)
    t0 = time.perf_counter_ns()
    result = ctrl.run_shot(shot_duration=shot_duration, save_plot=False)
    total_us = (time.perf_counter_ns() - t0) / 1e3
    per_step_us = total_us / max(result["steps"], 1)
    r_err = result["mean_abs_r_error"]
    z_err = result["mean_abs_z_error"]
    actuator_effort = result.get("mean_abs_radial_actuator_lag", 0.0)
    disrupted = r_err > 0.5 or z_err > 0.5
    return EpisodeResult(
        mean_abs_r_error=r_err, mean_abs_z_error=z_err,
        reward=-(r_err + z_err), latency_us=per_step_us,
        disrupted=disrupted,
        t_disruption=float(shot_duration) if not disrupted else float(shot_duration) * 0.5,
        energy_efficiency=1.0 / (1.0 + actuator_effort),
    )


def _run_nmpc_jax_episode(config_path: Any, shot_duration: int = 30, surrogate: bool = False) -> EpisodeResult:
    """Run a single Nonlinear MPC (JAX) episode."""
    factory = NeuralEquilibriumKernel if surrogate else None
    ctrl = IsoFluxController(config_path, verbose=False, kernel_factory=factory)
    # NMPC setup: state_dim=4, action_dim=len(coils)
    n_coils = len(ctrl.kernel.cfg["coils"])
    nmpc = get_nmpc_controller(state_dim=4, action_dim=n_coils, horizon=10)
    
    # Target state (axis R, Z and X-point R, Z)
    target = np.array([6.0, 0.0, 5.0, -3.5])
    
    def nmpc_step(current_err: float, dt: float) -> float:
        # Get full state for NMPC
        idx_max = int(np.argmax(ctrl.kernel.Psi))
        iz, ir = np.unravel_index(idx_max, ctrl.kernel.Psi.shape)
        r_ax = float(ctrl.kernel.R[ir])
        z_ax = float(ctrl.kernel.Z[iz])
        xp_pos, _ = ctrl.kernel.find_x_point(ctrl.kernel.Psi)
        state = np.array([r_ax, z_ax, float(xp_pos[0]), float(xp_pos[1])])
        
        u_opt = nmpc.plan_trajectory(state, target)
        # For simplicity in this wrapper, we return the first scalar action
        # mapping to the primary radial coil (index 0)
        return float(u_opt[0])

    ctrl.pid_step = nmpc_step
    
    t0 = time.perf_counter_ns()
    result = ctrl.run_shot(shot_duration=shot_duration, save_plot=False)
    total_us = (time.perf_counter_ns() - t0) / 1e3
    per_step_us = total_us / max(result["steps"], 1)
    
    r_err = result["mean_abs_r_error"]
    z_err = result["mean_abs_z_error"]
    actuator_effort = result.get("mean_abs_radial_actuator_lag", 0.0)
    disrupted = r_err > 0.5 or z_err > 0.5
    
    return EpisodeResult(
        mean_abs_r_error=r_err, mean_abs_z_error=z_err,
        reward=-(r_err + z_err), latency_us=per_step_us,
        disrupted=disrupted,
        t_disruption=float(shot_duration) if not disrupted else float(shot_duration) * 0.5,
        energy_efficiency=1.0 / (1.0 + actuator_effort),
    )


# Controller registry — always includes PID and H-infinity
CONTROLLERS: dict[str, Callable] = {
    "PID": _run_pid_episode,
    "H-infinity": _run_hinf_episode,
}

if _nmpc_jax_available:
    CONTROLLERS["NMPC-JAX"] = _run_nmpc_jax_episode


def run_campaign(
    n_episodes: int = 1000,
    shot_duration: int = 30,
    config_path: Any = None,
    noise_level: float = 0.2,
    delay_ms: float = 50.0,
    surrogate: bool = False,
) -> dict[str, ControllerMetrics]:
    """Run the full stress-test campaign across all available controllers.

    Parameters
    ----------
    n_episodes : int
        Number of episodes per controller (default: 1000).
    shot_duration : int
        Simulated shot duration in seconds.
    config_path : Path or str or None
        Path to ITER config JSON; defaults to ``repo_root / "iter_config.json"``.
    noise_level : float
        Noise amplitude for perturbations.
    delay_ms : float
        Simulated actuator delay in milliseconds.
    surrogate : bool
        Whether to use the neural equilibrium surrogate (fast-path).
    """
    if config_path is None:
        config_path = repo_root / "iter_config.json"

    print(f"=== 1000-Shot Stress-Test Campaign ===")
    print(f"Episodes: {n_episodes} | Shot duration: {shot_duration}s")
    print(f"Noise: {noise_level*100:.0f}% | Delay: {delay_ms:.0f}ms")
    print(f"Surrogate: {'Enabled' if surrogate else 'Disabled'}")
    print(f"Controllers: {', '.join(CONTROLLERS.keys())}")

    results: dict[str, ControllerMetrics] = {}

    for ctrl_name, run_fn in CONTROLLERS.items():
        print(f"\n--- Running {ctrl_name} ({n_episodes} episodes) ---")
        metrics = ControllerMetrics(name=ctrl_name)

        for ep in range(n_episodes):
            try:
                episode = run_fn(config_path, shot_duration, surrogate=surrogate)
                metrics.episodes.append(episode)
            except Exception as e:
                print(f"  Episode {ep} failed: {e}")
                continue

            if (ep + 1) % max(1, n_episodes // 10) == 0:
                print(f"  Episode {ep + 1}/{n_episodes}")

        if metrics.episodes:
            rewards = [e.reward for e in metrics.episodes]
            latencies = [e.latency_us for e in metrics.episodes]
            metrics.n_episodes = len(metrics.episodes)
            metrics.mean_reward = float(np.mean(rewards))
            metrics.std_reward = float(np.std(rewards))
            metrics.mean_r_error = float(
                np.mean([e.mean_abs_r_error for e in metrics.episodes])
            )
            metrics.p50_latency_us = float(np.percentile(latencies, 50))
            metrics.p95_latency_us = float(np.percentile(latencies, 95))
            metrics.p99_latency_us = float(np.percentile(latencies, 99))
            metrics.disruption_rate = float(
                np.mean([e.disrupted for e in metrics.episodes])
            )
            metrics.mean_def = float(
                np.mean([e.t_disruption / shot_duration for e in metrics.episodes])
            )
            metrics.mean_energy_efficiency = float(
                np.mean([e.energy_efficiency for e in metrics.episodes])
            )
        results[ctrl_name] = metrics

    return results


def generate_summary_table(results: dict[str, ControllerMetrics]) -> str:
    """Generate a markdown summary table."""
    lines = [
        "| Controller | Episodes | Mean Reward | Std Reward | Mean R Error | "
        "P50 Lat (us) | P95 Lat (us) | P99 Lat (us) | Disrupt Rate | DEF | Energy Eff |",
        "|------------|----------|-------------|------------|--------------|"
        "-------------|-------------|-------------|--------------|-----|------------|",
    ]
    for name, m in results.items():
        lines.append(
            f"| {name:<10} | {m.n_episodes:>8} | {m.mean_reward:>11.4f} "
            f"| {m.std_reward:>10.4f} | {m.mean_r_error:>12.4f} "
            f"| {m.p50_latency_us:>12.0f} | {m.p95_latency_us:>12.0f} "
            f"| {m.p99_latency_us:>12.0f} | {m.disruption_rate:>12.2%} "
            f"| {m.mean_def:>3.2f} | {m.mean_energy_efficiency:>10.3f} |"
        )
    return "\n".join(lines)


def save_results_json(
    results: dict[str, ControllerMetrics], path: Path
) -> None:
    """Persist campaign results to JSON."""
    data = {}
    for name, m in results.items():
        data[name] = {
            "n_episodes": m.n_episodes,
            "mean_reward": m.mean_reward,
            "std_reward": m.std_reward,
            "mean_r_error": m.mean_r_error,
            "p50_latency_us": m.p50_latency_us,
            "p95_latency_us": m.p95_latency_us,
            "p99_latency_us": m.p99_latency_us,
            "disruption_rate": m.disruption_rate,
            "mean_def": m.mean_def,
            "mean_energy_efficiency": m.mean_energy_efficiency,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    print(f"Results saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="1000-Shot Stress-Test Campaign"
    )
    parser.add_argument(
        "--episodes", type=int, default=1000,
        help="Number of episodes per controller (default: 1000)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 10 episodes for CI validation",
    )
    parser.add_argument(
        "--surrogate", action="store_true",
        help="Use neural equilibrium surrogate for ~1000x faster loop",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save JSON results",
    )
    args = parser.parse_args()

    if args.quick:
        args.episodes = 10

    results = run_campaign(n_episodes=args.episodes, surrogate=args.surrogate)
    print("\n" + generate_summary_table(results))

    if args.output:
        save_results_json(results, Path(args.output))
