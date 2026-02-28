# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — 4-Way Controller Comparison
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Step 2.6: 4-Way Controller Comparison.

Runs PID, MPC, SNN, and H-infinity on identical tokamak scenarios,
computing: mean reward, reward std, P95 latency, disruption rate,
disruption extension factor (DEF), and energy efficiency.

Produces markdown and LaTeX comparison tables.
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

# Optional controller imports
_mpc_available = False
_nmpc_jax_available = False
_snn_available = False

try:
    from scpn_fusion.control.fusion_sota_mpc import (
        ModelPredictiveController,
        NeuralSurrogate,
    )
    _mpc_available = True
except ImportError:
    ModelPredictiveController = None  # type: ignore[assignment]
    NeuralSurrogate = None  # type: ignore[assignment]

try:
    from scpn_fusion.control.fusion_nmpc_jax import get_nmpc_controller
    _nmpc_jax_available = True
except ImportError:
    get_nmpc_controller = None  # type: ignore[assignment]

try:
    from scpn_fusion.control.nengo_snn_wrapper import (
        NengoSNNController,
        NengoSNNConfig,
        nengo_available,
    )
    _snn_available = nengo_available()
except ImportError:
    NengoSNNController = None  # type: ignore[assignment]
    NengoSNNConfig = None  # type: ignore[assignment]


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
    p95_latency_us: float = 0.0
    disruption_rate: float = 0.0
    mean_def: float = 0.0
    mean_energy_efficiency: float = 0.0
    episodes: list = field(default_factory=list)


def _to_episode_result(
    run_result: dict[str, Any],
    total_us: float,
    shot_duration: int,
) -> EpisodeResult:
    per_step_us = total_us / max(int(run_result.get("steps", 0)), 1)
    r_err = float(run_result["mean_abs_r_error"])
    z_err = float(run_result["mean_abs_z_error"])
    actuator_effort = float(run_result.get("mean_abs_radial_actuator_lag", 0.0))
    disrupted = r_err > 0.5 or z_err > 0.5
    return EpisodeResult(
        mean_abs_r_error=r_err,
        mean_abs_z_error=z_err,
        reward=-(r_err + z_err),
        latency_us=per_step_us,
        disrupted=disrupted,
        t_disruption=float(shot_duration) if not disrupted else float(shot_duration) * 0.5,
        energy_efficiency=1.0 / (1.0 + actuator_effort),
    )


def _run_pid_episode(config_path: Any, shot_duration: int = 30) -> EpisodeResult:
    steps = int(shot_duration)
    ctrl = IsoFluxController(config_path, verbose=False)
    t0 = time.perf_counter_ns()
    result = ctrl.run_shot(shot_duration=steps, save_plot=False)
    total_us = (time.perf_counter_ns() - t0) / 1e3
    return _to_episode_result(result, total_us, steps)


def _run_hinf_episode(config_path: Any, shot_duration: int = 30) -> EpisodeResult:
    steps = int(shot_duration)
    dt = 0.05
    ctrl = IsoFluxController(config_path, verbose=False, control_dt_s=dt)
    # Maintain independent observer state for radial and vertical channels.
    hinf_r = get_radial_robust_controller()
    hinf_z = get_radial_robust_controller()
    hinf_r.step(0.0, dt)
    hinf_r.reset()
    hinf_z.step(0.0, dt)
    hinf_z.reset()
    kp_r = float(ctrl.pid_R["Kp"])
    kp_z = float(ctrl.pid_Z["Kp"])
    scale_r = kp_r / max(abs(float(hinf_r._Fd[0, 0])), 1.0e-12)
    scale_z = kp_z / max(abs(float(hinf_z._Fd[0, 0])), 1.0e-12)
    pid_r_id = id(ctrl.pid_R)

    def hinf_step(pid: Any, err: float) -> float:
        if id(pid) == pid_r_id:
            return scale_r * hinf_r.step(err, dt)
        return scale_z * hinf_z.step(err, dt)

    ctrl.pid_step = hinf_step
    t0 = time.perf_counter_ns()
    result = ctrl.run_shot(shot_duration=steps, save_plot=False)
    total_us = (time.perf_counter_ns() - t0) / 1e3
    return _to_episode_result(result, total_us, steps)


def _run_mpc_episode(config_path: Any, shot_duration: int = 30) -> EpisodeResult:
    if not _mpc_available:
        raise RuntimeError("MPC controller is unavailable in this environment.")
    steps = int(shot_duration)
    ctrl = IsoFluxController(config_path, verbose=False)
    n_coils = len(ctrl.kernel.cfg.get("coils", []))
    if n_coils < 1:
        raise RuntimeError("ITER config is missing coil definitions for MPC.")
    target = np.array([6.2, 0.0, 5.0, -3.5], dtype=np.float64)
    surrogate = NeuralSurrogate(n_coils=n_coils, n_state=4, verbose=False)
    surrogate.train_on_kernel(ctrl.kernel, perturbation=1.0)
    mpc = ModelPredictiveController(
        surrogate,
        target_state=target,
        prediction_horizon=6,
        learning_rate=0.25,
        iterations=8,
        action_limit=2.0,
    )

    def mpc_step(pid: Any, err: float) -> float:
        idx_max = int(np.argmax(ctrl.kernel.Psi))
        iz, ir = np.unravel_index(idx_max, ctrl.kernel.Psi.shape)
        r_ax = float(ctrl.kernel.R[ir])
        z_ax = float(ctrl.kernel.Z[iz])
        xp_pos, _ = ctrl.kernel.find_x_point(ctrl.kernel.Psi)
        state = np.array([r_ax, z_ax, float(xp_pos[0]), float(xp_pos[1])], dtype=np.float64)
        action = np.asarray(mpc.plan_trajectory(state), dtype=np.float64).reshape(-1)
        return float(action[0]) if action.size else 0.0

    ctrl.pid_step = mpc_step
    t0 = time.perf_counter_ns()
    result = ctrl.run_shot(shot_duration=steps, save_plot=False)
    total_us = (time.perf_counter_ns() - t0) / 1e3
    return _to_episode_result(result, total_us, steps)


def _run_nmpc_jax_episode(config_path: Any, shot_duration: int = 30) -> EpisodeResult:
    if not _nmpc_jax_available:
        raise RuntimeError("NMPC-JAX controller is unavailable in this environment.")
    steps = int(shot_duration)
    ctrl = IsoFluxController(config_path, verbose=False)
    n_coils = len(ctrl.kernel.cfg.get("coils", []))
    if n_coils < 1:
        raise RuntimeError("ITER config is missing coil definitions for NMPC-JAX.")
    target = np.array([6.2, 0.0, 5.0, -3.5], dtype=np.float64)
    nmpc = get_nmpc_controller(state_dim=4, action_dim=n_coils, horizon=10)

    def nmpc_step(pid: Any, err: float) -> float:
        idx_max = int(np.argmax(ctrl.kernel.Psi))
        iz, ir = np.unravel_index(idx_max, ctrl.kernel.Psi.shape)
        r_ax = float(ctrl.kernel.R[ir])
        z_ax = float(ctrl.kernel.Z[iz])
        xp_pos, _ = ctrl.kernel.find_x_point(ctrl.kernel.Psi)
        state = np.array([r_ax, z_ax, float(xp_pos[0]), float(xp_pos[1])], dtype=np.float64)
        action = np.asarray(nmpc.plan_trajectory(state, target), dtype=np.float64).reshape(-1)
        return float(action[0]) if action.size else 0.0

    ctrl.pid_step = nmpc_step
    t0 = time.perf_counter_ns()
    result = ctrl.run_shot(shot_duration=steps, save_plot=False)
    total_us = (time.perf_counter_ns() - t0) / 1e3
    return _to_episode_result(result, total_us, steps)


def _run_snn_episode(config_path: Any, shot_duration: int = 30) -> EpisodeResult:
    if not _snn_available:
        raise RuntimeError("Nengo-SNN controller is unavailable in this environment.")
    steps = int(shot_duration)
    ctrl = IsoFluxController(config_path, verbose=False)
    snn = NengoSNNController(NengoSNNConfig(n_neurons=200, n_channels=2))

    def snn_step(pid: Any, err: float) -> float:
        output = snn.step(np.array([float(err), 0.0], dtype=np.float64))
        return float(np.asarray(output, dtype=np.float64).reshape(-1)[0])

    ctrl.pid_step = snn_step
    t0 = time.perf_counter_ns()
    result = ctrl.run_shot(shot_duration=steps, save_plot=False)
    total_us = (time.perf_counter_ns() - t0) / 1e3
    return _to_episode_result(result, total_us, steps)


def _build_controller_registry() -> dict[str, Callable[[Any, int], EpisodeResult]]:
    controllers: dict[str, Callable[[Any, int], EpisodeResult]] = {
        "PID": _run_pid_episode,
        "H-infinity": _run_hinf_episode,
    }
    if _mpc_available:
        controllers["MPC"] = _run_mpc_episode
    if _nmpc_jax_available:
        controllers["NMPC-JAX"] = _run_nmpc_jax_episode
    if _snn_available:
        controllers["Nengo-SNN"] = _run_snn_episode
    return controllers


CONTROLLERS = _build_controller_registry()


def run_comparison(
    n_episodes: int = 100,
    shot_duration: int = 30,
    config_path: Any = None,
) -> dict[str, ControllerMetrics]:
    """Run all available controllers on identical scenarios."""
    if config_path is None:
        config_path = repo_root / "iter_config.json"

    if not CONTROLLERS:
        raise RuntimeError("No controllers are available for comparison.")

    print("=== 4-Way Controller Comparison ===")
    print(f"Episodes: {n_episodes} | Controllers: {', '.join(CONTROLLERS.keys())}")

    results: dict[str, ControllerMetrics] = {}
    for ctrl_name, run_fn in CONTROLLERS.items():
        print(f"--- Running {ctrl_name} ({n_episodes} episodes) ---")
        metrics = ControllerMetrics(name=ctrl_name)
        for ep in range(n_episodes):
            try:
                episode = run_fn(config_path, shot_duration)
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
            metrics.mean_r_error = float(np.mean([e.mean_abs_r_error for e in metrics.episodes]))
            metrics.p95_latency_us = float(np.percentile(latencies, 95))
            metrics.disruption_rate = float(np.mean([e.disrupted for e in metrics.episodes]))
            metrics.mean_def = float(np.mean([e.t_disruption / shot_duration for e in metrics.episodes]))
            metrics.mean_energy_efficiency = float(np.mean([e.energy_efficiency for e in metrics.episodes]))
        results[ctrl_name] = metrics

    return results


def generate_comparison_table(results: dict[str, ControllerMetrics]) -> str:
    """Generate markdown comparison table."""
    lines = [
        "| Controller | Episodes | Mean Reward | P95 Latency (us) | Disruption Rate | DEF | Energy Eff |",
        "|------------|----------|-------------|-------------------|-----------------|-----|------------|",
    ]
    for name, m in results.items():
        lines.append(
            f"| {name:<10} | {m.n_episodes:>8} | {m.mean_reward:>11.4f} "
            f"| {m.p95_latency_us:>17.0f} | {m.disruption_rate:>15.2%} "
            f"| {m.mean_def:>3.2f} | {m.mean_energy_efficiency:>10.3f} |"
        )
    return "\n".join(lines)


def generate_latex_table(results: dict[str, ControllerMetrics]) -> str:
    """Generate publication-ready LaTeX table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{4-Way Controller Comparison}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Controller & Episodes & Mean Reward & P95 Lat. & Disrupt. & DEF & Eff. \\",
        r"\midrule",
    ]
    for name, m in results.items():
        lines.append(
            f"  {name} & {m.n_episodes} & {m.mean_reward:.4f} "
            f"& {m.p95_latency_us:.0f} & {m.disruption_rate:.2%} "
            f"& {m.mean_def:.2f} & {m.mean_energy_efficiency:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def save_results_json(results: dict[str, ControllerMetrics], output_path: Path) -> None:
    payload: dict[str, dict[str, float | int]] = {}
    for name, metrics in results.items():
        payload[name] = {
            "n_episodes": metrics.n_episodes,
            "mean_reward": metrics.mean_reward,
            "std_reward": metrics.std_reward,
            "mean_r_error": metrics.mean_r_error,
            "p95_latency_us": metrics.p95_latency_us,
            "disruption_rate": metrics.disruption_rate,
            "mean_def": metrics.mean_def,
            "mean_energy_efficiency": metrics.mean_energy_efficiency,
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="4-Way Controller Comparison")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--quick", action="store_true", help="5 episodes for CI")
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional output path for JSON summary payload.",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Optional output path for markdown table.",
    )
    parser.add_argument(
        "--output-tex",
        type=str,
        default=None,
        help="Optional output path for LaTeX table.",
    )
    args = parser.parse_args()
    if args.quick:
        args.episodes = 5
    results = run_comparison(n_episodes=args.episodes)
    md_table = generate_comparison_table(results)
    print("\n" + md_table)

    if args.output_json:
        save_results_json(results, Path(args.output_json))
        print(f"\nJSON summary saved to {args.output_json}")
    if args.output_md:
        Path(args.output_md).write_text(md_table + "\n", encoding="utf-8")
        print(f"Markdown table saved to {args.output_md}")
    if args.output_tex:
        tex = generate_latex_table(results)
        Path(args.output_tex).write_text(tex + "\n", encoding="utf-8")
        print(f"LaTeX table saved to {args.output_tex}")
