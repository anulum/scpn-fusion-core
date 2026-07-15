# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Shielded Safety-Contract Comparison
"""SNN vs PID vs RL-shielded safety-contract comparison.

Runs each controller inside the same
:class:`~scpn_fusion.control.shielded_tokamak_env.ShieldedTokamakEnv` and reports
the *safety contract* rather than raw tracking performance: how often the plasma
entered a limit-breach state, how many actuator clamps the shield had to apply,
the disruption rate, and mean reward. The point of the table is to show that the
interlock+Lyapunov shield holds the hard-constraint contract across policies of
very different provenance (a hand-tuned PID law, a spiking-neural controller, and
a learned constrained-PPO policy).

The heavy equilibrium solve makes large real-env sweeps a validation-lane job;
the aggregation and table logic are unit-tested with injected episode results.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeAlias

import numpy as np
from numpy.typing import NDArray

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from scpn_fusion.control.shielded_tokamak_env import ShieldedTokamakEnv

FloatArray: TypeAlias = NDArray[np.float64]
GymPolicy: TypeAlias = Callable[[FloatArray], FloatArray]


@dataclass
class ShieldedEpisodeResult:
    """Safety-contract metrics from one shielded episode."""

    reward: float
    steps: int
    disrupted: bool
    limit_breach_steps: int
    shield_clamp_events: int


@dataclass
class ContractMetrics:
    """Aggregate safety-contract metrics for a controller."""

    name: str
    n_episodes: int = 0
    mean_reward: float = 0.0
    disruption_rate: float = 0.0
    mean_limit_breach_steps: float = 0.0
    mean_shield_clamp_events: float = 0.0
    episodes: list[ShieldedEpisodeResult] = field(default_factory=list)


def pid_gym_policy(
    obs: FloatArray,
    *,
    kp_r: float = 0.5,
    kp_z: float = 0.5,
    kp_beta: float = 0.2,
    beta_target: float = 1.0,
) -> FloatArray:
    """Deterministic proportional baseline mapping observation to coil/heating deltas.

    Observation ``[R, Z, Ip, Beta, dR, dZ, XP_R, XP_Z]`` -> action
    ``[PF1_top, PF3_radial, PF5_bottom, Heating]``. Top and bottom coils oppose to
    correct the vertical error; the radial coil corrects the radial error; heating
    tracks a beta target. All outputs are clipped to the normalised ``[-1, 1]`` band.
    """
    o = np.asarray(obs, dtype=np.float64).ravel()
    err_r, err_z, beta = float(o[4]), float(o[5]), float(o[3])
    pf_top = np.clip(kp_z * err_z, -1.0, 1.0)
    pf_radial = np.clip(kp_r * err_r, -1.0, 1.0)
    pf_bottom = np.clip(-kp_z * err_z, -1.0, 1.0)
    heating = np.clip(kp_beta * (beta_target - beta), -1.0, 1.0)
    return np.array([pf_top, pf_radial, pf_bottom, heating], dtype=np.float64)


def evaluate_contract(
    policy: GymPolicy,
    env: ShieldedTokamakEnv,
    shot_duration: int,
) -> ShieldedEpisodeResult:
    """Run one shielded episode and tally its safety-contract metrics."""
    obs, _ = env.reset()
    total_reward = 0.0
    limit_breach_steps = 0
    disrupted = False
    steps = 0
    for _ in range(int(shot_duration)):
        action = policy(np.asarray(obs, dtype=np.float64))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        allowed = info.get("interlock_allowed", {})
        if any(not enabled for enabled in allowed.values()):
            limit_breach_steps += 1
        if info.get("disrupted", False) or info.get("shield_halt", False):
            disrupted = True
        if terminated or truncated:
            break
    return ShieldedEpisodeResult(
        reward=total_reward,
        steps=steps,
        disrupted=disrupted,
        limit_breach_steps=limit_breach_steps,
        shield_clamp_events=env.report.clamp_events,
    )


def aggregate_contract(name: str, episodes: list[ShieldedEpisodeResult]) -> ContractMetrics:
    """Aggregate a controller's episodes into contract metrics."""
    metrics = ContractMetrics(name=name, episodes=list(episodes))
    if not episodes:
        return metrics
    metrics.n_episodes = len(episodes)
    metrics.mean_reward = float(np.mean([e.reward for e in episodes]))
    metrics.disruption_rate = float(np.mean([e.disrupted for e in episodes]))
    metrics.mean_limit_breach_steps = float(np.mean([e.limit_breach_steps for e in episodes]))
    metrics.mean_shield_clamp_events = float(np.mean([e.shield_clamp_events for e in episodes]))
    return metrics


def run_contract_comparison(
    controllers: dict[str, Callable[[int], ShieldedEpisodeResult]],
    n_episodes: int,
    shot_duration: int,
) -> dict[str, ContractMetrics]:
    """Run each controller's episode runner ``n_episodes`` times and aggregate."""
    if not controllers:
        raise RuntimeError("No controllers supplied for the shielded contract comparison.")
    results: dict[str, ContractMetrics] = {}
    for name, run_fn in controllers.items():
        episodes: list[ShieldedEpisodeResult] = []
        for _ in range(int(n_episodes)):
            episodes.append(run_fn(shot_duration))
        results[name] = aggregate_contract(name, episodes)
    return results


def generate_contract_table(results: dict[str, ContractMetrics]) -> str:
    """Render the safety-contract comparison as a markdown table."""
    lines = [
        "| Controller | Episodes | Mean Reward | Disruption Rate | "
        "Limit-Breach Steps | Shield Clamps |",
        "|------------|----------|-------------|-----------------|"
        "--------------------|---------------|",
    ]
    for name, m in results.items():
        lines.append(
            f"| {name:<10} | {m.n_episodes:>8} | {m.mean_reward:>11.4f} "
            f"| {m.disruption_rate:>15.2%} | {m.mean_limit_breach_steps:>18.2f} "
            f"| {m.mean_shield_clamp_events:>13.2f} |"
        )
    return "\n".join(lines)


def save_contract_json(results: dict[str, ContractMetrics], output_path: Path) -> None:
    """Serialise the contract metrics to a JSON artifact."""
    payload: dict[str, Any] = {"controllers": {}}
    for name, m in results.items():
        payload["controllers"][name] = {
            "n_episodes": m.n_episodes,
            "mean_reward": m.mean_reward,
            "disruption_rate": m.disruption_rate,
            "mean_limit_breach_steps": m.mean_limit_breach_steps,
            "mean_shield_clamp_events": m.mean_shield_clamp_events,
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_real_controllers(
    shot_duration: int,
) -> dict[str, Callable[[int], ShieldedEpisodeResult]]:  # pragma: no cover - heavy env lane
    """Wire PID, RL-shielded (and SNN when available) over the real TokamakEnv.

    Executed only on the validation lane; the equilibrium solve is too heavy for
    the unit-test surface, which exercises the aggregation logic directly.
    """
    from scpn_fusion.control.gym_tokamak_env import TokamakEnv
    from scpn_fusion.control.safe_rl_controller import (
        ConstrainedGymTokamakEnv,
        LagrangianPPO,
        default_safety_constraints,
    )

    def make_env() -> ShieldedTokamakEnv:
        return ShieldedTokamakEnv(TokamakEnv(max_steps=shot_duration))

    ppo = LagrangianPPO(
        ConstrainedGymTokamakEnv(TokamakEnv(max_steps=shot_duration), default_safety_constraints()),
        seed=0,
    )
    ppo.train(total_timesteps=shot_duration * 20)

    def pid_runner(duration: int) -> ShieldedEpisodeResult:
        return evaluate_contract(pid_gym_policy, make_env(), duration)

    def rl_runner(duration: int) -> ShieldedEpisodeResult:
        return evaluate_contract(lambda obs: ppo.predict(obs), make_env(), duration)

    controllers: dict[str, Callable[[int], ShieldedEpisodeResult]] = {
        "PID": pid_runner,
        "RL-shielded": rl_runner,
    }
    try:
        from scpn_fusion.control.nengo_snn_wrapper import (
            NengoSNNConfig,
            NengoSNNController,
            nengo_available,
        )

        if nengo_available():
            snn = NengoSNNController(NengoSNNConfig(n_neurons=200, n_channels=2))

            def snn_policy(obs: FloatArray) -> FloatArray:
                o = np.asarray(obs, dtype=np.float64).ravel()
                out = np.asarray(snn.step(np.array([float(o[4]), float(o[5])])), dtype=np.float64)
                pf = float(out.ravel()[0])
                return np.array([pf, pf, -pf, 0.0], dtype=np.float64)

            def snn_runner(duration: int) -> ShieldedEpisodeResult:
                return evaluate_contract(snn_policy, make_env(), duration)

            controllers["Nengo-SNN"] = snn_runner
    except ImportError:
        pass
    return controllers


if __name__ == "__main__":  # pragma: no cover - CLI validation lane
    parser = argparse.ArgumentParser(description="Shielded safety-contract comparison")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--shot-duration", type=int, default=30)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-md", type=str, default=None)
    args = parser.parse_args()

    controllers = _build_real_controllers(args.shot_duration)
    results = run_contract_comparison(controllers, args.episodes, args.shot_duration)
    table = generate_contract_table(results)
    print(table)
    if args.output_json:
        save_contract_json(results, Path(args.output_json))
    if args.output_md:
        Path(args.output_md).write_text(table + "\n", encoding="utf-8")
