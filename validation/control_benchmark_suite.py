# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Integrated Control Benchmark Suite
from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np


@dataclass
class BenchmarkScenario:
    name: str
    env_config: dict[str, Any]
    reference_trajectory: Callable[[float], np.ndarray]
    duration_s: float
    disturbances: list[tuple[float, str, float]] = field(default_factory=list)
    n_episodes: int = 10


class ControllerWrapper(Protocol):
    def reset(self) -> None: ...

    def step(self, obs: np.ndarray, ref: np.ndarray, dt: float) -> np.ndarray: ...

    def name(self) -> str: ...


@dataclass
class BenchmarkResults:
    controller_name: str
    scenario_name: str
    iae: float
    ise: float
    itae: float
    max_overshoot_pct: float
    settling_time_s: float
    control_effort: float
    violations: int
    computation_time_us: float


class BenchmarkRunner:
    def __init__(self, controllers: list[ControllerWrapper], scenarios: list[BenchmarkScenario]):
        self.controllers = controllers
        self.scenarios = scenarios
        self.results: list[BenchmarkResults] = []

    def run(self) -> list[BenchmarkResults]:
        for scenario in self.scenarios:
            for controller in self.controllers:
                print(f"Running {controller.name()} on {scenario.name}...")
                res = self._run_single(controller, scenario)
                self.results.append(res)
        return self.results

    def _run_single(
        self, controller: ControllerWrapper, scenario: BenchmarkScenario
    ) -> BenchmarkResults:
        # Simple mock of the Gym interface to run the test offline
        dt = scenario.env_config.get("dt", 0.05)
        n_steps = int(scenario.duration_s / dt)

        all_iae = []
        all_ise = []
        all_itae = []
        all_overshoot = []
        all_settling = []
        all_effort = []
        all_violations = []
        all_time = []

        for ep in range(scenario.n_episodes):
            controller.reset()

            # Simple 1D plant mock for standard comparison: x_dot = -a x + b u + d
            x = np.array([0.0])
            a = 0.5
            b = 1.0

            ep_iae = 0.0
            ep_ise = 0.0
            ep_itae = 0.0
            ep_effort = 0.0
            ep_violations = 0

            x_traj = np.zeros(n_steps)
            ref_traj = np.zeros(n_steps)

            comp_times = []

            for k in range(n_steps):
                t = k * dt
                ref = scenario.reference_trajectory(t)
                ref_arr = np.array([ref])

                # Disturbance logic
                dist = 0.0
                for d_t, d_type, d_mag in scenario.disturbances:
                    if t >= d_t:
                        if d_type == "step":
                            dist += d_mag

                # Add noise
                if scenario.name == "noise_robustness":
                    obs = x + np.random.randn(1) * 0.05
                else:
                    obs = x

                t0 = time.perf_counter()
                u = controller.step(obs, ref_arr, dt)
                t1 = time.perf_counter()

                comp_times.append((t1 - t0) * 1e6)

                # Actuator saturation mock
                if scenario.name == "actuator_saturation":
                    u = np.clip(u, -20.0, 20.0)

                ep_effort += float(np.sum(u**2)) * dt

                # Step plant
                x = x + dt * (-a * x + b * u + dist)

                err = float(x[0] - ref)
                abs_err = abs(err)

                ep_iae += abs_err * dt
                ep_ise += err**2 * dt
                ep_itae += t * abs_err * dt

                # Check constraints (mock beta limit at 3.5)
                if x[0] > 3.5:
                    ep_violations += 1

                x_traj[k] = x[0]
                ref_traj[k] = ref

            all_iae.append(ep_iae)
            all_ise.append(ep_ise)
            all_itae.append(ep_itae)
            all_effort.append(ep_effort)
            all_violations.append(ep_violations)
            all_time.append(np.mean(comp_times))

            # Overshoot and settling
            # Assume step response if ref changes
            if ref_traj[0] != ref_traj[-1]:
                final_ref = ref_traj[-1]
                max_val = np.max(x_traj)
                overshoot = max(0.0, (max_val - final_ref) / final_ref * 100.0)

                # Find settling time (last time it was outside 2% band)
                band = 0.02 * final_ref
                outside_idx = np.where(np.abs(x_traj - final_ref) > band)[0]
                settling = (outside_idx[-1] * dt) if len(outside_idx) > 0 else 0.0
            else:
                overshoot = 0.0
                settling = 0.0

            all_overshoot.append(overshoot)
            all_settling.append(settling)

        return BenchmarkResults(
            controller_name=controller.name(),
            scenario_name=scenario.name,
            iae=float(np.mean(all_iae)),
            ise=float(np.mean(all_ise)),
            itae=float(np.mean(all_itae)),
            max_overshoot_pct=float(np.mean(all_overshoot)),
            settling_time_s=float(np.mean(all_settling)),
            control_effort=float(np.mean(all_effort)),
            violations=int(np.mean(all_violations)),
            computation_time_us=float(np.mean(all_time)),
        )

    def save_json(self, path: Path):
        data = [dataclasses.asdict(r) for r in self.results]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def save_markdown(self, path: Path):
        with open(path, "w") as f:
            f.write(
                "| Controller | Scenario | IAE | Settling [s] | Overshoot [%] | Violations | Time [µs] |\n"
            )
            f.write(
                "|------------|----------|-----|-------------|---------------|------------|-----------|\n"
            )
            for r in self.results:
                f.write(
                    f"| {r.controller_name} | {r.scenario_name} | "
                    f"{r.iae:.2f} | {r.settling_time_s:.2f} | "
                    f"{r.max_overshoot_pct:.1f} | {r.violations} | {r.computation_time_us:.1f} |\n"
                )


# --- Standard Scenarios ---


def setpoint_tracking() -> BenchmarkScenario:
    # Step from 2.0 to 2.5 at t=5s
    def ref(t):
        return 2.5 if t >= 5.0 else 2.0

    return BenchmarkScenario(
        name="setpoint_tracking",
        env_config={"dt": 0.05},
        reference_trajectory=ref,
        duration_s=10.0,
        n_episodes=1,
    )


# --- Controller Wrappers ---


class PIDWrapper:
    def __init__(self, Kp: float = 1.0, Ki: float = 0.5):
        self.Kp = Kp
        self.Ki = Ki
        self.integral = 0.0

    def reset(self):
        self.integral = 0.0

    def step(self, obs: np.ndarray, ref: np.ndarray, dt: float) -> np.ndarray:
        err = ref[0] - obs[0]
        self.integral += err * dt
        u = self.Kp * err + self.Ki * self.integral
        return np.array([u])

    def name(self) -> str:
        return "PID"
