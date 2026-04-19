# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Feedforward Scenario Scheduler
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class ScenarioWaveform:
    """Piecewise-linear time-series for a single actuator channel."""

    name: str
    times: np.ndarray
    values: np.ndarray
    interp_kind: str = "linear"

    def __call__(self, t: float) -> float:
        return float(np.interp(t, self.times, self.values))


class ScenarioSchedule:
    """Collection of named waveforms defining a tokamak discharge scenario."""

    def __init__(self, waveforms: dict[str, ScenarioWaveform]):
        self.waveforms = waveforms

    def evaluate(self, t: float) -> dict[str, float]:
        """Interpolate all waveforms at time t."""
        return {name: wf(t) for name, wf in self.waveforms.items()}

    def duration(self) -> float:
        """Maximum endpoint across all waveforms [s]."""
        if not self.waveforms:
            return 0.0
        return float(max(wf.times[-1] for wf in self.waveforms.values()))

    def validate(self) -> list[str]:
        """Check monotonic time vectors and physical sign constraints."""
        errors = []
        for name, wf in self.waveforms.items():
            if not np.all(np.diff(wf.times) >= 0):
                errors.append(f"Waveform {name} has non-monotonic time vector.")

            if "Ip" in name and np.any(wf.values < 0):
                errors.append(f"Waveform {name} has negative plasma current.")

            if "P_" in name and np.any(wf.values < 0):
                errors.append(f"Waveform {name} has negative heating power.")

            if "n_e" in name and np.any(wf.values <= 0):
                errors.append(f"Waveform {name} has non-positive density.")

        return errors


class FeedforwardController:
    """Combines pre-computed feedforward trajectories with a feedback trim."""

    def __init__(self, schedule: ScenarioSchedule, feedback: Callable):
        self.schedule = schedule
        self.feedback = feedback

    def step(self, x: np.ndarray, t: float, dt: float) -> np.ndarray:
        """u = u_ff(t) + u_fb(x_err)."""
        ff_dict = self.schedule.evaluate(t)

        u_ff = np.zeros(3)
        u_ff[0] = ff_dict.get("P_aux", 0.0)
        u_ff[1] = ff_dict.get("Ip", 0.0)
        u_ff[2] = ff_dict.get("n_gas", 0.0)

        x_ref = np.zeros(len(x))
        x_ref[0] = ff_dict.get("Ip", 0.0)

        u_fb = self.feedback(x, x_ref, t, dt)

        return np.asarray(u_ff + u_fb)


class ScenarioOptimizer:
    """Offline trajectory design via Nelder-Mead."""

    def __init__(
        self, plant_model: Callable, target_state: np.ndarray, T_total: float, dt: float = 0.5
    ):
        self.plant_model = plant_model
        self.target_state = target_state
        self.T_total = T_total
        self.dt = dt

    def optimize(self, n_iter: int = 100) -> ScenarioSchedule:
        """Minimize state-tracking cost over T_total via Nelder-Mead on waveform knots."""
        times = np.array([0.0, self.T_total / 2.0, self.T_total])

        n_u = 2
        p0 = np.zeros(n_u * len(times))

        def objective(p: np.ndarray) -> float:
            p = p.reshape(n_u, len(times))
            wfs = {
                "P_aux": ScenarioWaveform("P_aux", times, p[0]),
                "Ip": ScenarioWaveform("Ip", times, p[1]),
            }
            sched = ScenarioSchedule(wfs)

            x = np.zeros(len(self.target_state))
            cost = 0.0

            t = 0.0
            while t < self.T_total:
                u_dict = sched.evaluate(t)
                u = np.array([u_dict["P_aux"], u_dict["Ip"]])

                x = self.plant_model(x, u, self.dt)

                err = x - self.target_state
                cost += np.sum(err**2) * self.dt

                t += self.dt

            return cost

        import scipy.optimize

        res = scipy.optimize.minimize(
            objective, p0, method="Nelder-Mead", options={"maxiter": n_iter}
        )
        p_opt = res.x.reshape(n_u, len(times))

        wfs = {
            "P_aux": ScenarioWaveform("P_aux", times, p_opt[0]),
            "Ip": ScenarioWaveform("Ip", times, p_opt[1]),
        }
        return ScenarioSchedule(wfs)


def iter_15ma_baseline() -> ScenarioSchedule:
    times = np.array([0, 10, 30, 60, 400, 430, 480], dtype=float)
    ip_vals = np.array([0.5, 5.0, 10.0, 15.0, 15.0, 10.0, 2.0])
    p_nbi = np.array([0.0, 0.0, 10.0, 33.0, 33.0, 10.0, 0.0])
    p_eccd = np.array([0.0, 0.0, 5.0, 17.0, 17.0, 5.0, 0.0])
    n_e = np.array([0.5, 1.0, 3.0, 5.0, 5.0, 3.0, 0.5])

    wfs = {
        "Ip": ScenarioWaveform("Ip", times, ip_vals),
        "P_NBI": ScenarioWaveform("P_NBI", times, p_nbi),
        "P_ECCD": ScenarioWaveform("P_ECCD", times, p_eccd),
        "n_e": ScenarioWaveform("n_e", times, n_e),
        "P_aux": ScenarioWaveform("P_aux", times, p_nbi + p_eccd),
    }
    return ScenarioSchedule(wfs)


def nstx_u_1ma_standard() -> ScenarioSchedule:
    times = np.array([0.0, 0.2, 0.5, 1.5, 1.8, 2.0])
    ip_vals = np.array([0.1, 0.5, 1.0, 1.0, 0.5, 0.1])
    p_aux = np.array([0.0, 2.0, 8.0, 8.0, 2.0, 0.0])

    wfs = {
        "Ip": ScenarioWaveform("Ip", times, ip_vals),
        "P_aux": ScenarioWaveform("P_aux", times, p_aux),
    }
    return ScenarioSchedule(wfs)
