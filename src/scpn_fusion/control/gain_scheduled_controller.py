# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Gain-Scheduled Multi-Regime Controller
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np


class OperatingRegime(Enum):
    RAMP_UP = auto()
    L_MODE_FLAT = auto()
    LH_TRANSITION = auto()
    H_MODE_FLAT = auto()
    RAMP_DOWN = auto()
    DISRUPTION_MITIGATION = auto()


@dataclass
class RegimeController:
    regime: OperatingRegime
    Kp: np.ndarray
    Ki: np.ndarray
    Kd: np.ndarray
    x_ref: np.ndarray
    constraints: dict[str, Any]


class RegimeDetector:
    def __init__(self, thresholds: dict[str, float] | None = None):
        self.thresholds = thresholds or {
            "ramp_rate": 0.1,
            "tau_e_L_mode": 1.0,
            "tau_e_jump": 1.5,
            "disruption_prob": 0.8,
        }
        self.history: list[OperatingRegime] = []
        self.history_len = 5

    def detect(
        self, state: np.ndarray, dstate_dt: np.ndarray, tau_E: float, p_disrupt: float
    ) -> OperatingRegime:
        dIp_dt = dstate_dt[0]

        if p_disrupt > self.thresholds["disruption_prob"]:
            new_reg = OperatingRegime.DISRUPTION_MITIGATION
        elif dIp_dt > self.thresholds["ramp_rate"]:
            new_reg = OperatingRegime.RAMP_UP
        elif dIp_dt < -self.thresholds["ramp_rate"]:
            new_reg = OperatingRegime.RAMP_DOWN
        else:
            if tau_E > self.thresholds["tau_e_jump"] * self.thresholds["tau_e_L_mode"]:
                new_reg = OperatingRegime.H_MODE_FLAT
            else:
                new_reg = OperatingRegime.L_MODE_FLAT

        self.history.append(new_reg)
        if len(self.history) > self.history_len:
            self.history.pop(0)

        if self.history.count(new_reg) == self.history_len:
            return new_reg
        elif len(set(self.history)) == 1:
            return self.history[0]
        else:
            return self.history[0] if len(self.history) > 0 else new_reg


class GainScheduledController:
    def __init__(self, controllers: dict[OperatingRegime, RegimeController]):
        self.controllers = controllers
        self.current_regime = OperatingRegime.RAMP_UP
        self.prev_regime = OperatingRegime.RAMP_UP

        self.Kp = self.controllers[self.current_regime].Kp.copy()
        self.Ki = self.controllers[self.current_regime].Ki.copy()
        self.Kd = self.controllers[self.current_regime].Kd.copy()

        self.integral_error = np.zeros_like(self.controllers[self.current_regime].x_ref)
        self.prev_error = np.zeros_like(self.integral_error)

        self.switch_time = -1.0
        self.tau_switch = 0.5

    def step(
        self,
        x: np.ndarray,
        t: float,
        dt: float,
        detected_regime: OperatingRegime,
    ) -> np.ndarray:
        if detected_regime != self.current_regime:
            self.prev_regime = self.current_regime
            self.current_regime = detected_regime
            self.switch_time = t

            if detected_regime == OperatingRegime.DISRUPTION_MITIGATION:
                self.integral_error.fill(0.0)

        if self.switch_time >= 0 and t - self.switch_time < self.tau_switch:
            alpha = (t - self.switch_time) / self.tau_switch
            ctrl_old = self.controllers[self.prev_regime]
            ctrl_new = self.controllers[self.current_regime]

            self.Kp = (1 - alpha) * ctrl_old.Kp + alpha * ctrl_new.Kp
            self.Ki = (1 - alpha) * ctrl_old.Ki + alpha * ctrl_new.Ki
            self.Kd = (1 - alpha) * ctrl_old.Kd + alpha * ctrl_new.Kd
            x_ref = (1 - alpha) * ctrl_old.x_ref + alpha * ctrl_new.x_ref
        else:
            ctrl_new = self.controllers[self.current_regime]
            self.Kp = ctrl_new.Kp
            self.Ki = ctrl_new.Ki
            self.Kd = ctrl_new.Kd
            x_ref = ctrl_new.x_ref

        error = x_ref - x
        self.integral_error += error * dt
        derror = (error - self.prev_error) / max(dt, 1e-6)

        u = self.Kp * error + self.Ki * self.integral_error + self.Kd * derror
        self.prev_error = error

        return np.asarray(u)


class ScenarioWaveform:
    def __init__(
        self, name: str, times: np.ndarray, values: np.ndarray, interp_kind: str = "linear"
    ):
        self.name = name
        self.times = times
        self.values = values
        self.interp_kind = interp_kind

    def __call__(self, t: float) -> float:
        return float(np.interp(t, self.times, self.values))


class ScenarioSchedule:
    def __init__(self, waveforms: dict[str, ScenarioWaveform]):
        self.waveforms = waveforms

    def evaluate(self, t: float) -> dict[str, float]:
        return {name: wf(t) for name, wf in self.waveforms.items()}

    def duration(self) -> float:
        if not self.waveforms:
            return 0.0
        return float(max(wf.times[-1] for wf in self.waveforms.values()))

    def validate(self) -> list[str]:
        errors = []
        for name, wf in self.waveforms.items():
            if not np.all(np.diff(wf.times) > 0):
                errors.append(f"Waveform {name} has non-monotonic times.")
        return errors


def iter_baseline_schedule() -> ScenarioSchedule:
    times = np.array([0, 10, 30, 60, 400, 430, 480], dtype=float)
    ip_vals = np.array([0.5, 5.0, 10.0, 15.0, 15.0, 10.0, 2.0])
    p_nbi = np.array([0.0, 0.0, 10.0, 33.0, 33.0, 10.0, 0.0])
    p_eccd = np.array([0.0, 0.0, 5.0, 17.0, 17.0, 5.0, 0.0])

    wfs = {
        "Ip": ScenarioWaveform("Ip", times, ip_vals),
        "P_NBI": ScenarioWaveform("P_NBI", times, p_nbi),
        "P_ECCD": ScenarioWaveform("P_ECCD", times, p_eccd),
    }
    return ScenarioSchedule(wfs)
