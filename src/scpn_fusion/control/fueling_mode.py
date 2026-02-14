# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Fueling Mode (GNEU-03)
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Ice-pellet fueling mode via Petri-to-SNN control path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from scpn_fusion.scpn.compiler import FusionCompiler
from scpn_fusion.scpn.contracts import (
    ControlObservation,
    ControlScales,
    ControlTargets,
)
from scpn_fusion.scpn.controller import NeuroSymbolicController
from scpn_fusion.scpn.structure import StochasticPetriNet


@dataclass(frozen=True)
class FuelingSimResult:
    final_density: float
    final_abs_error: float
    rmse: float
    steps: int
    dt_s: float
    history_density: list[float]
    history_command: list[float]


def _build_fueling_controller() -> NeuroSymbolicController:
    net = StochasticPetriNet()
    net.add_place("x_R_pos", initial_tokens=0.0)
    net.add_place("x_R_neg", initial_tokens=0.0)
    net.add_place("x_Z_pos", initial_tokens=0.0)
    net.add_place("x_Z_neg", initial_tokens=0.0)
    net.add_place("a_R_pos", initial_tokens=0.0)
    net.add_place("a_R_neg", initial_tokens=0.0)
    net.add_place("a_Z_pos", initial_tokens=0.0)
    net.add_place("a_Z_neg", initial_tokens=0.0)

    net.add_transition("T_Rp", threshold=0.1)
    net.add_transition("T_Rn", threshold=0.1)
    net.add_transition("T_Zp", threshold=0.1)
    net.add_transition("T_Zn", threshold=0.1)

    net.add_arc("x_R_pos", "T_Rp", weight=1.0)
    net.add_arc("x_R_neg", "T_Rn", weight=1.0)
    net.add_arc("x_Z_pos", "T_Zp", weight=1.0)
    net.add_arc("x_Z_neg", "T_Zn", weight=1.0)
    net.add_arc("T_Rp", "a_R_pos", weight=1.0)
    net.add_arc("T_Rn", "a_R_neg", weight=1.0)
    net.add_arc("T_Zp", "a_Z_pos", weight=1.0)
    net.add_arc("T_Zn", "a_Z_neg", weight=1.0)
    net.compile()

    compiled = FusionCompiler.with_reactor_lif_defaults(
        bitstream_length=1024,
        seed=77,
    ).compile(
        net, firing_mode="binary"
    )
    artifact = compiled.export_artifact(
        name="gneu03_fueling",
        dt_control_s=0.001,
        readout_config={
            "actions": [
                {"name": "dI_PF3_A", "pos_place": 4, "neg_place": 5},
                {"name": "dI_PF_topbot_A", "pos_place": 6, "neg_place": 7},
            ],
            "gains": [1000.0, 800.0],
            "abs_max": [5000.0, 5000.0],
            "slew_per_s": [1e6, 1e6],
        },
        injection_config=[
            {"place_id": 0, "source": "x_R_pos", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
            {"place_id": 1, "source": "x_R_neg", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
            {"place_id": 2, "source": "x_Z_pos", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
            {"place_id": 3, "source": "x_Z_neg", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
        ],
    )
    return NeuroSymbolicController(
        artifact=artifact,
        seed_base=987654321,
        targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
        scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
        sc_binary_margin=0.05,
    )


class IcePelletFuelingController:
    """Hybrid Petri-to-SNN + PI fueling controller."""

    def __init__(self, target_density: float = 1.0) -> None:
        td = float(target_density)
        if not np.isfinite(td) or td <= 0.0:
            raise ValueError("target_density must be finite and > 0.")
        self.target_density = td
        self.controller = _build_fueling_controller()
        self.integrator = 0.0

    def step(self, density: float, k: int, dt_s: float) -> tuple[float, float]:
        error = self.target_density - float(density)
        self.integrator += error * dt_s
        self.integrator = float(np.clip(self.integrator, -0.5, 0.5))

        # SNN pathway receives mapped pseudo-observation.
        obs: ControlObservation = {
            "R_axis_m": float(6.2 - 0.25 * np.clip(error, -1.0, 1.0)),
            "Z_axis_m": 0.0,
        }
        action = self.controller.step(obs, k)
        u_snn_raw = float(action["dI_PF3_A"]) / 5000.0
        # Keep neuromorphic actuation active away from setpoint while avoiding
        # persistent jitter-induced offset near final convergence.
        snn_gate = float(np.clip(abs(error) / 0.05, 0.0, 1.0))
        u_snn = 0.25 * snn_gate * u_snn_raw

        # PI path gives tight density convergence, SNN term perturbs/controls actuation.
        u_pi = 1.95 * error + 7.2 * self.integrator
        command = float(np.clip(u_pi + u_snn, -2.0, 2.0))
        return command, error


def simulate_iter_density_control(
    *,
    target_density: float = 1.0,
    initial_density: float = 0.82,
    steps: int = 3000,
    dt_s: float = 1e-3,
) -> FuelingSimResult:
    steps = max(int(steps), 8)
    raw_dt_s = float(dt_s)
    target_density = float(target_density)
    density = float(initial_density)
    if not np.isfinite(target_density) or target_density <= 0.0:
        raise ValueError("target_density must be finite and > 0.")
    if not np.isfinite(density) or density < 0.0:
        raise ValueError("initial_density must be finite and >= 0.")
    if not np.isfinite(raw_dt_s) or raw_dt_s <= 0.0:
        raise ValueError("dt_s must be finite and > 0.")
    dt_s = max(raw_dt_s, 1e-5)

    ctrl = IcePelletFuelingController(target_density=target_density)
    history_density: list[float] = []
    history_command: list[float] = []
    sq_err = 0.0

    # Reduced ITER-like 0D density dynamics.
    leak_coeff = 1.15
    fueling_gain = 1.15
    baseline = leak_coeff * target_density

    for k in range(steps):
        command, error = ctrl.step(density, k, dt_s)
        fueling_rate = baseline + fueling_gain * command
        density = density + dt_s * (fueling_rate - leak_coeff * density)
        density = float(max(density, 0.0))

        history_density.append(density)
        history_command.append(command)
        sq_err += error * error

    final_abs_error = abs(target_density - density)
    rmse = float(np.sqrt(sq_err / steps))
    return FuelingSimResult(
        final_density=density,
        final_abs_error=final_abs_error,
        rmse=rmse,
        steps=steps,
        dt_s=dt_s,
        history_density=history_density,
        history_command=history_command,
    )


def run_fueling_mode(
    *,
    target_density: float = 1.0,
    initial_density: float = 0.82,
    steps: int = 3000,
    dt_s: float = 1e-3,
) -> dict[str, Any]:
    """
    Run deterministic fueling simulation and return summary metrics.
    """
    result = simulate_iter_density_control(
        target_density=target_density,
        initial_density=initial_density,
        steps=steps,
        dt_s=dt_s,
    )
    dens = np.asarray(result.history_density, dtype=np.float64)
    cmd = np.asarray(result.history_command, dtype=np.float64)
    return {
        "target_density": float(target_density),
        "initial_density": float(initial_density),
        "steps": int(result.steps),
        "dt_s": float(result.dt_s),
        "final_density": float(result.final_density),
        "final_abs_error": float(result.final_abs_error),
        "rmse": float(result.rmse),
        "max_abs_command": float(np.max(np.abs(cmd))) if cmd.size else 0.0,
        "min_density": float(np.min(dens)) if dens.size else 0.0,
        "max_density": float(np.max(dens)) if dens.size else 0.0,
        "passes_thresholds": bool(result.final_abs_error <= 1e-3),
    }
