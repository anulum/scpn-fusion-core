# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — TORAX Hybrid Realtime Loop (GAI-02)
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Synthetic TORAX-hybrid realtime control lane for NSTX-U-like scenarios."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from scpn_fusion.control.disruption_predictor import predict_disruption_risk
from scpn_fusion.scpn.compiler import FusionCompiler
from scpn_fusion.scpn.contracts import ControlScales, ControlTargets
from scpn_fusion.scpn.controller import NeuroSymbolicController
from scpn_fusion.scpn.structure import StochasticPetriNet


@dataclass(frozen=True)
class ToraxPlasmaState:
    beta_n: float
    q95: float
    li3: float
    w_thermal_mj: float


@dataclass(frozen=True)
class ToraxHybridCampaignResult:
    episodes: int
    steps_per_episode: int
    disruption_avoidance_rate: float
    torax_parity_pct: float
    p95_loop_latency_ms: float
    mean_risk: float
    passes_thresholds: bool


def _build_hybrid_controller() -> NeuroSymbolicController:
    net = StochasticPetriNet()
    net.add_place("x_R_pos", initial_tokens=0.0)
    net.add_place("x_R_neg", initial_tokens=0.0)
    net.add_place("a_R_pos", initial_tokens=0.0)
    net.add_place("a_R_neg", initial_tokens=0.0)
    net.add_transition("T_Rp", threshold=0.1)
    net.add_transition("T_Rn", threshold=0.1)
    net.add_arc("x_R_pos", "T_Rp", weight=1.0)
    net.add_arc("x_R_neg", "T_Rn", weight=1.0)
    net.add_arc("T_Rp", "a_R_pos", weight=1.0)
    net.add_arc("T_Rn", "a_R_neg", weight=1.0)
    net.compile()

    compiled = FusionCompiler(bitstream_length=1024, seed=211).compile(net, firing_mode="binary")
    artifact = compiled.export_artifact(
        name="gai02_torax_hybrid",
        dt_control_s=0.001,
        readout_config={
            "actions": [{"name": "dI_PF3_A", "pos_place": 2, "neg_place": 3}],
            "gains": [2200.0],
            "abs_max": [4500.0],
            "slew_per_s": [1e6],
        },
        injection_config=[
            {"place_id": 0, "source": "x_R_pos", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
            {"place_id": 1, "source": "x_R_neg", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
        ],
    )
    return NeuroSymbolicController(
        artifact=artifact,
        seed_base=314159265,
        targets=ControlTargets(R_target_m=1.85, Z_target_m=0.0),
        scales=ControlScales(R_scale_m=0.8, Z_scale_m=1.0),
    )


def _torax_policy(state: ToraxPlasmaState) -> float:
    """Reduced TORAX-like policy head for beta/q tracking."""

    beta_err = 1.85 - state.beta_n
    q_err = state.q95 - 4.9
    cmd = 1.10 * beta_err - 0.32 * q_err
    return float(np.clip(cmd, -1.6, 1.6))


def _torax_step(
    state: ToraxPlasmaState,
    command: float,
    disturbance: float,
    rng: np.random.Generator,
) -> ToraxPlasmaState:
    """Reduced TORAX-like transport/equilibrium state update."""

    command = float(np.clip(command, -2.0, 2.0))
    beta_n = state.beta_n + 0.045 * (
        0.85 * command - (state.beta_n - 1.85) - 0.52 * disturbance + rng.normal(0.0, 0.004)
    )
    q95 = state.q95 + 0.060 * (
        0.18 - 0.33 * command + 0.62 * disturbance - 0.16 * (state.q95 - 4.9) + rng.normal(0.0, 0.006)
    )
    li3 = state.li3 + 0.050 * (0.06 * command - 0.11 * disturbance - 0.09 * (state.li3 - 0.95))
    w_thermal = state.w_thermal_mj + 0.110 * (
        10.0 * command - 5.0 * disturbance - 0.06 * (state.w_thermal_mj - 140.0)
    )
    return ToraxPlasmaState(
        beta_n=float(np.clip(beta_n, 0.6, 3.2)),
        q95=float(np.clip(q95, 2.8, 7.5)),
        li3=float(np.clip(li3, 0.45, 1.8)),
        w_thermal_mj=float(np.clip(w_thermal, 50.0, 260.0)),
    )


def _risk_signal(state: ToraxPlasmaState, disturbance: float) -> float:
    return float(
        0.40
        + 0.42 * max(state.beta_n - 2.05, 0.0)
        + 0.38 * max(4.4 - state.q95, 0.0)
        + 0.22 * max(state.li3 - 1.25, 0.0)
        + 0.30 * disturbance
    )


def run_nstxu_torax_hybrid_campaign(
    *,
    seed: int = 42,
    episodes: int = 16,
    steps_per_episode: int = 220,
) -> ToraxHybridCampaignResult:
    """Run deterministic NSTX-U-like realtime hybrid control campaign."""

    rng = np.random.default_rng(int(seed))
    controller = _build_hybrid_controller()
    episodes = max(int(episodes), 1)
    steps = max(int(steps_per_episode), 32)

    disruptions = 0
    parity_scores = []
    latencies_ms = []
    all_risks = []

    for ep in range(episodes):
        base = ToraxPlasmaState(
            beta_n=float(rng.uniform(1.65, 1.95)),
            q95=float(rng.uniform(4.6, 5.2)),
            li3=float(rng.uniform(0.85, 1.05)),
            w_thermal_mj=float(rng.uniform(120.0, 170.0)),
        )
        torax_state = base
        hybrid_state = base
        signal_history = []
        streak_high_risk = 0
        beta_delta_sq = []
        beta_ref_sq = []

        for k in range(steps):
            phase = k / max(steps - 1, 1)
            disturbance = 0.0
            if 0.35 <= phase <= 0.58:
                disturbance = float(0.22 + 0.15 * np.sin(np.pi * (phase - 0.35) / 0.23))

            # TORAX-only baseline branch
            torax_cmd = _torax_policy(torax_state)
            torax_state = _torax_step(torax_state, torax_cmd, disturbance, rng)

            # Hybrid branch = TORAX command + SNN correction
            t0 = time.perf_counter()
            base_cmd = _torax_policy(hybrid_state)
            obs = {"R_axis_m": hybrid_state.beta_n, "Z_axis_m": 0.0}
            action = controller.step(obs, ep * steps + k)
            snn_corr = float(np.clip(action["dI_PF3_A"] / 4500.0, -0.45, 0.45))
            cmd = float(np.clip(base_cmd + 0.30 * snn_corr, -2.0, 2.0))
            hybrid_state = _torax_step(hybrid_state, cmd, disturbance, rng)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            sig = _risk_signal(hybrid_state, disturbance)
            signal_history.append(sig)
            toroidal = {
                "toroidal_n1_amp": 0.04 + 0.40 * disturbance,
                "toroidal_n2_amp": 0.03 + 0.25 * disturbance,
                "toroidal_n3_amp": 0.02 + 0.12 * disturbance,
                "toroidal_asymmetry_index": 0.05 + 0.48 * disturbance,
                "toroidal_radial_spread": 0.02 + 0.08 * disturbance,
            }
            risk = float(predict_disruption_risk(signal_history, toroidal))
            all_risks.append(risk)

            if risk > 0.93:
                streak_high_risk += 1
            else:
                streak_high_risk = 0
            if streak_high_risk >= 3:
                disruptions += 1
                break

            beta_delta_sq.append((hybrid_state.beta_n - torax_state.beta_n) ** 2)
            beta_ref_sq.append(torax_state.beta_n**2)

        if beta_ref_sq:
            rmse = float(np.sqrt(np.mean(beta_delta_sq)))
            scale = float(np.sqrt(np.mean(beta_ref_sq)))
            parity = float(np.clip(100.0 * (1.0 - rmse / max(scale, 1e-9)), 0.0, 100.0))
            parity_scores.append(parity)

    avoidance_rate = float(1.0 - disruptions / episodes)
    torax_parity = float(np.mean(parity_scores) if parity_scores else 0.0)
    p95_latency = float(np.percentile(latencies_ms, 95) if latencies_ms else 0.0)
    mean_risk = float(np.mean(all_risks) if all_risks else 0.0)
    passes = bool(avoidance_rate >= 0.90 and torax_parity >= 95.0 and p95_latency <= 1.0)

    return ToraxHybridCampaignResult(
        episodes=episodes,
        steps_per_episode=steps,
        disruption_avoidance_rate=avoidance_rate,
        torax_parity_pct=torax_parity,
        p95_loop_latency_ms=p95_latency,
        mean_risk=mean_risk,
        passes_thresholds=passes,
    )
