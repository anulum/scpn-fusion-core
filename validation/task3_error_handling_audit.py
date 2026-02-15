# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Task 3 Error Handling Audit
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Task 3: fault-injection audit for SOC/RL uptime and 3D flux liveness."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scpn_fusion.control.advanced_soc_fusion_learning import (
    CoupledSandpileReactor,
    FusionAIAgent,
)
from scpn_fusion.control.disruption_predictor import apply_bit_flip_fault
from scpn_fusion.core.divertor_thermal_sim import DivertorLab
from scpn_fusion.core.equilibrium_3d import FourierMode3D, VMECStyleEquilibrium3D
from scpn_fusion.core.geometry_3d import Reactor3DBuilder
from scpn_fusion.scpn.compiler import FusionCompiler
from scpn_fusion.scpn.contracts import ControlScales, ControlTargets
from scpn_fusion.scpn.controller import NeuroSymbolicController
from scpn_fusion.scpn.structure import StochasticPetriNet


ROOT = Path(__file__).resolve().parents[1]


def _require_int(name: str, value: Any, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    out = int(value)
    if out < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return out


def _require_finite(name: str, value: Any, minimum: float | None = None) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite.")
    if minimum is not None and out < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return out


def _require_fraction(name: str, value: Any) -> float:
    out = _require_finite(name, value, 0.0)
    if out > 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return out


def _build_petri_liveness_controller() -> NeuroSymbolicController:
    net = StochasticPetriNet()
    net.add_place("x_R_pos", initial_tokens=0.0)
    net.add_place("x_R_neg", initial_tokens=0.0)
    net.add_place("a_R_pos", initial_tokens=0.0)
    net.add_place("a_R_neg", initial_tokens=0.0)
    net.add_transition("T_Rp", threshold=0.05)
    net.add_transition("T_Rn", threshold=0.05)
    net.add_arc("x_R_pos", "T_Rp", weight=1.0)
    net.add_arc("x_R_neg", "T_Rn", weight=1.0)
    net.add_arc("T_Rp", "a_R_pos", weight=1.0)
    net.add_arc("T_Rn", "a_R_neg", weight=1.0)
    net.compile()

    artifact = FusionCompiler(bitstream_length=1024, seed=315).compile(
        net, firing_mode="binary"
    ).export_artifact(
        name="task3_liveness",
        dt_control_s=0.001,
        readout_config={
            "actions": [{"name": "dI_PF3_A", "pos_place": 2, "neg_place": 3}],
            "gains": [1800.0],
            "abs_max": [3500.0],
            "slew_per_s": [1e6],
        },
        injection_config=[
            {
                "place_id": 0,
                "source": "x_R_pos",
                "scale": 1.0,
                "offset": 0.0,
                "clamp_0_1": True,
            },
            {
                "place_id": 1,
                "source": "x_R_neg",
                "scale": 1.0,
                "offset": 0.0,
                "clamp_0_1": True,
            },
        ],
    )
    return NeuroSymbolicController(
        artifact=artifact,
        seed_base=271828183,
        targets=ControlTargets(R_target_m=0.03, Z_target_m=0.0),
        scales=ControlScales(R_scale_m=0.08, Z_scale_m=1.0),
        runtime_backend="auto",
    )


def _init_divertor_lab() -> DivertorLab:
    # DivertorLab prints by design; keep validation output clean.
    with contextlib.redirect_stdout(io.StringIO()):
        return DivertorLab(P_sol_MW=35.0, R_major=1.4, B_pol=2.3)


def _simulate_overheat_fault(
    lab: DivertorLab,
    *,
    rng: np.random.Generator,
) -> tuple[bool, dict[str, float]]:
    flow_velocity = float(rng.uniform(0.0005, 0.05))
    expansion_factor = float(rng.uniform(9.0, 16.0))
    with contextlib.redirect_stdout(io.StringIO()):
        out = lab.simulate_temhd_liquid_metal(
            flow_velocity_m_s=flow_velocity,
            expansion_factor=expansion_factor,
        )
    overheat = bool(
        (not out["is_stable"])
        or out["surface_heat_flux_w_m2"] > 45.0e6
        or out["surface_temperature_c"] > 1400.0
    )
    return overheat, {
        "flow_velocity_m_s": float(flow_velocity),
        "expansion_factor": float(expansion_factor),
        "stability_index": float(out["stability_index"]),
        "surface_heat_flux_w_m2": float(out["surface_heat_flux_w_m2"]),
        "surface_temperature_c": float(out["surface_temperature_c"]),
    }


def _run_soc_fault_episode(
    *,
    seed: int,
    steps: int,
    injected_error_rate: float,
    error_check_interval: int,
    bitflip_fraction: float,
    noise_probability: float,
    overheat_downtime_steps: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    reactor = CoupledSandpileReactor(size=32)
    agent = FusionAIAgent(epsilon=0.06)
    divertor = _init_divertor_lab()

    current_ext_shear = 0.0
    uptime_steps = 0
    downtime_steps = 0
    forced_downtime = 0

    opportunities = 0
    injected_faults = 0
    bitflip_faults = 0
    overheat_faults = 0
    overheat_events = 0
    max_stability_index = 0.0

    for t in range(int(steps)):
        if t == 0:
            turb_prev = 0.0
            flow_prev = 0.0
        else:
            turb_prev = float(av_size)
            flow_prev = float(flow_val)

        state = agent.discretize_state(turb_prev, flow_prev)
        action_idx = agent.choose_action(state, rng)
        if action_idx == 0:
            current_ext_shear -= 0.05
        elif action_idx == 2:
            current_ext_shear += 0.05
        current_ext_shear = float(np.clip(current_ext_shear, 0.0, 1.0))

        if t % int(error_check_interval) == 0:
            opportunities += 1
            if float(rng.random()) < injected_error_rate:
                injected_faults += 1
                if float(rng.random()) < bitflip_fraction:
                    bitflip_faults += 1
                    idx = int(rng.integers(0, reactor.size))
                    bit = int(rng.integers(0, 52))
                    flipped = float(apply_bit_flip_fault(float(reactor.Z[idx]), bit))
                    reactor.Z[idx] = float(
                        np.clip(np.nan_to_num(flipped, nan=0.0, posinf=16.0, neginf=0.0), 0.0, 16.0)
                    )
                else:
                    overheat_faults += 1
                    overheat, overheat_meta = _simulate_overheat_fault(divertor, rng=rng)
                    max_stability_index = max(
                        max_stability_index, float(overheat_meta["stability_index"])
                    )
                    if overheat:
                        overheat_events += 1
                        forced_downtime = max(
                            int(forced_downtime), int(overheat_downtime_steps)
                        )

        reactor.drive()
        if float(rng.random()) < noise_probability:
            reactor.drive()

        av_size, flow_val, total_shear = reactor.step_physics(current_ext_shear)
        core_temp = reactor.get_profile_energy()
        reward = (core_temp * 0.1) - (float(av_size) * 0.5) - (current_ext_shear * 2.0)
        next_state = agent.discretize_state(float(av_size), float(flow_val))
        agent.learn(state, action_idx, next_state, reward)

        live_step = bool(
            np.isfinite(core_temp)
            and np.isfinite(flow_val)
            and np.isfinite(total_shear)
            and core_temp <= 360.0
            and float(flow_val) <= 4.8
            and forced_downtime == 0
        )
        if live_step:
            uptime_steps += 1
        else:
            downtime_steps += 1

        if forced_downtime > 0:
            forced_downtime -= 1

    uptime_rate = float(uptime_steps / max(steps, 1))
    observed_error_rate = float(injected_faults / max(opportunities, 1))
    return {
        "seed": int(seed),
        "steps": int(steps),
        "uptime_rate": uptime_rate,
        "downtime_steps": int(downtime_steps),
        "fault_opportunities": int(opportunities),
        "injected_faults": int(injected_faults),
        "observed_error_rate": observed_error_rate,
        "bitflip_faults": int(bitflip_faults),
        "overheat_faults": int(overheat_faults),
        "overheat_events": int(overheat_events),
        "max_stability_index": float(max_stability_index),
        "q_table_max_abs": float(np.max(np.abs(agent.q_table))),
        "total_reward": float(agent.total_reward),
    }


def _run_flux_liveness_campaign(
    *,
    seed: int,
    scenarios: int,
    injected_error_rate: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    controller = _build_petri_liveness_controller()

    liveness_ok = 0
    petri_activity_ok = 0
    actions: list[float] = []
    instabilities: list[float] = []

    for i in range(int(scenarios)):
        nfp = int(rng.integers(3, 7))
        mode = FourierMode3D(
            m=1,
            n=1,
            r_cos=float(rng.uniform(0.02, 0.09)),
            z_sin=float(rng.uniform(0.01, 0.07)),
        )
        eq = VMECStyleEquilibrium3D(
            r_axis=float(1.8 + rng.uniform(-0.2, 0.2)),
            z_axis=0.0,
            a_minor=float(0.45 + rng.uniform(-0.06, 0.08)),
            kappa=float(1.60 + rng.uniform(-0.15, 0.20)),
            triangularity=float(0.20 + rng.uniform(-0.10, 0.16)),
            nfp=nfp,
            modes=[mode],
        )
        builder = Reactor3DBuilder(equilibrium_3d=eq, solve_equilibrium=False)
        tracer = builder.create_fieldline_tracer(
            rotational_transform=float(0.40 + rng.uniform(-0.05, 0.08)),
            helical_coupling_scale=float(0.10 + rng.uniform(-0.03, 0.08)),
            radial_coupling_scale=float(0.05 + rng.uniform(-0.01, 0.03)),
            nfp=nfp,
        )
        trace = tracer.trace_line(
            rho0=0.92,
            theta0=0.07,
            phi0=0.0,
            toroidal_turns=6,
            steps_per_turn=120,
        )
        dtheta = np.diff(np.asarray(trace.theta, dtype=np.float64))
        instability = float(np.std(dtheta)) if dtheta.size else 1.0
        if float(rng.random()) < injected_error_rate:
            bit = int(rng.integers(0, 20))
            instability = float(apply_bit_flip_fault(instability, bit))
        instability = float(
            np.clip(np.nan_to_num(instability, nan=0.5, posinf=0.5, neginf=0.5), 0.0, 0.5)
        )
        instabilities.append(instability)

        xyz = np.asarray(trace.xyz, dtype=np.float64)
        scenario_live = bool(
            np.isfinite(np.asarray(trace.rho, dtype=np.float64)).all()
            and np.isfinite(np.asarray(trace.theta, dtype=np.float64)).all()
            and np.isfinite(np.asarray(trace.phi, dtype=np.float64)).all()
            and np.isfinite(xyz).all()
            and xyz.shape[0] >= 64
            and dtheta.size >= 32
        )

        seen_activity = False
        for k in range(12):
            obs_val = float(np.clip(instability * (1.0 + 0.10 * np.sin(0.5 * k)), 0.0, 1.0))
            obs = {"R_axis_m": obs_val, "Z_axis_m": 0.0}
            out = controller.step(obs, i * 100 + k)
            action = float(out["dI_PF3_A"])
            actions.append(action)
            if np.any(np.asarray(controller.last_sc_firing, dtype=np.float64) > 0.0):
                seen_activity = True
            scenario_live = bool(scenario_live and np.isfinite(action) and abs(action) <= 3500.0)

        if scenario_live:
            liveness_ok += 1
        if seen_activity:
            petri_activity_ok += 1

    return {
        "scenarios": int(scenarios),
        "liveness_pass_rate": float(liveness_ok / max(scenarios, 1)),
        "petri_activity_rate": float(petri_activity_ok / max(scenarios, 1)),
        "mean_instability_metric": float(np.mean(np.asarray(instabilities, dtype=np.float64))),
        "p95_abs_action_a": float(np.percentile(np.abs(np.asarray(actions, dtype=np.float64)), 95)),
    }


def run_campaign(
    *,
    seed: int = 42,
    episodes: int = 12,
    sim_seconds: int = 3600,
    dt_s: float = 1.0,
    injected_error_rate: float = 0.20,
    error_check_interval: int = 12,
    bitflip_fraction: float = 0.85,
    overheat_downtime_steps: int = 2,
    noise_probability: float = 0.01,
    flux_scenarios: int = 24,
) -> dict[str, Any]:
    t0 = time.perf_counter()

    seed_i = _require_int("seed", seed, 0)
    episodes_i = _require_int("episodes", episodes, 1)
    sim_seconds_i = _require_int("sim_seconds", sim_seconds, 60)
    dt = _require_finite("dt_s", dt_s, 1e-6)
    if dt > float(sim_seconds_i):
        raise ValueError("dt_s must be <= sim_seconds.")
    error_rate = _require_fraction("injected_error_rate", injected_error_rate)
    interval = _require_int("error_check_interval", error_check_interval, 1)
    bitflip_share = _require_fraction("bitflip_fraction", bitflip_fraction)
    overheat_downtime = _require_int("overheat_downtime_steps", overheat_downtime_steps, 1)
    noise = _require_fraction("noise_probability", noise_probability)
    flux_n = _require_int("flux_scenarios", flux_scenarios, 4)

    steps = int(sim_seconds_i / dt)
    if steps < 120:
        raise ValueError("sim_seconds / dt_s must produce at least 120 steps.")

    episodes_out: list[dict[str, Any]] = []
    for i in range(episodes_i):
        episodes_out.append(
            _run_soc_fault_episode(
                seed=seed_i + i,
                steps=steps,
                injected_error_rate=error_rate,
                error_check_interval=interval,
                bitflip_fraction=bitflip_share,
                noise_probability=noise,
                overheat_downtime_steps=overheat_downtime,
            )
        )

    uptime = np.asarray([float(e["uptime_rate"]) for e in episodes_out], dtype=np.float64)
    obs_err = np.asarray([float(e["observed_error_rate"]) for e in episodes_out], dtype=np.float64)
    q_abs = np.asarray([float(e["q_table_max_abs"]) for e in episodes_out], dtype=np.float64)

    soc_summary = {
        "episodes": int(episodes_i),
        "sim_seconds": int(sim_seconds_i),
        "dt_s": float(dt),
        "steps_per_episode": int(steps),
        "mean_uptime_rate": float(np.mean(uptime)),
        "p05_uptime_rate": float(np.percentile(uptime, 5)),
        "mean_observed_error_rate": float(np.mean(obs_err)),
        "total_injected_faults": int(sum(int(e["injected_faults"]) for e in episodes_out)),
        "total_fault_opportunities": int(sum(int(e["fault_opportunities"]) for e in episodes_out)),
        "total_bitflip_faults": int(sum(int(e["bitflip_faults"]) for e in episodes_out)),
        "total_overheat_faults": int(sum(int(e["overheat_faults"]) for e in episodes_out)),
        "total_overheat_events": int(sum(int(e["overheat_events"]) for e in episodes_out)),
        "max_q_table_abs": float(np.max(q_abs)),
    }

    flux_summary = _run_flux_liveness_campaign(
        seed=seed_i + 10000,
        scenarios=flux_n,
        injected_error_rate=error_rate,
    )

    thresholds = {
        "target_error_rate": 0.20,
        "target_uptime_rate": 0.99,
        "min_p05_uptime_rate": 0.985,
        "min_flux_liveness_rate": 0.99,
        "min_petri_activity_rate": 0.99,
        "max_p95_abs_action_a": 3500.0,
        "error_rate_tolerance": 0.07,
    }

    error_rate_ok = bool(
        abs(soc_summary["mean_observed_error_rate"] - thresholds["target_error_rate"])
        <= thresholds["error_rate_tolerance"]
    )
    passes = bool(
        soc_summary["mean_uptime_rate"] >= thresholds["target_uptime_rate"]
        and soc_summary["p05_uptime_rate"] >= thresholds["min_p05_uptime_rate"]
        and flux_summary["liveness_pass_rate"] >= thresholds["min_flux_liveness_rate"]
        and flux_summary["petri_activity_rate"] >= thresholds["min_petri_activity_rate"]
        and flux_summary["p95_abs_action_a"] <= thresholds["max_p95_abs_action_a"]
        and error_rate_ok
    )

    return {
        "seed": int(seed_i),
        "soc_monte_carlo": soc_summary,
        "flux_liveness": flux_summary,
        "thresholds": thresholds,
        "error_rate_within_tolerance": error_rate_ok,
        "passes_thresholds": passes,
        "runtime_seconds": float(time.perf_counter() - t0),
        "episodes": episodes_out,
    }


def generate_report(**kwargs: Any) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task3_error_handling_audit": run_campaign(**kwargs),
    }


def render_markdown(report: dict[str, Any]) -> str:
    g = report["task3_error_handling_audit"]
    soc = g["soc_monte_carlo"]
    flux = g["flux_liveness"]
    th = g["thresholds"]
    lines = [
        "# Task 3 Error Handling Audit",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Runtime: `{g['runtime_seconds']:.3f} s`",
        f"- Overall pass: `{'YES' if g['passes_thresholds'] else 'NO'}`",
        "",
        "## SOC Monte Carlo (RL Agent)",
        "",
        f"- 1h equivalent runtime per episode: `{soc['sim_seconds']} s`",
        f"- Episodes: `{soc['episodes']}`",
        f"- Injected error target: `{100.0 * th['target_error_rate']:.1f}%`",
        f"- Observed injected error rate: `{100.0 * soc['mean_observed_error_rate']:.2f}%`",
        f"- Mean uptime: `{100.0 * soc['mean_uptime_rate']:.3f}%` (threshold `>= {100.0 * th['target_uptime_rate']:.2f}%`)",
        f"- P05 uptime: `{100.0 * soc['p05_uptime_rate']:.3f}%` (threshold `>= {100.0 * th['min_p05_uptime_rate']:.2f}%`)",
        f"- Fault totals: bit-flips `{soc['total_bitflip_faults']}`, divertor faults `{soc['total_overheat_faults']}`, overheat events `{soc['total_overheat_events']}`",
        "",
        "## 3D Flux + Petri Liveness",
        "",
        f"- Flux liveness pass rate: `{100.0 * flux['liveness_pass_rate']:.2f}%` (threshold `>= {100.0 * th['min_flux_liveness_rate']:.2f}%`)",
        f"- Petri activity rate: `{100.0 * flux['petri_activity_rate']:.2f}%` (threshold `>= {100.0 * th['min_petri_activity_rate']:.2f}%`)",
        f"- P95 |actuator action|: `{flux['p95_abs_action_a']:.2f} A` (threshold `<= {th['max_p95_abs_action_a']:.1f} A`)",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--sim-seconds", type=int, default=3600)
    parser.add_argument("--dt-s", type=float, default=1.0)
    parser.add_argument("--injected-error-rate", type=float, default=0.20)
    parser.add_argument("--error-check-interval", type=int, default=12)
    parser.add_argument("--bitflip-fraction", type=float, default=0.85)
    parser.add_argument("--overheat-downtime-steps", type=int, default=2)
    parser.add_argument("--noise-probability", type=float, default=0.01)
    parser.add_argument("--flux-scenarios", type=int, default=24)
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / "task3_error_handling_audit.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "task3_error_handling_audit.md"),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    report = generate_report(
        seed=args.seed,
        episodes=args.episodes,
        sim_seconds=args.sim_seconds,
        dt_s=args.dt_s,
        injected_error_rate=args.injected_error_rate,
        error_check_interval=args.error_check_interval,
        bitflip_fraction=args.bitflip_fraction,
        overheat_downtime_steps=args.overheat_downtime_steps,
        noise_probability=args.noise_probability,
        flux_scenarios=args.flux_scenarios,
    )
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    g = report["task3_error_handling_audit"]
    soc = g["soc_monte_carlo"]
    flux = g["flux_liveness"]
    print("Task 3 error handling audit complete.")
    print(
        "Summary -> "
        f"uptime_mean={soc['mean_uptime_rate']:.5f}, "
        f"error_rate={soc['mean_observed_error_rate']:.5f}, "
        f"flux_liveness={flux['liveness_pass_rate']:.5f}, "
        f"petri_activity={flux['petri_activity_rate']:.5f}, "
        f"passes_thresholds={g['passes_thresholds']}"
    )

    if args.strict and not g["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
