#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — End-to-End Control Latency Benchmark
# ──────────────────────────────────────────────────────────────────────
"""Deterministic end-to-end latency benchmark for SCPN/PID/MPC-lite."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "src"))

from scpn_fusion.scpn.compiler import FusionCompiler
from scpn_fusion.scpn.contracts import ControlScales, ControlTargets
from scpn_fusion.scpn.controller import NeuroSymbolicController
from scpn_fusion.scpn.structure import StochasticPetriNet


FloatArray = NDArray[np.float64]

_FULL_PHYSICS_MATRIX = np.array(
    [
        [1.20, 0.05, 0.01, 0.00, 0.00, 0.00],
        [0.05, 1.10, 0.03, 0.00, 0.00, 0.00],
        [0.01, 0.03, 1.05, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 1.08, 0.04, 0.01],
        [0.00, 0.00, 0.00, 0.04, 1.07, 0.02],
        [0.00, 0.00, 0.00, 0.01, 0.02, 1.06],
    ],
    dtype=np.float64,
)


def _build_scpn_controller() -> NeuroSymbolicController:
    net = StochasticPetriNet()
    net.add_place("x_R_pos", initial_tokens=0.0)
    net.add_place("x_R_neg", initial_tokens=0.0)
    net.add_place("x_Z_pos", initial_tokens=0.0)
    net.add_place("x_Z_neg", initial_tokens=0.0)
    net.add_place("a_R_pos", initial_tokens=0.0)
    net.add_place("a_R_neg", initial_tokens=0.0)
    net.add_place("a_Z_pos", initial_tokens=0.0)
    net.add_place("a_Z_neg", initial_tokens=0.0)

    net.add_transition("T_Rp", threshold=0.1, delay_ticks=1024)
    net.add_transition("T_Rn", threshold=0.1, delay_ticks=1024)
    net.add_transition("T_Zp", threshold=0.1, delay_ticks=1024)
    net.add_transition("T_Zn", threshold=0.1, delay_ticks=1024)

    net.add_arc("x_R_pos", "T_Rp", weight=1.0)
    net.add_arc("x_R_neg", "T_Rn", weight=1.0)
    net.add_arc("x_Z_pos", "T_Zp", weight=1.0)
    net.add_arc("x_Z_neg", "T_Zn", weight=1.0)
    net.add_arc("T_Rp", "a_R_pos", weight=1.0)
    net.add_arc("T_Rn", "a_R_neg", weight=1.0)
    net.add_arc("T_Zp", "a_Z_pos", weight=1.0)
    net.add_arc("T_Zn", "a_Z_neg", weight=1.0)
    net.compile(validate_topology=True)

    compiler = FusionCompiler(
        bitstream_length=512,
        seed=42,
        lif_tau_mem=10.0,
        lif_noise_std=0.1,
        lif_dt=1.0,
        lif_resistance=1.0,
        lif_refractory_period=1,
    )
    compiled = compiler.compile(net, firing_mode="fractional", firing_margin=0.25)
    artifact = compiled.export_artifact(
        name="scpn_end_to_end_latency",
        dt_control_s=0.05,
        readout_config={
            "actions": [
                {"name": "dI_PF3_A", "pos_place": 0, "neg_place": 1},
                {"name": "dI_PF_topbot_A", "pos_place": 2, "neg_place": 3},
            ],
            "gains": [5.0, 0.0],
            "abs_max": [1.0, 1.0],
            "slew_per_s": [60.0, 60.0],
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
        seed_base=123456,
        targets=ControlTargets(R_target_m=0.0, Z_target_m=0.0),
        scales=ControlScales(R_scale_m=1.0, Z_scale_m=1.0),
        sc_n_passes=16,
        sc_bitflip_rate=0.0,
        sc_binary_margin=0.05,
        runtime_profile="traceable",
        runtime_backend="auto",
    )


@dataclass
class _PIDState:
    kp: float
    ki: float
    kd: float
    integral: float = 0.0
    last_err: float = 0.0

    def step(self, err: float, dt: float, u_limit: float) -> float:
        self.integral += err * dt
        d_err = (err - self.last_err) / max(dt, 1e-9)
        self.last_err = err
        u = self.kp * err + self.ki * self.integral + self.kd * d_err
        return float(np.clip(u, -u_limit, u_limit))


def _disturbance(k: int, steps: int) -> float:
    return float(0.02 * math.sin(0.08 * k) + (0.04 if k >= steps // 2 else 0.0))


def _mpc_lite_action(
    x: float, k: int, steps: int, horizon: int, u_limit: float
) -> float:
    candidates = np.linspace(-u_limit, u_limit, 31)
    best_u = 0.0
    best_cost = float("inf")
    for u in candidates:
        x_pred = x
        cost = 0.0
        for h in range(horizon):
            d = _disturbance(k + h, steps)
            x_pred = 0.95 * x_pred + 0.12 * float(u) + d
            cost += x_pred * x_pred + 0.06 * float(u * u)
        if cost < best_cost:
            best_cost = float(cost)
            best_u = float(u)
    return best_u


def _sensor_preprocess(x: float, last_filtered: float) -> tuple[float, float]:
    filt = 0.84 * last_filtered + 0.16 * x
    return float(np.clip(filt, -1.5, 1.5)), float(filt)


def _actuator_lag(u_cmd: float, u_prev: float, dt: float) -> float:
    tau = 0.04
    alpha = dt / (tau + dt)
    return float(np.clip(u_prev + alpha * (u_cmd - u_prev), -1.0, 1.0))


def _physics_step(x: float, u: float, d: float, mode: str) -> float:
    base = 0.95 * x + 0.12 * u + d
    if mode == "surrogate":
        return float(base)

    rhs = np.array([x, u, d, x * x, u * u, d * d], dtype=np.float64)
    correction = np.linalg.solve(_FULL_PHYSICS_MATRIX, rhs)
    damping = 0.004 * float(np.tanh(correction[0]))
    act_gain = 0.004 * float(np.tanh(correction[1]))
    drift = 0.002 * float(np.tanh(correction[2]))
    return float((0.95 + damping) * x + (0.12 + act_gain) * u + d + drift)


def _pctl(arr: FloatArray, pct: float) -> float:
    return float(np.percentile(arr, pct))


def _controller_runner(
    name: str,
    *,
    scpn: NeuroSymbolicController | None,
) -> Callable[[float, int, int, float], float]:
    if name == "SNN":
        if scpn is None:
            raise ValueError("SNN controller requires SCPN runtime.")

        def _run(x_obs: float, k: int, steps: int, dt: float) -> float:
            del steps, dt
            action_vec = scpn.step_traceable((float(x_obs), 0.0), k)
            return float(np.clip(action_vec[0], -1.0, 1.0))

        return _run

    if name == "PID":
        state = _PIDState(kp=1.15, ki=0.24, kd=0.04)

        def _run(x_obs: float, k: int, steps: int, dt: float) -> float:
            del k, steps
            err = -x_obs
            return state.step(err, dt=dt, u_limit=1.0)

        return _run

    if name == "MPC-lite":

        def _run(x_obs: float, k: int, steps: int, dt: float) -> float:
            del dt
            return _mpc_lite_action(x_obs, k, steps, horizon=6, u_limit=1.0)

        return _run

    raise ValueError(f"Unknown controller: {name}")


def _run_lane(
    *,
    controller_name: str,
    physics_mode: str,
    steps: int,
    dt: float,
    x0: float,
    scpn: NeuroSymbolicController | None,
) -> dict[str, float]:
    if physics_mode not in {"surrogate", "full"}:
        raise ValueError("physics_mode must be 'surrogate' or 'full'.")

    runner = _controller_runner(controller_name, scpn=scpn)
    x = float(x0)
    sensor_state = float(x0)
    u_prev = 0.0

    sensor_ms = np.zeros(steps, dtype=np.float64)
    controller_ms = np.zeros(steps, dtype=np.float64)
    actuator_ms = np.zeros(steps, dtype=np.float64)
    physics_ms = np.zeros(steps, dtype=np.float64)
    loop_ms = np.zeros(steps, dtype=np.float64)
    errors = np.zeros(steps, dtype=np.float64)

    for k in range(steps):
        d = _disturbance(k, steps)
        t_loop0 = time.perf_counter()

        t0 = time.perf_counter()
        x_obs, sensor_state = _sensor_preprocess(x, sensor_state)
        sensor_ms[k] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        u_cmd = runner(x_obs, k, steps, dt)
        controller_ms[k] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        u = _actuator_lag(u_cmd, u_prev, dt)
        actuator_ms[k] = (time.perf_counter() - t0) * 1e3
        u_prev = u

        t0 = time.perf_counter()
        x = _physics_step(x, u, d, physics_mode)
        physics_ms[k] = (time.perf_counter() - t0) * 1e3

        errors[k] = -x
        loop_ms[k] = (time.perf_counter() - t_loop0) * 1e3

    return {
        "rmse": float(np.sqrt(np.mean(errors * errors))),
        "p50_loop_ms": _pctl(loop_ms, 50),
        "p95_loop_ms": _pctl(loop_ms, 95),
        "p99_loop_ms": _pctl(loop_ms, 99),
        "p95_sensor_ms": _pctl(sensor_ms, 95),
        "p95_controller_ms": _pctl(controller_ms, 95),
        "p95_actuator_ms": _pctl(actuator_ms, 95),
        "p95_physics_ms": _pctl(physics_ms, 95),
    }


def run_campaign(*, seed: int = 42, steps: int = 320) -> dict[str, Any]:
    seed = int(seed)
    steps = int(steps)
    if steps < 32:
        raise ValueError("steps must be >= 32.")

    del seed  # Deterministic fixed benchmark path.
    dt = 0.05
    x0 = 0.35
    controllers = ("SNN", "PID", "MPC-lite")
    modes = ("surrogate", "full")

    out: dict[str, dict[str, dict[str, float]]] = {}
    for mode in modes:
        out_mode: dict[str, dict[str, float]] = {}
        for ctrl in controllers:
            scpn = _build_scpn_controller() if ctrl == "SNN" else None
            if scpn is not None:
                scpn.reset()
            out_mode[ctrl] = _run_lane(
                controller_name=ctrl,
                physics_mode=mode,
                steps=steps,
                dt=dt,
                x0=x0,
                scpn=scpn,
            )
        out[mode] = out_mode

    thresholds = {
        "max_snn_p95_loop_ms_surrogate": 8.0,
        "max_snn_p95_loop_ms_full": 12.0,
        "max_snn_full_to_surrogate_ratio": 8.0,
    }
    snn_surrogate = out["surrogate"]["SNN"]["p95_loop_ms"]
    snn_full = out["full"]["SNN"]["p95_loop_ms"]
    ratio = snn_full / max(snn_surrogate, 1e-9)
    passes = bool(
        snn_surrogate <= thresholds["max_snn_p95_loop_ms_surrogate"]
        and snn_full <= thresholds["max_snn_p95_loop_ms_full"]
        and ratio <= thresholds["max_snn_full_to_surrogate_ratio"]
    )

    return {
        "seed": 42,
        "steps": steps,
        "modes": out,
        "ratios": {"snn_full_to_surrogate_p95_ratio": float(ratio)},
        "thresholds": thresholds,
        "passes_thresholds": passes,
    }


def generate_report(**kwargs: Any) -> dict[str, Any]:
    t0 = time.perf_counter()
    campaign = run_campaign(**kwargs)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": float(time.perf_counter() - t0),
        "scpn_end_to_end_latency": campaign,
    }


def render_markdown(report: dict[str, Any]) -> str:
    g = report["scpn_end_to_end_latency"]
    lines = [
        "# SCPN End-to-End Latency Benchmark",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Runtime: `{report['runtime_seconds']:.3f} s`",
        f"- Steps: `{g['steps']}`",
        f"- Overall pass: `{'YES' if g['passes_thresholds'] else 'NO'}`",
        f"- SNN full/surrogate p95 ratio: `{g['ratios']['snn_full_to_surrogate_p95_ratio']:.3f}`",
        "",
    ]
    for mode in ("surrogate", "full"):
        lines.extend(
            [
                f"## {mode.capitalize()} Physics Mode",
                "",
                "| Controller | RMSE | p95 loop [ms] | p95 sensor [ms] | p95 controller [ms] | p95 actuator [ms] | p95 physics [ms] |",
                "|------------|------|---------------|------------------|---------------------|-------------------|------------------|",
            ]
        )
        for ctrl in ("SNN", "PID", "MPC-lite"):
            rec = g["modes"][mode][ctrl]
            lines.append(
                f"| {ctrl} | {rec['rmse']:.6f} | {rec['p95_loop_ms']:.6f} | "
                f"{rec['p95_sensor_ms']:.6f} | {rec['p95_controller_ms']:.6f} | "
                f"{rec['p95_actuator_ms']:.6f} | {rec['p95_physics_ms']:.6f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / "scpn_end_to_end_latency.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "scpn_end_to_end_latency.md"),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    report = generate_report(seed=args.seed, steps=args.steps)
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    g = report["scpn_end_to_end_latency"]
    print("SCPN end-to-end latency benchmark complete.")
    print(
        "snn_p95_surrogate={s:.6f}ms, snn_p95_full={f:.6f}ms, pass={p}".format(
            s=g["modes"]["surrogate"]["SNN"]["p95_loop_ms"],
            f=g["modes"]["full"]["SNN"]["p95_loop_ms"],
            p=g["passes_thresholds"],
        )
    )
    if args.strict and not g["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
