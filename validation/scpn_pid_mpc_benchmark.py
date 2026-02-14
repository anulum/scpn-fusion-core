# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — SCPN vs PID/MPC Benchmark
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Deterministic control benchmark comparing SCPN controller against PID/MPC."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]

from scpn_fusion.scpn.compiler import FusionCompiler
from scpn_fusion.scpn.contracts import ControlScales, ControlTargets
from scpn_fusion.scpn.controller import NeuroSymbolicController
from scpn_fusion.scpn.structure import StochasticPetriNet


def _build_scpn_controller(
    *,
    runtime_profile: str = "adaptive",
    runtime_backend: str = "auto",
) -> NeuroSymbolicController:
    net = StochasticPetriNet()
    net.add_place("x_R_pos", initial_tokens=0.0)
    net.add_place("x_R_neg", initial_tokens=0.0)
    net.add_place("x_Z_pos", initial_tokens=0.0)
    net.add_place("x_Z_neg", initial_tokens=0.0)
    net.add_place("a_R_pos", initial_tokens=0.0)
    net.add_place("a_R_neg", initial_tokens=0.0)
    net.add_place("a_Z_pos", initial_tokens=0.0)
    net.add_place("a_Z_neg", initial_tokens=0.0)

    # Use long transition delays so readout follows injected features while
    # still exercising timed-transition state tracking.
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
        name="scpn_pid_mpc_benchmark",
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
    runtime_profile_norm = runtime_profile.strip().lower()
    controller_kwargs: dict[str, Any] = {
        "artifact": artifact,
        "seed_base": 123456,
        "targets": ControlTargets(R_target_m=0.0, Z_target_m=0.0),
        "scales": ControlScales(R_scale_m=1.0, Z_scale_m=1.0),
        "sc_n_passes": 16,
        "sc_bitflip_rate": 0.0,
    }
    if runtime_profile_norm == "traceable":
        controller_kwargs.update(
            FusionCompiler.traceable_runtime_kwargs(runtime_backend=runtime_backend)
        )
    else:
        controller_kwargs.update(
            {
                "runtime_profile": runtime_profile_norm,
                "runtime_backend": runtime_backend,
                "sc_binary_margin": 0.05,
            }
        )
    return NeuroSymbolicController(**controller_kwargs)


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


def _mpc_action(x: float, k: int, steps: int, horizon: int, u_limit: float) -> float:
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


def _metrics(errors: NDArray[np.float64], controls: NDArray[np.float64]) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(np.mean(errors * errors))),
        "iae": float(np.mean(np.abs(errors))),
        "peak_abs_error": float(np.max(np.abs(errors))),
        "mean_abs_control": float(np.mean(np.abs(controls))),
    }


def run_campaign(
    *,
    seed: int = 42,
    steps: int = 320,
    scpn_runtime_profile: str = "traceable",
    scpn_runtime_backend: str = "auto",
) -> dict[str, Any]:
    np.random.seed(int(seed))
    steps = max(int(steps), 32)
    u_limit = 1.0
    dt = 0.05
    x0 = 0.35

    scpn = _build_scpn_controller(
        runtime_profile=scpn_runtime_profile,
        runtime_backend=scpn_runtime_backend,
    )
    pid = _PIDState(kp=1.15, ki=0.24, kd=0.04)

    x_scpn = float(x0)
    x_pid = float(x0)
    x_mpc = float(x0)

    e_scpn = np.zeros(steps, dtype=np.float64)
    e_pid = np.zeros(steps, dtype=np.float64)
    e_mpc = np.zeros(steps, dtype=np.float64)
    u_scpn_hist = np.zeros(steps, dtype=np.float64)
    u_pid_hist = np.zeros(steps, dtype=np.float64)
    u_mpc_hist = np.zeros(steps, dtype=np.float64)

    for k in range(steps):
        d = _disturbance(k, steps)

        err_pid = -x_pid
        u_pid = pid.step(err_pid, dt=dt, u_limit=u_limit)
        x_pid = 0.95 * x_pid + 0.12 * u_pid + d

        u_mpc = _mpc_action(x_mpc, k, steps, horizon=6, u_limit=u_limit)
        x_mpc = 0.95 * x_mpc + 0.12 * u_mpc + d

        if scpn.runtime_profile_name == "traceable":
            action_vec = scpn.step_traceable((float(x_scpn), 0.0), k)
            u_scpn = float(np.clip(action_vec[0], -u_limit, u_limit))
        else:
            action = scpn.step({"R_axis_m": float(x_scpn), "Z_axis_m": 0.0}, k)
            u_scpn = float(np.clip(action["dI_PF3_A"], -u_limit, u_limit))
        x_scpn = 0.95 * x_scpn + 0.12 * u_scpn + d

        e_pid[k] = -x_pid
        e_mpc[k] = -x_mpc
        e_scpn[k] = -x_scpn
        u_pid_hist[k] = u_pid
        u_mpc_hist[k] = u_mpc
        u_scpn_hist[k] = u_scpn

    pid_metrics = _metrics(e_pid, u_pid_hist)
    mpc_metrics = _metrics(e_mpc, u_mpc_hist)
    scpn_metrics = _metrics(e_scpn, u_scpn_hist)

    ratios = {
        "scpn_vs_pid_rmse_ratio": scpn_metrics["rmse"] / max(pid_metrics["rmse"], 1e-12),
        "scpn_vs_mpc_rmse_ratio": scpn_metrics["rmse"] / max(mpc_metrics["rmse"], 1e-12),
    }
    thresholds = {
        "max_scpn_vs_pid_rmse_ratio": 1.10,
        "max_scpn_vs_mpc_rmse_ratio": 1.35,
        "max_scpn_peak_abs_error": 0.65,
        "require_mpc_beats_pid": True,
    }
    passes = bool(
        ratios["scpn_vs_pid_rmse_ratio"] <= thresholds["max_scpn_vs_pid_rmse_ratio"]
        and ratios["scpn_vs_mpc_rmse_ratio"] <= thresholds["max_scpn_vs_mpc_rmse_ratio"]
        and scpn_metrics["peak_abs_error"] <= thresholds["max_scpn_peak_abs_error"]
        and mpc_metrics["rmse"] <= pid_metrics["rmse"]
    )

    return {
        "seed": int(seed),
        "steps": int(steps),
        "runtime_lane": {
            "runtime_profile": scpn.runtime_profile_name,
            "runtime_backend": scpn.runtime_backend_name,
            "uses_traceable_step": bool(scpn.runtime_profile_name == "traceable"),
        },
        "pid": pid_metrics,
        "mpc": mpc_metrics,
        "scpn": scpn_metrics,
        "ratios": ratios,
        "thresholds": thresholds,
        "passes_thresholds": passes,
    }


def generate_report(**kwargs: Any) -> dict[str, Any]:
    t0 = time.perf_counter()
    campaign = run_campaign(**kwargs)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": float(time.perf_counter() - t0),
        "scpn_pid_mpc_benchmark": campaign,
    }


def render_markdown(report: dict[str, Any]) -> str:
    r = report["scpn_pid_mpc_benchmark"]
    lines = [
        "# SCPN vs PID/MPC Benchmark",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Runtime: `{report['runtime_seconds']:.3f} s`",
        f"- Steps: `{r['steps']}`",
        f"- SCPN runtime profile: `{r['runtime_lane']['runtime_profile']}`",
        f"- SCPN runtime backend: `{r['runtime_lane']['runtime_backend']}`",
        f"- Traceable step API: `{'YES' if r['runtime_lane']['uses_traceable_step'] else 'NO'}`",
        "",
        "## RMSE",
        "",
        f"- PID: `{r['pid']['rmse']:.6f}`",
        f"- MPC: `{r['mpc']['rmse']:.6f}`",
        f"- SCPN: `{r['scpn']['rmse']:.6f}`",
        "",
        "## Ratios",
        "",
        f"- SCPN / PID RMSE: `{r['ratios']['scpn_vs_pid_rmse_ratio']:.4f}`",
        f"- SCPN / MPC RMSE: `{r['ratios']['scpn_vs_mpc_rmse_ratio']:.4f}`",
        "",
        "## Threshold Pass",
        "",
        f"- Pass: `{'YES' if r['passes_thresholds'] else 'NO'}`",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument(
        "--scpn-runtime-profile",
        choices=["adaptive", "deterministic", "traceable"],
        default="traceable",
    )
    parser.add_argument(
        "--scpn-runtime-backend",
        choices=["auto", "numpy", "rust"],
        default="auto",
    )
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / "scpn_pid_mpc_benchmark.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "scpn_pid_mpc_benchmark.md"),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    report = generate_report(
        seed=args.seed,
        steps=args.steps,
        scpn_runtime_profile=args.scpn_runtime_profile,
        scpn_runtime_backend=args.scpn_runtime_backend,
    )
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    r = report["scpn_pid_mpc_benchmark"]
    print("SCPN vs PID/MPC benchmark complete.")
    print(
        "rmse(pid/mpc/scpn)="
        f"{r['pid']['rmse']:.6f}/{r['mpc']['rmse']:.6f}/{r['scpn']['rmse']:.6f}, "
        f"passes_thresholds={r['passes_thresholds']}"
    )
    if args.strict and not r["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
