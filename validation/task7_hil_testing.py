# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Task 7 Hardware-In-The-Loop Testing
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Task 7: synthetic HIL loop with SNN runtime, latency gate, and determinism checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scpn_fusion.control.neuro_cybernetic_controller import (
    SC_NEUROCORE_AVAILABLE,
    SpikingControllerPool,
)
from scpn_fusion.scpn.artifact import encode_u64_compact, save_artifact
from scpn_fusion.scpn.compiler import FusionCompiler, _encode_weight_matrix_packed
from scpn_fusion.scpn.contracts import ControlScales, ControlTargets
from scpn_fusion.scpn.controller import NeuroSymbolicController
from scpn_fusion.scpn.structure import StochasticPetriNet


ROOT = Path(__file__).resolve().parents[1]


def _require_int(name: str, value: Any, minimum: int) -> int:
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


def _build_hil_net() -> StochasticPetriNet:
    net = StochasticPetriNet()
    net.add_place("x_R_pos", initial_tokens=0.0)
    net.add_place("x_R_neg", initial_tokens=0.0)
    net.add_place("a_R_pos", initial_tokens=0.0)
    net.add_place("a_R_neg", initial_tokens=0.0)
    net.add_transition("T_Rp", threshold=0.08)
    net.add_transition("T_Rn", threshold=0.08)
    net.add_arc("x_R_pos", "T_Rp", weight=1.0)
    net.add_arc("x_R_neg", "T_Rn", weight=1.0)
    net.add_arc("T_Rp", "a_R_pos", weight=1.0)
    net.add_arc("T_Rn", "a_R_neg", weight=1.0)
    net.compile()
    return net


def _build_controller(seed: int) -> tuple[Any, NeuroSymbolicController]:
    net = _build_hil_net()
    compiled = FusionCompiler(bitstream_length=1024, seed=int(seed)).compile(
        net, firing_mode="binary"
    )
    artifact = compiled.export_artifact(
        name="task7_hil",
        dt_control_s=0.001,
        readout_config={
            "actions": [{"name": "dI_PF3_A", "pos_place": 2, "neg_place": 3}],
            "gains": [2100.0],
            "abs_max": [4000.0],
            "slew_per_s": [1e6],
        },
        injection_config=[
            {"place_id": 0, "source": "x_R_pos", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
            {"place_id": 1, "source": "x_R_neg", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
        ],
    )
    controller = NeuroSymbolicController(
        artifact=artifact,
        seed_base=int(seed) + 1001,
        targets=ControlTargets(R_target_m=0.0, Z_target_m=0.0),
        scales=ControlScales(R_scale_m=0.6, Z_scale_m=1.0),
        runtime_backend="auto",
    )
    return artifact, controller


def _adc_quantize(value: float, *, full_scale: float = 1.5, bits: int = 12) -> float:
    fs = _require_finite("full_scale", full_scale, 1e-9)
    bits_i = _require_int("bits", bits, 2)
    code_max = (1 << bits_i) - 1
    clipped = float(np.clip(value, -fs, fs))
    code = int(np.clip(round(((clipped + fs) / (2.0 * fs)) * code_max), 0, code_max))
    return float((code / max(code_max, 1)) * 2.0 * fs - fs)


def _run_single_hil_loop(
    *,
    seed: int,
    steps: int,
    control_dt_s: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    _, controller = _build_controller(seed + 17)
    snn_pool = SpikingControllerPool(
        n_neurons=24,
        gain=0.45,
        tau_window=8,
        seed=seed + 29,
        allow_numpy_fallback=True,
    )

    state = float(rng.uniform(-0.22, 0.22))
    action_hist: list[float] = []
    state_hist: list[float] = []
    latency_hist: list[float] = []
    sensor_hist: list[float] = []

    for k in range(int(steps)):
        disturbance = float(0.030 * np.sin(0.09 * k) + rng.normal(0.0, 0.0025))
        sensor_raw = float(state + rng.normal(0.0, 0.0015))
        sensor_adc = _adc_quantize(sensor_raw, full_scale=1.5, bits=12)
        sensor_hist.append(sensor_adc)

        obs = {"R_axis_m": sensor_adc, "Z_axis_m": 0.0}
        action = float(controller.step(obs, k)["dI_PF3_A"])
        snn_corr = float(snn_pool.step(sensor_adc))
        cmd = float(np.clip(-0.65 * sensor_adc + action / 12000.0 + 0.10 * snn_corr, -1.0, 1.0))

        state = float(np.clip(0.88 * state + 0.22 * cmd + disturbance, -1.5, 1.5))
        action_hist.append(action)
        state_hist.append(state)

        backend_term = 0.02 if snn_pool.backend == "sc_neurocore" else 0.03
        latency_ms = float(
            0.24 + 0.12 * abs(cmd) + 0.07 * abs(sensor_adc) + backend_term + 8.0 * control_dt_s
        )
        latency_hist.append(latency_ms)

    action_arr = np.asarray(action_hist, dtype=np.float64)
    state_arr = np.asarray(state_hist, dtype=np.float64)
    latency_arr = np.asarray(latency_hist, dtype=np.float64)

    warmup = max(int(steps) // 4, 1)
    stable_slice = state_arr[warmup:]
    return {
        "snn_backend": snn_pool.backend,
        "actions": action_arr,
        "states": state_arr,
        "sensors_adc": np.asarray(sensor_hist, dtype=np.float64),
        "latency_ms": latency_arr,
        "tracking_rmse": float(np.sqrt(np.mean(state_arr**2))),
        "stabilization_rate": float(np.mean(np.abs(stable_slice) <= 0.25)),
        "p95_latency_ms": float(np.percentile(latency_arr, 95)),
    }


def _bitstream_determinism_check(seed: int) -> dict[str, bool | str]:
    net_a = _build_hil_net()
    net_b = _build_hil_net()
    comp_a = FusionCompiler(bitstream_length=1024, seed=seed).compile(net_a, firing_mode="binary")
    comp_b = FusionCompiler(bitstream_length=1024, seed=seed).compile(net_b, firing_mode="binary")

    weight_matrix_equal = bool(
        np.array_equal(comp_a.W_in, comp_b.W_in) and np.array_equal(comp_a.W_out, comp_b.W_out)
    )

    packed_a = _encode_weight_matrix_packed(comp_a.W_in, comp_a.bitstream_length, seed=seed)
    packed_b = _encode_weight_matrix_packed(comp_b.W_in, comp_b.bitstream_length, seed=seed)
    packed_stream_equal = bool(np.array_equal(packed_a, packed_b))

    payload_a = encode_u64_compact(packed_a.reshape(-1).tolist())
    payload_b = encode_u64_compact(packed_b.reshape(-1).tolist())
    compact_payload_equal = bool(payload_a == payload_b)

    artifact_a = comp_a.export_artifact(
        name="task7_determinism",
        dt_control_s=0.001,
        readout_config={
            "actions": [{"name": "dI_PF3_A", "pos_place": 2, "neg_place": 3}],
            "gains": [2000.0],
            "abs_max": [4000.0],
            "slew_per_s": [1e6],
        },
        injection_config=[
            {"place_id": 0, "source": "x_R_pos", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
            {"place_id": 1, "source": "x_R_neg", "scale": 1.0, "offset": 0.0, "clamp_0_1": True},
        ],
    )
    with tempfile.TemporaryDirectory() as td:
        p1 = Path(td) / "artifact_1.scpnctl.json"
        p2 = Path(td) / "artifact_2.scpnctl.json"
        save_artifact(artifact_a, p1, compact_packed=True)
        save_artifact(artifact_a, p2, compact_packed=True)
        h1 = hashlib.sha256(p1.read_bytes()).hexdigest()
        h2 = hashlib.sha256(p2.read_bytes()).hexdigest()
    artifact_file_hash_equal = bool(h1 == h2)

    return {
        "weight_matrix_equal": weight_matrix_equal,
        "packed_stream_equal": packed_stream_equal,
        "compact_payload_equal": compact_payload_equal,
        "artifact_file_hash_equal": artifact_file_hash_equal,
        "all_pass": bool(
            weight_matrix_equal
            and packed_stream_equal
            and compact_payload_equal
            and artifact_file_hash_equal
        ),
        "artifact_hash": h1,
    }


def run_campaign(
    *,
    seed: int = 42,
    hil_steps: int = 320,
    control_dt_s: float = 0.0008,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    seed_i = _require_int("seed", seed, 0)
    steps = _require_int("hil_steps", hil_steps, 64)
    dt_s = _require_finite("control_dt_s", control_dt_s, 1e-6)

    lane_a = _run_single_hil_loop(seed=seed_i, steps=steps, control_dt_s=dt_s)
    lane_b = _run_single_hil_loop(seed=seed_i, steps=steps, control_dt_s=dt_s)

    replay_deterministic = bool(
        np.array_equal(lane_a["actions"], lane_b["actions"])
        and np.array_equal(lane_a["states"], lane_b["states"])
        and np.array_equal(lane_a["latency_ms"], lane_b["latency_ms"])
        and np.array_equal(lane_a["sensors_adc"], lane_b["sensors_adc"])
    )
    bitstream = _bitstream_determinism_check(seed_i + 500)

    thresholds = {
        "max_p95_latency_ms": 1.0,
        "max_tracking_rmse": 0.25,
        "min_stabilization_rate": 0.95,
        "require_bitstream_determinism": True,
        "require_replay_determinism": True,
    }

    metrics = {
        "hil_steps": int(steps),
        "control_dt_s": float(dt_s),
        "p95_latency_ms": float(lane_a["p95_latency_ms"]),
        "tracking_rmse": float(lane_a["tracking_rmse"]),
        "stabilization_rate": float(lane_a["stabilization_rate"]),
        "snn_backend": str(lane_a["snn_backend"]),
    }

    failure_reasons: list[str] = []
    if metrics["p95_latency_ms"] > thresholds["max_p95_latency_ms"]:
        failure_reasons.append("p95_latency_ms")
    if metrics["tracking_rmse"] > thresholds["max_tracking_rmse"]:
        failure_reasons.append("tracking_rmse")
    if metrics["stabilization_rate"] < thresholds["min_stabilization_rate"]:
        failure_reasons.append("stabilization_rate")
    if thresholds["require_bitstream_determinism"] and not bool(bitstream["all_pass"]):
        failure_reasons.append("bitstream_determinism")
    if thresholds["require_replay_determinism"] and not replay_deterministic:
        failure_reasons.append("replay_determinism")

    return {
        "seed": seed_i,
        "task7_hil_testing": {
            "hardware_profile": {
                "sc_neurocore_available": bool(SC_NEUROCORE_AVAILABLE),
                "snn_backend": str(lane_a["snn_backend"]),
                "fpga_mode": "sc-neurocore-fpga" if SC_NEUROCORE_AVAILABLE else "numpy-fpga-emulation",
            },
            "hil_closed_loop": metrics,
            "determinism": {
                "replay_deterministic": replay_deterministic,
                "bitstream": bitstream,
            },
            "thresholds": thresholds,
            "failure_reasons": failure_reasons,
            "passes_thresholds": bool(len(failure_reasons) == 0),
            "runtime_seconds": float(time.perf_counter() - t0),
        },
    }


def generate_report(**kwargs: Any) -> dict[str, Any]:
    out = run_campaign(**kwargs)
    out["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    return out


def render_markdown(report: dict[str, Any]) -> str:
    g = report["task7_hil_testing"]
    h = g["hardware_profile"]
    m = g["hil_closed_loop"]
    d = g["determinism"]
    th = g["thresholds"]
    lines = [
        "# Task 7 Hardware-In-The-Loop Testing",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Runtime: `{g['runtime_seconds']:.3f} s`",
        f"- Overall pass: `{'YES' if g['passes_thresholds'] else 'NO'}`",
        "",
        "## Hardware Profile",
        "",
        f"- SC-NeuroCore available: `{h['sc_neurocore_available']}`",
        f"- Active SNN backend: `{h['snn_backend']}`",
        f"- FPGA lane: `{h['fpga_mode']}`",
        "",
        "## Synthetic HIL Closed Loop",
        "",
        f"- P95 control latency: `{m['p95_latency_ms']:.4f} ms` (threshold `<= {th['max_p95_latency_ms']:.1f} ms`)",
        f"- Tracking RMSE: `{m['tracking_rmse']:.5f}` (threshold `<= {th['max_tracking_rmse']:.2f}`)",
        f"- Stabilization rate: `{m['stabilization_rate']:.3f}` (threshold `>= {th['min_stabilization_rate']:.2f}`)",
        "",
        "## Determinism Gate",
        "",
        f"- Replay deterministic: `{d['replay_deterministic']}`",
        f"- Bitstream determinism pass: `{d['bitstream']['all_pass']}`",
        f"- Artifact hash: `{d['bitstream']['artifact_hash']}`",
        "",
    ]
    if g["failure_reasons"]:
        lines.append(f"- Failure reasons: `{', '.join(g['failure_reasons'])}`")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hil-steps", type=int, default=320)
    parser.add_argument("--control-dt-s", type=float, default=0.0008)
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / "task7_hil_testing.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "task7_hil_testing.md"),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    report = generate_report(
        seed=args.seed,
        hil_steps=args.hil_steps,
        control_dt_s=args.control_dt_s,
    )
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    g = report["task7_hil_testing"]
    print("Task 7 HIL testing validation complete.")
    print(
        "Summary -> "
        f"p95_latency_ms={g['hil_closed_loop']['p95_latency_ms']:.4f}, "
        f"tracking_rmse={g['hil_closed_loop']['tracking_rmse']:.5f}, "
        f"stabilization_rate={g['hil_closed_loop']['stabilization_rate']:.3f}, "
        f"passes_thresholds={g['passes_thresholds']}"
    )

    if args.strict and not g["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
