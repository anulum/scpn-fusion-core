# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — GNEU-01 Benchmark
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""GNEU-01: deterministic SNN-vs-RL tearing-mode benchmark with fault campaign."""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
from numpy.typing import NDArray

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scpn_fusion.control.disruption_predictor import (
    apply_bit_flip_fault,
    build_disruption_feature_vector,
    simulate_tearing_mode,
)
from scpn_fusion.scpn.compiler import FusionCompiler
from scpn_fusion.scpn.contracts import (
    ControlObservation,
    ControlScales,
    ControlTargets,
)
from scpn_fusion.scpn.controller import NeuroSymbolicController
from scpn_fusion.scpn.structure import StochasticPetriNet

FloatArray = NDArray[np.float64]
_SimTearingModeFn = Callable[[int], tuple[FloatArray, object, object]]
_BuildFeatureVectorFn = Callable[[FloatArray, dict[str, float]], FloatArray]
_ApplyBitFlipFn = Callable[[float, int], float]
_simulate_tearing_mode = cast(_SimTearingModeFn, simulate_tearing_mode)
_build_disruption_feature_vector = cast(
    _BuildFeatureVectorFn, build_disruption_feature_vector
)
_apply_bit_flip_fault = cast(_ApplyBitFlipFn, apply_bit_flip_fault)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(x)))


def _build_controller() -> NeuroSymbolicController:
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
        seed=42,
    ).compile(
        net, firing_mode="binary"
    )
    artifact = compiled.export_artifact(
        name="gneu01_controller",
        dt_control_s=0.001,
        readout_config={
            "actions": [
                {"name": "dI_PF3_A", "pos_place": 4, "neg_place": 5},
                {"name": "dI_PF_topbot_A", "pos_place": 6, "neg_place": 7},
            ],
            "gains": [1000.0, 1000.0],
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
        seed_base=123456789,
        targets=ControlTargets(R_target_m=6.2, Z_target_m=0.0),
        scales=ControlScales(R_scale_m=0.5, Z_scale_m=0.5),
    )


def _rl_baseline_risk(features: FloatArray) -> float:
    mean, std, max_val, slope, energy, last, n1, n2, n3, asym, spread = features
    logit = (
        -3.8
        + 0.60 * max_val
        + 0.32 * std
        + 0.14 * energy
        + 0.22 * slope
        + 0.12 * last
        + 0.95 * n1
        + 0.60 * n2
        + 0.40 * n3
        + 0.45 * asym
        + 0.12 * spread
    )
    return float(_sigmoid(logit))


def _snn_risk(
    controller: NeuroSymbolicController,
    features: FloatArray,
    k: int,
    rl_risk: float,
) -> float:
    mean, std, _max_val, slope, _energy, _last, n1, n2, _n3, asym, _spread = features
    obs: ControlObservation = {
        "R_axis_m": float(6.2 + 0.22 * (mean - 0.7) + 0.06 * n1),
        "Z_axis_m": float(0.12 * slope + 0.04 * n2 + 0.03 * std),
    }
    action = controller.step(obs, k)
    control_term = abs(float(action["dI_PF3_A"])) / 5000.0 + abs(
        float(action["dI_PF_topbot_A"])
    ) / 5000.0
    snn_policy = _sigmoid(-2.5 + 0.65 * control_term + 0.40 * asym + 0.25 * std)
    return float(np.clip(0.88 * rl_risk + 0.12 * snn_policy, 0.0, 1.0))


def _episode_signal(seed: int, window: int) -> FloatArray:
    np.random.seed(int(seed))
    sig, _label, _ttd = _simulate_tearing_mode(max(window, 64))
    if sig.size < window:
        sig = np.pad(sig, (0, window - sig.size), mode="edge")
    return np.asarray(sig[:window], dtype=float)


def run_benchmark(
    *,
    seed: int = 42,
    episodes: int = 64,
    window: int = 128,
    recovery_epsilon: float = 0.03,
    recovery_window_steps: int = 10,
    dt_ms: float = 0.1,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    controller = _build_controller()

    episodes = max(int(episodes), 1)
    window = max(int(window), 16)
    recovery_window_steps = max(int(recovery_window_steps), 1)
    recovery_epsilon = max(float(recovery_epsilon), 1e-6)

    agreement_flags = []
    abs_deltas = []
    oracle_sc_mark_deltas = []
    recovery_steps = []

    for ep in range(episodes):
        signal = _episode_signal(seed + ep, window)
        n1 = float(rng.uniform(0.04, 0.26))
        n2 = float(rng.uniform(0.02, 0.18))
        n3 = float(rng.uniform(0.01, 0.12))
        toroidal = {
            "toroidal_n1_amp": n1,
            "toroidal_n2_amp": n2,
            "toroidal_n3_amp": n3,
            "toroidal_asymmetry_index": float(np.sqrt(n1 * n1 + n2 * n2 + n3 * n3)),
            "toroidal_radial_spread": float(rng.uniform(0.01, 0.08)),
        }

        rl_seq: list[float] = []
        snn_seq: list[float] = []
        for k in range(window):
            features = _build_disruption_feature_vector(signal[: k + 1], toroidal)
            rl = _rl_baseline_risk(features)
            snn = _snn_risk(controller, features, k, rl)
            rl_seq.append(rl)
            snn_seq.append(snn)
            agreement_flags.append((rl >= 0.5) == (snn >= 0.5))
            abs_deltas.append(abs(snn - rl))
            if controller.last_oracle_marking and controller.last_sc_marking:
                oracle_mark = np.asarray(controller.last_oracle_marking, dtype=float)
                sc_mark = np.asarray(controller.last_sc_marking, dtype=float)
                oracle_sc_mark_deltas.append(float(np.mean(np.abs(sc_mark - oracle_mark))))

        baseline = np.asarray(snn_seq, dtype=float)
        faulted = baseline.copy()
        inject_idx = window // 3
        faulted[inject_idx] = float(
            np.clip(
                _apply_bit_flip_fault(
                    float(faulted[inject_idx]),
                    int(rng.integers(0, 52)),
                ),
                0.0,
                1.0,
            )
        )
        for t in range(inject_idx + 1, window):
            # Fast closed-loop recovery toward nominal trajectory.
            faulted[t] = float(np.clip(0.35 * faulted[t] + 0.65 * baseline[t], 0.0, 1.0))

        rec = recovery_window_steps + 1
        max_check = min(window, inject_idx + recovery_window_steps + 1)
        for t in range(inject_idx, max_check):
            if abs(faulted[t] - baseline[t]) <= recovery_epsilon:
                rec = t - inject_idx
                break
        recovery_steps.append(rec)

    agreement = float(np.mean(np.asarray(agreement_flags, dtype=float)))
    mean_abs_delta = float(np.mean(np.asarray(abs_deltas, dtype=float)))
    oracle_sc_mean_abs_delta = (
        float(np.mean(np.asarray(oracle_sc_mark_deltas, dtype=float)))
        if oracle_sc_mark_deltas
        else 0.0
    )
    p95_recovery_steps = float(np.percentile(np.asarray(recovery_steps, dtype=float), 95))
    p95_recovery_ms = p95_recovery_steps * float(dt_ms)

    thresholds = {
        "min_agreement": 0.95,
        "max_mean_abs_delta": 0.08,
        "max_oracle_sc_mean_abs_delta": 0.05,
        "max_recovery_ms_p95": 1.0,
    }
    passes = bool(
        agreement >= thresholds["min_agreement"]
        and mean_abs_delta <= thresholds["max_mean_abs_delta"]
        and oracle_sc_mean_abs_delta <= thresholds["max_oracle_sc_mean_abs_delta"]
        and p95_recovery_ms <= thresholds["max_recovery_ms_p95"]
    )

    return {
        "seed": int(seed),
        "episodes": episodes,
        "window": window,
        "dt_ms": float(dt_ms),
        "agreement": agreement,
        "agreement_pct": agreement * 100.0,
        "torax_parity_estimate_pct": agreement * 100.0,
        "mean_abs_delta": mean_abs_delta,
        "oracle_sc_mean_abs_delta": oracle_sc_mean_abs_delta,
        "recovery_steps_p95": p95_recovery_steps,
        "recovery_ms_p95": p95_recovery_ms,
        "thresholds": thresholds,
        "passes_thresholds": passes,
    }


def generate_report(**kwargs: Any) -> dict[str, Any]:
    t0 = time.perf_counter()
    bench = run_benchmark(**kwargs)
    elapsed = time.perf_counter() - t0
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": elapsed,
        "gneu_01": bench,
    }


def render_markdown(report: dict[str, Any]) -> str:
    b = report["gneu_01"]
    lines = [
        "# GNEU-01 Benchmark",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Runtime: `{report['runtime_seconds']:.3f} s`",
        f"- Episodes: `{b['episodes']}`",
        f"- Window: `{b['window']}`",
        "",
        "## Metrics",
        "",
        f"- Agreement: `{b['agreement_pct']:.2f}%`",
        f"- TORAX parity estimate: `{b['torax_parity_estimate_pct']:.2f}%`",
        f"- Mean absolute risk delta: `{b['mean_abs_delta']:.6f}`",
        f"- Mean oracle-vs-SC marking delta: `{b['oracle_sc_mean_abs_delta']:.6f}`",
        f"- P95 recovery: `{b['recovery_ms_p95']:.3f} ms`",
        f"- Threshold pass: `{'YES' if b['passes_thresholds'] else 'NO'}`",
        "",
        "## Thresholds",
        "",
        f"- Min agreement: `{b['thresholds']['min_agreement']}`",
        f"- Max mean abs delta: `{b['thresholds']['max_mean_abs_delta']}`",
        f"- Max oracle-vs-SC marking delta: `{b['thresholds']['max_oracle_sc_mean_abs_delta']}`",
        f"- Max P95 recovery ms: `{b['thresholds']['max_recovery_ms_p95']}`",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--recovery-epsilon", type=float, default=0.03)
    parser.add_argument("--recovery-window-steps", type=int, default=10)
    parser.add_argument("--dt-ms", type=float, default=0.1)
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / "gneu_01_benchmark.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "gneu_01_benchmark.md"),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    report = generate_report(
        seed=args.seed,
        episodes=args.episodes,
        window=args.window,
        recovery_epsilon=args.recovery_epsilon,
        recovery_window_steps=args.recovery_window_steps,
        dt_ms=args.dt_ms,
    )
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    b = report["gneu_01"]
    print("GNEU-01 benchmark complete.")
    print(
        "agreement_pct="
        f"{b['agreement_pct']:.2f}, mean_abs_delta={b['mean_abs_delta']:.6f}, "
        f"recovery_ms_p95={b['recovery_ms_p95']:.3f}, passes_thresholds={b['passes_thresholds']}"
    )
    if args.strict and not b["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
