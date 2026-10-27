# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — SNN/RL Tearing-Mode Fault Benchmark
"""Benchmark deterministic SNN/RL tearing-mode decisions and fault recovery."""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]

from scpn_fusion.control.disruption_risk_runtime import (
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

FloatArray: TypeAlias = NDArray[np.float64]
REPORT_SCHEMA_VERSION = 2
REPORT_KIND = "snn_rl_tearing_mode_fault_benchmark"
_SimTearingModeFn = Callable[..., tuple[FloatArray, object, object]]
_BuildFeatureVectorFn = Callable[[FloatArray, dict[str, float]], FloatArray]
_ApplyBitFlipFn = Callable[[float, int], float]
_simulate_tearing_mode = cast(_SimTearingModeFn, simulate_tearing_mode)
_build_disruption_feature_vector = cast(_BuildFeatureVectorFn, build_disruption_feature_vector)
_apply_bit_flip_fault = cast(_ApplyBitFlipFn, apply_bit_flip_fault)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(x)))


def build_tearing_mode_fault_controller() -> NeuroSymbolicController:
    """Build the deterministic SCPN controller used by the benchmark."""
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
    ).compile(net, firing_mode="binary")
    artifact = compiled.export_artifact(
        name="snn_rl_tearing_mode_fault_controller",
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
        sc_binary_margin=0.05,
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
    action = controller.step(cast(Mapping[str, float], obs), k)
    control_term = (
        abs(float(action["dI_PF3_A"])) / 5000.0 + abs(float(action["dI_PF_topbot_A"])) / 5000.0
    )
    snn_policy = _sigmoid(-2.5 + 0.65 * control_term + 0.40 * asym + 0.25 * std)
    return float(np.clip(0.88 * rl_risk + 0.12 * snn_policy, 0.0, 1.0))


def _episode_signal(seed: int, window: int) -> FloatArray:
    local_rng = np.random.default_rng(int(seed))
    signal, _label, _ttd = _simulate_tearing_mode(window, rng=local_rng)
    return np.asarray(signal, dtype=float)


def _finite_threshold(name: str, value: float, *, minimum: float, maximum: float) -> float:
    threshold = float(value)
    if not np.isfinite(threshold):
        raise ValueError(f"{name} must be finite.")
    if threshold < minimum or threshold > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return threshold


def run_benchmark(
    *,
    seed: int = 42,
    episodes: int = 64,
    window: int = 128,
    recovery_epsilon: float = 0.03,
    recovery_window_steps: int = 10,
    dt_ms: float = 0.1,
    min_decision_agreement: float = 0.95,
    max_mean_abs_delta: float = 0.08,
    max_stochastic_float_equivalence_error: float = 0.05,
    max_oracle_sc_mean_abs_delta: float = 0.05,
    max_oracle_sc_firing_delta: float = 0.05,
    max_recovery_ms_p95: float = 1.0,
) -> dict[str, Any]:
    """Run the deterministic SNN/RL fault campaign and return its scorecard."""
    episodes = int(episodes)
    if episodes < 1:
        raise ValueError("episodes must be >= 1.")
    window = int(window)
    if window < 16:
        raise ValueError("window must be >= 16.")
    recovery_window_steps = int(recovery_window_steps)
    if recovery_window_steps < 1:
        raise ValueError("recovery_window_steps must be >= 1.")
    recovery_epsilon = float(recovery_epsilon)
    if not np.isfinite(recovery_epsilon) or recovery_epsilon <= 0.0:
        raise ValueError("recovery_epsilon must be finite and > 0.")
    dt_ms = float(dt_ms)
    if not np.isfinite(dt_ms) or dt_ms <= 0.0:
        raise ValueError("dt_ms must be finite and > 0.")

    thresholds = {
        "min_decision_agreement": _finite_threshold(
            "min_decision_agreement", min_decision_agreement, minimum=0.0, maximum=1.0
        ),
        "max_mean_abs_delta": _finite_threshold(
            "max_mean_abs_delta", max_mean_abs_delta, minimum=0.0, maximum=1.0
        ),
        "max_stochastic_float_equivalence_error": _finite_threshold(
            "max_stochastic_float_equivalence_error",
            max_stochastic_float_equivalence_error,
            minimum=0.0,
            maximum=1.0,
        ),
        "max_oracle_sc_mean_abs_delta": _finite_threshold(
            "max_oracle_sc_mean_abs_delta",
            max_oracle_sc_mean_abs_delta,
            minimum=0.0,
            maximum=1.0,
        ),
        "max_oracle_sc_firing_delta": _finite_threshold(
            "max_oracle_sc_firing_delta",
            max_oracle_sc_firing_delta,
            minimum=0.0,
            maximum=1.0,
        ),
        "max_recovery_ms_p95": _finite_threshold(
            "max_recovery_ms_p95", max_recovery_ms_p95, minimum=0.0, maximum=1_000.0
        ),
    }

    rng = np.random.default_rng(int(seed))
    controller = build_tearing_mode_fault_controller()

    agreement_flags: list[bool] = []
    abs_deltas: list[float] = []
    oracle_sc_mark_deltas: list[float] = []
    oracle_sc_firing_deltas: list[float] = []
    recovery_steps: list[int] = []

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

        snn_seq: list[float] = []
        for k in range(window):
            features = _build_disruption_feature_vector(signal[: k + 1], toroidal)
            rl = _rl_baseline_risk(features)
            snn = _snn_risk(controller, features, k, rl)
            snn_seq.append(snn)
            agreement_flags.append((rl >= 0.5) == (snn >= 0.5))
            abs_deltas.append(abs(snn - rl))
            oracle_mark = np.asarray(controller.last_oracle_marking, dtype=float)
            sc_mark = np.asarray(controller.last_sc_marking, dtype=float)
            oracle_sc_mark_deltas.append(float(np.mean(np.abs(sc_mark - oracle_mark))))
            oracle_firing = np.asarray(controller.last_oracle_firing, dtype=float)
            sc_firing = np.asarray(controller.last_sc_firing, dtype=float)
            oracle_sc_firing_deltas.append(float(np.mean(np.abs(sc_firing - oracle_firing))))

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
        residual = float(faulted[inject_idx] - baseline[inject_idx])
        for t in range(inject_idx + 1, window):
            # Propagate a decaying fault residual toward the nominal trajectory.
            residual *= 0.35
            faulted[t] = float(np.clip(baseline[t] + residual, 0.0, 1.0))

        rec = recovery_window_steps + 1
        max_check = min(window, inject_idx + recovery_window_steps + 1)
        for t in range(inject_idx, max_check):
            if abs(faulted[t] - baseline[t]) <= recovery_epsilon:
                rec = t - inject_idx
                break
        recovery_steps.append(rec)

    decision_agreement = float(np.mean(np.asarray(agreement_flags, dtype=float)))
    mean_abs_delta = float(np.mean(np.asarray(abs_deltas, dtype=float)))
    oracle_sc_mean_abs_delta = float(np.mean(np.asarray(oracle_sc_mark_deltas, dtype=float)))
    oracle_sc_firing_mean_abs_delta = float(
        np.mean(np.asarray(oracle_sc_firing_deltas, dtype=float))
    )
    stochastic_float_equivalence_error = float(
        max(oracle_sc_mean_abs_delta, oracle_sc_firing_mean_abs_delta)
    )
    p95_recovery_steps = float(np.percentile(np.asarray(recovery_steps, dtype=float), 95))
    p95_recovery_ms = p95_recovery_steps * float(dt_ms)

    gate_results = {
        "decision_agreement": decision_agreement >= thresholds["min_decision_agreement"],
        "mean_abs_delta": mean_abs_delta <= thresholds["max_mean_abs_delta"],
        "stochastic_float_equivalence_error": stochastic_float_equivalence_error
        <= thresholds["max_stochastic_float_equivalence_error"],
        "oracle_sc_mean_abs_delta": oracle_sc_mean_abs_delta
        <= thresholds["max_oracle_sc_mean_abs_delta"],
        "oracle_sc_firing_mean_abs_delta": oracle_sc_firing_mean_abs_delta
        <= thresholds["max_oracle_sc_firing_delta"],
        "recovery_ms_p95": p95_recovery_ms <= thresholds["max_recovery_ms_p95"],
    }
    passes = all(gate_results.values())

    return {
        "seed": int(seed),
        "episodes": episodes,
        "window": window,
        "dt_ms": float(dt_ms),
        "decision_agreement": decision_agreement,
        "decision_agreement_pct": decision_agreement * 100.0,
        "mean_abs_delta": mean_abs_delta,
        "stochastic_float_equivalence_error": stochastic_float_equivalence_error,
        "stochastic_float_equivalence_error_pct": stochastic_float_equivalence_error * 100.0,
        "oracle_sc_mean_abs_delta": oracle_sc_mean_abs_delta,
        "oracle_sc_firing_mean_abs_delta": oracle_sc_firing_mean_abs_delta,
        "recovery_steps_p95": p95_recovery_steps,
        "recovery_ms_p95": p95_recovery_ms,
        "thresholds": thresholds,
        "gate_results": gate_results,
        "passes_thresholds": passes,
    }


def generate_report(**kwargs: Any) -> dict[str, Any]:
    """Generate a schema-versioned SNN/RL fault-benchmark report."""
    t0 = time.perf_counter()
    bench = run_benchmark(**kwargs)
    elapsed = time.perf_counter() - t0
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_kind": REPORT_KIND,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": elapsed,
        REPORT_KIND: bench,
    }


def validate_report(report: dict[str, Any]) -> dict[str, Any]:
    """Return a validated benchmark payload and refuse stale report schemas."""
    expected = {
        "schema_version",
        "report_kind",
        "generated_at_utc",
        "runtime_seconds",
        REPORT_KIND,
    }
    if set(report) != expected:
        raise ValueError("report keys do not match the current descriptive contract")
    if report["schema_version"] != REPORT_SCHEMA_VERSION:
        raise ValueError(f"unsupported report schema_version: {report['schema_version']!r}")
    if report["report_kind"] != REPORT_KIND:
        raise ValueError(f"unsupported report_kind: {report['report_kind']!r}")
    generated_at = report["generated_at_utc"]
    if not isinstance(generated_at, str):
        raise ValueError("generated_at_utc must be a non-empty string")
    if not generated_at:
        raise ValueError("generated_at_utc must be a non-empty string")
    runtime_seconds = report["runtime_seconds"]
    if not isinstance(runtime_seconds, (int, float)):
        raise ValueError("runtime_seconds must be a finite non-negative number")
    if not np.isfinite(runtime_seconds) or runtime_seconds < 0.0:
        raise ValueError("runtime_seconds must be a finite non-negative number")
    payload = report[REPORT_KIND]
    if not isinstance(payload, dict):
        raise ValueError(f"{REPORT_KIND} payload must be an object")
    return cast(dict[str, Any], payload)


def render_markdown(report: dict[str, Any]) -> str:
    """Render a validated benchmark report into compact Markdown."""
    benchmark = validate_report(report)
    lines = [
        "# SNN/RL Tearing-Mode Fault Benchmark",
        "",
        f"- Report schema: `{report['schema_version']}`",
        f"- Report kind: `{report['report_kind']}`",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Runtime: `{report['runtime_seconds']:.3f} s`",
        f"- Episodes: `{benchmark['episodes']}`",
        f"- Window: `{benchmark['window']}`",
        "",
        "## Metrics",
        "",
        f"- SNN/RL decision agreement: `{benchmark['decision_agreement_pct']:.2f}%`",
        f"- Mean absolute risk delta: `{benchmark['mean_abs_delta']:.6f}`",
        "- Stochastic-vs-float equivalence error: "
        f"`{benchmark['stochastic_float_equivalence_error']:.6f}` "
        f"(`{benchmark['stochastic_float_equivalence_error_pct']:.2f}%`)",
        f"- Mean oracle-vs-SC marking delta: `{benchmark['oracle_sc_mean_abs_delta']:.6f}`",
        f"- Mean oracle-vs-SC firing delta: `{benchmark['oracle_sc_firing_mean_abs_delta']:.6f}`",
        f"- P95 recovery: `{benchmark['recovery_ms_p95']:.3f} ms`",
        f"- Threshold pass: `{'YES' if benchmark['passes_thresholds'] else 'NO'}`",
        "",
        "## Thresholds",
        "",
        f"- Min decision agreement: `{benchmark['thresholds']['min_decision_agreement']}`",
        f"- Max mean abs delta: `{benchmark['thresholds']['max_mean_abs_delta']}`",
        "- Max stochastic-vs-float equivalence error: "
        f"`{benchmark['thresholds']['max_stochastic_float_equivalence_error']}`",
        "- Max oracle-vs-SC marking delta: "
        f"`{benchmark['thresholds']['max_oracle_sc_mean_abs_delta']}`",
        "- Max oracle-vs-SC firing delta: "
        f"`{benchmark['thresholds']['max_oracle_sc_firing_delta']}`",
        f"- Max P95 recovery ms: `{benchmark['thresholds']['max_recovery_ms_p95']}`",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the fault benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--recovery-epsilon", type=float, default=0.03)
    parser.add_argument("--recovery-window-steps", type=int, default=10)
    parser.add_argument("--dt-ms", type=float, default=0.1)
    parser.add_argument("--min-decision-agreement", type=float, default=0.95)
    parser.add_argument("--max-mean-abs-delta", type=float, default=0.08)
    parser.add_argument("--max-stochastic-float-equivalence-error", type=float, default=0.05)
    parser.add_argument("--max-oracle-sc-mean-abs-delta", type=float, default=0.05)
    parser.add_argument("--max-oracle-sc-firing-delta", type=float, default=0.05)
    parser.add_argument("--max-recovery-ms-p95", type=float, default=1.0)
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / f"{REPORT_KIND}.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / f"{REPORT_KIND}.md"),
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the fault benchmark CLI and export schema-versioned reports."""
    args = parse_args(argv)

    report = generate_report(
        seed=args.seed,
        episodes=args.episodes,
        window=args.window,
        recovery_epsilon=args.recovery_epsilon,
        recovery_window_steps=args.recovery_window_steps,
        dt_ms=args.dt_ms,
        min_decision_agreement=args.min_decision_agreement,
        max_mean_abs_delta=args.max_mean_abs_delta,
        max_stochastic_float_equivalence_error=args.max_stochastic_float_equivalence_error,
        max_oracle_sc_mean_abs_delta=args.max_oracle_sc_mean_abs_delta,
        max_oracle_sc_firing_delta=args.max_oracle_sc_firing_delta,
        max_recovery_ms_p95=args.max_recovery_ms_p95,
    )
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(f"{json.dumps(report, indent=2)}\n", encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    benchmark = validate_report(report)
    print("SNN/RL tearing-mode fault benchmark complete.")
    print(
        "decision_agreement_pct="
        f"{benchmark['decision_agreement_pct']:.2f}, "
        f"mean_abs_delta={benchmark['mean_abs_delta']:.6f}, "
        "stochastic_float_equivalence_error="
        f"{benchmark['stochastic_float_equivalence_error']:.6f}, "
        f"recovery_ms_p95={benchmark['recovery_ms_p95']:.3f}, "
        f"passes_thresholds={benchmark['passes_thresholds']}"
    )
    if args.strict and not benchmark["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
