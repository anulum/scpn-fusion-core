#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Disruption Replay Pipeline Benchmark
# ──────────────────────────────────────────────────────────────────────
"""Contract benchmark for sensor-preprocess + actuator-lag replay defaults."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "validation"))

from validate_real_shots import validate_disruption


def run_benchmark(*, disruption_dir: Path) -> dict[str, Any]:
    default_a = validate_disruption(disruption_dir)
    default_b = validate_disruption(disruption_dir)
    disabled = validate_disruption(
        disruption_dir,
        replay_pipeline={
            "sensor_preprocess_enabled": False,
            "actuator_lag_enabled": False,
        },
    )

    p_default = default_a.get("pipeline", {})
    p_disabled = disabled.get("pipeline", {})
    default_det = bool(
        default_a.get("recall") == default_b.get("recall")
        and default_a.get("false_positive_rate") == default_b.get("false_positive_rate")
        and default_a.get("true_positives") == default_b.get("true_positives")
        and default_a.get("false_positives") == default_b.get("false_positives")
    )
    enabled_flags_ok = bool(
        p_default.get("sensor_preprocess_enabled", False)
        and p_default.get("actuator_lag_enabled", False)
    )
    disabled_flags_ok = bool(
        not p_disabled.get("sensor_preprocess_enabled", True)
        and not p_disabled.get("actuator_lag_enabled", True)
    )
    disabled_invariants_ok = bool(
        float(p_disabled.get("mean_abs_sensor_delta", -1.0)) == 0.0
        and float(p_disabled.get("mean_abs_actuator_lag", -1.0)) == 0.0
    )
    enabled_metrics_ok = bool(
        float(p_default.get("mean_abs_sensor_delta", -1.0)) >= 0.0
        and float(p_default.get("mean_abs_actuator_lag", -1.0)) >= 0.0
    )
    passes = bool(
        default_det
        and enabled_flags_ok
        and disabled_flags_ok
        and enabled_metrics_ok
        and disabled_invariants_ok
        and int(default_a.get("n_shots", 0)) > 0
    )

    return {
        "disruption_replay_pipeline_benchmark": {
            "disruption_dir": str(disruption_dir),
            "default_deterministic_pass": default_det,
            "enabled_flags_pass": enabled_flags_ok,
            "disabled_flags_pass": disabled_flags_ok,
            "enabled_metrics_pass": enabled_metrics_ok,
            "disabled_invariants_pass": disabled_invariants_ok,
            "passes_thresholds": passes,
            "default": {
                "recall": default_a.get("recall"),
                "false_positive_rate": default_a.get("false_positive_rate"),
                "pipeline": p_default,
            },
            "disabled": {
                "recall": disabled.get("recall"),
                "false_positive_rate": disabled.get("false_positive_rate"),
                "pipeline": p_disabled,
            },
        }
    }


def render_markdown(report: dict[str, Any]) -> str:
    g = report["disruption_replay_pipeline_benchmark"]
    d0 = g["default"]
    d1 = g["disabled"]
    p0 = d0["pipeline"]
    p1 = d1["pipeline"]
    lines = [
        "# Disruption Replay Pipeline Benchmark",
        "",
        f"- Disruption directory: `{g['disruption_dir']}`",
        f"- Deterministic replay pass: `{'YES' if g['default_deterministic_pass'] else 'NO'}`",
        f"- Enabled flags pass: `{'YES' if g['enabled_flags_pass'] else 'NO'}`",
        f"- Disabled flags pass: `{'YES' if g['disabled_flags_pass'] else 'NO'}`",
        f"- Disabled invariants pass: `{'YES' if g['disabled_invariants_pass'] else 'NO'}`",
        f"- Overall pass: `{'YES' if g['passes_thresholds'] else 'NO'}`",
        "",
        "| Lane | Recall | FPR | Sensor preprocess | Actuator lag | Mean |sensor delta| | Mean |actuator lag| |",
        "|------|--------|-----|-------------------|--------------|----------------------|--------------------|",
        (
            f"| default | {float(d0['recall']):.2f} | {float(d0['false_positive_rate']):.2f} | "
            f"{'ON' if p0.get('sensor_preprocess_enabled', False) else 'OFF'} | "
            f"{'ON' if p0.get('actuator_lag_enabled', False) else 'OFF'} | "
            f"{float(p0.get('mean_abs_sensor_delta', 0.0)):.6f} | "
            f"{float(p0.get('mean_abs_actuator_lag', 0.0)):.6f} |"
        ),
        (
            f"| disabled | {float(d1['recall']):.2f} | {float(d1['false_positive_rate']):.2f} | "
            f"{'ON' if p1.get('sensor_preprocess_enabled', False) else 'OFF'} | "
            f"{'ON' if p1.get('actuator_lag_enabled', False) else 'OFF'} | "
            f"{float(p1.get('mean_abs_sensor_delta', 0.0)):.6f} | "
            f"{float(p1.get('mean_abs_actuator_lag', 0.0)):.6f} |"
        ),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--disruption-dir",
        default=str(ROOT / "validation" / "reference_data" / "diiid" / "disruption_shots"),
    )
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / "disruption_replay_pipeline_benchmark.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "disruption_replay_pipeline_benchmark.md"),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    report = run_benchmark(disruption_dir=Path(args.disruption_dir))
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    g = report["disruption_replay_pipeline_benchmark"]
    print("Disruption replay pipeline benchmark complete.")
    print(
        "default_det={det}, enabled_flags={ef}, disabled_flags={df}, pass={p}".format(
            det=g["default_deterministic_pass"],
            ef=g["enabled_flags_pass"],
            df=g["disabled_flags_pass"],
            p=g["passes_thresholds"],
        )
    )
    if args.strict and not g["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
