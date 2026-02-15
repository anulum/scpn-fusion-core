# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Traceable Runtime Backend Parity
# ──────────────────────────────────────────────────────────────────────
"""Validate backend parity for traceable runtime rollouts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scpn_fusion.control.jax_traceable_runtime import (
    TraceableRuntimeSpec,
    available_traceable_backends,
    validate_traceable_backend_parity,
)


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Traceable Runtime Backend Parity")
    lines.append("")
    lines.append(f"- timestamp_utc: `{summary['timestamp_utc']}`")
    lines.append(f"- steps: `{summary['steps']}`")
    lines.append(f"- batch: `{summary['batch']}`")
    lines.append(f"- seed: `{summary['seed']}`")
    lines.append(f"- atol: `{summary['atol']}`")
    lines.append(f"- available_backends: `{summary['available_backends']}`")
    lines.append("")
    lines.append("| backend | single_max_abs_err | batch_max_abs_err | single_within_tol | batch_within_tol |")
    lines.append("|---|---:|---:|:---:|:---:|")
    for rec in summary["reports"]:
        lines.append(
            f"| {rec['backend']} | {rec['single_max_abs_err']:.3e} | {rec['batch_max_abs_err']:.3e} | "
            f"{'yes' if rec['single_within_tol'] else 'no'} | {'yes' if rec['batch_within_tol'] else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def run_parity_check(
    *,
    steps: int,
    batch: int,
    seed: int,
    atol: float,
    dt_s: float,
    tau_s: float,
    gain: float,
    command_limit: float,
) -> dict[str, Any]:
    spec = TraceableRuntimeSpec(
        dt_s=float(dt_s),
        tau_s=float(tau_s),
        gain=float(gain),
        command_limit=float(command_limit),
    )
    reports_map = validate_traceable_backend_parity(
        steps=int(steps),
        batch=int(batch),
        seed=int(seed),
        spec=spec,
        atol=float(atol),
    )
    reports = [asdict(v) for _, v in sorted(reports_map.items(), key=lambda kv: kv[0])]
    strict_ok = all(bool(r["single_within_tol"] and r["batch_within_tol"]) for r in reports)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "steps": int(steps),
        "batch": int(batch),
        "seed": int(seed),
        "atol": float(atol),
        "spec": asdict(spec),
        "available_backends": available_traceable_backends(),
        "reports": reports,
        "strict_ok": bool(strict_ok),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Traceable runtime backend parity checker.")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--dt-s", type=float, default=1.0e-3)
    parser.add_argument("--tau-s", type=float, default=5.0e-3)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--command-limit", type=float, default=1.0)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    summary = run_parity_check(
        steps=args.steps,
        batch=args.batch,
        seed=args.seed,
        atol=args.atol,
        dt_s=args.dt_s,
        tau_s=args.tau_s,
        gain=args.gain,
        command_limit=args.command_limit,
    )

    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(_summary_markdown(summary), encoding="utf-8")

    if args.strict and not summary["strict_ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

