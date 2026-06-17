#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Rust Multigrid Scaling Validation
"""Collect bounded Rust multigrid convergence/scaling evidence."""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUST_WORKSPACE = ROOT / "scpn-fusion-rs"
REPORT_DIR = ROOT / "validation" / "reports"
JSON_REPORT = REPORT_DIR / "rust_multigrid_scaling.json"
MD_REPORT = REPORT_DIR / "rust_multigrid_scaling.md"


def _run_text(command: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable"


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return platform.processor() or "unknown"
    for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _load_average() -> str:
    loadavg = Path("/proc/loadavg")
    if not loadavg.exists():
        return "unavailable"
    return loadavg.read_text(encoding="utf-8").strip()


def _parse_json_lines(stdout: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    header: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if payload.get("schema") == "rust-multigrid-scaling.v1":
            header = payload
        elif "grid" in payload:
            rows.append(payload)
    if header is None:
        raise ValueError("multigrid scaling example did not emit schema header")
    if not rows:
        raise ValueError("multigrid scaling example did not emit grid rows")
    return header, rows


def _row_passes(row: dict[str, Any]) -> bool:
    initial = float(row["initial_residual"])
    final = float(row["final_residual"])
    contraction = float(row["contraction_factor"])
    return (
        bool(row["converged"])
        and initial > 0.0
        and final >= 0.0
        and final < initial
        and 0.0 <= contraction < 1.0
        and int(row["cycles"]) <= 30
        and float(row["wall_time_ms"]) > 0.0
    )


def _write_markdown(report: dict[str, Any]) -> None:
    rows = report["rows"]
    lines = [
        "# Rust Multigrid Scaling Validation",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Status: `{report['status']}`",
        f"- Claim boundary: `{report['claim_boundary']}`",
        f"- Benchmark context: `{report['benchmark_context']}`",
        f"- Command: `{' '.join(report['command'])}`",
        f"- CPU: `{report['environment']['cpu_model']}`",
        f"- Kernel: `{report['environment']['kernel']}`",
        f"- Rust: `{report['environment']['rustc']}`",
        f"- Cargo: `{report['environment']['cargo']}`",
        f"- Load average: `{report['environment']['load_average']}`",
        "",
        "| Grid | Points | Cycles | Converged | Initial residual | Final residual | Contraction | Wall time ms |",
        "|------|--------|--------|-----------|------------------|----------------|-------------|--------------|",
    ]
    for row in rows:
        lines.append(
            "| {grid}x{grid} | {points} | {cycles} | {converged} | {initial_residual:.6e} | "
            "{final_residual:.6e} | {contraction_factor:.6e} | {wall_time_ms:.3f} |".format(
                grid=int(row["grid"]),
                points=int(row["points"]),
                cycles=int(row["cycles"]),
                converged=str(bool(row["converged"])).lower(),
                initial_residual=float(row["initial_residual"]),
                final_residual=float(row["final_residual"]),
                contraction_factor=float(row["contraction_factor"]),
                wall_time_ms=float(row["wall_time_ms"]),
            )
        )
    lines.extend(
        [
            "",
            "This report validates convergence instrumentation and local scaling shape only.",
            "It is not an isolated release-performance claim and must not be cited as a production speedup.",
        ]
    )
    MD_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Run the Rust multigrid scaling example and persist local evidence."""
    command = [
        "cargo",
        "run",
        "--quiet",
        "-p",
        "fusion-math",
        "--example",
        "multigrid_scaling",
    ]
    env = os.environ.copy()
    env.setdefault("CARGO_TARGET_DIR", "/tmp/scpn-fusion-rs-target")
    completed = subprocess.run(
        command,
        cwd=RUST_WORKSPACE,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    header, rows = _parse_json_lines(completed.stdout)
    checks = [{"grid": row["grid"], "passed": _row_passes(row)} for row in rows]
    status = "passed" if all(check["passed"] for check in checks) else "failed"

    report: dict[str, Any] = {
        "schema": "rust-multigrid-scaling-report.v1",
        "generated_at_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        "status": status,
        "claim_boundary": "convergence instrumentation and local non-isolated scaling evidence only",
        "benchmark_context": "single-run local workstation; not CPU-isolated; not a production speedup claim",
        "command": command,
        "rust_example_header": header,
        "environment": {
            "platform": platform.platform(),
            "kernel": platform.release(),
            "cpu_model": _cpu_model(),
            "rustc": _run_text(["rustc", "--version"]),
            "cargo": _run_text(["cargo", "--version"]),
            "load_average": _load_average(),
            "cargo_target_dir": env["CARGO_TARGET_DIR"],
        },
        "checks": checks,
        "rows": rows,
        "stderr": completed.stderr.strip(),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(report)

    print(f"Rust multigrid scaling validation {status}: {MD_REPORT.relative_to(ROOT)}")
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
