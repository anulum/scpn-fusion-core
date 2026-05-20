#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Formal Safety Contract Replay Benchmark
"""Replay benchmark for canonical SCPN safety-interlock contracts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scpn_fusion.scpn.safety_interlocks import (  # noqa: E402
    CONTROL_TRANSITIONS,
    SAFETY_CHANNELS,
    SafetyInterlockRuntime,
    default_safety_contracts,
)

SCHEMA_PATH = "schemas/formal_safety_contract_replay_benchmark.schema.json"
SCHEMA_VERSION = "1.0.0"


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return proc.stdout.strip() or "unknown"


def _contract_manifest() -> list[dict[str, str]]:
    return [asdict(contract) for contract in default_safety_contracts()]


def _provenance() -> dict[str, Any]:
    return {
        "source_commit": _source_commit(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "schema": SCHEMA_PATH,
        "schema_version": SCHEMA_VERSION,
        "contract_manifest_checksum_sha256": _sha256_json(_contract_manifest()),
        "replay_input_checksum_sha256": _sha256_json(_replay_states()),
    }


def _replay_states() -> tuple[dict[str, Any], ...]:
    return (
        {
            "state_id": "nominal",
            "state": {"T_e": 12.0, "n_e": 6.0, "beta_N": 1.8, "I_p": 9.0, "dZ_dt": 0.1},
        },
        {
            "state_id": "thermal_limit",
            "state": {"T_e": 31.0, "n_e": 6.0, "beta_N": 1.8, "I_p": 9.0, "dZ_dt": 0.1},
        },
        {
            "state_id": "density_limit",
            "state": {"T_e": 12.0, "n_e": 13.0, "beta_N": 1.8, "I_p": 9.0, "dZ_dt": 0.1},
        },
        {
            "state_id": "beta_limit",
            "state": {"T_e": 12.0, "n_e": 6.0, "beta_N": 3.2, "I_p": 9.0, "dZ_dt": 0.1},
        },
        {
            "state_id": "current_limit",
            "state": {"T_e": 12.0, "n_e": 6.0, "beta_N": 1.8, "I_p": 16.0, "dZ_dt": 0.1},
        },
        {
            "state_id": "vertical_velocity_limit",
            "state": {"T_e": 12.0, "n_e": 6.0, "beta_N": 1.8, "I_p": 9.0, "dZ_dt": 1.4},
        },
    )


def _run_once() -> list[dict[str, Any]]:
    runtime = SafetyInterlockRuntime()
    rows: list[dict[str, Any]] = []
    for item in _replay_states():
        enabled = runtime.update_from_state(item["state"])
        rows.append(
            {
                "state_id": str(item["state_id"]),
                "tokens": dict(runtime.last_tokens),
                "transition_enabled": enabled,
                "contract_violations": list(runtime.last_contract_violations),
            }
        )
    return rows


def run_benchmark() -> dict[str, Any]:
    """Run deterministic safety-contract replay over canonical state cases."""

    first = _run_once()
    second = _run_once()
    violations = [
        f"{row['state_id']}:{violation}"
        for row in first
        for violation in row["contract_violations"]
    ]
    contracts = default_safety_contracts()
    return {
        "SPDX-License-Identifier": "AGPL-3.0-or-later",
        "formal_safety_contract_replay_benchmark": {
            "schema_version": SCHEMA_VERSION,
            "n_contracts": len(contracts),
            "n_states": len(first),
            "deterministic_replay_pass": first == second,
            "contract_manifest": _contract_manifest(),
            "states": first,
            "trace_integrity": {
                "states_checksum_sha256": _sha256_json(first),
                "transition_enablement_checksum_sha256": _sha256_json(
                    {row["state_id"]: row["transition_enabled"] for row in first}
                ),
                "tokens_checksum_sha256": _sha256_json(
                    {row["state_id"]: row["tokens"] for row in first}
                ),
                "state_count": len(first),
                "safety_channel_count": len(SAFETY_CHANNELS),
                "control_transition_count": len(CONTROL_TRANSITIONS),
            },
            "violations": violations,
            "provenance": _provenance(),
            "passes_thresholds": bool(first == second and not violations),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    bench = report["formal_safety_contract_replay_benchmark"]
    lines = [
        "# Formal Safety Contract Replay Benchmark",
        "",
        f"- Schema version: `{bench['schema_version']}`",
        f"- Contracts: `{bench['n_contracts']}`",
        f"- States: `{bench['n_states']}`",
        f"- Deterministic replay pass: `{'YES' if bench['deterministic_replay_pass'] else 'NO'}`",
        f"- Overall pass: `{'YES' if bench['passes_thresholds'] else 'NO'}`",
        f"- States checksum: `{bench['trace_integrity']['states_checksum_sha256']}`",
        "",
        "| State | Active safety tokens | Disabled transitions | Violations |",
        "|-------|----------------------|----------------------|------------|",
    ]
    for row in bench["states"]:
        active = ",".join(k for k, v in row["tokens"].items() if float(v) > 0.0) or "none"
        disabled = ",".join(k for k, v in row["transition_enabled"].items() if not v) or "none"
        violations = ",".join(row["contract_violations"]) or "none"
        lines.append(f"| {row['state_id']} | {active} | {disabled} | {violations} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        default=str(
            ROOT / "validation" / "reports" / "formal_safety_contract_replay_benchmark.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "formal_safety_contract_replay_benchmark.md"),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    report = run_benchmark()
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    bench = report["formal_safety_contract_replay_benchmark"]
    print("Formal safety contract replay benchmark complete.")
    print(
        "deterministic={det}, states={states}, violations={violations}, pass={passed}".format(
            det=bench["deterministic_replay_pass"],
            states=bench["n_states"],
            violations=len(bench["violations"]),
            passed=bench["passes_thresholds"],
        )
    )
    if args.strict and not bench["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
