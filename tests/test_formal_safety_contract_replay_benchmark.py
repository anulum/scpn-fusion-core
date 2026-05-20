# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Formal Safety Contract Replay Benchmark Tests
"""Tests for the formal safety-contract replay benchmark."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "formal_safety_contract_replay_benchmark.py"
SPEC = importlib.util.spec_from_file_location(
    "formal_safety_contract_replay_benchmark", MODULE_PATH
)
assert SPEC and SPEC.loader
formal_safety_contract_replay_benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = formal_safety_contract_replay_benchmark
SPEC.loader.exec_module(formal_safety_contract_replay_benchmark)


def test_formal_safety_replay_is_deterministic_and_complete() -> None:
    first = formal_safety_contract_replay_benchmark.run_benchmark()
    second = formal_safety_contract_replay_benchmark.run_benchmark()

    assert first == second
    bench = first["formal_safety_contract_replay_benchmark"]
    assert bench["schema_version"] == "1.0.0"
    assert bench["deterministic_replay_pass"] is True
    assert bench["passes_thresholds"] is True
    assert bench["n_contracts"] >= 5
    assert bench["n_states"] == 6
    assert bench["violations"] == []
    assert bench["trace_integrity"]["states_checksum_sha256"]
    assert bench["trace_integrity"]["transition_enablement_checksum_sha256"]
    assert bench["provenance"]["source_commit"]
    assert bench["provenance"]["schema"] == (
        "schemas/formal_safety_contract_replay_benchmark.schema.json"
    )


def test_vertical_limit_blocks_position_move_only() -> None:
    report = formal_safety_contract_replay_benchmark.run_benchmark()
    states = report["formal_safety_contract_replay_benchmark"]["states"]
    vertical = next(row for row in states if row["state_id"] == "vertical_velocity_limit")

    assert vertical["tokens"]["vertical_limit"] == 1.0
    assert vertical["transition_enabled"]["position_move"] is False
    assert vertical["transition_enabled"]["heat_ramp"] is True
    assert vertical["transition_enabled"]["density_ramp"] is True


def test_each_canonical_limit_blocks_only_its_contract_transition() -> None:
    report = formal_safety_contract_replay_benchmark.run_benchmark()
    states = {
        row["state_id"]: row for row in report["formal_safety_contract_replay_benchmark"]["states"]
    }
    expected = {
        "thermal_limit": ("thermal_limit", "heat_ramp"),
        "density_limit": ("density_limit", "density_ramp"),
        "beta_limit": ("beta_limit", "power_ramp"),
        "current_limit": ("current_limit", "current_ramp"),
        "vertical_velocity_limit": ("vertical_limit", "position_move"),
    }

    for state_id, (token, blocked_transition) in expected.items():
        row = states[state_id]
        assert row["tokens"][token] == 1.0
        assert row["transition_enabled"][blocked_transition] is False
        enabled_elsewhere = {
            name: enabled
            for name, enabled in row["transition_enabled"].items()
            if name != blocked_transition
        }
        assert all(enabled_elsewhere.values())


def test_json_schema_contains_required_contract_keys() -> None:
    schema_path = ROOT / "schemas" / "formal_safety_contract_replay_benchmark.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["title"] == "SCPN Fusion Core Formal Safety Contract Replay Benchmark"
    required = set(schema["properties"]["formal_safety_contract_replay_benchmark"]["required"])
    assert {
        "schema_version",
        "n_contracts",
        "n_states",
        "deterministic_replay_pass",
        "states",
        "trace_integrity",
        "provenance",
        "violations",
        "passes_thresholds",
    } <= required
    bench_schema = schema["properties"]["formal_safety_contract_replay_benchmark"]
    assert bench_schema["additionalProperties"] is False
    assert bench_schema["properties"]["states"]["items"]["$ref"] == "#/$defs/state_row"
    assert schema["$defs"]["state_row"]["additionalProperties"] is False


def test_report_validates_against_committed_json_schema() -> None:
    schema_path = ROOT / "schemas" / "formal_safety_contract_replay_benchmark.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    report = formal_safety_contract_replay_benchmark.run_benchmark()

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(report)


def test_cli_writes_reports_and_strict_passes(tmp_path: Path) -> None:
    out_json = tmp_path / "formal_safety_contract_replay_benchmark.json"
    out_md = tmp_path / "formal_safety_contract_replay_benchmark.md"
    proc = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--strict",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["formal_safety_contract_replay_benchmark"]["passes_thresholds"] is True
    assert "# Formal Safety Contract Replay Benchmark" in out_md.read_text(encoding="utf-8")
