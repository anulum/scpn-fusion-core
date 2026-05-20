# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Vertical Control Replay Benchmark Tests
"""Tests for the vertical-control replay benchmark scaffold."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from jsonschema import Draft202012Validator
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "vertical_control_replay_benchmark.py"
SPEC = importlib.util.spec_from_file_location("vertical_control_replay_benchmark", MODULE_PATH)
assert SPEC and SPEC.loader
vertical_control_replay_benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = vertical_control_replay_benchmark
SPEC.loader.exec_module(vertical_control_replay_benchmark)


def test_run_benchmark_is_deterministic_and_reports_limits() -> None:
    first = vertical_control_replay_benchmark.run_benchmark()
    second = vertical_control_replay_benchmark.run_benchmark()

    assert first == second
    bench = first["vertical_control_replay_benchmark"]
    assert bench["schema_version"] == "1.0.0"
    assert bench["deterministic_replay_pass"] is True
    assert bench["passes_thresholds"] is True
    assert bench["actuator_limits"]["max_abs_command"] == pytest.approx(0.45)
    assert bench["actuator_limits"]["max_slew_per_step"] == pytest.approx(0.035)
    assert bench["uncertainty_report"]["n_scenarios"] >= 16
    assert {
        "growth_scale",
        "damping_scale",
        "actuator_scale",
        "sensor_bias_m",
        "latency_steps",
    } <= set(bench["uncertainty_report"]["grid_axes"])
    assert bench["uncertainty_report"]["p95_abs_z_m_p95"] <= bench["thresholds"]["max_p95_abs_z_m"]
    assert bench["uncertainty_report"]["worst_case"]["controller_id"] in {
        "pid",
        "super_twisting",
        "sliding_mode_vertical",
    }
    assert bench["uncertainty_report"]["max_p95_abs_z_m"] <= bench["thresholds"]["max_p95_abs_z_m"]
    assert set(bench["controllers"]) == {
        "pid",
        "super_twisting",
        "sliding_mode_vertical",
        "no_control",
    }
    assert bench["controller_roles"]["no_control"] == "diagnostic_only"
    assert bench["controller_passes"]["no_control"] is False
    assert bench["machine_profile"]["profile_id"] == "iter_like"
    assert bench["machine_profile"]["provenance"]
    assert bench["plant_contract"]["contract_id"] == "rzip_vertical_state_space_v1"
    assert bench["plant_contract"]["source_module"] == "scpn_fusion.control.rzip_model"
    assert bench["plant_contract"]["deterministic_state_trajectory_pass"] is True
    assert len(bench["plant_contract"]["state_space_checksum_sha256"]) == 64
    assert bench["fairness_report"]["passes_fairness_checks"] is True
    assert bench["trace_integrity"]["state_trace_checksum_sha256"]
    assert bench["trace_integrity"]["command_trace_checksum_sha256"]
    assert bench["trace_integrity"]["disturbance_trace_checksum_sha256"]
    assert bench["provenance"]["source_commit"]
    assert bench["provenance"]["schema"] == "schemas/vertical_control_replay_benchmark.schema.json"


def test_controller_results_include_replay_metrics_and_uncertainty() -> None:
    report = vertical_control_replay_benchmark.run_benchmark()
    bench = report["vertical_control_replay_benchmark"]

    for controller_id, result in bench["controllers"].items():
        assert result["controller_id"] == controller_id
        assert result["n_steps"] == bench["scenario"]["n_steps"]
        assert result["max_abs_command"] <= bench["actuator_limits"]["max_abs_command"] + 1.0e-12
        assert result["max_abs_slew"] <= bench["actuator_limits"]["max_slew_per_step"] + 1.0e-12
        if bench["controller_roles"][controller_id] == "primary":
            assert result["p95_abs_z_m"] <= bench["thresholds"]["max_p95_abs_z_m"]
            assert result["final_abs_z_m"] <= bench["thresholds"]["max_final_abs_z_m"]
        assert result["uncertainty"]["max_abs_z_m"] >= result["uncertainty"]["nominal_abs_z_m"]
        assert result["uncertainty"]["n_cases"] == bench["uncertainty_report"]["n_scenarios"]
        assert result["uncertainty"]["p95_abs_z_m_p95"] <= max(
            bench["thresholds"]["max_p95_abs_z_m"], result["uncertainty"]["max_p95_abs_z_m"]
        )
        assert len(result["trace_checksums"]["state_z_m_sha256"]) == 64
        assert len(result["trace_checksums"]["command_sha256"]) == 64
        assert len(result["trace_checksums"]["disturbance_sha256"]) == 64
        assert len(result["trace_checksums"]["raw_command_sha256"]) == 64
        assert result["actuator_limit_application"]["applied_after_controller_output"] is True
        assert (
            result["actuator_limit_application"]["max_bounded_abs_command"]
            <= bench["actuator_limits"]["max_abs_command"]
        )
        assert (
            result["actuator_limit_application"]["max_bounded_abs_slew"]
            <= bench["actuator_limits"]["max_slew_per_step"]
        )


def test_fairness_report_proves_shared_inputs_and_controller_reset() -> None:
    report = vertical_control_replay_benchmark.run_benchmark()
    bench = report["vertical_control_replay_benchmark"]
    fairness = bench["fairness_report"]

    assert fairness["passes_fairness_checks"] is True
    assert fairness["shared_initial_measurement"] == {
        "z_m": bench["scenario"]["initial_z_m"],
        "dz_dt_m_per_s": bench["scenario"]["initial_dz_dt_m_per_s"],
    }
    assert fairness["controller_state_reset_pass"] is True
    assert fairness["all_controllers_share_disturbance_trace"] is True
    assert len(set(fairness["disturbance_trace_checksums"].values())) == 1
    assert set(fairness["controller_trace_roles"]) == set(bench["controllers"])


def test_rzip_plant_contract_is_deterministic_and_wired_to_replay() -> None:
    report = vertical_control_replay_benchmark.run_benchmark()
    bench = report["vertical_control_replay_benchmark"]
    contract = bench["plant_contract"]

    assert contract["state_variables"] == ["z_m", "dz_dt_m_per_s"]
    assert contract["input_variables"] == ["normalised_vertical_actuator_command"]
    assert contract["open_loop_growth_rate_s_inv"] > 0.0
    assert contract["effective_growth_rate_s_inv"] == pytest.approx(
        bench["scenario"]["vertical_growth_rate_s_inv"]
    )
    assert contract["effective_actuator_gain_m_per_s2"] == pytest.approx(
        bench["scenario"]["actuator_gain_m_per_s2"]
    )
    assert (
        contract["deterministic_state_trajectory_checksum_sha256"]
        == contract["repeat_state_trajectory_checksum_sha256"]
    )


def test_replay_checksums_change_when_scenario_changes() -> None:
    baseline = vertical_control_replay_benchmark.run_benchmark()
    changed = vertical_control_replay_benchmark.run_benchmark(
        scenario=vertical_control_replay_benchmark.ReplayScenario(disturbance_accel_m_per_s2=0.24)
    )

    baseline_integrity = baseline["vertical_control_replay_benchmark"]["trace_integrity"]
    changed_integrity = changed["vertical_control_replay_benchmark"]["trace_integrity"]
    assert (
        baseline_integrity["state_trace_checksum_sha256"]
        != changed_integrity["state_trace_checksum_sha256"]
    )
    assert (
        baseline_integrity["disturbance_trace_checksum_sha256"]
        != changed_integrity["disturbance_trace_checksum_sha256"]
    )


def test_invalid_contract_values_are_rejected() -> None:
    scenario = vertical_control_replay_benchmark.ReplayScenario(n_steps=2)
    with pytest.raises(ValueError, match="n_steps"):
        vertical_control_replay_benchmark.run_benchmark(scenario=scenario)

    limits = vertical_control_replay_benchmark.ActuatorLimits(max_abs_command=0.0)
    with pytest.raises(ValueError, match="max_abs_command"):
        vertical_control_replay_benchmark.run_benchmark(actuator_limits=limits)

    profile = vertical_control_replay_benchmark.MachineProfile(profile_id="bad", provenance="")
    with pytest.raises(ValueError, match="provenance"):
        vertical_control_replay_benchmark.run_benchmark(machine_profile=profile)


def test_machine_profiles_are_available_and_change_replay_contract() -> None:
    profiles = vertical_control_replay_benchmark.machine_profiles()
    assert set(profiles) == {"iter_like", "diii_d_like", "compact_tokamak"}
    assert all(profile.provenance for profile in profiles.values())

    iter_report = vertical_control_replay_benchmark.run_benchmark(
        machine_profile=profiles["iter_like"]
    )
    compact_report = vertical_control_replay_benchmark.run_benchmark(
        machine_profile=profiles["compact_tokamak"]
    )

    iter_bench = iter_report["vertical_control_replay_benchmark"]
    compact_bench = compact_report["vertical_control_replay_benchmark"]
    assert iter_bench["machine_profile"]["profile_id"] == "iter_like"
    assert compact_bench["machine_profile"]["profile_id"] == "compact_tokamak"
    assert (
        iter_bench["provenance"]["scenario_checksum_sha256"]
        != compact_bench["provenance"]["scenario_checksum_sha256"]
    )


def test_json_schema_contains_required_contract_keys() -> None:
    schema_path = ROOT / "schemas" / "vertical_control_replay_benchmark.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["title"] == "SCPN Fusion Core Vertical Control Replay Benchmark"
    required = set(schema["properties"]["vertical_control_replay_benchmark"]["required"])
    assert {
        "schema_version",
        "scenario",
        "actuator_limits",
        "machine_profile",
        "plant_contract",
        "controllers",
        "controller_roles",
        "fairness_report",
        "trace_integrity",
        "provenance",
        "uncertainty_report",
        "passes_thresholds",
    } <= required
    assert (
        schema["properties"]["vertical_control_replay_benchmark"]["additionalProperties"] is False
    )
    assert (
        schema["properties"]["vertical_control_replay_benchmark"]["properties"]["scenario"][
            "additionalProperties"
        ]
        is False
    )


def test_report_validates_against_committed_json_schema() -> None:
    schema_path = ROOT / "schemas" / "vertical_control_replay_benchmark.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    report = vertical_control_replay_benchmark.run_benchmark()

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(report)


def test_cli_writes_json_and_markdown_reports(tmp_path: Path) -> None:
    out_json = tmp_path / "vertical_control_replay_benchmark.json"
    out_md = tmp_path / "vertical_control_replay_benchmark.md"
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
    assert payload["vertical_control_replay_benchmark"]["passes_thresholds"] is True
    assert "# Vertical Control Replay Benchmark" in out_md.read_text(encoding="utf-8")
