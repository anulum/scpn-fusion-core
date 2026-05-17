# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Stellarator Control Replay Benchmark Tests

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import scpn_fusion.cli as cli_mod
from scpn_fusion.control.stellarator_control_contracts import (
    ActuatorChannel,
    ActuatorSet,
    ControlObjective,
    DiagnosticChannel,
    DiagnosticFrame,
    MagneticConfiguration,
    ReplayScenario,
)
from validation.stellarator_control_replay_benchmark import (
    DEFAULT_THRESHOLDS,
    generate_report,
    load_benchmark_config,
    load_config_schema,
    load_report_schema,
    render_markdown,
    validate_benchmark_config,
    validate_report_against_schema,
)

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_CONFIG = ROOT / "validation" / "reference_data" / "stellarator_control_replay_public_config.json"


def test_geometry_neutral_contracts_reject_invalid_actuator_limits() -> None:
    with pytest.raises(ValueError, match="max_value must be greater"):
        ActuatorChannel(
            name="trim_coil_1",
            unit="A",
            min_value=10.0,
            max_value=5.0,
            slew_rate_per_s=100.0,
            latency_steps=1,
        )


def test_replay_scenario_is_geometry_neutral_and_validates_channel_names() -> None:
    config = MagneticConfiguration(
        name="public_w7x_like",
        device_class="stellarator",
        field_periods=5,
        coordinate_system="boozer_vmec_like",
        reference="public synthetic W7-X-like reduced-order fixture",
    )
    actuators = ActuatorSet(
        channels=(
            ActuatorChannel(
                name="helical_trim_A",
                unit="A",
                min_value=-1200.0,
                max_value=1200.0,
                slew_rate_per_s=4.0e5,
                latency_steps=2,
            ),
        )
    )
    objective = ControlObjective(
        target_metrics={"fieldline_spread": 0.015},
        weights={"fieldline_spread": 1.0},
        constraints={"max_abs_current_A": 1200.0},
    )
    frame = DiagnosticFrame(
        step=0,
        time_s=0.0,
        channels=(
            DiagnosticChannel(
                name="fieldline_spread",
                value=0.04,
                unit="rad",
                sigma=0.002,
                provenance="public_synthetic",
            ),
        ),
    )

    scenario = ReplayScenario(
        name="w7x_like_replay",
        seed=1234,
        steps=6,
        dt_s=0.001,
        magnetic_configuration=config,
        actuator_set=actuators,
        objective=objective,
        initial_frame=frame,
        fault_schedule={3: {"helical_trim_A": "stuck"}},
    )

    assert scenario.magnetic_configuration.device_class == "stellarator"
    assert scenario.actuator_set.by_name("helical_trim_A").latency_steps == 2
    assert "R_axis_m" not in scenario.initial_frame.as_mapping()


def test_benchmark_report_is_deterministic_and_schema_valid() -> None:
    first = generate_report(steps=9, seed=4242, thresholds=DEFAULT_THRESHOLDS)
    second = generate_report(steps=9, seed=4242, thresholds=DEFAULT_THRESHOLDS)

    assert first == second
    validate_report_against_schema(first, load_report_schema())
    bench = first["stellarator_control_replay_benchmark"]
    assert bench["schema_version"] == "stellarator-control-replay-benchmark.v1"
    assert bench["replay"]["deterministic"] is True
    assert bench["actuators"]["max_abs_current_A"] <= bench["thresholds"]["max_abs_current_A"]
    assert bench["uncertainty"]["fieldline_spread_p95_high"] > bench["uncertainty"]["fieldline_spread_p95_low"]
    assert bench["data_provenance"]["geometry"] == "public_synthetic"


def test_public_config_file_drives_geometry_and_report_provenance() -> None:
    config = load_benchmark_config(PUBLIC_CONFIG)
    validate_benchmark_config(config, load_config_schema())

    report = generate_report(config_path=PUBLIC_CONFIG)

    bench = report["stellarator_control_replay_benchmark"]
    assert bench["magnetic_configuration"]["name"] == "public_w7x_like_reduced_order"
    assert bench["physics_context"]["field_periods"] == 5
    assert bench["replay"]["steps"] == config["steps"]
    assert bench["benchmark_config"]["schema_version"] == "stellarator-control-replay-config.v1"
    assert bench["benchmark_config"]["primary_control_actuator"] == "helical_trim_A"
    assert bench["benchmark_config"]["data_provenance"]["external_company_data"] == "none"
    assert "external_company_data" in json.dumps(report, sort_keys=True)


def test_public_config_uses_named_primary_actuator_when_decoy_channel_is_first(tmp_path: Path) -> None:
    payload = load_benchmark_config(PUBLIC_CONFIG)
    payload["actuators"].insert(
        0,
        {
            "name": "trim_probe_decoy",
            "unit": "A",
            "min_value": -1.0,
            "max_value": 1.0,
            "slew_rate_per_s": 1.0,
            "latency_steps": 0,
            "failure_mode": "none",
        },
    )
    config_path = tmp_path / "multi_actuator_config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    report = generate_report(config_path=config_path)

    bench = report["stellarator_control_replay_benchmark"]
    assert bench["benchmark_config"]["primary_control_actuator"] == "helical_trim_A"
    assert bench["actuators"]["primary_control_actuator"] == "helical_trim_A"
    assert bench["metrics"]["max_abs_current_A"] > 1.0


def test_invalid_config_rejects_unknown_primary_actuator(tmp_path: Path) -> None:
    payload = load_benchmark_config(PUBLIC_CONFIG)
    payload["primary_control_actuator"] = "missing_trim"
    bad_config = tmp_path / "bad_primary_actuator_config.json"
    bad_config.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown primary_control_actuator"):
        load_benchmark_config(bad_config)


def test_invalid_config_rejects_unknown_top_level_key(tmp_path: Path) -> None:
    payload = load_benchmark_config(PUBLIC_CONFIG)
    payload["operator_notes"] = "not part of the public contract"
    bad_config = tmp_path / "bad_extra_key_config.json"
    bad_config.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected config key: operator_notes"):
        load_benchmark_config(bad_config)


def test_invalid_config_rejects_unknown_nested_keys(tmp_path: Path) -> None:
    payload = load_benchmark_config(PUBLIC_CONFIG)
    payload["magnetic_configuration"]["operator_notes"] = "not part of the public contract"
    bad_config = tmp_path / "bad_nested_key_config.json"
    bad_config.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"unexpected config key: magnetic_configuration\.operator_notes"):
        load_benchmark_config(bad_config)


def test_invalid_config_rejects_unknown_actuator_keys(tmp_path: Path) -> None:
    payload = load_benchmark_config(PUBLIC_CONFIG)
    payload["actuators"][0]["operator_notes"] = "not part of the public contract"
    bad_config = tmp_path / "bad_actuator_key_config.json"
    bad_config.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"unexpected config key: actuators\[0\]\.operator_notes"):
        load_benchmark_config(bad_config)


def test_invalid_config_rejects_missing_nested_required_key(tmp_path: Path) -> None:
    payload = load_benchmark_config(PUBLIC_CONFIG)
    del payload["diagnostics"]["n_phi"]
    bad_config = tmp_path / "bad_missing_nested_key_config.json"
    bad_config.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"missing required config key: diagnostics\.n_phi"):
        load_benchmark_config(bad_config)


def test_invalid_config_rejects_missing_actuator_required_key(tmp_path: Path) -> None:
    payload = load_benchmark_config(PUBLIC_CONFIG)
    del payload["actuators"][0]["slew_rate_per_s"]
    bad_config = tmp_path / "bad_missing_actuator_key_config.json"
    bad_config.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"missing required config key: actuators\[0\]\.slew_rate_per_s"):
        load_benchmark_config(bad_config)


def test_invalid_config_rejects_private_data_and_unknown_fault_channel(tmp_path: Path) -> None:
    payload = load_benchmark_config(PUBLIC_CONFIG)
    payload["data_provenance"]["external_company_data"] = "proprietary"
    payload["fault_schedule"] = [{"step": 3, "channel": "missing_trim", "mode": "stuck"}]
    bad_config = tmp_path / "bad_stellarator_replay_config.json"
    bad_config.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="external_company_data must be 'none'"):
        load_benchmark_config(bad_config)


def test_invalid_config_rejects_nonfinite_threshold(tmp_path: Path) -> None:
    payload = load_benchmark_config(PUBLIC_CONFIG)
    payload["thresholds"]["max_final_fieldline_spread"] = "NaN"
    bad_config = tmp_path / "bad_threshold_config.json"
    bad_config.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="thresholds.max_final_fieldline_spread must be finite"):
        load_benchmark_config(bad_config)


def test_report_schema_rejects_unknown_top_level_key() -> None:
    report = generate_report(steps=7, seed=17, thresholds=DEFAULT_THRESHOLDS)
    report["stellarator_control_replay_benchmark"]["operator_notes"] = "not part of the public report contract"

    with pytest.raises(ValueError, match="unexpected benchmark report key: operator_notes"):
        validate_report_against_schema(report, load_report_schema())


def test_report_schema_rejects_incomplete_trace_row() -> None:
    report = generate_report(steps=7, seed=17, thresholds=DEFAULT_THRESHOLDS)
    del report["stellarator_control_replay_benchmark"]["replay"]["trace"][0]["latency_us"]

    with pytest.raises(ValueError, match=r"missing required benchmark report key: replay\.trace\[0\]\.latency_us"):
        validate_report_against_schema(report, load_report_schema())


def test_report_schema_rejects_missing_nested_required_key() -> None:
    report = generate_report(steps=7, seed=17, thresholds=DEFAULT_THRESHOLDS)
    del report["stellarator_control_replay_benchmark"]["uncertainty"]["samples"]

    with pytest.raises(ValueError, match=r"missing required benchmark report key: uncertainty\.samples"):
        validate_report_against_schema(report, load_report_schema())


def test_report_schema_rejects_extra_trace_row_key() -> None:
    report = generate_report(steps=7, seed=17, thresholds=DEFAULT_THRESHOLDS)
    report["stellarator_control_replay_benchmark"]["replay"]["trace"][0]["operator_notes"] = "not part of the public report contract"

    with pytest.raises(ValueError, match=r"unexpected benchmark report key: replay\.trace\[0\]\.operator_notes"):
        validate_report_against_schema(report, load_report_schema())


def test_markdown_report_contains_limits_and_no_company_specific_claim() -> None:
    report = generate_report(steps=7, seed=17, thresholds=DEFAULT_THRESHOLDS)
    md = render_markdown(report)

    assert "Stellarator Control Replay Benchmark" in md
    assert "Limitations" in md
    assert "not a production plant-control system" in md
    assert "External company data: `none`" in md


def test_cli_mode_exposes_one_command_benchmark() -> None:
    spec = cli_mod.MODE_SPECS["stellarator-control-replay-benchmark"]
    assert spec.module == "validation.stellarator_control_replay_benchmark"
    assert spec.maturity == "public"


def test_module_cli_writes_json_and_markdown_outputs(tmp_path: Path) -> None:
    output_json = tmp_path / "stellarator_report.json"
    output_md = tmp_path / "stellarator_report.md"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "validation.stellarator_control_replay_benchmark",
            "--steps",
            "8",
            "--seed",
            "99",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--strict",
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    validate_report_against_schema(payload, load_report_schema())
    assert output_md.read_text(encoding="utf-8").startswith("# Stellarator Control Replay Benchmark")


def test_module_cli_accepts_public_config_file(tmp_path: Path) -> None:
    output_json = tmp_path / "configured_stellarator_report.json"
    output_md = tmp_path / "configured_stellarator_report.md"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "validation.stellarator_control_replay_benchmark",
            "--config",
            str(PUBLIC_CONFIG),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--strict",
        ],
        check=False,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    bench = payload["stellarator_control_replay_benchmark"]
    assert bench["benchmark_config"]["source_path"].endswith("stellarator_control_replay_public_config.json")
    assert output_md.read_text(encoding="utf-8").startswith("# Stellarator Control Replay Benchmark")
