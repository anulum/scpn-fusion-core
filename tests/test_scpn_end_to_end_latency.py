# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — End-to-End Latency Benchmark Tests
"""Tests for validation/scpn_end_to_end_latency.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "scpn_end_to_end_latency.py"
SPEC = importlib.util.spec_from_file_location("scpn_end_to_end_latency", MODULE_PATH)
assert SPEC and SPEC.loader
scpn_end_to_end_latency = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scpn_end_to_end_latency
SPEC.loader.exec_module(scpn_end_to_end_latency)


def test_campaign_returns_expected_structure_and_passes_smoke() -> None:
    """Verify the legacy controller-loop campaign reports latency structure."""
    out = scpn_end_to_end_latency.run_campaign(seed=42, steps=200)
    assert out["passes_thresholds"] is True
    assert "surrogate" in out["modes"]
    assert "full" in out["modes"]
    for mode in ("surrogate", "full"):
        for ctrl in ("SNN", "PID", "MPC-lite"):
            rec = out["modes"][mode][ctrl]
            assert rec["rmse"] >= 0.0
            assert rec["p95_loop_ms"] > 0.0
            assert rec["p95_sensor_ms"] >= 0.0
            assert rec["p95_controller_ms"] >= 0.0
            assert rec["p95_actuator_ms"] >= 0.0
            assert rec["p95_physics_ms"] >= 0.0


def test_digital_twin_latency_campaign_reports_cpu_rust_gpu_boundaries() -> None:
    """Verify digital-twin timing reports local, simulated, and blocked boundaries."""
    out = scpn_end_to_end_latency.run_digital_twin_latency_campaign(steps=64)
    assert out["schema"] == "scpn-fusion-core.digital_twin_control_latency.v1"
    assert out["simulated_hil_scaffold_ready"] is True
    assert out["physical_hil_ready"] is False
    assert out["fpga_timing_ready"] is False
    assert out["codac_timing_ready"] is False
    assert out["actuator_hardware_timing_ready"] is False
    assert "isolation" in out["measurement_context"]["host_before"]
    assert "cpu_frequency" in out["measurement_context"]["host_before"]
    assert out["cpu"]["status"] == "measured"
    assert out["cpu"]["p95_loop_ms"] > 0.0
    assert out["cpu"]["p99_loop_ms"] >= out["cpu"]["p95_loop_ms"]
    assert out["gpu"]["status"].startswith("blocked_") or out["gpu"]["status"] == "measured"
    assert out["rust"]["status"].startswith("blocked_") or out["rust"]["status"] == "measured"
    assert out["hil"]["status"] == "measured_simulated_hil"
    assert out["hil"]["hardware_status"] == "simulated_host_adc_dac_loop"
    assert out["hil"]["actuator_count"] == 256
    assert all(row["passes_semantics"] for row in out["hil"]["scenarios"].values())
    assert out["thresholds"]["hil_wall_clock_threshold_required_for_report_acceptance"] is False


def test_host_snapshot_records_taskset_isolation_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify host snapshot records benchmark isolation metadata."""
    monkeypatch.setenv(
        "SCPN_BENCHMARK_ISOLATION_METHOD", "taskset_affinity_operator_reserved_cores"
    )
    monkeypatch.setenv("SCPN_BENCHMARK_CPUSET", "10,11")
    monkeypatch.setenv(
        "SCPN_BENCHMARK_COMMAND",
        "taskset -c 10,11 python validation/scpn_end_to_end_latency.py --strict",
    )
    monkeypatch.setenv(
        "SCPN_BENCHMARK_CONCURRENT_HEAVY_JOBS", "none_intentionally_started_by_this_task"
    )
    monkeypatch.setenv("SCPN_BENCHMARK_CLAIM_BOUNDARY", "Taskset affinity benchmark evidence.")

    snapshot = scpn_end_to_end_latency._host_snapshot()

    assert snapshot["isolation"] == "taskset_affinity_operator_reserved_cores"
    assert snapshot["reserved_cpu_set"] == "10,11"
    assert snapshot["benchmark_command"].startswith("taskset -c 10,11")
    assert snapshot["concurrent_heavy_jobs"] == "none_intentionally_started_by_this_task"
    assert snapshot["claim_boundary"] == "Taskset affinity benchmark evidence."


def test_digital_twin_degraded_modes_fail_closed_with_safe_outputs() -> None:
    """Verify degraded digital-twin modes produce safe outputs."""
    out = scpn_end_to_end_latency.run_digital_twin_latency_campaign(steps=64)
    degraded = out["cpu"]["degraded_modes"]
    for case in scpn_end_to_end_latency._DEGRADED_MODE_CASES:
        rec = degraded[case]
        assert rec["safe_output_rate"] == 1.0
        assert rec["passes_semantics"] is True
        if case != "nominal":
            assert rec["fallback_count"] > 0


def test_digital_twin_cpu_stage_schema_is_complete() -> None:
    """Verify CPU timing stages expose complete percentile summaries."""
    out = scpn_end_to_end_latency.run_digital_twin_latency_campaign(steps=64)
    stages = out["cpu"]["stages"]
    assert set(stages) == set(scpn_end_to_end_latency._PIPELINE_STAGE_KEYS)
    for rec in stages.values():
        assert set(rec) == {"p50_ms", "p95_ms", "p99_ms"}
        assert rec["p50_ms"] >= 0.0
        assert rec["p95_ms"] >= rec["p50_ms"]
        assert rec["p99_ms"] >= rec["p95_ms"]


def test_actuator_scaling_reaches_more_than_two_hundred_channels() -> None:
    """Verify actuator scaling includes the 256-channel reduced-order row."""
    out = scpn_end_to_end_latency.run_actuator_scaling_campaign(steps=64)
    assert out["schema"] == "scpn-fusion-core.digital_twin_actuator_scaling.v1"
    rows = {row["actuator_count"]: row for row in out["rows"]}
    assert 256 in rows
    row_256 = rows[256]
    assert row_256["cpu"]["status"] == "measured"
    assert row_256["cpu"]["safe_output_rate"] == 1.0
    assert (
        row_256["rust"]["status"].startswith("blocked_") or row_256["rust"]["status"] == "measured"
    )
    assert row_256["gpu"]["status"].startswith("blocked_") or row_256["gpu"]["status"] == "measured"


def test_predictive_horizon_campaign_covers_competitor_range() -> None:
    """Verify predictive-horizon timing covers the 50 ms and 100 ms rows."""
    out = scpn_end_to_end_latency.run_predictive_horizon_campaign(steps=64)
    assert out["schema"] == "scpn-fusion-core.digital_twin_predictive_horizon.v1"
    rows = {row["horizon_ms"]: row for row in out["rows"]}
    assert set(rows) == {50, 100}
    for row in rows.values():
        assert row["p95_forecast_ms"] > 0.0
        assert row["p95_real_time_factor"] > 1.0
        assert row["passes_realtime"] is True


def test_campaign_has_deterministic_rmse_for_seed() -> None:
    """Verify campaign RMSE values are deterministic for a fixed seed."""
    a = scpn_end_to_end_latency.run_campaign(seed=42, steps=180)
    b = scpn_end_to_end_latency.run_campaign(seed=42, steps=180)
    for mode in ("surrogate", "full"):
        for ctrl in ("SNN", "PID", "MPC-lite"):
            assert a["modes"][mode][ctrl]["rmse"] == b["modes"][mode][ctrl]["rmse"]


def test_full_mode_ratio_is_finite_and_positive() -> None:
    """Verify the full-to-surrogate SNN latency ratio is finite."""
    out = scpn_end_to_end_latency.run_campaign(seed=42, steps=180)
    ratio = out["ratios"]["snn_full_to_surrogate_p95_ratio"]
    assert ratio > 0.0


def test_campaign_rejects_invalid_steps() -> None:
    """Verify invalid campaign step counts fail fast."""
    for steps in (0, 8, 31):
        try:
            scpn_end_to_end_latency.run_campaign(seed=42, steps=steps)
        except ValueError as exc:
            assert "steps" in str(exc)
        else:
            raise AssertionError(f"steps={steps} should be rejected")


def test_render_markdown_contains_latency_sections() -> None:
    """Verify rendered Markdown includes scoped report sections."""
    report = scpn_end_to_end_latency.generate_report(seed=11, steps=120)
    text = scpn_end_to_end_latency.render_markdown(report)
    assert "# SCPN End-to-End Latency Benchmark" in text
    assert "local_reduced_order_latency_report" in text
    assert "Physical HIL ready: `NO`" in text
    assert "FPGA timing ready: `NO`" in text
    assert "CODAC timing ready: `NO`" in text
    assert "Actuator hardware timing ready: `NO`" in text
    assert "Digital-Twin Sensor-to-Control Path" in text
    assert "Simulated HIL Sensor-to-Actuator Scaffold" in text
    assert "Actuator-Count Scaling" in text
    assert "Predictive-Horizon Timing" in text
    assert "Degraded Modes" in text
    assert "Python CPU" in text
    assert "Rust native" in text
    assert "Surrogate Physics Mode" in text
    assert "Full Physics Mode" in text
    assert "p95 loop [ms]" in text


def test_cli_writes_scoped_reports(tmp_path: Path) -> None:
    """Verify the CLI writes scoped JSON and Markdown reports."""
    out_json = tmp_path / "scpn_end_to_end_latency.json"
    out_md = tmp_path / "scpn_end_to_end_latency.md"
    cmd = [
        sys.executable,
        str(MODULE_PATH),
        "--steps",
        "200",
        "--output-json",
        str(out_json),
        "--output-md",
        str(out_md),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert out_json.exists()
    assert out_md.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["schema"] == "scpn-fusion-core.end_to_end_latency_report.v1"
    assert payload["status"] in {
        "accepted_local_reduced_order_latency_report",
        "blocked_local_reduced_order_latency_report",
    }
    assert payload["passes_thresholds"] is (
        payload["status"] == "accepted_local_reduced_order_latency_report"
    )
    assert payload["physical_hil_ready"] is False
    assert payload["fpga_timing_ready"] is False
    assert payload["codac_timing_ready"] is False
    assert payload["actuator_hardware_timing_ready"] is False
    assert "scpn_end_to_end_latency" in payload
    assert "digital_twin_control_latency" in payload
    assert "actuator_count_scaling" in payload
    assert "predictive_horizon" in payload
    assert payload["digital_twin_control_latency"]["hil"]["status"] == "measured_simulated_hil"
    assert payload["digital_twin_control_latency"]["physical_hil_ready"] is False
    assert payload["digital_twin_control_latency"]["fpga_timing_ready"] is False
    assert payload["digital_twin_control_latency"]["codac_timing_ready"] is False
    assert payload["digital_twin_control_latency"]["actuator_hardware_timing_ready"] is False
