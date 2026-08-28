# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Deterministic CPU/GPU-Sim Runtime Benchmark Tests
"""Real-surface tests for deterministic CPU/GPU-sim runtime validation."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
import runpy
import sys
from typing import Any, cast

import numpy as np
import pytest

from scpn_fusion.core.gpu_runtime import GPURuntimeBridge


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "deterministic_cpu_gpu_sim_runtime_benchmark.py"
SPEC = importlib.util.spec_from_file_location(
    "deterministic_cpu_gpu_sim_runtime_benchmark",
    MODULE_PATH,
)
assert SPEC and SPEC.loader
validation_cli = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validation_cli
SPEC.loader.exec_module(validation_cli)


def test_runtime_bridge_reports_deterministic_speedup_estimates() -> None:
    bridge = GPURuntimeBridge(seed=42)
    first = bridge.benchmark_pair(trials=8, grid_size=16)
    second = bridge.benchmark_pair(trials=8, grid_size=16)
    gpu_sim = cast(dict[str, float], first["gpu_sim"])
    cpu = cast(dict[str, float], first["cpu"])

    assert first["multigrid_speedup_est"] == second["multigrid_speedup_est"]
    assert first["snn_speedup_est"] == second["snn_speedup_est"]
    assert first["multigrid_speedup_est"] == pytest.approx(12.5)
    assert first["snn_speedup_est"] == pytest.approx(12.5)
    assert gpu_sim["multigrid_p95_ms_est"] < cpu["multigrid_p95_ms_est"]


@pytest.mark.parametrize("backend", ["cpu", "gpu_sim"])
def test_runtime_bridge_public_benchmark_reports_both_kernel_lanes(backend: str) -> None:
    result = GPURuntimeBridge(seed=7).benchmark(backend=backend, trials=8, grid_size=16)

    assert result.backend == backend
    assert result.multigrid_p95_ms_est > 0.0
    assert result.snn_p95_ms_est > 0.0
    assert result.multigrid_mean_ms_wall > 0.0
    assert result.snn_mean_ms_wall > 0.0


@pytest.mark.parametrize(
    ("backend", "trials", "grid_size", "match"),
    [
        ("cuda", 8, 16, "backend"),
        ("cpu", 7, 16, "trials"),
        ("cpu", 8, 15, "grid_size"),
    ],
)
def test_runtime_bridge_public_benchmark_rejects_invalid_inputs(
    backend: str, trials: int, grid_size: int, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        GPURuntimeBridge(seed=42).benchmark(
            backend=backend,
            trials=trials,
            grid_size=grid_size,
        )


def test_equilibrium_latency_public_surface_reports_fault_campaign() -> None:
    bridge = GPURuntimeBridge(seed=42)
    result = bridge.benchmark_equilibrium_latency(
        backend="gpu_sim",
        trials=8,
        grid_size=16,
        iterations=1,
        fault_runs=2,
        sensor_noise_std=0.01,
        bit_flips_per_run=2,
        seed=123,
    )

    assert result.backend == "gpu_sim"
    assert result.trials == 8
    assert result.grid_size == 16
    assert result.fault_runs == 2
    assert result.p95_ms_est > 0.0
    assert result.fault_p95_ms_est > result.p95_ms_est


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: GPURuntimeBridge().benchmark_equilibrium_latency(backend="cuda"), "backend"),
        (lambda: GPURuntimeBridge().benchmark_equilibrium_latency(trials=7), "trials"),
        (lambda: GPURuntimeBridge().benchmark_equilibrium_latency(grid_size=15), "grid_size"),
        (lambda: GPURuntimeBridge().benchmark_equilibrium_latency(iterations=0), "iterations"),
        (lambda: GPURuntimeBridge().benchmark_equilibrium_latency(fault_runs=0), "fault_runs"),
        (
            lambda: GPURuntimeBridge().benchmark_equilibrium_latency(sensor_noise_std=-0.1),
            "sensor_noise_std",
        ),
        (
            lambda: GPURuntimeBridge().benchmark_equilibrium_latency(sensor_noise_std=float("nan")),
            "sensor_noise_std",
        ),
    ],
)
def test_equilibrium_latency_public_surface_rejects_invalid_inputs(
    call: Callable[[], object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        call()


def test_available_equilibrium_backends_are_executable() -> None:
    bridge = GPURuntimeBridge(seed=42)
    available = bridge.available_equilibrium_backends()

    assert available[:2] == ("cpu", "gpu_sim")
    for backend in available:
        result = bridge.benchmark_equilibrium_latency(
            backend=backend,
            trials=8,
            grid_size=16,
            iterations=1,
            fault_runs=1,
            bit_flips_per_run=0,
        )
        assert result.backend == backend
        assert np.isfinite(result.mean_ms_wall)


def test_campaign_passes_default_thresholds() -> None:
    result = validation_cli.run_campaign(trials=8, grid_size=16)

    assert result["passes_thresholds"] is True
    assert result["thresholds"] == {
        "max_gpu_sim_multigrid_p95_ms_est": 2.0,
        "max_gpu_sim_snn_p95_ms_est": 1.0,
        "min_multigrid_speedup_est": 4.0,
        "min_snn_speedup_est": 4.0,
    }


def test_campaign_strict_public_threshold_can_fail_closed() -> None:
    result = validation_cli.run_campaign(
        trials=8,
        grid_size=16,
        min_snn_speedup_est=100.0,
    )

    assert result["passes_thresholds"] is False
    assert result["thresholds"]["min_snn_speedup_est"] == 100.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_gpu_sim_multigrid_p95_ms_est": 0.0}, "max_gpu_sim_multigrid"),
        ({"max_gpu_sim_multigrid_p95_ms_est": float("nan")}, "max_gpu_sim_multigrid"),
        ({"max_gpu_sim_snn_p95_ms_est": -1.0}, "max_gpu_sim_snn"),
        ({"max_gpu_sim_snn_p95_ms_est": float("inf")}, "max_gpu_sim_snn"),
        ({"min_multigrid_speedup_est": 0.0}, "min_multigrid_speedup"),
        ({"min_multigrid_speedup_est": float("nan")}, "min_multigrid_speedup"),
        ({"min_snn_speedup_est": -1.0}, "min_snn_speedup"),
        ({"min_snn_speedup_est": float("inf")}, "min_snn_speedup"),
    ],
)
def test_campaign_rejects_invalid_thresholds(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validation_cli.run_campaign(trials=8, grid_size=16, **kwargs)


def test_report_contract_and_markdown_are_descriptive() -> None:
    report = validation_cli.generate_report(trials=8, grid_size=16)
    payload = validation_cli.validate_report(report)
    markdown = validation_cli.render_markdown(report)

    assert report["schema_version"] == 2
    assert report["report_kind"] == "deterministic_cpu_gpu_sim_runtime_benchmark"
    assert payload["passes_thresholds"] is True
    assert markdown.startswith("# Deterministic CPU/GPU-Sim Runtime Benchmark")
    assert "## GPU-Sim P95 Estimate" in markdown
    assert "## Estimated Speedups" in markdown


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schema_version": 1}, "unsupported report schema_version"),
        ({"report_kind": "obsolete"}, "unsupported report_kind"),
        ({"generated_at_utc": ""}, "generated_at_utc must be"),
        ({"generated_at_utc": None}, "generated_at_utc must be"),
        ({"deterministic_cpu_gpu_sim_runtime_benchmark": []}, "must be an object"),
    ],
)
def test_report_contract_rejects_invalid_current_payloads(
    change: dict[str, Any], message: str
) -> None:
    report = validation_cli.generate_report(trials=8, grid_size=16)
    report.update(change)
    with pytest.raises(ValueError, match=message):
        validation_cli.validate_report(report)


def test_report_contract_rejects_obsolete_coded_payload_exactly() -> None:
    stale_report = {
        "generated_at_utc": "2026-08-26T00:00:00+00:00",
        "gdep_02": {"passes_thresholds": True},
    }
    with pytest.raises(ValueError, match="current descriptive contract"):
        validation_cli.validate_report(stale_report)


def test_cli_writes_current_contract_and_enforces_strict_thresholds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_json = tmp_path / "runtime-benchmark.json"
    output_md = tmp_path / "runtime-benchmark.md"
    common_args = [
        "--trials",
        "8",
        "--grid-size",
        "16",
        "--output-json",
        str(output_json),
        "--output-md",
        str(output_md),
    ]

    assert validation_cli.main([*common_args, "--strict"]) == 0
    written = json.loads(output_json.read_text(encoding="utf-8"))
    validation_cli.validate_report(written)
    assert output_md.read_text(encoding="utf-8").startswith(
        "# Deterministic CPU/GPU-Sim Runtime Benchmark"
    )
    assert "runtime benchmark complete" in capsys.readouterr().out

    assert (
        validation_cli.main(
            [
                *common_args,
                "--strict",
                "--min-snn-speedup-est",
                "100.0",
            ]
        )
        == 2
    )


def test_script_entry_point_runs_the_real_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_json = tmp_path / "script-report.json"
    output_md = tmp_path / "script-report.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(MODULE_PATH),
            "--trials",
            "8",
            "--grid-size",
            "16",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--strict",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(MODULE_PATH), run_name="__main__")

    assert exc_info.value.code == 0
    assert output_json.is_file()
    assert output_md.is_file()
