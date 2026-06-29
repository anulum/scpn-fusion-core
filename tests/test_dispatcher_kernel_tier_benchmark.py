# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Dispatcher Kernel-Tier Benchmark Tests
"""Regression tests for the dispatcher kernel-tier benchmark deck."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "benchmarks" / "bench_dispatcher_kernel_tiers.py"
REPORT_PATH = ROOT / "validation" / "reports" / "dispatcher_kernel_tiers_benchmark.json"
EXPECTED_SCHEMA = "scpn-fusion-core.dispatcher-kernel-tiers-benchmark.v1"
EXPECTED_KERNELS = {
    "shafranov_bv",
    "solve_coil_currents",
    "measure_magnetics",
    "multigrid_solve",
    "simulate_tearing_mode",
}


def _load_benchmark_module() -> ModuleType:
    """Load the benchmark module from disk without requiring package installation."""
    spec = importlib.util.spec_from_file_location("bench_dispatcher_kernel_tiers", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_benchmark_deck_declares_all_a2_dispatch_kernels() -> None:
    """Keep the benchmark deck aligned with the A2 canonical dispatcher kernels."""
    module = _load_benchmark_module()

    assert set(module.KERNEL_NAMES) == EXPECTED_KERNELS


def test_build_report_records_tiers_timings_and_fallback_telemetry() -> None:
    """The benchmark report proves tier selection and fallback telemetry together."""
    module = _load_benchmark_module()

    report = module.build_report(repeats=1)

    assert report["schema"] == EXPECTED_SCHEMA
    assert report["gate"]["passes"] is True
    assert report["gate"]["kernel_count"] == len(EXPECTED_KERNELS)
    assert report["gate"]["all_numpy_floor_passed"] is True
    assert report["gate"]["fallback_telemetry_validation_passed"] is True

    rows = report["kernels"]
    assert {row["kernel"] for row in rows} == EXPECTED_KERNELS
    for row in rows:
        assert row["passes"] is True
        assert row["selected_tier"] in set(row["registered_tiers"])
        assert "rust" in row["registered_tiers"]
        assert "numpy" in row["registered_tiers"]
        assert row["repeats"] == 1
        assert row["wall_time_s_min"] >= 0.0
        assert row["output_checksum"] == pytest.approx(float(row["output_checksum"]))

    telemetry = report["fallback_telemetry_validation"]
    assert telemetry["passes"] is True
    assert telemetry["forced_unavailable_tier"] == "rust"
    assert telemetry["expected_selected_tier"] == "numpy"
    assert telemetry["fallback_event_count"] == len(EXPECTED_KERNELS)
    assert {row["kernel"] for row in telemetry["rows"]} == EXPECTED_KERNELS


def test_main_writes_benchmark_report(tmp_path: Path) -> None:
    """The CLI writes the same schema as the in-process builder."""
    module = _load_benchmark_module()
    output = tmp_path / "dispatcher_kernel_tiers_benchmark.json"

    rc = module.main(["--output", str(output), "--repeats", "1"])

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == EXPECTED_SCHEMA
    assert payload["gate"]["passes"] is True


def test_helper_defensive_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise benchmark helper guards that are host- or input-dependent."""
    module = _load_benchmark_module()

    def raise_load_average() -> tuple[float, float, float]:
        raise OSError("load average unavailable")

    monkeypatch.setattr(module.os, "getloadavg", raise_load_average)

    assert module._host_load_average() == []
    assert module._checksum(np.float64(2.5)) == pytest.approx(2.5)
    assert module._output_shape([1.0, np.array([2.0])]) == "list[float,ndarray(1,)]"
    assert module._output_shape(object()) == "object"
    with pytest.raises(TypeError, match="Unsupported benchmark output type"):
        module._checksum(object())
    with pytest.raises(ValueError, match="repeats"):
        module.build_report(repeats=0)


def test_tracked_report_and_docs_reference_benchmark_command() -> None:
    """Keep the tracked report and public docs cross-wired to the benchmark deck."""
    payload = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert payload["schema"] == EXPECTED_SCHEMA
    assert payload["gate"]["passes"] is True
    assert {row["kernel"] for row in payload["kernels"]} == EXPECTED_KERNELS

    docs = (ROOT / "docs" / "BENCHMARKS.md").read_text(encoding="utf-8")
    readme = (ROOT / "benchmarks" / "README.md").read_text(encoding="utf-8")
    architecture = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    command = "PYTHONPATH=src python benchmarks/bench_dispatcher_kernel_tiers.py"

    assert command in docs
    assert command in readme
    assert "dispatcher_kernel_tiers_benchmark.json" in docs
    assert "fallback_telemetry_validation" in architecture
