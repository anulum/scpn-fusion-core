# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Linear Reduced-MHD Benchmark Tests
"""Tests for the linear reduced-MHD benchmark deck and its tracked local report."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "benchmarks" / "bench_linear_mhd.py"
REPORT_PATH = ROOT / "validation" / "reports" / "linear_mhd_benchmark.local.json"
UNAVAILABLE = {"rust", "mojo", "julia", "go"}


def _load() -> ModuleType:
    """Load the benchmark module from disk without requiring package installation."""
    spec = importlib.util.spec_from_file_location("bench_linear_mhd", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_statistics_and_timing_helpers() -> None:
    """Percentiles come from the samples; warm-up calls are untimed and the last result is kept."""
    module = _load()
    stats = module._stats([1_000_000, 2_000_000, 3_000_000])
    assert stats["p50_ms"] == pytest.approx(2.0) and stats["min_ms"] == pytest.approx(1.0)
    assert stats["max_ms"] == pytest.approx(3.0) and stats["throughput_per_s"] == pytest.approx(
        500.0
    )
    calls: list[int] = []

    def call() -> int:
        calls.append(1)
        return len(calls)

    samples, last = module._time(call, 2, 3)
    assert len(samples) == 3 and len(calls) == 5 and last == 5


def test_provenance_helpers_fall_back_honestly(tmp_path: Path) -> None:
    """Outside a git checkout the commit is unknown; without a model line the platform string is used."""
    module = _load()
    assert module._commit(tmp_path) == "unknown"
    assert len(module._commit()) == 40
    no_model = tmp_path / "cpuinfo"
    no_model.write_text("processor : 0\n", encoding="utf-8")
    assert module._cpu_model(no_model) == module._cpu_model(tmp_path / "missing")
    with_model = tmp_path / "cpuinfo_model"
    with_model.write_text("model name : Test CPU 1\n", encoding="utf-8")
    assert module._cpu_model(with_model) == "Test CPU 1"


def test_smoke_run_writes_a_complete_report_with_every_backend(tmp_path: Path) -> None:
    """A real small run measures assembly and both search paths and lists absent backends."""
    module = _load()
    output = tmp_path / "report.json"
    module.main(
        [
            "--smoke",
            "--warmup",
            "0",
            "--repeats",
            "1",
            "--assembly-repeats",
            "1",
            "--output",
            str(output),
        ]
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "complete" and report["isolation"]["isolated"] is False
    measured = {row["name"] for row in report["results"] if row["status"] == "measured"}
    assert measured == {
        "1d_tearing_2_1_n_side_10/assembly",
        "1d_tearing_2_1_n_side_10/search-dense-qz",
        "1d_tearing_2_1_n_side_10/search-shift-invert",
    }
    absent = {row["backend"] for row in report["results"] if row["stats"] is None}
    assert absent == UNAVAILABLE
    growth = [row["growth_rate"] for row in report["results"] if "search-" in row["name"]]
    # ARPACK converges mu = 1/(lambda - sigma) to tol 1e-10; for sigma >> lambda the relative
    # error of lambda reaches ~1e-7 (4.3e-8 measured on this coarse grid)
    assert growth[0] == pytest.approx(growth[1], rel=1e-6)


def test_production_case_set_is_declared() -> None:
    """The default deck holds the production grid, the iterative-path grid and a 2-D case."""
    cases = _load()._cases()
    assert [case["name"] for case in cases] == [
        "1d_tearing_2_1_n_side_200",
        "1d_tearing_2_1_n_side_60",
        "2d_circular_m2_n_side_60",
    ]
    assert [case["paths"] for case in cases] == [
        ("dense-qz",),
        ("dense-qz", "shift-invert"),
        ("dense-qz",),
    ]


def test_tracked_local_report_is_complete_and_labelled_non_isolated() -> None:
    """The committed report carries provenance, the full case set and its claim boundary."""
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert report["schema"] == "scpn_fusion.linear_mhd_benchmark.v1"
    assert report["status"] == "complete" and report["isolation"]["isolated"] is False
    assert "not a production performance claim" in report["claim_boundary"]
    names = {row["name"] for row in report["results"] if row["status"] == "measured"}
    assert "1d_tearing_2_1_n_side_200/search-dense-qz" in names
    assert "1d_tearing_2_1_n_side_60/search-shift-invert" in names
    assert {row["backend"] for row in report["results"] if row["stats"] is None} == UNAVAILABLE
