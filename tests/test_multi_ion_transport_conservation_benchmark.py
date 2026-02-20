# ----------------------------------------------------------------------
# SCPN Fusion Core -- Multi-Ion Transport Conservation Benchmark Tests
# ----------------------------------------------------------------------
"""Tests for validation/benchmark_multi_ion_transport_conservation.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "benchmark_multi_ion_transport_conservation.py"
SPEC = importlib.util.spec_from_file_location(
    "benchmark_multi_ion_transport_conservation", MODULE_PATH
)
assert SPEC and SPEC.loader
benchmark_multi_ion_transport_conservation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark_multi_ion_transport_conservation
SPEC.loader.exec_module(benchmark_multi_ion_transport_conservation)


def test_run_benchmark_passes_contracts() -> None:
    report = benchmark_multi_ion_transport_conservation.run_benchmark()
    g = report["multi_ion_transport_conservation_benchmark"]
    assert g["passes_thresholds"] is True
    assert g["finite_pass"] is True
    assert g["positivity_pass"] is True
    assert g["quasineutral_pass"] is True
    assert g["late_energy_pass"] is True
    assert g["he_ash_pass"] is True


def test_render_markdown_contains_sections() -> None:
    report = benchmark_multi_ion_transport_conservation.run_benchmark()
    text = benchmark_multi_ion_transport_conservation.render_markdown(report)
    assert "# Multi-Ion Transport Conservation Benchmark" in text
    assert "Quasi-neutrality pass" in text
    assert "| Metric | Value | Threshold |" in text


def test_cli_writes_reports_and_strict_passes(tmp_path: Path) -> None:
    out_json = tmp_path / "multi_ion_transport_conservation_benchmark.json"
    out_md = tmp_path / "multi_ion_transport_conservation_benchmark.md"
    cmd = [
        sys.executable,
        str(MODULE_PATH),
        "--output-json",
        str(out_json),
        "--output-md",
        str(out_md),
        "--strict",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0
    assert out_json.exists()
    assert out_md.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert "multi_ion_transport_conservation_benchmark" in payload
