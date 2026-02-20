# ----------------------------------------------------------------------
# SCPN Fusion Core -- Transport Uncertainty Envelope Benchmark Tests
# ----------------------------------------------------------------------
"""Tests for validation/benchmark_transport_uncertainty_envelope.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "benchmark_transport_uncertainty_envelope.py"
SPEC = importlib.util.spec_from_file_location(
    "benchmark_transport_uncertainty_envelope", MODULE_PATH
)
assert SPEC and SPEC.loader
benchmark_transport_uncertainty_envelope = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark_transport_uncertainty_envelope
SPEC.loader.exec_module(benchmark_transport_uncertainty_envelope)


def test_run_benchmark_passes_contracts() -> None:
    itpa_csv = ROOT / "validation" / "reference_data" / "itpa" / "hmode_confinement.csv"
    report = benchmark_transport_uncertainty_envelope.run_benchmark(itpa_csv=itpa_csv)
    g = report["transport_uncertainty_envelope_benchmark"]
    assert g["passes_thresholds"] is True
    assert g["transport_pass"] is True
    assert g["envelope_fields_pass"] is True
    assert g["coverage_pass"] is True
    assert g["abs_relative_p95_pass"] is True
    assert g["zscore_p95_pass"] is True


def test_render_markdown_contains_sections() -> None:
    itpa_csv = ROOT / "validation" / "reference_data" / "itpa" / "hmode_confinement.csv"
    report = benchmark_transport_uncertainty_envelope.run_benchmark(itpa_csv=itpa_csv)
    text = benchmark_transport_uncertainty_envelope.render_markdown(report)
    assert "# Transport Uncertainty Envelope Benchmark" in text
    assert "Envelope fields pass" in text
    assert "| Metric | Value | Threshold |" in text


def test_cli_writes_reports_and_strict_passes(tmp_path: Path) -> None:
    itpa_csv = ROOT / "validation" / "reference_data" / "itpa" / "hmode_confinement.csv"
    out_json = tmp_path / "transport_uncertainty_envelope_benchmark.json"
    out_md = tmp_path / "transport_uncertainty_envelope_benchmark.md"
    cmd = [
        sys.executable,
        str(MODULE_PATH),
        "--itpa-csv",
        str(itpa_csv),
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
    assert "transport_uncertainty_envelope_benchmark" in payload
