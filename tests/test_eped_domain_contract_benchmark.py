# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — EPED Domain Contract Benchmark Tests
# ──────────────────────────────────────────────────────────────────────
"""Tests for validation/benchmark_eped_domain_contract.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "benchmark_eped_domain_contract.py"
SPEC = importlib.util.spec_from_file_location("benchmark_eped_domain_contract", MODULE_PATH)
assert SPEC and SPEC.loader
benchmark_eped_domain_contract = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark_eped_domain_contract
SPEC.loader.exec_module(benchmark_eped_domain_contract)


def test_run_benchmark_passes_and_flags_out_of_domain_cases() -> None:
    report = benchmark_eped_domain_contract.run_benchmark()
    g = report["eped_domain_contract_benchmark"]
    assert g["passes_thresholds"] is True
    assert g["in_domain_pass"] is True
    assert g["out_domain_flag_pass"] is True
    assert g["penalty_bounds_pass"] is True

    by_id = {case["case_id"]: case for case in g["cases"]}
    assert by_id["in_ref"]["in_domain"] is True
    assert by_id["in_ref"]["extrapolation_penalty"] == 1.0
    assert by_id["out_density"]["in_domain"] is False
    assert by_id["out_density"]["extrapolation_penalty"] < 1.0
    assert len(by_id["out_density"]["domain_violations"]) >= 1


def test_render_markdown_contains_contract_sections() -> None:
    report = benchmark_eped_domain_contract.run_benchmark()
    text = benchmark_eped_domain_contract.render_markdown(report)
    assert "# EPED Domain Contract Benchmark" in text
    assert "In-domain contract pass" in text
    assert "Penalty bounds pass" in text
    assert "| Case | In domain | Score | Penalty |" in text


def test_cli_writes_reports_and_exits_zero_in_strict_mode(tmp_path: Path) -> None:
    out_json = tmp_path / "eped_domain_contract_benchmark.json"
    out_md = tmp_path / "eped_domain_contract_benchmark.md"
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
    assert "eped_domain_contract_benchmark" in payload
