# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — End-to-End Latency Benchmark Tests
# ──────────────────────────────────────────────────────────────────────
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


def test_campaign_has_deterministic_rmse_for_seed() -> None:
    a = scpn_end_to_end_latency.run_campaign(seed=42, steps=180)
    b = scpn_end_to_end_latency.run_campaign(seed=42, steps=180)
    for mode in ("surrogate", "full"):
        for ctrl in ("SNN", "PID", "MPC-lite"):
            assert a["modes"][mode][ctrl]["rmse"] == b["modes"][mode][ctrl]["rmse"]


def test_full_mode_ratio_is_finite_and_positive() -> None:
    out = scpn_end_to_end_latency.run_campaign(seed=42, steps=180)
    ratio = out["ratios"]["snn_full_to_surrogate_p95_ratio"]
    assert ratio > 0.0


@pytest.mark.parametrize("steps", [0, 8, 31])
def test_campaign_rejects_invalid_steps(steps: int) -> None:
    with pytest.raises(ValueError, match="steps"):
        scpn_end_to_end_latency.run_campaign(seed=42, steps=steps)


def test_render_markdown_contains_latency_sections() -> None:
    report = scpn_end_to_end_latency.generate_report(seed=11, steps=120)
    text = scpn_end_to_end_latency.render_markdown(report)
    assert "# SCPN End-to-End Latency Benchmark" in text
    assert "Surrogate Physics Mode" in text
    assert "Full Physics Mode" in text
    assert "p95 loop [ms]" in text


def test_cli_writes_reports_and_strict_passes(tmp_path: Path) -> None:
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
        "--strict",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0
    assert out_json.exists()
    assert out_md.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert "scpn_end_to_end_latency" in payload
