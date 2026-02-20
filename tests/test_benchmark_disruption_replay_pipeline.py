# ----------------------------------------------------------------------
# SCPN Fusion Core -- Disruption Replay Pipeline Benchmark Tests
# ----------------------------------------------------------------------
"""Tests for validation/benchmark_disruption_replay_pipeline.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "benchmark_disruption_replay_pipeline.py"
SPEC = importlib.util.spec_from_file_location(
    "benchmark_disruption_replay_pipeline", MODULE_PATH
)
assert SPEC and SPEC.loader
benchmark_disruption_replay_pipeline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark_disruption_replay_pipeline
SPEC.loader.exec_module(benchmark_disruption_replay_pipeline)


def _build_shot_dir(tmp_path: Path) -> Path:
    n = 180
    t = np.linspace(0.0, 0.179, n, dtype=np.float64)
    base = 0.42 + 0.05 * np.sin(2.0 * np.pi * 3.5 * t)
    bump = 0.30 * np.exp(-(((t - 0.158) / 0.011) ** 2))
    shot_dir = tmp_path / "shots"
    shot_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        shot_dir / "shot_disrupt.npz",
        time_s=t,
        dBdt_gauss_per_s=base + bump,
        n1_amp=0.15 + 0.55 * bump,
        n2_amp=0.05 + 0.18 * bump,
        is_disruption=np.array(True),
        disruption_time_idx=np.array(165),
    )
    np.savez(
        shot_dir / "shot_safe.npz",
        time_s=t,
        dBdt_gauss_per_s=base,
        n1_amp=np.full(n, 0.12, dtype=np.float64),
        n2_amp=np.full(n, 0.05, dtype=np.float64),
        is_disruption=np.array(False),
        disruption_time_idx=np.array(-1),
    )
    return shot_dir


def test_run_benchmark_passes_contract_invariants(tmp_path: Path) -> None:
    shot_dir = _build_shot_dir(tmp_path)
    report = benchmark_disruption_replay_pipeline.run_benchmark(disruption_dir=shot_dir)
    g = report["disruption_replay_pipeline_benchmark"]
    assert g["passes_thresholds"] is True
    assert g["default_deterministic_pass"] is True
    assert g["enabled_flags_pass"] is True
    assert g["disabled_flags_pass"] is True
    assert g["disabled_invariants_pass"] is True


def test_render_markdown_contains_sections(tmp_path: Path) -> None:
    shot_dir = _build_shot_dir(tmp_path)
    report = benchmark_disruption_replay_pipeline.run_benchmark(disruption_dir=shot_dir)
    text = benchmark_disruption_replay_pipeline.render_markdown(report)
    assert "# Disruption Replay Pipeline Benchmark" in text
    assert "Deterministic replay pass" in text
    assert "| Lane | Recall | FPR |" in text


def test_cli_writes_reports_and_strict_passes(tmp_path: Path) -> None:
    shot_dir = _build_shot_dir(tmp_path)
    out_json = tmp_path / "disruption_replay_pipeline_benchmark.json"
    out_md = tmp_path / "disruption_replay_pipeline_benchmark.md"
    cmd = [
        sys.executable,
        str(MODULE_PATH),
        "--disruption-dir",
        str(shot_dir),
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
    assert "disruption_replay_pipeline_benchmark" in payload
