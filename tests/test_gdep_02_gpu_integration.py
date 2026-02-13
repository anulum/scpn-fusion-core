# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — GDEP-02 GPU Integration Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for GDEP-02 GPU runtime bridge and validation lane."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from scpn_fusion.core.gpu_runtime import GPURuntimeBridge


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "validation" / "gdep_02_gpu_integration.py"
SPEC = importlib.util.spec_from_file_location("gdep_02_gpu_integration", MODULE_PATH)
assert SPEC and SPEC.loader
gdep_02_gpu_integration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gdep_02_gpu_integration)


def test_gpu_runtime_bridge_speedup_estimates() -> None:
    bridge = GPURuntimeBridge(seed=42)
    out = bridge.benchmark_pair(trials=32, grid_size=48)
    assert out["multigrid_speedup_est"] > 1.0
    assert out["snn_speedup_est"] > 1.0
    assert out["gpu_sim"]["multigrid_p95_ms_est"] < out["cpu"]["multigrid_p95_ms_est"]


def test_gdep_02_campaign_passes_thresholds() -> None:
    out = gdep_02_gpu_integration.run_campaign(trials=48, grid_size=64)
    assert out["passes_thresholds"] is True
