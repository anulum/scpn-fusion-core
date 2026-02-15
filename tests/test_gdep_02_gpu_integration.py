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

import numpy as np
import pytest

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


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"trials": 7}, "trials"),
        ({"grid_size": 15}, "grid_size"),
    ],
)
def test_gpu_runtime_bridge_rejects_invalid_benchmark_inputs(
    kwargs: dict[str, int], match: str
) -> None:
    bridge = GPURuntimeBridge(seed=42)
    params = {"backend": "cpu", "trials": 32, "grid_size": 64}
    params.update(kwargs)
    with pytest.raises(ValueError, match=match):
        bridge.benchmark(**params)


def test_gpu_runtime_bridge_rejects_invalid_multigrid_iterations() -> None:
    bridge = GPURuntimeBridge(seed=42)
    field = np.linspace(0.0, 1.0, 16 * 16, dtype=np.float64).reshape(16, 16)
    with pytest.raises(ValueError, match="iterations"):
        bridge._gpu_sim_multigrid(field, iterations=0)
    with pytest.raises(ValueError, match="iterations"):
        bridge._cpu_multigrid(field, iterations=0)


def test_equilibrium_latency_auto_backend_resolves_and_reports_fault_runs() -> None:
    bridge = GPURuntimeBridge(seed=42)
    out = bridge.benchmark_equilibrium_latency(
        backend="auto",
        trials=16,
        grid_size=32,
        fault_runs=10,
        seed=123,
    )
    assert out.backend in {"gpu_sim", "torch_fallback"}
    assert out.fault_runs == 10
    assert out.p95_ms_est > 0.0
    assert out.fault_p95_ms_est > 0.0
