# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — GS Residual Gating Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for GS residual telemetry and convergence gating in FusionKernel."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scpn_fusion.core.fusion_kernel import FusionKernel


BASE_CONFIG: dict[str, object] = {
    "reactor_name": "GS-Residual-Gate-Test",
    "grid_resolution": [24, 24],
    "dimensions": {"R_min": 4.0, "R_max": 8.0, "Z_min": -3.0, "Z_max": 3.0},
    "physics": {"plasma_current_target": 15.0, "vacuum_permeability": 1.0},
    "coils": [
        {"name": "CS", "r": 1.7, "z": 0.0, "current": 0.12},
    ],
    "solver": {
        "solver_method": "sor",
        "max_iterations": 40,
        "convergence_threshold": 1e-4,
        "relaxation_factor": 0.15,
        "sor_omega": 1.6,
    },
}


def _write_config(tmp_path: Path, cfg: dict[str, object], name: str) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


def test_solve_reports_gs_residual_metrics(tmp_path: Path) -> None:
    cfg = copy.deepcopy(BASE_CONFIG)
    fk = FusionKernel(str(_write_config(tmp_path, cfg, "base.json")))
    result = fk.solve_equilibrium()

    assert "gs_residual" in result
    assert "gs_residual_best" in result
    assert "gs_residual_history" in result

    hist = result["gs_residual_history"]
    assert len(hist) == len(result["residual_history"])
    assert len(hist) >= 1
    assert np.isfinite(result["gs_residual"])
    assert np.isfinite(result["gs_residual_best"])
    assert all(np.isfinite(x) and x >= 0.0 for x in hist)
    assert result["gs_residual"] == pytest.approx(hist[-1])
    assert result["gs_residual_best"] <= result["gs_residual"] + 1e-12


def test_gs_residual_gate_blocks_update_only_convergence(tmp_path: Path) -> None:
    # With extremely loose update tolerance, solver can terminate on iter 1.
    cfg_loose = copy.deepcopy(BASE_CONFIG)
    cfg_loose["solver"]["max_iterations"] = 5
    cfg_loose["solver"]["convergence_threshold"] = 1e6
    fk_loose = FusionKernel(str(_write_config(tmp_path, cfg_loose, "loose.json")))
    result_loose = fk_loose.solve_equilibrium()
    assert result_loose["converged"]
    assert result_loose["iterations"] == 1

    # Enable GS residual gate with an intentionally strict threshold.
    cfg_gate = copy.deepcopy(cfg_loose)
    cfg_gate["solver"]["require_gs_residual"] = True
    cfg_gate["solver"]["gs_residual_threshold"] = 1e-12
    fk_gate = FusionKernel(str(_write_config(tmp_path, cfg_gate, "gate.json")))
    result_gate = fk_gate.solve_equilibrium()

    assert not result_gate["converged"]
    assert result_gate["iterations"] == cfg_gate["solver"]["max_iterations"]
    assert result_gate["gs_residual"] > cfg_gate["solver"]["gs_residual_threshold"]


def test_gs_residual_gate_rejects_non_positive_threshold(tmp_path: Path) -> None:
    cfg = copy.deepcopy(BASE_CONFIG)
    cfg["solver"]["require_gs_residual"] = True
    cfg["solver"]["gs_residual_threshold"] = 0.0
    fk = FusionKernel(str(_write_config(tmp_path, cfg, "invalid.json")))

    with pytest.raises(ValueError, match="gs_residual_threshold"):
        fk.solve_equilibrium()
