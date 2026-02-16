# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Rust Multigrid Wiring Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Phase 4 verification: Rust multigrid wiring through FusionKernel."""

import json
from pathlib import Path

import numpy as np
import pytest

from scpn_fusion.core._rust_compat import _rust_available
from scpn_fusion.core.fusion_kernel import FusionKernel

MOCK_CONFIG = {
    "reactor_name": "RustMG-Test",
    "grid_resolution": [32, 32],
    "dimensions": {"R_min": 4.0, "R_max": 8.0, "Z_min": -4.0, "Z_max": 4.0},
    "physics": {"plasma_current_target": 15.0, "vacuum_permeability": 1.0},
    "coils": [
        {"name": "PF1", "r": 3.0, "z": 7.6, "current": 8.0},
        {"name": "PF2", "r": 8.2, "z": 6.7, "current": -1.32},
        {"name": "CS", "r": 1.7, "z": 0.0, "current": 0.15},
    ],
    "solver": {
        "max_iterations": 200,
        "convergence_threshold": 1e-4,
        "relaxation_factor": 0.1,
        "solver_method": "rust_multigrid",
    },
}

requires_rust = pytest.mark.skipif(
    not _rust_available(), reason="Rust backend (scpn_fusion_rs) not installed"
)


@pytest.fixture
def cfg_path(tmp_path: Path) -> Path:
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(MOCK_CONFIG), encoding="utf-8")
    return p


def _make_sor_config(tmp_path: Path) -> Path:
    cfg = MOCK_CONFIG.copy()
    cfg["solver"] = {**cfg["solver"], "solver_method": "sor"}
    p = tmp_path / "sor_cfg.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


# ── Tests ─────────────────────────────────────────────────────────

@requires_rust
def test_rust_multigrid_returns_finite(cfg_path: Path):
    """Rust multigrid should return finite Psi, no NaN."""
    fk = FusionKernel(str(cfg_path))
    result = fk.solve_equilibrium()
    assert np.all(np.isfinite(fk.Psi)), "Rust multigrid produced NaN in Psi"
    assert result["solver_method"] == "rust_multigrid"


@requires_rust
def test_rust_multigrid_convergence(cfg_path: Path):
    """Rust multigrid should converge for standard config."""
    fk = FusionKernel(str(cfg_path))
    result = fk.solve_equilibrium()
    assert result["converged"], f"Did not converge: residual={result['residual']}"


def test_rust_multigrid_fallback_to_sor(tmp_path: Path):
    """Without Rust, rust_multigrid should gracefully fall back to SOR."""
    if _rust_available():
        pytest.skip("Rust IS available — cannot test fallback")
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(MOCK_CONFIG), encoding="utf-8")
    fk = FusionKernel(str(p))
    result = fk.solve_equilibrium()
    assert result["solver_method"] in ("sor", "rust_multigrid")
    assert np.all(np.isfinite(fk.Psi))


@requires_rust
def test_rust_multigrid_faster_than_sor(tmp_path: Path):
    """Rust multigrid should be faster than Python SOR."""
    # SOR run
    sor_path = _make_sor_config(tmp_path)
    fk_sor = FusionKernel(str(sor_path))
    r_sor = fk_sor.solve_equilibrium()

    # Rust multigrid run
    mg_path = tmp_path / "mg_cfg.json"
    mg_path.write_text(json.dumps(MOCK_CONFIG), encoding="utf-8")
    fk_mg = FusionKernel(str(mg_path))
    r_mg = fk_mg.solve_equilibrium()

    assert r_mg["wall_time_s"] < r_sor["wall_time_s"], (
        f"Rust MG ({r_mg['wall_time_s']:.3f}s) not faster than "
        f"SOR ({r_sor['wall_time_s']:.3f}s)"
    )


def test_rust_multigrid_result_keys(tmp_path: Path):
    """Result dict must have all expected keys regardless of backend."""
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(MOCK_CONFIG), encoding="utf-8")
    fk = FusionKernel(str(p))
    result = fk.solve_equilibrium()
    expected = {"psi", "converged", "iterations", "residual",
                "residual_history", "wall_time_s", "solver_method"}
    assert expected.issubset(result.keys()), f"Missing keys: {expected - result.keys()}"
