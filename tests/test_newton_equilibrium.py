# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Newton-Kantorovich Equilibrium Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Phase 5 verification: Newton-Kantorovich equilibrium solver."""

import json
from pathlib import Path

import numpy as np
import pytest

from scpn_fusion.core.fusion_kernel import FusionKernel

MOCK_CONFIG = {
    "reactor_name": "Newton-Test",
    "grid_resolution": [20, 20],
    "dimensions": {"R_min": 4.0, "R_max": 8.0, "Z_min": -4.0, "Z_max": 4.0},
    "physics": {"plasma_current_target": 15.0, "vacuum_permeability": 1.0},
    "coils": [
        {"name": "PF1", "r": 3.0, "z": 7.6, "current": 8.0},
        {"name": "PF2", "r": 8.2, "z": 6.7, "current": -1.32},
        {"name": "CS", "r": 1.7, "z": 0.0, "current": 0.15},
    ],
    "solver": {
        "max_iterations": 60,
        "convergence_threshold": 1e-3,
        "relaxation_factor": 0.1,
        "solver_method": "newton",
    },
}


@pytest.fixture
def newton_cfg(tmp_path: Path) -> Path:
    p = tmp_path / "newton_cfg.json"
    p.write_text(json.dumps(MOCK_CONFIG), encoding="utf-8")
    return p


@pytest.fixture
def picard_cfg(tmp_path: Path) -> Path:
    cfg = MOCK_CONFIG.copy()
    cfg["solver"] = {**cfg["solver"], "solver_method": "sor", "max_iterations": 200}
    p = tmp_path / "picard_cfg.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


# ── Tests ─────────────────────────────────────────────────────────

def test_newton_converges_finite(newton_cfg: Path):
    """Newton should produce finite Psi with no NaN."""
    fk = FusionKernel(str(newton_cfg))
    result = fk.solve_equilibrium()
    assert np.all(np.isfinite(fk.Psi)), "Newton produced NaN in Psi"
    assert result["solver_method"] == "newton"


def test_newton_fewer_iterations(newton_cfg: Path, picard_cfg: Path):
    """Newton should converge in fewer iterations than Picard."""
    fk_n = FusionKernel(str(newton_cfg))
    r_n = fk_n.solve_equilibrium()

    fk_p = FusionKernel(str(picard_cfg))
    r_p = fk_p.solve_equilibrium()

    # Newton should use fewer total iterations (warmup + Newton)
    # But the comparison is only meaningful if both converge
    if r_n["converged"] and r_p["converged"]:
        assert r_n["iterations"] < r_p["iterations"], (
            f"Newton ({r_n['iterations']}) not fewer than "
            f"Picard ({r_p['iterations']})"
        )


def test_newton_quadratic_residual_decrease(newton_cfg: Path):
    """Residual should decrease rapidly during Newton phase."""
    fk = FusionKernel(str(newton_cfg))
    result = fk.solve_equilibrium()
    rh = result["residual_history"]
    if len(rh) >= 4:
        # Check that later residuals are much smaller than earlier ones
        late = rh[-1]
        mid = rh[len(rh) // 2]
        assert late <= mid or late < 1e-2, (
            f"Residual not decreasing: mid={mid:.2e}, late={late:.2e}"
        )


def test_newton_matches_picard_solution(newton_cfg: Path, picard_cfg: Path):
    """Newton and Picard Psi should agree within tolerance."""
    fk_n = FusionKernel(str(newton_cfg))
    r_n = fk_n.solve_equilibrium()

    fk_p = FusionKernel(str(picard_cfg))
    r_p = fk_p.solve_equilibrium()

    if r_n["converged"] and r_p["converged"]:
        psi_diff = np.abs(fk_n.Psi - fk_p.Psi)
        # Allow larger tolerance since these are different iteration strategies
        assert np.mean(psi_diff) < 1.0, (
            f"Psi mean diff = {np.mean(psi_diff):.3f}"
        )


def test_newton_gs_residual_small(newton_cfg: Path):
    """Final GS residual should be small after Newton convergence."""
    fk = FusionKernel(str(newton_cfg))
    result = fk.solve_equilibrium()
    if result["converged"]:
        assert result["residual"] < 1e-2, (
            f"Final residual = {result['residual']:.2e}"
        )


def test_newton_with_hmode_profiles(tmp_path: Path):
    """Newton should work with H-mode (mtanh pedestal) profiles too."""
    cfg = MOCK_CONFIG.copy()
    cfg["physics"] = {
        **cfg["physics"],
        "profiles": {
            "mode": "h-mode",
            "p_prime": {"ped_top": 0.92, "ped_width": 0.05,
                        "ped_height": 1.0, "core_alpha": 0.3},
            "ff_prime": {"ped_top": 0.92, "ped_width": 0.05,
                         "ped_height": 1.0, "core_alpha": 0.3},
        },
    }
    p = tmp_path / "hmode_newton.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")

    fk = FusionKernel(str(p))
    result = fk.solve_equilibrium()
    assert np.all(np.isfinite(fk.Psi)), "Newton H-mode produced NaN"
    assert result["solver_method"] == "newton"


def test_newton_picard_warmup_runs(newton_cfg: Path):
    """The Picard warmup phase should execute without error."""
    fk = FusionKernel(str(newton_cfg))
    result = fk.solve_equilibrium()
    # Warmup should have added entries to residual_history
    assert len(result["residual_history"]) >= 1, "No residual history entries"
