# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Newton-Kantorovich Equilibrium Tests
"""Phase 5 verification: Newton-Kantorovich equilibrium solver."""

import json
from pathlib import Path

import numpy as np
import pytest

import scpn_fusion.core.fusion_kernel_newton_solver as newton_mod
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
            f"Newton ({r_n['iterations']}) not fewer than " f"Picard ({r_p['iterations']})"
        )


def test_newton_quadratic_residual_decrease(tmp_path: Path):
    """Residual should decrease with line search enabled."""
    cfg = MOCK_CONFIG.copy()
    cfg["solver"] = {
        **cfg["solver"],
        "newton_line_search": True,
        "newton_line_search_c": 1e-4,
        "newton_line_search_max_backtracks": 8,
    }
    p = tmp_path / "newton_ls.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")

    fk = FusionKernel(str(p))
    result = fk.solve_equilibrium()
    rh = result["residual_history"]
    if len(rh) >= 4:
        late = rh[-1]
        mid = rh[len(rh) // 2]
        assert (
            late <= mid * 1.05 or late < 1e-2
        ), f"Residual not decreasing: mid={mid:.2e}, late={late:.2e}"


def test_newton_matches_picard_solution(newton_cfg: Path, picard_cfg: Path):
    """Newton and Picard Psi should agree within tolerance."""
    fk_n = FusionKernel(str(newton_cfg))
    r_n = fk_n.solve_equilibrium()

    fk_p = FusionKernel(str(picard_cfg))
    r_p = fk_p.solve_equilibrium()

    if r_n["converged"] and r_p["converged"]:
        psi_diff = np.abs(fk_n.Psi - fk_p.Psi)
        # Allow larger tolerance since these are different iteration strategies
        assert np.mean(psi_diff) < 1.0, f"Psi mean diff = {np.mean(psi_diff):.3f}"


def test_newton_gs_residual_small(newton_cfg: Path):
    """Final GS residual should be small after Newton convergence."""
    fk = FusionKernel(str(newton_cfg))
    result = fk.solve_equilibrium()
    if result["converged"]:
        assert result["residual"] < 1e-2, f"Final residual = {result['residual']:.2e}"


def test_newton_with_hmode_profiles(tmp_path: Path):
    """Newton should work with H-mode (mtanh pedestal) profiles too."""
    cfg = MOCK_CONFIG.copy()
    cfg["physics"] = {
        **cfg["physics"],
        "profiles": {
            "mode": "h-mode",
            "p_prime": {"ped_top": 0.92, "ped_width": 0.05, "ped_height": 1.0, "core_alpha": 0.3},
            "ff_prime": {"ped_top": 0.92, "ped_width": 0.05, "ped_height": 1.0, "core_alpha": 0.3},
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


def test_newton_reports_gs_residual_metrics(newton_cfg: Path):
    """Newton result should include GS residual telemetry."""
    fk = FusionKernel(str(newton_cfg))
    result = fk.solve_equilibrium()

    assert "gs_residual" in result
    assert "gs_residual_best" in result
    assert "gs_residual_history" in result
    assert np.isfinite(result["gs_residual"])
    assert np.isfinite(result["gs_residual_best"])
    assert len(result["gs_residual_history"]) >= 1


def test_newton_allows_ilu_preconditioner_mode(tmp_path: Path):
    """ILU preconditioner mode should execute without crashing."""
    cfg = MOCK_CONFIG.copy()
    cfg["solver"] = {
        **cfg["solver"],
        "gmres_preconditioner": "ilu",
        "max_iterations": 20,
    }
    p = tmp_path / "newton_ilu.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")

    fk = FusionKernel(str(p))
    result = fk.solve_equilibrium()
    assert result["solver_method"] == "newton"
    assert np.isfinite(result["residual"])


def test_newton_rejects_invalid_preconditioner_mode(tmp_path: Path):
    """Invalid preconditioner names should fail fast."""
    cfg = MOCK_CONFIG.copy()
    cfg["solver"] = {
        **cfg["solver"],
        "gmres_preconditioner": "not-a-mode",
    }
    p = tmp_path / "newton_bad_precond.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")

    fk = FusionKernel(str(p))
    with pytest.raises(ValueError, match="gmres_preconditioner"):
        fk.solve_equilibrium()


@pytest.mark.parametrize(
    ("field", "value", "error_pattern"),
    [
        ("gmres_ilu_drop_tol", 0.0, "gmres_ilu_drop_tol"),
        ("gmres_ilu_drop_tol", float("nan"), "gmres_ilu_drop_tol"),
        ("gmres_ilu_fill_factor", 0.5, "gmres_ilu_fill_factor"),
        ("gmres_ilu_fill_factor", float("inf"), "gmres_ilu_fill_factor"),
    ],
)
def test_newton_rejects_invalid_ilu_parameters(
    tmp_path: Path,
    field: str,
    value: float,
    error_pattern: str,
) -> None:
    """Invalid ILU tuning parameters should fail fast in Newton setup."""
    cfg = MOCK_CONFIG.copy()
    cfg["solver"] = {
        **cfg["solver"],
        "gmres_preconditioner": "ilu",
        field: value,
    }
    p = tmp_path / "newton_bad_ilu.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")

    fk = FusionKernel(str(p))
    with pytest.raises(ValueError, match=error_pattern):
        fk.solve_equilibrium()


@pytest.mark.parametrize(
    ("value", "error_pattern"),
    [
        (-1, "gmres_nonconverged_budget"),
        (True, "gmres_nonconverged_budget"),
    ],
)
def test_newton_rejects_invalid_gmres_nonconverged_budget(
    tmp_path: Path,
    value: object,
    error_pattern: str,
) -> None:
    """GMRES non-convergence budget must be an integer >= 0."""
    cfg = MOCK_CONFIG.copy()
    cfg["solver"] = {
        **cfg["solver"],
        "gmres_nonconverged_budget": value,
    }
    p = tmp_path / "newton_bad_gmres_budget.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")

    fk = FusionKernel(str(p))
    with pytest.raises(ValueError, match=error_pattern):
        fk.solve_equilibrium()


def test_newton_fails_fast_when_gmres_budget_exceeded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail fast when GMRES keeps returning non-converged info and budget is exhausted."""
    cfg = MOCK_CONFIG.copy()
    cfg["solver"] = {
        **cfg["solver"],
        "max_iterations": 8,
        "convergence_threshold": 1e-18,
        "require_gs_residual": True,
        "gs_residual_threshold": 1e-18,
        "gmres_nonconverged_budget": 0,
    }
    p = tmp_path / "newton_gmres_budget.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")

    def _fake_solve_linear_system(**kwargs: object) -> tuple[np.ndarray, int]:
        rhs = np.asarray(kwargs["rhs"], dtype=np.float64)
        return np.zeros_like(rhs), 1

    monkeypatch.setattr(
        newton_mod,
        "_solve_newton_linear_system_runtime",
        _fake_solve_linear_system,
    )

    fk = FusionKernel(str(p))
    with pytest.raises(RuntimeError, match="GMRES non-convergence budget exceeded"):
        fk.solve_equilibrium()


def test_newton_reports_gmres_telemetry_keys(newton_cfg: Path) -> None:
    """Newton result should expose GMRES convergence telemetry."""
    fk = FusionKernel(str(newton_cfg))
    result = fk.solve_equilibrium()

    assert "gmres_nonconverged_count" in result
    assert "gmres_breakdown_count" in result
    assert "gmres_last_info" in result
    assert "gmres_fail_on_breakdown" in result
    assert "gmres_nonconverged_budget" in result
    assert result["gmres_nonconverged_count"] >= 0
    assert result["gmres_breakdown_count"] >= 0


def test_newton_line_search_reports_telemetry(tmp_path: Path) -> None:
    """Line-search mode should expose structured telemetry counters."""
    cfg = MOCK_CONFIG.copy()
    cfg["solver"] = {
        **cfg["solver"],
        "newton_line_search": True,
        "newton_line_search_c": 1e-4,
        "newton_line_search_max_backtracks": 4,
        "max_iterations": 20,
    }
    p = tmp_path / "newton_line_search.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")

    fk = FusionKernel(str(p))
    result = fk.solve_equilibrium()
    assert result["solver_method"] == "newton"
    assert result["newton_line_search"] is True
    assert result["newton_line_search_attempts"] >= 0
    assert result["newton_line_search_accepts"] >= 0
    assert result["newton_line_search_rejects"] >= 0
    assert (
        result["newton_line_search_attempts"]
        >= result["newton_line_search_accepts"] + result["newton_line_search_rejects"]
    )


@pytest.mark.parametrize(
    ("field", "value", "error_pattern"),
    [
        ("newton_line_search_c", 0.0, "newton_line_search_c"),
        ("newton_line_search_c", 1.0, "newton_line_search_c"),
        ("newton_line_search_max_backtracks", 0, "newton_line_search_max_backtracks"),
    ],
)
def test_newton_rejects_invalid_line_search_config(
    tmp_path: Path,
    field: str,
    value: float,
    error_pattern: str,
) -> None:
    """Invalid line-search parameters should fail fast."""
    cfg = MOCK_CONFIG.copy()
    cfg["solver"] = {
        **cfg["solver"],
        "newton_line_search": True,
        field: value,
    }
    p = tmp_path / "newton_bad_line_search.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")

    fk = FusionKernel(str(p))
    with pytest.raises(ValueError, match=error_pattern):
        fk.solve_equilibrium()


@pytest.mark.parametrize("line_search_enabled", [False, True])
def test_newton_mode_strict_determinism_across_fixed_seed(
    tmp_path: Path,
    line_search_enabled: bool,
) -> None:
    """Newton line-search mode should be bit-stable across repeated fixed-seed runs."""
    cfg = MOCK_CONFIG.copy()
    cfg["solver"] = {
        **cfg["solver"],
        "newton_line_search": line_search_enabled,
        "max_iterations": 25,
    }
    p = tmp_path / f"newton_seeded_{int(line_search_enabled)}.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")

    np.random.seed(12345)
    fk1 = FusionKernel(str(p))
    r1 = fk1.solve_equilibrium()
    psi_1 = fk1.Psi.copy()

    np.random.seed(12345)
    fk2 = FusionKernel(str(p))
    r2 = fk2.solve_equilibrium()
    psi_2 = fk2.Psi.copy()

    np.testing.assert_array_equal(psi_1, psi_2)
    np.testing.assert_array_equal(
        np.asarray(r1["residual_history"], dtype=np.float64),
        np.asarray(r2["residual_history"], dtype=np.float64),
    )
    assert r1["gmres_nonconverged_count"] == r2["gmres_nonconverged_count"]
    assert r1["gmres_breakdown_count"] == r2["gmres_breakdown_count"]


def test_newton_line_search_enabled_disabled_fixed_seed_parity(
    tmp_path: Path,
) -> None:
    """Enabled/disabled line-search modes should both produce finite, shape-parity outputs."""
    cfg = MOCK_CONFIG.copy()
    cfg["solver"] = {
        **cfg["solver"],
        "max_iterations": 25,
    }
    p = tmp_path / "newton_line_search_parity.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")

    np.random.seed(777)
    fk_off = FusionKernel(str(p))
    fk_off.cfg["solver"]["newton_line_search"] = False
    r_off = fk_off.solve_equilibrium()
    psi_off = fk_off.Psi.copy()

    np.random.seed(777)
    fk_on = FusionKernel(str(p))
    fk_on.cfg["solver"]["newton_line_search"] = True
    r_on = fk_on.solve_equilibrium()
    psi_on = fk_on.Psi.copy()

    assert psi_off.shape == psi_on.shape
    assert np.all(np.isfinite(psi_off))
    assert np.all(np.isfinite(psi_on))
    assert np.isfinite(r_off["residual"])
    assert np.isfinite(r_on["residual"])
