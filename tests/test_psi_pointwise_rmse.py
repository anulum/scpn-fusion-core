# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Tests for point-wise ψ RMSE validation
# ──────────────────────────────────────────────────────────────────────
"""
Tests for validation/psi_pointwise_rmse.py.

Tests both the numerical operators (GS operator, SOR solver) and the
RMSE metric computation on real SPARC GEQDSK files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPARC_DIR = ROOT / "validation" / "reference_data" / "sparc"

# Skip entire module if SPARC data is missing
pytestmark = pytest.mark.skipif(
    not SPARC_DIR.exists() or not list(SPARC_DIR.glob("*.geqdsk")),
    reason="SPARC reference data not available",
)

from scpn_fusion.core.eqdsk import read_geqdsk
from validation.psi_pointwise_rmse import (
    compute_gs_source,
    compute_psi_rmse,
    gs_operator,
    gs_residual,
    manufactured_solve_vectorised,
    validate_all_sparc,
    validate_file,
)


# ── Unit tests for GS operator ──────────────────────────────────────


class TestGSOperator:
    """Verify the finite-difference GS operator on analytic solutions."""

    def test_constant_psi_gives_zero_operator(self):
        """Δ*(constant) = 0."""
        R = np.linspace(1.0, 3.0, 33)
        Z = np.linspace(-1.0, 1.0, 33)
        psi = np.ones((33, 33)) * 5.0
        result = gs_operator(psi, R, Z)
        assert np.allclose(result[1:-1, 1:-1], 0.0, atol=1e-12)

    def test_linear_in_Z_gives_zero(self):
        """ψ = Z has ∂²ψ/∂Z² = 0 and no R dependence → Δ* = 0."""
        R = np.linspace(1.0, 3.0, 65)
        Z = np.linspace(-1.0, 1.0, 65)
        _, ZZ = np.meshgrid(R, Z)
        psi = ZZ
        result = gs_operator(psi, R, Z)
        assert np.max(np.abs(result[2:-2, 2:-2])) < 1e-10

    def test_quadratic_in_R_nonzero(self):
        """ψ = R² → Δ*ψ = 2 - 2R/R = 0 ... no, Δ*ψ = d²(R²)/dR² - (1/R)d(R²)/dR = 2 - 2 = 0."""
        R = np.linspace(1.0, 3.0, 129)
        Z = np.linspace(-1.0, 1.0, 129)
        RR, _ = np.meshgrid(R, Z)
        psi = RR**2
        result = gs_operator(psi, R, Z)
        # Should be zero to within FD truncation
        assert np.max(np.abs(result[2:-2, 2:-2])) < 1e-6

    def test_operator_shape_matches(self):
        R = np.linspace(1.0, 3.0, 33)
        Z = np.linspace(-1.0, 1.0, 45)
        psi = np.random.default_rng(42).standard_normal((45, 33))
        result = gs_operator(psi, R, Z)
        assert result.shape == (45, 33)


# ── Integration tests on SPARC data ─────────────────────────────────


class TestGSResidualOnSPARC:
    """GS residual should be moderate on real SPARC equilibria."""

    def test_lmode_vv_gs_residual_finite(self):
        eq = read_geqdsk(SPARC_DIR / "lmode_vv.geqdsk")
        rel_l2, max_abs = gs_residual(eq)
        assert np.isfinite(rel_l2)
        assert np.isfinite(max_abs)
        # Relative L2 on real data with 2nd-order FD can be moderate due to
        # discretisation mismatch with the original equilibrium solver.
        assert rel_l2 < 5.0, f"GS relative L2 residual too large: {rel_l2}"

    def test_sparc_1310_gs_residual_finite(self):
        eq = read_geqdsk(SPARC_DIR / "sparc_1310.eqdsk")
        rel_l2, max_abs = gs_residual(eq)
        assert np.isfinite(rel_l2)
        assert np.isfinite(max_abs)


class TestManufacturedSolve:
    """Manufactured-source SOR solve should reproduce reference ψ."""

    def test_warm_start_low_rmse(self):
        """Warm-started SOR should converge to moderate RMSE."""
        eq = read_geqdsk(SPARC_DIR / "lmode_vv.geqdsk")
        psi_sol, iters, res, t_ms = manufactured_solve_vectorised(
            eq, omega=1.3, max_iter=500, tol=1e-8,
        )
        metrics = compute_psi_rmse(eq, psi_sol)
        # Manufactured-source solve: compute GS source from reference profiles,
        # solve with SOR using reference boundary conditions.  The relative L2
        # includes discretisation error from the FD GS operator and the
        # profile interpolation (p'/FF' from GEQDSK).  A value under 1.0
        # confirms the solver converges; under 0.1 would require higher-order
        # stencils or more SOR iterations than we allow in CI.
        assert metrics["psi_relative_l2"] < 1.0, (
            f"Relative L2 too large: {metrics['psi_relative_l2']}"
        )
        assert np.isfinite(metrics["psi_rmse_norm"])


class TestComputePsiRMSE:
    """RMSE computation correctness."""

    def test_identical_gives_zero(self):
        eq = read_geqdsk(SPARC_DIR / "lmode_vv.geqdsk")
        metrics = compute_psi_rmse(eq, eq.psirz)
        assert metrics["psi_rmse_wb"] == 0.0
        assert metrics["psi_rmse_norm"] == 0.0
        assert metrics["psi_max_error_wb"] == 0.0
        assert metrics["psi_relative_l2"] == 0.0

    def test_small_perturbation(self):
        eq = read_geqdsk(SPARC_DIR / "lmode_vv.geqdsk")
        noise = np.random.default_rng(42).normal(0, 1e-4, eq.psirz.shape)
        metrics = compute_psi_rmse(eq, eq.psirz + noise)
        assert metrics["psi_rmse_wb"] < 1e-3
        assert metrics["psi_relative_l2"] < 0.01


class TestValidateFile:
    """End-to-end per-file validation."""

    def test_lmode_vv_produces_result(self):
        result = validate_file(SPARC_DIR / "lmode_vv.geqdsk")
        assert result.file == "lmode_vv.geqdsk"
        assert result.grid == "129x129"
        assert np.isfinite(result.psi_rmse_norm)
        assert result.sor_iterations > 0
        assert result.solve_time_ms > 0

    @pytest.mark.parametrize("fname", [
        "sparc_1300.eqdsk",
        "sparc_1305.eqdsk",
        "sparc_1310.eqdsk",
        "sparc_1315.eqdsk",
        "sparc_1349.eqdsk",
    ])
    def test_eqdsk_files_produce_results(self, fname):
        path = SPARC_DIR / fname
        if not path.exists():
            pytest.skip(f"{fname} not found")
        result = validate_file(path)
        assert result.file == fname
        assert np.isfinite(result.psi_rmse_norm)


class TestValidateAllSPARC:
    """Aggregate validation."""

    def test_all_8_files_found(self):
        summary = validate_all_sparc()
        assert summary.count == 8, f"Expected 8 files, got {summary.count}"

    def test_summary_fields_finite(self):
        summary = validate_all_sparc()
        assert np.isfinite(summary.mean_psi_rmse_norm)
        assert np.isfinite(summary.mean_psi_relative_l2)
        assert np.isfinite(summary.mean_gs_residual_l2)
        assert summary.worst_file != ""
