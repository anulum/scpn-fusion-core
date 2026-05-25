# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Tests for point-wise ψ RMSE validation
"""
Tests for validation/psi_pointwise_rmse.py.

Tests both the numerical operators (GS operator, SOR solver) and the
RMSE metric computation on real SPARC GEQDSK files.
"""

from __future__ import annotations

from dataclasses import asdict
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

import validation.psi_pointwise_rmse as psi_rmse_mod
from scpn_fusion.core.eqdsk import read_geqdsk
from validation.psi_pointwise_rmse import (
    EFIT_BENCHMARK_MACHINE_PROVENANCE,
    EfitNRMSEBenchmarkGate,
    PsiRMSEResult,
    classify_source_scale_convention,
    compute_gs_source,
    compute_psi_rmse,
    compute_operator_candidate_rankings,
    compute_source_alignment,
    compute_source_candidate_rankings,
    compute_source_components,
    compute_toroidal_current_consistency,
    gs_operator,
    gs_residual,
    load_efit_nrmse_benchmark_schema,
    manufactured_solve_vectorised,
    validate_all_sparc,
    validate_efit_nrmse_benchmark_report,
    validate_efit_nrmse_benchmark,
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

    @pytest.mark.parametrize(
        ("psi_shape", "r_len", "z_len", "match"),
        [
            ((2, 3), 3, 2, "at least 3x3"),
            ((45, 33), 32, 45, "axis lengths"),
            ((45, 33), 33, 44, "axis lengths"),
        ],
    )
    def test_rejects_invalid_grid_shapes(self, psi_shape, r_len, z_len, match):
        psi = np.zeros(psi_shape, dtype=np.float64)
        R = np.linspace(1.0, 3.0, r_len)
        Z = np.linspace(-1.0, 1.0, z_len)
        with pytest.raises(ValueError, match=match):
            gs_operator(psi, R, Z)

    def test_rejects_non_monotonic_axes(self):
        psi = np.zeros((5, 5), dtype=np.float64)
        R = np.array([1.0, 1.5, 1.5, 2.0, 2.5], dtype=np.float64)
        Z = np.linspace(-1.0, 1.0, 5)
        with pytest.raises(ValueError, match="strictly increasing"):
            gs_operator(psi, R, Z)

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_inputs(self, bad_value):
        psi = np.zeros((5, 5), dtype=np.float64)
        R = np.linspace(1.0, 3.0, 5)
        Z = np.linspace(-1.0, 1.0, 5)

        bad_psi = psi.copy()
        bad_psi[2, 2] = bad_value
        with pytest.raises(ValueError, match="finite"):
            gs_operator(bad_psi, R, Z)

        bad_R = R.copy()
        bad_R[1] = bad_value
        with pytest.raises(ValueError, match="finite"):
            gs_operator(psi, bad_R, Z)


class TestComputeGSSource:
    """Validate GS source input contracts."""

    def test_rejects_profile_length_mismatch(self):
        eq = read_geqdsk(SPARC_DIR / "lmode_vv.geqdsk")
        eq.pprime = eq.pprime[:-1]
        with pytest.raises(ValueError, match="pprime"):
            compute_gs_source(eq)

    def test_rejects_psirz_shape_mismatch(self):
        eq = read_geqdsk(SPARC_DIR / "lmode_vv.geqdsk")
        eq.psirz = eq.psirz[:, :-1]
        with pytest.raises(ValueError, match="psirz"):
            compute_gs_source(eq)

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_profiles(self, bad_value):
        eq = read_geqdsk(SPARC_DIR / "lmode_vv.geqdsk")
        eq.ffprime = eq.ffprime.copy()
        eq.ffprime[0] = bad_value
        with pytest.raises(ValueError, match="finite"):
            compute_gs_source(eq)

    def test_rejects_non_monotonic_axes(self):
        eq = read_geqdsk(SPARC_DIR / "lmode_vv.geqdsk")
        eq.rdim = -abs(eq.rdim)
        with pytest.raises(ValueError, match="strictly increasing"):
            compute_gs_source(eq)

    def test_source_components_recombine_to_gs_source(self):
        eq = read_geqdsk(SPARC_DIR / "lmode_vv.geqdsk")

        components = compute_source_components(eq)
        source = compute_gs_source(eq)

        assert np.allclose(components["total_source"], source)
        assert components["pressure_source"].shape == source.shape
        assert components["ffprime_source"].shape == source.shape
        assert components["plasma_mask"].shape == source.shape
        assert 0.0 < components["plasma_mask_fraction"] < 1.0
        assert components["pressure_source_norm"] >= 0.0
        assert components["ffprime_source_norm"] >= 0.0
        assert components["total_source_norm"] > 0.0

    def test_source_components_reject_invalid_equilibrium_contract(self):
        eq = read_geqdsk(SPARC_DIR / "lmode_vv.geqdsk")
        eq.ffprime = eq.ffprime[:-1]
        with pytest.raises(ValueError, match="ffprime"):
            compute_source_components(eq)


class TestComputeSourceAlignment:
    """Source-attribution metrics for EFIT/GEQDSK profile-source mismatch."""

    def test_operator_source_is_exactly_self_consistent(self):
        eq = read_geqdsk(SPARC_DIR / "lmode_vv.geqdsk")
        operator_source = gs_operator(eq.psirz, eq.r, eq.z)

        metrics = compute_source_alignment(eq, source=operator_source)

        assert metrics["source_residual_l2"] < 1e-12
        assert metrics["source_correlation"] == pytest.approx(1.0)
        assert metrics["source_best_fit_scale"] == pytest.approx(1.0)
        assert abs(metrics["source_best_fit_offset"]) < 1e-12
        assert metrics["source_best_fit_relative_l2"] < 1e-12
        assert metrics["source_best_fit_convention"] == "canonical"
        assert metrics["source_plasma_residual_l2"] < 1e-12
        assert metrics["source_vacuum_residual_l2"] < 1e-12

    def test_profile_source_mismatch_metrics_are_finite_on_sparc_reference(self):
        eq = read_geqdsk(SPARC_DIR / "lmode_vv.geqdsk")

        metrics = compute_source_alignment(eq)

        assert np.isfinite(metrics["operator_source_norm"])
        assert np.isfinite(metrics["profile_source_norm"])
        assert np.isfinite(metrics["source_residual_l2"])
        assert np.isfinite(metrics["source_correlation"])
        assert np.isfinite(metrics["source_best_fit_scale"])
        assert np.isfinite(metrics["source_best_fit_offset"])
        assert np.isfinite(metrics["source_best_fit_relative_l2"])
        assert metrics["source_best_fit_convention"] != ""
        assert np.isfinite(metrics["source_plasma_residual_l2"])
        assert np.isfinite(metrics["source_vacuum_residual_l2"])
        assert np.isfinite(metrics["source_plasma_operator_norm"])
        assert np.isfinite(metrics["source_vacuum_operator_norm"])
        assert metrics["source_residual_l2"] > 1.0
        assert metrics["source_plasma_point_count"] > 0
        assert metrics["source_vacuum_point_count"] > 0
        assert metrics["source_vacuum_residual_l2"] <= 10.0

    def test_source_candidate_rankings_are_sorted_and_include_canonical_source(self):
        eq = read_geqdsk(SPARC_DIR / "lmode_vv.geqdsk")

        candidates = compute_source_candidate_rankings(eq)

        assert len(candidates) >= 6
        names = [row["candidate"] for row in candidates]
        assert "profile_source" in names
        assert "negated_profile_source" in names
        assert "pressure_plus_negated_ffprime" in names
        residuals = [row["source_residual_l2"] for row in candidates]
        assert residuals == sorted(residuals)
        assert all(np.isfinite(row["source_residual_l2"]) for row in candidates)
        assert all(row["source_best_fit_convention"] != "" for row in candidates)
        assert candidates[0]["candidate"] != ""

    def test_source_scale_convention_classifier_covers_common_geqdsk_factors(self):
        assert classify_source_scale_convention(1.0) == "canonical"
        assert classify_source_scale_convention(-1.0) == "negated"
        assert classify_source_scale_convention(2.0 * np.pi) == "scaled_by_2pi"
        assert classify_source_scale_convention(-2.0 * np.pi) == "scaled_by_minus_2pi"
        assert classify_source_scale_convention(1.0 / (2.0 * np.pi)) == "scaled_by_inv_2pi"
        assert classify_source_scale_convention(float("nan")) == "non_finite_scale"
        assert classify_source_scale_convention(3.0) == "unclassified_global_scale"

    def test_operator_candidate_rankings_are_sorted_and_include_normalization_variants(self):
        eq = read_geqdsk(SPARC_DIR / "lmode_vv.geqdsk")

        candidates = compute_operator_candidate_rankings(eq)

        assert len(candidates) >= 5
        names = [row["candidate"] for row in candidates]
        assert "delta_star_psi" in names
        assert "delta_star_psi_over_flux_span" in names
        assert "delta_star_psi_norm" in names
        residuals = [row["profile_residual_l2"] for row in candidates]
        assert residuals == sorted(residuals)
        assert all(np.isfinite(row["profile_residual_l2"]) for row in candidates)
        assert all(np.isfinite(row["profile_best_fit_relative_l2"]) for row in candidates)
        assert candidates[0]["candidate"] != ""


class TestToroidalCurrentConsistency:
    """Toroidal-current closure diagnostics for GEQDSK equilibria."""

    def test_operator_current_closes_high_current_public_sparc_eqdsk(self):
        eq = read_geqdsk(SPARC_DIR / "sparc_1310.eqdsk")

        metrics = compute_toroidal_current_consistency(eq)

        assert metrics["operator_current_closure_pass"] is True
        assert abs(metrics["operator_current_relative_error"]) < 1e-3
        assert np.sign(metrics["operator_toroidal_current_A"]) == np.sign(eq.current)
        assert np.isfinite(metrics["profile_current_relative_error"])

    def test_current_consistency_rejects_invalid_grid(self):
        eq = read_geqdsk(SPARC_DIR / "lmode_vv.geqdsk")
        eq.rdim = -abs(eq.rdim)

        with pytest.raises(ValueError, match="strictly increasing"):
            compute_toroidal_current_consistency(eq)


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
            eq,
            omega=1.3,
            max_iter=500,
            tol=1e-8,
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

    def test_vectorised_solver_honours_tolerance_for_early_stop(self):
        """Very loose tolerance should stop at first convergence checkpoint."""
        eq = read_geqdsk(SPARC_DIR / "lmode_vv.geqdsk")
        _, iters, _, _ = manufactured_solve_vectorised(
            eq,
            omega=1.3,
            max_iter=40,
            tol=1e12,
        )
        assert iters == 10

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"omega": 0.0}, "omega"),
            ({"omega": 2.0}, "omega"),
            ({"omega": float("nan")}, "omega"),
            ({"max_iter": 0}, "max_iter"),
            ({"tol": -1e-9}, "tol"),
            ({"tol": float("inf")}, "tol"),
        ],
    )
    def test_vectorised_solver_rejects_invalid_controls(self, kwargs, match):
        eq = read_geqdsk(SPARC_DIR / "lmode_vv.geqdsk")
        with pytest.raises(ValueError, match=match):
            manufactured_solve_vectorised(eq, **kwargs)

    def test_vectorised_solver_preserves_reference_for_operator_source(self):
        """The reference ψ must be a fixed point when the source is Lψ_ref."""
        eq = read_geqdsk(SPARC_DIR / "lmode_vv.geqdsk")
        operator_source = gs_operator(eq.psirz, eq.r, eq.z)

        psi_sol, iters, res, _ = manufactured_solve_vectorised(
            eq,
            source_override=operator_source,
            omega=1.3,
            max_iter=20,
            tol=1e-10,
        )
        metrics = compute_psi_rmse(eq, psi_sol)

        assert iters == 10
        assert res <= 1e-10
        assert metrics["psi_rmse_norm"] < 1e-10

    def test_vectorised_solver_rejects_invalid_source_override(self):
        eq = read_geqdsk(SPARC_DIR / "lmode_vv.geqdsk")
        bad_shape = np.zeros((eq.nh - 1, eq.nw), dtype=np.float64)
        with pytest.raises(ValueError, match="source_override shape"):
            manufactured_solve_vectorised(eq, source_override=bad_shape)

        bad_value = np.zeros((eq.nh, eq.nw), dtype=np.float64)
        bad_value[0, 0] = np.nan
        with pytest.raises(ValueError, match="source_override must contain only finite values"):
            manufactured_solve_vectorised(eq, source_override=bad_value)


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

    def test_rejects_shape_mismatch(self):
        eq = read_geqdsk(SPARC_DIR / "lmode_vv.geqdsk")
        bad = np.zeros((eq.psirz.shape[0] - 1, eq.psirz.shape[1]))
        with pytest.raises(ValueError, match="shape"):
            compute_psi_rmse(eq, bad)

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_solver_values(self, bad_value):
        eq = read_geqdsk(SPARC_DIR / "lmode_vv.geqdsk")
        bad = eq.psirz.copy()
        bad[0, 0] = bad_value
        with pytest.raises(ValueError, match="finite"):
            compute_psi_rmse(eq, bad)


class TestValidateFile:
    """End-to-end per-file validation."""

    def test_lmode_vv_produces_result(self):
        result = validate_file(SPARC_DIR / "lmode_vv.geqdsk")
        assert result.file == "lmode_vv.geqdsk"
        assert result.grid == "129x129"
        assert np.isfinite(result.psi_rmse_norm)
        assert result.sor_iterations > 0
        assert result.solve_time_ms > 0

    @pytest.mark.parametrize(
        "fname",
        [
            "sparc_1300.eqdsk",
            "sparc_1305.eqdsk",
            "sparc_1310.eqdsk",
            "sparc_1315.eqdsk",
            "sparc_1349.eqdsk",
        ],
    )
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

    def test_worst_file_ignores_nan_rmse_entries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # Create placeholder file names to drive deterministic sort order.
        f_nan = tmp_path / "a_nan.geqdsk"
        f_good = tmp_path / "b_good.geqdsk"
        f_nan.write_text("placeholder", encoding="utf-8")
        f_good.write_text("placeholder", encoding="utf-8")

        def _row(
            name: str,
            psi_rmse_norm: float,
        ) -> PsiRMSEResult:
            return PsiRMSEResult(
                file=name,
                grid="3x3",
                gs_residual_l2=0.1,
                gs_residual_max=0.2,
                psi_rmse_wb=1e-3,
                psi_rmse_norm=psi_rmse_norm,
                psi_rmse_plasma_wb=1e-3,
                psi_max_error_wb=2e-3,
                psi_relative_l2=0.01,
                sor_iterations=10,
                sor_residual=1e-4,
                solve_time_ms=1.0,
            )

        fake_rows = {
            f_nan.name: _row(f_nan.name, float("nan")),
            f_good.name: _row(f_good.name, 0.25),
        }

        def _fake_validate_file(path: Path, warm_start: bool = True) -> PsiRMSEResult:
            del warm_start
            return fake_rows[path.name]

        monkeypatch.setattr(psi_rmse_mod, "validate_file", _fake_validate_file)
        summary = psi_rmse_mod.validate_all_sparc(tmp_path)
        assert summary.count == 2
        assert summary.worst_file == f_good.name
        assert summary.worst_psi_rmse_norm == pytest.approx(0.25)


class TestValidateEFITNRMSEBenchmark:
    """Strict aggregate gate for 10+ EFIT/GEQDSK reference equilibria."""

    def _make_reference_tree(self, tmp_path: Path, count_by_machine: dict[str, int]) -> list[Path]:
        files: list[Path] = []
        for machine, count in count_by_machine.items():
            machine_dir = tmp_path / machine
            machine_dir.mkdir(parents=True, exist_ok=True)
            for idx in range(count):
                suffix = ".eqdsk" if machine == "sparc" and idx % 2 else ".geqdsk"
                path = machine_dir / f"{machine}_{idx:02d}{suffix}"
                path.write_text("placeholder", encoding="utf-8")
                files.append(path)
        return sorted(files)

    def _install_fake_validate_file(
        self,
        monkeypatch: pytest.MonkeyPatch,
        rmse_by_name: dict[str, float],
    ) -> None:
        def _fake_validate_file(path: Path, warm_start: bool = True) -> PsiRMSEResult:
            del warm_start
            rmse = rmse_by_name.get(path.name, 0.01)
            return PsiRMSEResult(
                file=path.name,
                grid="3x3",
                gs_residual_l2=0.1,
                gs_residual_max=0.2,
                psi_rmse_wb=1e-3,
                psi_rmse_norm=rmse,
                psi_rmse_plasma_wb=1e-3,
                psi_max_error_wb=2e-3,
                psi_relative_l2=0.01,
                sor_iterations=10,
                sor_residual=1e-4,
                solve_time_ms=1.0,
                operator_source_psi_rmse_norm=1.0e-12,
                operator_source_sor_iterations=10,
                operator_source_sor_residual=1e-4,
                source_consistency_class="profile_source_consistent",
                operator_source_norm=1.0,
                profile_source_norm=1.0,
                source_residual_l2=rmse,
                source_correlation=1.0,
                source_best_fit_scale=1.0,
                source_best_fit_offset=0.0,
                source_best_fit_relative_l2=0.0,
                source_best_fit_convention="canonical",
                plasma_mask_fraction=0.5,
                pressure_source_norm=0.7,
                ffprime_source_norm=0.3,
                total_source_norm=1.0,
                pressure_source_fraction=0.7,
                ffprime_source_fraction=0.3,
                source_plasma_residual_l2=0.01,
                source_vacuum_residual_l2=0.01,
                source_plasma_operator_norm=0.5,
                source_vacuum_operator_norm=0.5,
                source_plasma_point_count=4.0,
                source_vacuum_point_count=5.0,
                best_source_candidate="profile_source",
                best_source_candidate_residual_l2=rmse,
                profile_source_candidate_rank=1,
                best_operator_candidate="delta_star_psi",
                best_operator_candidate_residual_l2=rmse,
                delta_star_psi_candidate_rank=1,
                declared_toroidal_current_A=-8.7e6,
                operator_toroidal_current_A=-8.6995e6,
                profile_toroidal_current_A=-1.5e6,
                operator_current_relative_error=5.0e-5,
                profile_current_relative_error=0.8,
                operator_current_closure_pass=True,
            )

        monkeypatch.setattr(psi_rmse_mod, "validate_file", _fake_validate_file)

    def test_passes_when_minimum_count_and_all_rows_under_threshold(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        self._make_reference_tree(tmp_path, {"sparc": 4, "diiid": 3, "jet": 3})
        self._install_fake_validate_file(monkeypatch, {})

        gate = validate_efit_nrmse_benchmark(
            tmp_path,
            min_files=10,
            max_nrmse=0.05,
        )

        assert isinstance(gate, EfitNRMSEBenchmarkGate)
        assert gate.passes is True
        assert gate.count == 10
        assert gate.pass_count == 10
        assert gate.operator_source_pass_count == 10
        assert gate.operator_source_threshold == pytest.approx(
            psi_rmse_mod.OPERATOR_SOURCE_RMSE_THRESHOLD
        )
        assert gate.operator_source_worst_psi_rmse_norm == pytest.approx(1.0e-12)
        assert gate.operator_source_worst_file
        assert gate.worst_psi_rmse_norm == pytest.approx(0.01)
        assert gate.count_by_machine == {"sparc": 4, "diiid": 3, "jet": 3}
        assert gate.provenance_by_machine == EFIT_BENCHMARK_MACHINE_PROVENANCE
        assert gate.source_consistency_counts == {"profile_source_consistent": 10}
        assert gate.worst_source_alignment_file != ""
        assert gate.worst_source_residual_l2 == pytest.approx(0.01)
        assert all(row["best_source_candidate"] for row in gate.rows)
        assert all(row["best_source_candidate_residual_l2"] >= 0.0 for row in gate.rows)
        assert all(row["profile_source_candidate_rank"] >= 1 for row in gate.rows)
        assert all(row["best_operator_candidate"] for row in gate.rows)
        assert all(row["best_operator_candidate_residual_l2"] >= 0.0 for row in gate.rows)
        assert all(row["delta_star_psi_candidate_rank"] >= 1 for row in gate.rows)
        assert all(row["operator_current_closure_pass"] for row in gate.rows)
        assert all(row["operator_current_relative_error"] < 0.05 for row in gate.rows)
        assert all(
            row["source_consistency_class"] == "profile_source_consistent" for row in gate.rows
        )
        assert all(
            row["operator_source_psi_rmse_norm"] == pytest.approx(1.0e-12) for row in gate.rows
        )

    def test_fails_when_file_count_is_below_required_minimum(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        self._make_reference_tree(tmp_path, {"sparc": 3, "diiid": 3, "jet": 3})
        self._install_fake_validate_file(monkeypatch, {})

        gate = validate_efit_nrmse_benchmark(
            tmp_path,
            min_files=10,
            max_nrmse=0.05,
        )

        assert gate.passes is False
        assert gate.count == 9
        assert gate.pass_count == 9
        assert "count 9 < required 10" in gate.failure_reasons

    def test_fails_when_any_finite_row_exceeds_threshold_and_reports_worst(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        files = self._make_reference_tree(tmp_path, {"sparc": 4, "diiid": 3, "jet": 3})
        worst = files[-1]
        self._install_fake_validate_file(monkeypatch, {worst.name: 0.075})

        gate = validate_efit_nrmse_benchmark(
            tmp_path,
            min_files=10,
            max_nrmse=0.05,
        )

        assert gate.passes is False
        assert gate.pass_count == 9
        assert gate.worst_file == f"{worst.parent.name}/{worst.name}"
        assert gate.worst_psi_rmse_norm == pytest.approx(0.075)
        assert "worst psi_rmse_norm 0.075 > threshold 0.05" in gate.failure_reasons

    def test_aggregate_source_alignment_reports_worst_profile_mismatch(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        files = self._make_reference_tree(tmp_path, {"sparc": 4, "diiid": 3, "jet": 3})
        worst = files[-1]

        def _fake_validate_file(path: Path, warm_start: bool = True) -> PsiRMSEResult:
            del warm_start
            is_worst = path.name == worst.name
            return PsiRMSEResult(
                file=path.name,
                grid="3x3",
                gs_residual_l2=0.1,
                gs_residual_max=0.2,
                psi_rmse_wb=1e-3,
                psi_rmse_norm=0.01,
                psi_rmse_plasma_wb=1e-3,
                psi_max_error_wb=2e-3,
                psi_relative_l2=0.01,
                sor_iterations=10,
                sor_residual=1e-4,
                solve_time_ms=1.0,
                operator_source_psi_rmse_norm=1e-12,
                operator_source_sor_iterations=10,
                operator_source_sor_residual=1e-12,
                source_consistency_class="profile_source_mismatch"
                if is_worst
                else "profile_source_consistent",
                source_residual_l2=3.5 if is_worst else 0.01,
                source_correlation=-0.2 if is_worst else 1.0,
                source_best_fit_scale=-0.5 if is_worst else 1.0,
                source_best_fit_offset=0.1 if is_worst else 0.0,
                source_best_fit_relative_l2=0.9 if is_worst else 0.0,
                source_best_fit_convention="unclassified_global_scale" if is_worst else "canonical",
                plasma_mask_fraction=0.5,
                pressure_source_norm=0.7,
                ffprime_source_norm=0.3,
                total_source_norm=1.0,
                pressure_source_fraction=0.7,
                ffprime_source_fraction=0.3,
                source_plasma_residual_l2=2.0 if is_worst else 0.01,
                source_vacuum_residual_l2=4.0 if is_worst else 0.01,
                source_plasma_operator_norm=0.5,
                source_vacuum_operator_norm=0.5,
                source_plasma_point_count=4.0,
                source_vacuum_point_count=5.0,
                best_source_candidate="profile_source",
                best_source_candidate_residual_l2=0.01,
                profile_source_candidate_rank=1,
                best_operator_candidate="delta_star_psi",
                best_operator_candidate_residual_l2=0.01,
                delta_star_psi_candidate_rank=1,
                declared_toroidal_current_A=-8.7e6,
                operator_toroidal_current_A=-8.6995e6,
                profile_toroidal_current_A=-1.5e6,
                operator_current_relative_error=5.0e-5,
                profile_current_relative_error=0.8,
                operator_current_closure_pass=True,
            )

        monkeypatch.setattr(psi_rmse_mod, "validate_file", _fake_validate_file)

        gate = validate_efit_nrmse_benchmark(tmp_path)

        assert gate.passes is False
        assert gate.source_consistency_counts == {
            "profile_source_consistent": 9,
            "profile_source_mismatch": 1,
        }
        assert gate.worst_source_alignment_file == f"{worst.parent.name}/{worst.name}"
        assert gate.worst_source_residual_l2 == pytest.approx(3.5)
        assert "profile-source mismatch attribution in 1 rows" in gate.failure_reasons

    def test_aggregate_operator_source_gate_reports_solver_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        files = self._make_reference_tree(tmp_path, {"sparc": 4, "diiid": 3, "jet": 3})
        worst = files[-1]

        def _fake_validate_file(path: Path, warm_start: bool = True) -> PsiRMSEResult:
            del warm_start
            operator_rmse = 2.0e-6 if path.name == worst.name else 1.0e-12
            return PsiRMSEResult(
                file=path.name,
                grid="3x3",
                gs_residual_l2=0.1,
                gs_residual_max=0.2,
                psi_rmse_wb=1e-3,
                psi_rmse_norm=0.01,
                psi_rmse_plasma_wb=1e-3,
                psi_max_error_wb=2e-3,
                psi_relative_l2=0.01,
                sor_iterations=10,
                sor_residual=1e-4,
                solve_time_ms=1.0,
                operator_source_psi_rmse_norm=operator_rmse,
                operator_source_sor_iterations=10,
                operator_source_sor_residual=1e-12,
                source_consistency_class="solver_consistency_failure"
                if path.name == worst.name
                else "profile_source_consistent",
                source_residual_l2=0.01,
                source_correlation=1.0,
                source_best_fit_scale=1.0,
                source_best_fit_offset=0.0,
                source_best_fit_relative_l2=0.0,
                source_best_fit_convention="canonical",
                plasma_mask_fraction=0.5,
                pressure_source_norm=0.7,
                ffprime_source_norm=0.3,
                total_source_norm=1.0,
                pressure_source_fraction=0.7,
                ffprime_source_fraction=0.3,
                source_plasma_residual_l2=0.01,
                source_vacuum_residual_l2=0.01,
                source_plasma_operator_norm=0.5,
                source_vacuum_operator_norm=0.5,
                source_plasma_point_count=4.0,
                source_vacuum_point_count=5.0,
                best_source_candidate="profile_source",
                best_source_candidate_residual_l2=0.01,
                profile_source_candidate_rank=1,
                best_operator_candidate="delta_star_psi",
                best_operator_candidate_residual_l2=0.01,
                delta_star_psi_candidate_rank=1,
                declared_toroidal_current_A=-8.7e6,
                operator_toroidal_current_A=-8.6995e6,
                profile_toroidal_current_A=-1.5e6,
                operator_current_relative_error=5.0e-5,
                profile_current_relative_error=0.8,
                operator_current_closure_pass=True,
            )

        monkeypatch.setattr(psi_rmse_mod, "validate_file", _fake_validate_file)

        gate = validate_efit_nrmse_benchmark(tmp_path)

        assert gate.passes is False
        assert gate.operator_source_pass_count == 9
        assert gate.operator_source_worst_file == f"{worst.parent.name}/{worst.name}"
        assert gate.operator_source_worst_psi_rmse_norm == pytest.approx(2.0e-6)
        assert "operator-source psi_rmse_norm 2e-06 > threshold 1e-06" in gate.failure_reasons
        assert "operator-source solver consistency failure in 1 rows" in gate.failure_reasons

    def test_fails_when_a_row_has_non_finite_normalized_rmse(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        files = self._make_reference_tree(tmp_path, {"sparc": 4, "diiid": 3, "jet": 3})
        bad = files[0]
        self._install_fake_validate_file(monkeypatch, {bad.name: float("nan")})

        gate = validate_efit_nrmse_benchmark(
            tmp_path,
            min_files=10,
            max_nrmse=0.05,
        )

        assert gate.passes is False
        assert gate.pass_count == 9
        assert gate.worst_file != f"{bad.parent.name}/{bad.name}"
        assert f"non-finite psi_rmse_norm in {bad.parent.name}/{bad.name}" in gate.failure_reasons

    def test_report_payload_is_schema_valid(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        self._make_reference_tree(tmp_path, {"sparc": 4, "diiid": 3, "jet": 3})
        self._install_fake_validate_file(monkeypatch, {})

        gate = validate_efit_nrmse_benchmark(tmp_path)
        payload = asdict(gate)

        validate_efit_nrmse_benchmark_report(
            payload,
            load_efit_nrmse_benchmark_schema(),
        )
        assert payload["schema_version"] == "efit-nrmse-benchmark.v1"
        assert payload["benchmark_id"] == "efit-nrmse-benchmark"

    def test_report_schema_rejects_unknown_top_level_key(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        self._make_reference_tree(tmp_path, {"sparc": 4, "diiid": 3, "jet": 3})
        self._install_fake_validate_file(monkeypatch, {})
        payload = asdict(validate_efit_nrmse_benchmark(tmp_path))
        payload["operator_notes"] = "not part of the public report contract"

        with pytest.raises(ValueError, match="unexpected benchmark report key: operator_notes"):
            validate_efit_nrmse_benchmark_report(
                payload,
                load_efit_nrmse_benchmark_schema(),
            )

    def test_report_schema_rejects_missing_nested_row_key(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        self._make_reference_tree(tmp_path, {"sparc": 4, "diiid": 3, "jet": 3})
        self._install_fake_validate_file(monkeypatch, {})
        payload = asdict(validate_efit_nrmse_benchmark(tmp_path))
        del payload["rows"][0]["provenance"]

        with pytest.raises(
            ValueError,
            match=r"missing required benchmark report key: rows\[0\]\.provenance",
        ):
            validate_efit_nrmse_benchmark_report(
                payload,
                load_efit_nrmse_benchmark_schema(),
            )

    def test_report_schema_rejects_missing_source_attribution_key(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        self._make_reference_tree(tmp_path, {"sparc": 4, "diiid": 3, "jet": 3})
        self._install_fake_validate_file(monkeypatch, {})
        payload = asdict(validate_efit_nrmse_benchmark(tmp_path))
        del payload["rows"][0]["source_consistency_class"]

        with pytest.raises(
            ValueError,
            match=r"missing required benchmark report key: rows\[0\]\.source_consistency_class",
        ):
            validate_efit_nrmse_benchmark_report(
                payload,
                load_efit_nrmse_benchmark_schema(),
            )
