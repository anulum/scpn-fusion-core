"""Tests for P2.2: Enhanced FreeGS / Solov'ev benchmark."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "validation"))

from benchmark_vs_freegs import (
    FREEGS_AXIS_ERROR_M,
    FREEGS_PSI_NRMSE_THRESHOLD,
    FREEGS_Q_NRMSE_THRESHOLD,
    FREEGS_SEPARATRIX_NRMSE,
    PSI_NRMSE_THRESHOLD,
    CASES,
    TokamakCase,
    compare_case,
    compare_separatrix,
    estimate_axis_pressure_pa,
    generate_per_metric_report,
    nrmse,
    run_benchmark,
    solovev_psi,
)


# ── Threshold constants ────────────────────────────────────────────────


class TestThresholds:
    def test_freegs_psi_tighter_than_solovev(self):
        assert FREEGS_PSI_NRMSE_THRESHOLD < PSI_NRMSE_THRESHOLD

    def test_freegs_psi_threshold_value(self):
        assert FREEGS_PSI_NRMSE_THRESHOLD == pytest.approx(0.005)

    def test_freegs_q_threshold_value(self):
        assert FREEGS_Q_NRMSE_THRESHOLD == pytest.approx(0.10)

    def test_freegs_axis_error_value(self):
        assert FREEGS_AXIS_ERROR_M == pytest.approx(0.10)

    def test_freegs_separatrix_value(self):
        assert FREEGS_SEPARATRIX_NRMSE == pytest.approx(0.05)


class TestFreeGSAxisPressureEstimate:
    def test_pressure_is_positive_and_finite(self):
        for case in CASES:
            p_axis = estimate_axis_pressure_pa(case)
            assert np.isfinite(p_axis)
            assert p_axis > 0.0

    def test_pressure_scales_up_with_higher_field(self):
        low_field = TokamakCase("lowB", R0=1.8, a=0.5, B0=2.0, Ip=2.0, kappa=1.8)
        high_field = TokamakCase("highB", R0=1.8, a=0.5, B0=8.0, Ip=2.0, kappa=1.8)
        assert estimate_axis_pressure_pa(high_field) > estimate_axis_pressure_pa(low_field)


class TestFreeGSPassContract:
    def test_freegs_pass_contract_enforces_separatrix_gate(self, monkeypatch):
        R = np.linspace(4.0, 8.0, 8)
        Z = np.linspace(-2.0, 2.0, 8)
        RR, ZZ = np.meshgrid(R, Z)
        psi = solovev_psi(RR, ZZ, R0=6.2, a=1.8, kappa=1.7)
        q_proxy = np.linspace(1.0, 3.0, len(R))

        def _our_stub(_case):  # type: ignore[no-untyped-def]
            return {
                "psi": psi,
                "R": R,
                "Z": Z,
                "RR": RR,
                "ZZ": ZZ,
                "R_axis": 6.2,
                "Z_axis": 0.0,
                "q_proxy": q_proxy,
                "converged": True,
                "residual": 0.0,
            }

        def _freegs_stub(_case):  # type: ignore[no-untyped-def]
            return {
                "psi": psi.copy(),
                "R": R.copy(),
                "Z": Z.copy(),
                "R_axis": 6.2,
                "Z_axis": 0.0,
                "q_proxy": q_proxy.copy(),
                "axis_pressure_pa": 1.0e5,
            }

        monkeypatch.setattr("benchmark_vs_freegs.run_our_solver", _our_stub)
        monkeypatch.setattr("benchmark_vs_freegs.run_freegs_case", _freegs_stub)
        monkeypatch.setattr(
            "benchmark_vs_freegs.compare_separatrix",
            lambda *_args, **_kwargs: FREEGS_SEPARATRIX_NRMSE + 0.01,
        )

        result = compare_case(CASES[0], use_freegs=True)
        assert result["psi_nrmse"] < FREEGS_PSI_NRMSE_THRESHOLD
        assert result["q_profile_nrmse"] < FREEGS_Q_NRMSE_THRESHOLD
        assert result["axis_error_m"] < FREEGS_AXIS_ERROR_M
        assert result["separatrix_nrmse"] > FREEGS_SEPARATRIX_NRMSE
        assert result["passes"] is False

    def test_freegs_report_includes_mode_specific_thresholds(self, monkeypatch):
        monkeypatch.setattr("benchmark_vs_freegs.HAS_FREEGS", True)

        def _compare_case_stub(case, *, use_freegs=False):  # type: ignore[no-untyped-def]
            return {
                "name": case.name,
                "mode": "freegs" if use_freegs else "solovev_manufactured_source",
                "comparison_backend": "fusion_kernel_nonlinear",
                "reference_backend": "freegs",
                "psi_nrmse": 0.001,
                "q_profile_nrmse": 0.02,
                "axis_error_m": 0.01,
                "separatrix_nrmse": 0.01,
                "our_converged": True,
                "our_residual": 1e-6,
                "passes": True,
            }

        monkeypatch.setattr("benchmark_vs_freegs.compare_case", _compare_case_stub)
        report = run_benchmark(force_solovev=False)

        assert report["mode"] == "freegs"
        assert report["psi_nrmse_threshold"] == pytest.approx(FREEGS_PSI_NRMSE_THRESHOLD)
        assert report["thresholds"]["q_profile_nrmse"] == pytest.approx(FREEGS_Q_NRMSE_THRESHOLD)
        assert report["thresholds"]["axis_error_m"] == pytest.approx(FREEGS_AXIS_ERROR_M)
        assert report["thresholds"]["separatrix_nrmse"] == pytest.approx(FREEGS_SEPARATRIX_NRMSE)


# ── compare_separatrix ─────────────────────────────────────────────────


class TestCompareSeparatrix:
    def test_identical_psi_gives_zero(self):
        R = np.linspace(4.0, 8.0, 65)
        Z = np.linspace(-3.0, 3.0, 65)
        RR, ZZ = np.meshgrid(R, Z)
        psi = solovev_psi(RR, ZZ, R0=6.2, a=2.0, kappa=1.7)
        result = compare_separatrix(psi, psi, R, Z)
        assert result < 1e-10

    def test_different_psi_nonzero(self):
        R = np.linspace(4.0, 8.0, 65)
        Z = np.linspace(-3.0, 3.0, 65)
        RR, ZZ = np.meshgrid(R, Z)
        psi1 = solovev_psi(RR, ZZ, R0=6.2, a=2.0, kappa=1.7)
        psi2 = solovev_psi(RR, ZZ, R0=6.2, a=1.8, kappa=1.7)  # different a
        result = compare_separatrix(psi1, psi2, R, Z)
        assert result > 0.0

    def test_degenerate_returns_one(self):
        R = np.linspace(4.0, 8.0, 10)
        Z = np.linspace(-3.0, 3.0, 10)
        # Flat psi
        psi = np.ones((10, 10))
        result = compare_separatrix(psi, psi, R, Z)
        assert result == 1.0

    def test_result_bounded(self):
        R = np.linspace(4.0, 8.0, 65)
        Z = np.linspace(-3.0, 3.0, 65)
        RR, ZZ = np.meshgrid(R, Z)
        psi1 = solovev_psi(RR, ZZ, R0=6.2, a=2.0, kappa=1.7)
        psi2 = solovev_psi(RR, ZZ, R0=6.2, a=2.0, kappa=2.5)
        result = compare_separatrix(psi1, psi2, R, Z)
        assert 0.0 <= result <= 2.0  # NRMSE can exceed 1 but should be bounded


# ── generate_per_metric_report ─────────────────────────────────────────


class TestPerMetricReport:
    def test_generates_markdown(self):
        report = {
            "cases": [
                {
                    "name": "Test-case",
                    "mode": "solovev",
                    "psi_nrmse": 0.05,
                    "q_profile_nrmse": 0.08,
                    "axis_error_m": 0.02,
                    "separatrix_nrmse": 0.03,
                    "passes": True,
                }
            ]
        }
        md = generate_per_metric_report(report)
        assert "## FreeGS / Solov'ev Benchmark" in md
        assert "Test-case" in md
        assert "PASS" in md
        assert "0.0500" in md

    def test_empty_cases(self):
        report = {"cases": []}
        md = generate_per_metric_report(report)
        assert "Per-Metric Report" in md

    def test_fail_status(self):
        report = {
            "cases": [
                {
                    "name": "Fail-case",
                    "mode": "freegs",
                    "psi_nrmse": 0.05,
                    "q_profile_nrmse": 0.20,
                    "axis_error_m": 0.15,
                    "separatrix_nrmse": 0.10,
                    "passes": False,
                }
            ]
        }
        md = generate_per_metric_report(report)
        assert "FAIL" in md


# ── Solov'ev analytic tests ────────────────────────────────────────────


class TestSolovevAnalytic:
    def test_psi_nonnegative(self):
        R = np.linspace(4.0, 8.0, 65)
        Z = np.linspace(-3.0, 3.0, 65)
        RR, ZZ = np.meshgrid(R, Z)
        psi = solovev_psi(RR, ZZ, R0=6.2, a=2.0, kappa=1.7)
        assert np.all(psi >= 0.0)

    def test_psi_max_near_axis(self):
        R = np.linspace(4.0, 8.0, 65)
        Z = np.linspace(-3.0, 3.0, 65)
        RR, ZZ = np.meshgrid(R, Z)
        psi = solovev_psi(RR, ZZ, R0=6.2, a=2.0, kappa=1.7)
        idx = np.unravel_index(np.argmax(psi), psi.shape)
        R_ax = R[idx[1]]
        Z_ax = Z[idx[0]]
        assert abs(R_ax - 6.2) < 0.5  # within 50 cm of R0
        assert abs(Z_ax) < 0.5  # near midplane

    def test_psi_zero_outside_plasma(self):
        R = np.linspace(4.0, 8.0, 65)
        Z = np.linspace(-3.0, 3.0, 65)
        RR, ZZ = np.meshgrid(R, Z)
        psi = solovev_psi(RR, ZZ, R0=6.2, a=2.0, kappa=1.7)
        # Far corners should be zero
        assert psi[0, 0] == 0.0
        assert psi[-1, -1] == 0.0
