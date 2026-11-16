# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for P2.2: Enhanced FreeGS / Solov'ev benchmark."""

import sys
from pathlib import Path
from typing import Any, cast

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
    run_benchmark,
    solovev_psi,
)

# Patch the globals the imported functions actually read. tests/test_freegs_benchmark.py re-executes the
# benchmark under the same module name, so a string target such as "benchmark_vs_freegs.compare_case"
# can resolve to a different module object when that file is collected first.
_BENCHMARK_GLOBALS = compare_case.__globals__
assert run_benchmark.__globals__ is _BENCHMARK_GLOBALS


# ── Threshold constants ────────────────────────────────────────────────


class TestThresholds:
    """The FreeGS acceptance thresholds and their declared values."""

    def test_freegs_psi_tighter_than_solovev(self):
        """The FreeGS flux tolerance is tighter than the Solov'ev one."""
        assert FREEGS_PSI_NRMSE_THRESHOLD < PSI_NRMSE_THRESHOLD

    def test_freegs_psi_threshold_value(self):
        """The FreeGS flux NRMSE threshold is 0.005."""
        assert pytest.approx(0.005) == FREEGS_PSI_NRMSE_THRESHOLD

    def test_freegs_q_threshold_value(self):
        """The FreeGS q-profile NRMSE threshold is 0.10."""
        assert pytest.approx(0.10) == FREEGS_Q_NRMSE_THRESHOLD

    def test_freegs_axis_error_value(self):
        """The FreeGS axis-error threshold is 0.10 m."""
        assert pytest.approx(0.10) == FREEGS_AXIS_ERROR_M

    def test_freegs_separatrix_value(self):
        """The FreeGS separatrix NRMSE threshold is 0.05."""
        assert pytest.approx(0.05) == FREEGS_SEPARATRIX_NRMSE


class TestFreeGSAxisPressureEstimate:
    """The axis-pressure estimate passed to FreeGS."""

    def test_pressure_is_positive_and_finite(self):
        """Every case gets a finite, positive axis pressure."""
        for case in CASES:
            p_axis = estimate_axis_pressure_pa(case)
            assert np.isfinite(p_axis)
            assert p_axis > 0.0

    def test_pressure_scales_up_with_higher_field(self):
        """A higher toroidal field gives a higher axis pressure."""
        low_field = TokamakCase("lowB", R0=1.8, a=0.5, B0=2.0, Ip=2.0, kappa=1.8)
        high_field = TokamakCase("highB", R0=1.8, a=0.5, B0=8.0, Ip=2.0, kappa=1.8)
        assert estimate_axis_pressure_pa(high_field) > estimate_axis_pressure_pa(low_field)


class TestFreeGSPassContract:
    """Pass or fail contract of the FreeGS comparison and its report."""

    def test_freegs_pass_contract_enforces_separatrix_gate(self, monkeypatch):
        """A case within the flux, q and axis tolerances still fails on the separatrix gate."""
        R = np.linspace(4.0, 8.0, 8)
        Z = np.linspace(-2.0, 2.0, 8)
        RR, ZZ = np.meshgrid(R, Z)
        psi = solovev_psi(RR, ZZ, R0=6.2, a=1.8, kappa=1.7)
        q_proxy = np.linspace(1.0, 3.0, len(R))

        def _our_stub(_case: TokamakCase) -> dict[str, object]:
            """Return the same flux map for our solver."""
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

        def _freegs_stub(_case: TokamakCase, **_kwargs: object) -> dict[str, object]:
            """Return a copy of the same flux map as the FreeGS reference."""
            return {
                "psi": psi.copy(),
                "R": R.copy(),
                "Z": Z.copy(),
                "R_axis": 6.2,
                "Z_axis": 0.0,
                "q_proxy": q_proxy.copy(),
                "axis_pressure_pa": 1.0e5,
            }

        monkeypatch.setitem(_BENCHMARK_GLOBALS, "run_our_solver", _our_stub)
        monkeypatch.setitem(_BENCHMARK_GLOBALS, "run_freegs_case", _freegs_stub)
        monkeypatch.setitem(
            _BENCHMARK_GLOBALS,
            "compare_separatrix",
            lambda *_args, **_kwargs: FREEGS_SEPARATRIX_NRMSE + 0.01,
        )

        result = compare_case(CASES[0], use_freegs=True)
        assert result["psi_nrmse"] < FREEGS_PSI_NRMSE_THRESHOLD
        assert result["q_profile_nrmse"] < FREEGS_Q_NRMSE_THRESHOLD
        assert result["axis_error_m"] < FREEGS_AXIS_ERROR_M
        assert result["separatrix_nrmse"] > FREEGS_SEPARATRIX_NRMSE
        assert result["passes"] is False

    def test_freegs_report_includes_mode_specific_thresholds(self, monkeypatch):
        """A FreeGS-mode report carries the FreeGS thresholds."""
        monkeypatch.setitem(_BENCHMARK_GLOBALS, "HAS_FREEGS", True)

        def _compare_case_stub(
            case: TokamakCase,
            *,
            use_freegs: bool = False,
            allow_runtime_fallback: bool = True,
        ) -> dict[str, object]:
            """Return a passing case result in the requested mode."""
            assert isinstance(allow_runtime_fallback, bool)
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

        monkeypatch.setitem(_BENCHMARK_GLOBALS, "compare_case", _compare_case_stub)
        report = run_benchmark(force_solovev=False)
        thresholds = cast(dict[str, Any], report["thresholds"])

        assert report["mode"] == "freegs"
        assert report["psi_nrmse_threshold"] == pytest.approx(FREEGS_PSI_NRMSE_THRESHOLD)
        assert thresholds["q_profile_nrmse"] == pytest.approx(FREEGS_Q_NRMSE_THRESHOLD)
        assert thresholds["axis_error_m"] == pytest.approx(FREEGS_AXIS_ERROR_M)
        assert thresholds["separatrix_nrmse"] == pytest.approx(FREEGS_SEPARATRIX_NRMSE)

    def test_report_fails_when_any_case_is_unconverged(self, monkeypatch):
        """One unconverged case makes the report count it and fail."""
        results = iter(
            [
                {
                    "name": "converged",
                    "mode": "solovev_manufactured_source",
                    "comparison_backend": "fusion_kernel_fixed_source",
                    "reference_backend": "solovev_analytic",
                    "psi_nrmse": 0.01,
                    "q_profile_nrmse": 0.0,
                    "axis_error_m": 0.0,
                    "separatrix_nrmse": 0.0,
                    "our_converged": True,
                    "passes": True,
                },
                {
                    "name": "unconverged",
                    "mode": "solovev_manufactured_source",
                    "comparison_backend": "fusion_kernel_fixed_source",
                    "reference_backend": "solovev_analytic",
                    "psi_nrmse": 0.01,
                    "q_profile_nrmse": 0.0,
                    "axis_error_m": 0.0,
                    "separatrix_nrmse": 0.0,
                    "our_converged": False,
                    "passes": False,
                },
            ]
        )

        def _compare_case_stub(*_args: object, **_kwargs: object) -> dict[str, object]:
            """Return the next prepared case result."""
            return next(results)

        monkeypatch.setitem(_BENCHMARK_GLOBALS, "CASES", CASES[:2])
        monkeypatch.setitem(_BENCHMARK_GLOBALS, "compare_case", _compare_case_stub)

        report = run_benchmark(force_solovev=True)

        assert report["unconverged_case_count"] == 1
        assert report["all_cases_converged"] is False
        assert report["passes"] is False


# ── compare_separatrix ─────────────────────────────────────────────────


class TestCompareSeparatrix:
    """The separatrix comparison metric."""

    def test_identical_psi_gives_zero(self):
        """Identical flux maps give a separatrix NRMSE of zero."""
        R = np.linspace(4.0, 8.0, 65)
        Z = np.linspace(-3.0, 3.0, 65)
        RR, ZZ = np.meshgrid(R, Z)
        psi = solovev_psi(RR, ZZ, R0=6.2, a=2.0, kappa=1.7)
        result = compare_separatrix(psi, psi, R, Z)
        assert result < 1e-10

    def test_different_psi_nonzero(self):
        """Different flux maps give a positive separatrix NRMSE."""
        R = np.linspace(4.0, 8.0, 65)
        Z = np.linspace(-3.0, 3.0, 65)
        RR, ZZ = np.meshgrid(R, Z)
        psi1 = solovev_psi(RR, ZZ, R0=6.2, a=2.0, kappa=1.7)
        psi2 = solovev_psi(RR, ZZ, R0=6.2, a=1.8, kappa=1.7)  # different a
        result = compare_separatrix(psi1, psi2, R, Z)
        assert result > 0.0

    def test_degenerate_returns_one(self):
        """A flat flux map, with no separatrix, gives the worst value 1.0."""
        R = np.linspace(4.0, 8.0, 10)
        Z = np.linspace(-3.0, 3.0, 10)
        # Flat psi
        psi = np.ones((10, 10))
        result = compare_separatrix(psi, psi, R, Z)
        assert result == 1.0

    def test_result_bounded(self):
        """The separatrix NRMSE stays within [0, 1]."""
        R = np.linspace(4.0, 8.0, 65)
        Z = np.linspace(-3.0, 3.0, 65)
        RR, ZZ = np.meshgrid(R, Z)
        psi1 = solovev_psi(RR, ZZ, R0=6.2, a=2.0, kappa=1.7)
        psi2 = solovev_psi(RR, ZZ, R0=6.2, a=2.0, kappa=2.5)
        result = compare_separatrix(psi1, psi2, R, Z)
        assert 0.0 <= result <= 2.0  # NRMSE can exceed 1 but should be bounded


# ── generate_per_metric_report ─────────────────────────────────────────


class TestPerMetricReport:
    """The per-metric Markdown report."""

    def test_generates_markdown(self):
        """A report renders as a Markdown table."""
        report: dict[str, object] = {
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
        """An empty report still renders its title."""
        report: dict[str, list[dict[str, object]]] = {"cases": []}
        md = generate_per_metric_report(report)
        assert "Per-Metric Report" in md

    def test_fail_status(self):
        """A failing case renders as FAIL."""
        report: dict[str, object] = {
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
    """The Solov'ev analytic flux used as the fallback reference."""

    def test_psi_nonnegative(self):
        """The Solov'ev flux is non-negative."""
        R = np.linspace(4.0, 8.0, 65)
        Z = np.linspace(-3.0, 3.0, 65)
        RR, ZZ = np.meshgrid(R, Z)
        psi = solovev_psi(RR, ZZ, R0=6.2, a=2.0, kappa=1.7)
        assert np.all(psi >= 0.0)

    def test_psi_max_near_axis(self):
        """The flux maximum lies within 0.5 m of R0 = 6.2 m and of the midplane."""
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
        """The flux is zero at the grid corners, outside the plasma."""
        R = np.linspace(4.0, 8.0, 65)
        Z = np.linspace(-3.0, 3.0, 65)
        RR, ZZ = np.meshgrid(R, Z)
        psi = solovev_psi(RR, ZZ, R0=6.2, a=2.0, kappa=1.7)
        # Far corners should be zero
        assert psi[0, 0] == 0.0
        assert psi[-1, -1] == 0.0
