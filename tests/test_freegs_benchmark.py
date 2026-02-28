#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — FreeGS / Solov'ev Benchmark Tests
# (c) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for validation/benchmark_vs_freegs.py.

The Solov'ev analytic comparison runs unconditionally (no FreeGS
dependency).  FreeGS-dependent tests are skipped when the package
is not installed.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

# ── Dynamic import of the benchmark module ───────────────────────────

MODULE_PATH = ROOT / "validation" / "benchmark_vs_freegs.py"
_spec = importlib.util.spec_from_file_location("benchmark_vs_freegs", MODULE_PATH)
assert _spec is not None and _spec.loader is not None
benchmark_vs_freegs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(benchmark_vs_freegs)

# Re-export key symbols for convenience
solovev_psi = benchmark_vs_freegs.solovev_psi
solovev_jphi = benchmark_vs_freegs.solovev_jphi
nrmse = benchmark_vs_freegs.nrmse
TokamakCase = benchmark_vs_freegs.TokamakCase
CASES = benchmark_vs_freegs.CASES
build_config = benchmark_vs_freegs.build_config
run_solovev_case = benchmark_vs_freegs.run_solovev_case
compare_case = benchmark_vs_freegs.compare_case
run_solovev_benchmark = benchmark_vs_freegs.run_solovev_benchmark
HAS_FREEGS = benchmark_vs_freegs.HAS_FREEGS
PSI_NRMSE_THRESHOLD = benchmark_vs_freegs.PSI_NRMSE_THRESHOLD


# ── Solov'ev analytic tests (always run) ─────────────────────────────


class TestSolovevAnalytic:
    """Verify properties of the Solov'ev analytic equilibrium."""

    @pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
    def test_solovev_psi_vanishes_at_outer_midplane(self, case: TokamakCase) -> None:
        """Psi should be ~0 at the outer midplane boundary (R0+a, 0)."""
        cfg = build_config(case)
        dims = cfg["dimensions"]
        R_1d = np.linspace(dims["R_min"], dims["R_max"], case.NR)
        Z_1d = np.linspace(dims["Z_min"], dims["Z_max"], case.NZ)
        RR, ZZ = np.meshgrid(R_1d, Z_1d)

        psi = solovev_psi(RR, ZZ, case.R0, case.a, case.kappa)

        # Find grid point closest to (R0+a, 0)
        ir_bdy = int(np.argmin(np.abs(R_1d - (case.R0 + case.a))))
        iz_mid = case.NZ // 2

        psi_range = float(np.max(psi) - np.min(psi))
        if psi_range > 1e-12:
            psi_bdy_norm = abs(psi[iz_mid, ir_bdy]) / psi_range
            assert psi_bdy_norm < 0.15, (
                f"Psi at outer midplane boundary too large: "
                f"{psi_bdy_norm:.4f} (normalised)"
            )

    @pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
    def test_solovev_psi_has_maximum_inside_plasma(self, case: TokamakCase) -> None:
        """Psi maximum should be near (R0, 0), i.e. inside the plasma."""
        cfg = build_config(case)
        dims = cfg["dimensions"]
        R_1d = np.linspace(dims["R_min"], dims["R_max"], case.NR)
        Z_1d = np.linspace(dims["Z_min"], dims["Z_max"], case.NZ)
        RR, ZZ = np.meshgrid(R_1d, Z_1d)

        psi = solovev_psi(RR, ZZ, case.R0, case.a, case.kappa)
        idx_max = int(np.argmax(psi))
        iz_ax, ir_ax = np.unravel_index(idx_max, psi.shape)
        R_axis = float(R_1d[ir_ax])
        Z_axis = float(Z_1d[iz_ax])

        # Axis should be within ~a of R0
        assert abs(R_axis - case.R0) < 1.5 * case.a, (
            f"Magnetic axis R={R_axis:.3f} too far from R0={case.R0}"
        )
        assert abs(Z_axis) < case.kappa * case.a, (
            f"Magnetic axis Z={Z_axis:.3f} too far from midplane"
        )

    @pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
    def test_solovev_psi_is_finite(self, case: TokamakCase) -> None:
        """Psi field should contain no NaN or Inf."""
        cfg = build_config(case)
        dims = cfg["dimensions"]
        R_1d = np.linspace(dims["R_min"], dims["R_max"], case.NR)
        Z_1d = np.linspace(dims["Z_min"], dims["Z_max"], case.NZ)
        RR, ZZ = np.meshgrid(R_1d, Z_1d)

        psi = solovev_psi(RR, ZZ, case.R0, case.a, case.kappa)
        assert np.all(np.isfinite(psi)), "Solov'ev Psi contains NaN/Inf"

    def test_solovev_jphi_interior_nonzero(self) -> None:
        """Analytic j_phi should be non-trivially nonzero inside the plasma."""
        case = CASES[0]  # ITER-like
        cfg = build_config(case)
        dims = cfg["dimensions"]
        R_1d = np.linspace(dims["R_min"], dims["R_max"], case.NR)
        Z_1d = np.linspace(dims["Z_min"], dims["Z_max"], case.NZ)
        RR, ZZ = np.meshgrid(R_1d, Z_1d)

        jphi = solovev_jphi(RR, ZZ, case.R0, case.a, case.kappa)
        assert np.all(np.isfinite(jphi)), "j_phi contains NaN/Inf"
        # Interior should have significant current density
        interior = jphi[2:-2, 2:-2]
        assert float(np.max(np.abs(interior))) > 1e-6, (
            "j_phi is essentially zero everywhere"
        )


class TestNRMSEUtility:
    """Verify the NRMSE helper."""

    def test_identical_arrays_give_zero(self) -> None:
        a = np.random.default_rng(42).standard_normal(100)
        assert nrmse(a, a) == pytest.approx(0.0, abs=1e-12)

    def test_nrmse_scales_with_noise(self) -> None:
        rng = np.random.default_rng(7)
        base = np.linspace(0, 10, 200)
        noisy = base + rng.normal(0, 0.1, base.shape)
        val = nrmse(base, noisy)
        assert 0.0 < val < 0.05  # ~1% noise on range 10

    def test_nrmse_constant_array_safe(self) -> None:
        """If y_true is constant, NRMSE should not produce Inf/NaN."""
        a = np.ones(50)
        b = a + 0.01
        result = nrmse(a, b)
        assert np.isfinite(result)


class TestBuildConfig:
    """Config builder produces valid FusionKernel configs."""

    @pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
    def test_config_has_required_keys(self, case: TokamakCase) -> None:
        cfg = build_config(case)
        assert "reactor_name" in cfg
        assert "grid_resolution" in cfg
        assert "dimensions" in cfg
        assert "physics" in cfg
        assert "coils" in cfg
        assert "solver" in cfg

    @pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
    def test_config_R_range_positive(self, case: TokamakCase) -> None:
        cfg = build_config(case)
        assert cfg["dimensions"]["R_min"] > 0
        assert cfg["dimensions"]["R_max"] > cfg["dimensions"]["R_min"]


class TestSolovevReference:
    """Verify run_solovev_case returns well-formed data."""

    @pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
    def test_returns_expected_keys(self, case: TokamakCase) -> None:
        result = run_solovev_case(case)
        for key in ("psi", "R", "Z", "R_axis", "Z_axis", "q_proxy"):
            assert key in result, f"Missing key: {key}"

    @pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
    def test_psi_shape_matches_grid(self, case: TokamakCase) -> None:
        result = run_solovev_case(case)
        assert result["psi"].shape == (case.NZ, case.NR)


# ── Solov'ev benchmark integration test ──────────────────────────────


class TestSolovevBenchmarkIntegration:
    """End-to-end test of the Solov'ev fallback benchmark."""

    def test_run_solovev_benchmark_produces_valid_report(self) -> None:
        """Full Solov'ev benchmark runs without error and returns a report."""
        report = run_solovev_benchmark()

        assert "cases" in report
        assert "overall_psi_nrmse" in report
        assert "passes" in report
        assert report["mode"] == "solovev_manufactured_source"
        assert len(report["cases"]) == len(CASES)

        # Each case should have expected fields
        for case_result in report["cases"]:
            assert "name" in case_result
            assert "psi_nrmse" in case_result
            assert "passes" in case_result
            assert "comparison_backend" in case_result
            assert "reference_backend" in case_result

    def test_report_is_json_serialisable(self) -> None:
        """Report can be serialised to JSON without error."""
        report = run_solovev_benchmark()
        text = json.dumps(report, indent=2, default=str)
        assert len(text) > 100
        # Round-trip
        parsed = json.loads(text)
        assert parsed["mode"] == "solovev_manufactured_source"


# ── FreeGS-dependent tests (skipped when not installed) ──────────────


@pytest.mark.skipif(not HAS_FREEGS, reason="freegs not installed")
class TestFreeGSComparison:
    """These tests only run when FreeGS is available."""

    def test_freegs_case_returns_expected_keys(self) -> None:
        from benchmark_vs_freegs import run_freegs_case

        case = CASES[0]
        result = run_freegs_case(case)
        for key in ("psi", "R", "Z", "R_axis", "Z_axis", "q_proxy"):
            assert key in result

    def test_full_freegs_benchmark(self) -> None:
        from benchmark_vs_freegs import run_benchmark

        report = run_benchmark(force_solovev=False)
        assert report["mode"] == "freegs"
        assert len(report["cases"]) == len(CASES)
