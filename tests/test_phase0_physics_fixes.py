# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Phase 0 Physics Fixes Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Tests for v3.1.0 Phase 0 physics fixes:

P0.1 — TBR realism (port coverage + streaming correction)
P0.2 — Q-scan Greenwald density limit + temperature cap + Q ceiling
P0.6 — Energy conservation enforcement in transport solver
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from scpn_fusion.nuclear.blanket_neutronics import (
    BreedingBlanket,
    MultiGroupBlanket,
    VolumetricBlanketReport,
)
from scpn_fusion.core.fusion_ignition_sim import DynamicBurnModel


# ════════════════════════════════════════════════════════════════════════
# P0.1 — TBR Realism
# ════════════════════════════════════════════════════════════════════════


class TestTBRCorrection:
    """Verify port-coverage and streaming correction factors."""

    def test_volumetric_tbr_correction_applied(self):
        """Default correction factors should reduce TBR by ~32%."""
        blanket = BreedingBlanket(thickness_cm=80.0, li6_enrichment=0.9)
        report = blanket.calculate_volumetric_tbr()  # default resolution
        expected_ratio = 0.80 * 0.85  # port_coverage * streaming
        actual_ratio = report.tbr / report.tbr_ideal
        assert abs(actual_ratio - expected_ratio) < 1e-6, (
            f"Correction ratio {actual_ratio:.6f} != expected {expected_ratio}"
        )

    def test_multigroup_tbr_corrected_in_realistic_range(self):
        """3-group blanket with default corrections should give TBR in [1.0, 1.4]."""
        blanket = MultiGroupBlanket(thickness_cm=80.0, li6_enrichment=0.9)
        result = blanket.solve_transport()
        assert 1.0 <= result["tbr"] <= 1.4, (
            f"Corrected 3-group TBR {result['tbr']:.3f} outside "
            f"Fischer/DEMO range [1.0, 1.4]"
        )

    def test_volumetric_tbr_ideal_unchanged_with_unit_factors(self):
        """When correction factors = 1.0, tbr should equal tbr_ideal."""
        blanket = BreedingBlanket(thickness_cm=80.0, li6_enrichment=0.9)
        report = blanket.calculate_volumetric_tbr(
            port_coverage_factor=1.0, streaming_factor=1.0,
            blanket_fill_factor=1.0,
        )
        assert abs(report.tbr - report.tbr_ideal) < 1e-12

    def test_volumetric_tbr_ideal_field_present(self):
        """VolumetricBlanketReport includes tbr_ideal."""
        blanket = BreedingBlanket(thickness_cm=80.0, li6_enrichment=0.9)
        report = blanket.calculate_volumetric_tbr()
        assert hasattr(report, "tbr_ideal")
        assert report.tbr_ideal > report.tbr  # ideal > corrected

    def test_multigroup_tbr_corrected_above_unity(self):
        """3-group blanket with corrections should still be above 1.0."""
        blanket = MultiGroupBlanket(thickness_cm=80.0, li6_enrichment=0.9)
        result = blanket.solve_transport()
        assert result["tbr"] > 1.0, (
            f"Corrected 3-group TBR {result['tbr']:.3f} below self-sufficiency"
        )

    def test_multigroup_tbr_ideal_field_present(self):
        """3-group result includes tbr_ideal."""
        blanket = MultiGroupBlanket(thickness_cm=80.0, li6_enrichment=0.9)
        result = blanket.solve_transport()
        assert "tbr_ideal" in result
        assert result["tbr_ideal"] > result["tbr"]

    def test_multigroup_tbr_unit_factors_equal_ideal(self):
        """With correction=1.0, corrected TBR equals ideal."""
        blanket = MultiGroupBlanket(thickness_cm=80.0, li6_enrichment=0.9)
        result = blanket.solve_transport(
            port_coverage_factor=1.0, streaming_factor=1.0,
        )
        assert abs(result["tbr"] - result["tbr_ideal"]) < 1e-12

    def test_blanket_fill_factor_reduces_tbr(self):
        """Lower fill factor should reduce corrected TBR."""
        blanket = BreedingBlanket(thickness_cm=80.0, li6_enrichment=0.9)
        full = blanket.calculate_volumetric_tbr(
            radial_cells=8, poloidal_cells=16, toroidal_cells=12,
            blanket_fill_factor=1.0,
        )
        partial = blanket.calculate_volumetric_tbr(
            radial_cells=8, poloidal_cells=16, toroidal_cells=12,
            blanket_fill_factor=0.65,
        )
        assert partial.tbr < full.tbr

    @pytest.mark.parametrize("bad_kwargs,match", [
        ({"port_coverage_factor": 0.0}, "port_coverage_factor"),
        ({"port_coverage_factor": 1.5}, "port_coverage_factor"),
        ({"streaming_factor": -0.1}, "streaming_factor"),
        ({"blanket_fill_factor": 0.0}, "blanket_fill_factor"),
    ])
    def test_volumetric_rejects_invalid_correction_factors(self, bad_kwargs, match):
        blanket = BreedingBlanket(thickness_cm=80.0, li6_enrichment=0.9)
        with pytest.raises(ValueError, match=match):
            blanket.calculate_volumetric_tbr(
                radial_cells=8, poloidal_cells=16, toroidal_cells=12,
                **bad_kwargs,
            )


# ════════════════════════════════════════════════════════════════════════
# P0.2 — Q-scan Greenwald & Temperature Limits
# ════════════════════════════════════════════════════════════════════════


class TestQScanLimits:
    """Verify Greenwald density limit, temperature cap, and Q ceiling."""

    def test_q_ceiling_at_15(self):
        """Q factor must not exceed 15 in dynamic burn model."""
        model = DynamicBurnModel(
            R0=6.2, a=2.0, B_t=5.3, I_p=15.0, n_e20=1.0,
        )
        result = model.simulate(P_aux_mw=10.0, duration_s=50.0, dt_s=0.05)
        assert result["Q_peak"] <= 15.0, (
            f"Q_peak={result['Q_peak']:.1f} exceeds ceiling of 15"
        )

    def test_temperature_capped_at_25_kev(self):
        """Temperature must be capped at 25 keV."""
        model = DynamicBurnModel(
            R0=6.2, a=2.0, B_t=5.3, I_p=15.0, n_e20=1.0,
        )
        result = model.simulate(P_aux_mw=70.0, duration_s=50.0, dt_s=0.05)
        assert max(result["T_keV"]) <= 25.0 + 0.01, (
            f"T_peak={max(result['T_keV']):.1f} keV exceeds 25 keV cap"
        )

    def test_temperature_warning_emitted(self):
        """Warning emitted when temperature hits cap."""
        model = DynamicBurnModel(
            R0=6.2, a=2.0, B_t=5.3, I_p=15.0, n_e20=1.0,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.simulate(P_aux_mw=70.0, duration_s=50.0, dt_s=0.05)
            temp_warnings = [x for x in w if "25 keV" in str(x.message)]
            assert len(temp_warnings) > 0

    def test_q_scan_with_iter_params_q_below_15(self):
        """Full Q-scan with ITER parameters should produce Q <= 15."""
        result = DynamicBurnModel.find_q10_operating_point(
            R0=6.2, a=2.0, B_t=5.3, I_p=15.0, kappa=1.7,
        )
        assert result["best"] is not None
        assert result["best"]["Q_final"] <= 15.0

    def test_greenwald_limit_skip(self):
        """Greenwald limit correctly computed and applied."""
        result = DynamicBurnModel.find_q10_operating_point(
            R0=6.2, a=2.0, B_t=5.3, I_p=15.0, kappa=1.7,
        )
        assert "n_greenwald" in result
        n_gw = result["n_greenwald"]
        # For ITER: n_GW = 15 / (pi * 4) ~ 1.19
        assert 1.0 < n_gw < 2.0

        # All scan results should be at densities <= 1.2 * n_GW
        for r in result["scan_results"]:
            assert r["n_e20"] <= 1.2 * n_gw + 1e-6

    def test_greenwald_limit_extreme_low_current(self):
        """Very low current should skip high-density points."""
        # I_p = 1 MA, a = 5 m → n_GW = 1/(pi*25) ≈ 0.013
        # All densities [0.8, 1.0, 1.2] >> 1.2*0.013 → all skipped
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = DynamicBurnModel.find_q10_operating_point(
                R0=6.2, a=5.0, B_t=5.3, I_p=1.0, kappa=1.7,
            )
        assert result["best"] is None
        assert result["q10_achieved"] is False
        assert len(result["scan_results"]) == 0


# ════════════════════════════════════════════════════════════════════════
# P0.6 — Energy Conservation in Transport
# ════════════════════════════════════════════════════════════════════════


class TestEnergyConservation:
    """Verify energy conservation enforcement in CN transport solver."""

    @pytest.fixture()
    def solver(self):
        """Create a TransportSolver with equilibrium solved."""
        from pathlib import Path
        from scpn_fusion.core.integrated_transport_solver import TransportSolver
        config = str(Path(__file__).resolve().parents[1] / "iter_config.json")
        ts = TransportSolver(config)
        ts.solve_equilibrium()
        return ts

    def test_conservation_error_attribute_exists(self, solver):
        """Solver has _last_conservation_error attribute."""
        assert hasattr(solver, "_last_conservation_error")
        assert solver._last_conservation_error == 0.0

    def test_conservation_error_updated_after_step(self, solver):
        """After evolve_profiles, conservation error is computed."""
        solver.update_transport_model(50.0)
        solver.evolve_profiles(0.01, 50.0)
        # Should be non-negative and finite
        assert np.isfinite(solver._last_conservation_error)
        assert solver._last_conservation_error >= 0.0

    def test_zero_heating_monotonic_cooling(self, solver):
        """With zero heating and no impurities, temperature should decrease."""
        solver.n_impurity[:] = 0.0
        solver.Ti = 5.0 * (1.0 - solver.rho**2)
        solver.Te = solver.Ti.copy()

        T_before = np.mean(solver.Ti)
        for _ in range(10):
            solver.update_transport_model(0.0)
            solver.evolve_profiles(0.01, 0.0)

        T_after = np.mean(solver.Ti)
        # With zero heating and Dirichlet edge BC, temperature should drop
        # (diffusion spreads heat to edge where it's lost)
        assert T_after <= T_before

    def test_finite_heating_conservation_below_threshold(self, solver):
        """With finite heating, conservation error should be small."""
        solver.update_transport_model(50.0)
        solver.evolve_profiles(0.01, 50.0)
        # Conservation error includes boundary losses, so allow some slack
        # but it should be well below 100% (i.e., no catastrophic energy leak)
        assert solver._last_conservation_error < 0.5

    def test_enforce_conservation_parameter_accepted(self, solver):
        """enforce_conservation=False should not raise."""
        solver.update_transport_model(50.0)
        solver.evolve_profiles(0.01, 50.0, enforce_conservation=False)

    def test_physics_error_importable(self):
        """PhysicsError can be imported from the transport module."""
        from scpn_fusion.core.integrated_transport_solver import PhysicsError
        assert issubclass(PhysicsError, RuntimeError)
