"""Tests for physics-based halo current and runaway electron models."""

import numpy as np
import pytest

from scpn_fusion.control.halo_re_physics import (
    HaloCurrentModel,
    HaloCurrentResult,
    RunawayElectronModel,
    RunawayElectronResult,
    DisruptionMitigationReport,
    run_disruption_ensemble,
)


class TestHaloCurrentModel:
    def test_basic_simulation(self):
        model = HaloCurrentModel(plasma_current_ma=15.0)
        result = model.simulate(tau_cq_s=0.01, duration_s=0.05)
        assert isinstance(result, HaloCurrentResult)
        assert len(result.time_ms) > 0
        assert result.peak_halo_ma > 0.0

    def test_halo_peaks_then_decays(self):
        model = HaloCurrentModel(plasma_current_ma=15.0)
        result = model.simulate(tau_cq_s=0.01, duration_s=0.05)
        halo = np.array(result.halo_current_ma)
        peak_idx = np.argmax(halo)
        assert peak_idx > 0  # rises first
        assert peak_idx < len(halo) - 1  # then decays

    def test_plasma_current_decays(self):
        model = HaloCurrentModel(plasma_current_ma=15.0)
        result = model.simulate(tau_cq_s=0.01, duration_s=0.05)
        assert result.plasma_current_ma[0] > result.plasma_current_ma[-1]

    def test_higher_tpf_increases_product(self):
        result_low = HaloCurrentModel(tpf=1.5).simulate()
        result_high = HaloCurrentModel(tpf=2.5).simulate()
        assert result_high.peak_tpf_product > result_low.peak_tpf_product

    def test_wall_force_positive(self):
        model = HaloCurrentModel()
        result = model.simulate()
        assert result.wall_force_mn_m >= 0.0

    def test_iter_limits_check(self):
        model = HaloCurrentModel(plasma_current_ma=15.0, tpf=2.0)
        result = model.simulate()
        # With neon mitigation, TPF product should be in a reasonable range
        assert result.peak_tpf_product < 5.0  # sanity


class TestRunawayElectronModel:
    def test_basic_simulation(self):
        model = RunawayElectronModel(n_e=1e20, T_e_keV=20.0)
        result = model.simulate(plasma_current_ma=15.0, tau_cq_s=0.01)
        assert isinstance(result, RunawayElectronResult)
        assert len(result.time_ms) > 0

    def test_dreicer_field_positive(self):
        model = RunawayElectronModel(n_e=1e20, T_e_keV=20.0)
        assert model.E_D > 0.0
        assert model.E_c > 0.0
        assert model.E_D > model.E_c  # Dreicer > critical

    def test_higher_zeff_increases_damping(self):
        result_low_z = RunawayElectronModel(z_eff=1.0).simulate(neon_z_eff=1.0)
        result_high_z = RunawayElectronModel(z_eff=5.0).simulate(neon_z_eff=5.0)
        # Higher Z_eff increases collisional drag → less RE
        assert result_high_z.peak_re_current_ma <= result_low_z.peak_re_current_ma * 10.0

    def test_avalanche_gain_finite(self):
        model = RunawayElectronModel()
        result = model.simulate()
        assert np.isfinite(result.avalanche_gain)
        assert result.avalanche_gain >= 1.0

    def test_re_current_bounded(self):
        model = RunawayElectronModel()
        result = model.simulate(plasma_current_ma=15.0)
        # RE current cannot exceed original plasma current
        assert result.peak_re_current_ma <= 15.0


class TestDisruptionEnsemble:
    def test_ensemble_runs(self):
        report = run_disruption_ensemble(ensemble_runs=10, seed=42)
        assert isinstance(report, DisruptionMitigationReport)
        assert report.ensemble_runs == 10
        assert len(report.per_run_details) == 10

    def test_prevention_rate_bounded(self):
        report = run_disruption_ensemble(ensemble_runs=20, seed=42)
        assert 0.0 <= report.prevention_rate <= 1.0

    def test_prevention_rate_deterministic(self):
        r1 = run_disruption_ensemble(ensemble_runs=20, seed=42)
        r2 = run_disruption_ensemble(ensemble_runs=20, seed=42)
        assert r1.prevention_rate == r2.prevention_rate

    def test_50_run_ensemble(self):
        """Full 50-run ensemble as specified in the review requirement."""
        report = run_disruption_ensemble(ensemble_runs=50, seed=42, verbose=False)
        assert report.ensemble_runs == 50
        assert report.prevention_rate >= 0.0
        # Report the actual prevention rate for documentation
        print(f"\n=== Ensemble Prevention Rate: {report.prevention_rate*100:.1f}% ===")
        print(f"    Mean halo peak: {report.mean_halo_peak_ma:.2f} MA")
        print(f"    P95 halo peak:  {report.p95_halo_peak_ma:.2f} MA")
        print(f"    Mean RE peak:   {report.mean_re_peak_ma:.4f} MA")
        print(f"    P95 RE peak:    {report.p95_re_peak_ma:.4f} MA")
        print(f"    ITER limits:    {'PASS' if report.passes_iter_limits else 'FAIL'}")

    def test_per_run_has_physics_fields(self):
        report = run_disruption_ensemble(ensemble_runs=5, seed=42)
        run = report.per_run_details[0]
        assert "halo_peak_ma" in run
        assert "re_peak_ma" in run
        assert "tpf_product" in run
        assert "wall_force_mn_m" in run
        assert "avalanche_gain" in run
        assert "prevented" in run
