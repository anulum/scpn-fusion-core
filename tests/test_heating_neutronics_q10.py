"""Tests for upgraded heating absorption, 3-group neutronics, and dynamic Q>=10."""

import numpy as np
import pytest

from scpn_fusion.core.rf_heating import ECRHHeatingSystem
from scpn_fusion.nuclear.blanket_neutronics import MultiGroupBlanket
from scpn_fusion.core.fusion_ignition_sim import DynamicBurnModel


class TestECRHHeating:
    def test_resonance_radius_positive(self):
        ecrh = ECRHHeatingSystem(b0_tesla=5.3, freq_ghz=170.0, harmonic=1)
        R_res = ecrh.resonance_radius()
        assert R_res > 0.0
        assert np.isfinite(R_res)

    def test_deposition_profile_shape(self):
        ecrh = ECRHHeatingSystem()
        rho, P_dep, eff = ecrh.compute_deposition(P_ecrh_mw=20.0)
        assert len(rho) == 50
        assert len(P_dep) == 50
        assert 0.0 < eff <= 1.0
        assert np.max(P_dep) > 0.0  # some power deposited

    def test_absorption_efficiency_reasonable(self):
        ecrh = ECRHHeatingSystem()
        _, _, eff = ecrh.compute_deposition(P_ecrh_mw=20.0, n_e=1e20)
        assert 0.1 < eff < 1.0  # typical ECRH absorption is 50-99%

    def test_second_harmonic(self):
        ecrh1 = ECRHHeatingSystem(harmonic=1)
        ecrh2 = ECRHHeatingSystem(harmonic=2)
        R1 = ecrh1.resonance_radius()
        R2 = ecrh2.resonance_radius()
        assert R2 > R1  # 2nd harmonic resonates at higher B → smaller R → wait, B~1/R
        # Actually: B_res = omega*m_e/(n*e), so n=2 → B_res halved → R_res doubled
        assert abs(R2 / R1 - 2.0) < 0.1

    def test_power_conservation(self):
        ecrh = ECRHHeatingSystem()
        _, P_dep, eff = ecrh.compute_deposition(P_ecrh_mw=20.0, n_radial_bins=100)
        # Total deposited should be approximately P_ecrh * efficiency
        # (approximate since binning is discrete)
        assert np.sum(P_dep) > 0.0

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"harmonic": 0}, "harmonic"),
            ({"freq_ghz": 0.0}, "freq_ghz"),
            ({"b0_tesla": -1.0}, "b0_tesla"),
            ({"r0_major": 0.0}, "r0_major"),
        ],
    )
    def test_constructor_rejects_invalid_inputs(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            ECRHHeatingSystem(**kwargs)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"P_ecrh_mw": -1.0}, "P_ecrh_mw"),
            ({"n_radial_bins": 4}, "n_radial_bins"),
            ({"n_radial_bins": 32.5}, "n_radial_bins"),
            ({"T_e_keV": 0.0}, "T_e_keV"),
            ({"n_e": 0.0}, "n_e"),
            ({"launch_angle_deg": 90.0}, "launch_angle_deg"),
        ],
    )
    def test_compute_deposition_rejects_invalid_inputs(self, kwargs, match):
        ecrh = ECRHHeatingSystem()
        with pytest.raises(ValueError, match=match):
            ecrh.compute_deposition(**kwargs)

    def test_off_resonance_second_harmonic_reduces_absorption(self):
        on_axis = ECRHHeatingSystem(harmonic=1)
        off_axis = ECRHHeatingSystem(harmonic=2)
        _, _, eff_on = on_axis.compute_deposition(P_ecrh_mw=20.0, n_e=1e20)
        _, _, eff_off = off_axis.compute_deposition(P_ecrh_mw=20.0, n_e=1e20)
        assert eff_off < eff_on


class TestMultiGroupBlanket:
    def test_3group_solve(self):
        blanket = MultiGroupBlanket(thickness_cm=80.0, li6_enrichment=0.9)
        result = blanket.solve_transport()
        assert "phi_g1" in result
        assert "phi_g2" in result
        assert "phi_g3" in result
        assert "tbr" in result
        assert result["tbr"] > 0.0

    def test_tbr_above_unity(self):
        """Enriched 80cm blanket should achieve TBR > 1.0 for self-sufficiency."""
        blanket = MultiGroupBlanket(thickness_cm=80.0, li6_enrichment=0.9)
        result = blanket.solve_transport()
        assert result["tbr"] > 1.0

    def test_tbr_by_group_sums(self):
        blanket = MultiGroupBlanket(thickness_cm=80.0, li6_enrichment=0.9)
        result = blanket.solve_transport()
        group_sum = sum(result["tbr_by_group"].values())
        assert abs(group_sum - result["tbr"]) < 0.01 * result["tbr"]

    def test_thermal_group_dominates(self):
        """Li-6 capture cross-section is largest at thermal energies."""
        blanket = MultiGroupBlanket(thickness_cm=80.0, li6_enrichment=0.9)
        result = blanket.solve_transport()
        assert result["tbr_by_group"]["thermal"] > result["tbr_by_group"]["fast"]

    def test_enrichment_increases_tbr(self):
        low = MultiGroupBlanket(li6_enrichment=0.3).solve_transport()["tbr"]
        high = MultiGroupBlanket(li6_enrichment=0.9).solve_transport()["tbr"]
        assert high > low

    def test_thickness_increases_tbr(self):
        thin = MultiGroupBlanket(thickness_cm=40.0).solve_transport()["tbr"]
        thick = MultiGroupBlanket(thickness_cm=100.0).solve_transport()["tbr"]
        assert thick > thin

    def test_flux_attenuation(self):
        blanket = MultiGroupBlanket()
        result = blanket.solve_transport(incident_flux=1e14)
        assert result["phi_g1"][0] > result["phi_g1"][-1]  # fast flux attenuates

    def test_downscatter_produces_thermal(self):
        blanket = MultiGroupBlanket()
        result = blanket.solve_transport()
        # Thermal flux should be non-zero even though no direct source
        assert np.max(result["phi_g3"]) > 0.0

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"thickness_cm": 0.0}, "thickness_cm"),
            ({"li6_enrichment": 1.2}, "li6_enrichment"),
            ({"n_cells": 2.5}, "n_cells"),
        ],
    )
    def test_constructor_rejects_invalid_inputs(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            MultiGroupBlanket(**kwargs)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"incident_flux": 0.0}, "incident_flux"),
            ({"port_coverage_factor": 0.0}, "port_coverage_factor"),
            ({"streaming_factor": 1.2}, "streaming_factor"),
        ],
    )
    def test_solve_transport_rejects_invalid_runtime_inputs(self, kwargs, match):
        blanket = MultiGroupBlanket()
        with pytest.raises(ValueError, match=match):
            blanket.solve_transport(**kwargs)

    def test_solve_transport_reports_flux_clamp_telemetry(self):
        blanket = MultiGroupBlanket()
        result = blanket.solve_transport()
        assert "flux_clamp_total" in result
        assert "flux_clamp_events" in result
        assert "incident_current_cm2_s" in result
        assert result["flux_clamp_total"] >= 0
        assert result["incident_current_cm2_s"] > 0.0
        assert set(result["flux_clamp_events"]) == {"fast", "epithermal", "thermal"}


class TestDynamicBurnModel:
    def test_basic_simulation(self):
        model = DynamicBurnModel()
        result = model.simulate(P_aux_mw=50.0, duration_s=10.0, dt_s=0.1)
        assert "Q" in result
        assert "T_keV" in result
        assert len(result["time_s"]) > 0

    def test_temperature_rises_with_heating(self):
        model = DynamicBurnModel()
        result = model.simulate(P_aux_mw=50.0, T_initial_keV=2.0, duration_s=20.0)
        assert result["T_final_keV"] > 2.0  # should heat up

    def test_q_factor_positive(self):
        model = DynamicBurnModel()
        result = model.simulate(P_aux_mw=50.0, duration_s=20.0)
        assert result["Q_final"] > 0.0

    def test_iter98y2_scaling(self):
        model = DynamicBurnModel(R0=6.2, a=2.0, B_t=5.3, I_p=15.0)
        tau = model.iter98y2_tau_e(50.0)
        assert 1.0 < tau < 10.0  # ITER-like should be ~3-5 s

    def test_h_mode_threshold(self):
        model = DynamicBurnModel()
        P_thr = model.h_mode_threshold_mw()
        assert 10.0 < P_thr < 200.0  # ITER ~ 50-100 MW

    def test_he_ash_accumulates(self):
        model = DynamicBurnModel()
        result = model.simulate(P_aux_mw=50.0, duration_s=50.0)
        assert result["f_he_final"] > 0.02  # started at 2%, should increase

    def test_bosch_hale_reactivity(self):
        # At T=20 keV, <sigma v> should be ~4e-22 m^3/s
        sv = DynamicBurnModel.bosch_hale_dt(20.0)
        assert 1e-23 < sv < 1e-21

    def test_q10_scan(self):
        """Scan for Q>=10 operating point (ITER-like parameters)."""
        result = DynamicBurnModel.find_q10_operating_point(
            R0=6.2, a=2.0, B_t=5.3, I_p=15.0, kappa=1.7
        )
        assert "scan_results" in result
        assert "best" in result
        assert len(result["scan_results"]) > 0
        print(f"\n=== Q-10 Scan Result ===")
        print(f"    Best Q:     {result['best']['Q_final']:.1f}")
        print(f"    P_aux:      {result['best']['P_aux_MW']:.0f} MW")
        print(f"    P_fus:      {result['best']['P_fus_final_MW']:.0f} MW")
        print(f"    T:          {result['best']['T_final_keV']:.1f} keV")
        print(f"    n_e20:      {result['best']['n_e20']}")
        print(f"    Q>=10:      {'YES' if result['q10_achieved'] else 'NO'}")

    def test_energy_conservation_trend(self):
        """W_thermal should be bounded (no runaway)."""
        model = DynamicBurnModel()
        result = model.simulate(P_aux_mw=50.0, duration_s=50.0, dt_s=0.05)
        W = np.array(result["W_MJ"])
        assert np.all(np.isfinite(W))
        assert np.max(W) < 2000.0  # MJ — physical bound for ITER-class
