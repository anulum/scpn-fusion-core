# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — IMAS IDS Round-trip Tests
# ──────────────────────────────────────────────────────────────────────
"""Round-trip: state -> IDS -> state for all supported IDS types."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.io.imas_connector import (
    digital_twin_summary_to_ids,
    ids_to_digital_twin_summary,
    digital_twin_state_to_ids,
    ids_to_digital_twin_state,
    state_to_imas_core_profiles,
    state_to_imas_core_transport,
    imas_core_transport_to_state,
    state_to_imas_summary,
)


def _base_summary():
    return {
        "steps": 100,
        "final_islands_px": 3,
        "final_reward": 0.85,
        "reward_mean_last_50": 0.82,
        "final_avg_temp": 12.5,
        "final_axis_r": 6.2,
        "final_axis_z": 0.01,
    }


def _base_state():
    s = _base_summary()
    s.update({
        "rho_norm": [0.0, 0.25, 0.5, 0.75, 1.0],
        "electron_temp_keV": [10.0, 8.0, 5.0, 2.0, 0.5],
        "electron_density_1e20_m3": [1.0, 0.9, 0.7, 0.4, 0.1],
    })
    return s


class TestSummaryRoundTrip:
    def test_summary_round_trip(self):
        original = _base_summary()
        ids = digital_twin_summary_to_ids(original)
        recovered = ids_to_digital_twin_summary(ids)
        assert recovered["steps"] == original["steps"]
        assert abs(recovered["final_reward"] - original["final_reward"]) < 1e-10
        assert abs(recovered["final_avg_temp"] - original["final_avg_temp"]) < 1e-10


class TestStateRoundTrip:
    def test_state_with_profiles_round_trip(self):
        original = _base_state()
        ids = digital_twin_state_to_ids(original)
        recovered = ids_to_digital_twin_state(ids)
        assert recovered["steps"] == original["steps"]
        np.testing.assert_allclose(
            recovered["rho_norm"], original["rho_norm"], atol=1e-10,
        )
        np.testing.assert_allclose(
            recovered["electron_temp_keV"],
            original["electron_temp_keV"],
            atol=1e-10,
        )


class TestCoreProfilesRoundTrip:
    def test_core_profiles_has_correct_units(self):
        state = _base_state()
        ids = state_to_imas_core_profiles(state)
        p1d = ids["profiles_1d"][0]
        # keV -> eV conversion
        np.testing.assert_allclose(
            p1d["electrons"]["temperature"],
            [v * 1e3 for v in state["electron_temp_keV"]],
            atol=1e-6,
        )
        # 1e20 m^-3 -> m^-3
        np.testing.assert_allclose(
            p1d["electrons"]["density"],
            [v * 1e20 for v in state["electron_density_1e20_m3"]],
            atol=1.0,
        )


class TestCoreTransportRoundTrip:
    def test_core_transport_round_trip(self):
        rho = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        state = {
            "rho_norm": rho,
            "chi_e": [0.5, 1.0, 2.0, 3.0, 2.5, 1.5],
            "chi_i": [0.3, 0.8, 1.5, 2.5, 2.0, 1.0],
            "d_e": [0.1, 0.2, 0.5, 0.8, 0.6, 0.3],
        }
        ids = state_to_imas_core_transport(state)
        recovered = imas_core_transport_to_state(ids)

        np.testing.assert_allclose(recovered["rho_norm"], rho, atol=1e-10)
        np.testing.assert_allclose(recovered["chi_e"], state["chi_e"], atol=1e-10)
        np.testing.assert_allclose(recovered["chi_i"], state["chi_i"], atol=1e-10)
        np.testing.assert_allclose(recovered["d_e"], state["d_e"], atol=1e-10)

    def test_core_transport_partial_profiles(self):
        state = {
            "rho_norm": [0.0, 0.5, 1.0],
            "chi_e": [1.0, 2.0, 0.5],
        }
        ids = state_to_imas_core_transport(state)
        recovered = imas_core_transport_to_state(ids)
        assert "chi_e" in recovered
        assert "chi_i" not in recovered
        assert "d_e" not in recovered

    def test_core_transport_missing_rho_raises(self):
        with pytest.raises(ValueError, match="rho_norm"):
            state_to_imas_core_transport({"chi_e": [1.0, 2.0]})


class TestSummaryIDS:
    def test_summary_ids_fields(self):
        state = {
            "power_fusion_MW": 500.0,
            "q_sci": 10.0,
            "beta_n": 1.8,
            "plasma_current_MA": 15.0,
        }
        ids = state_to_imas_summary(state)
        gq = ids["global_quantities"]
        assert abs(gq["power_fusion"] - 500.0) < 1e-10
        assert abs(gq["q"] - 10.0) < 1e-10
        assert abs(gq["beta_n"] - 1.8) < 1e-10
        assert abs(gq["ip"] - 15.0) < 1e-10
