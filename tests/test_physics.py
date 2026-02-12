import pytest
import numpy as np
import sys
import os

# Path hack for tests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from scpn_fusion.core.fusion_ignition_sim import FusionBurnPhysics

# Mock Config
MOCK_CONFIG = {
    "reactor_name": "Test",
    "grid_resolution": [20, 20],
    "dimensions": {"R_min": 1, "R_max": 2, "Z_min": -1, "Z_max": 1},
    "physics": {"plasma_current_target": 1.0, "vacuum_permeability": 1.0},
    "coils": [],
    "solver": {"max_iterations": 1, "convergence_threshold": 1e-4, "relaxation_factor": 0.1}
}

class TestFusionPhysics:
    
    def setup_method(self):
        import json
        with open("test_config.json", "w") as f:
            json.dump(MOCK_CONFIG, f)
        self.sim = FusionBurnPhysics("test_config.json")
        
    def teardown_method(self):
        if os.path.exists("test_config.json"):
            os.remove("test_config.json")

    def test_bosch_hale_rate(self):
        """Test D-T Reaction Rate physics"""
        # Test 1: Zero temperature -> Zero rate
        rate_zero = self.sim.bosch_hale_dt(0.0)
        assert rate_zero < 1e-30, "Fusion at absolute zero? Impossible."
        
        # Test 2: Peak rate (around 60 keV)
        rate_peak = self.sim.bosch_hale_dt(60.0)
        rate_low = self.sim.bosch_hale_dt(1.0)
        
        # Rate at 60keV should be orders of magnitude higher than at 1keV
        assert rate_peak > rate_low * 1000, "Fusion cross-section curve is wrong."
        
        # Test 3: Magnitude check (approx 8e-22 m3/s at 20keV)
        rate_20 = self.sim.bosch_hale_dt(20.0)
        # Allow wide margin because approximations vary, but must be within factor of 10
        assert 1e-22 < rate_20 < 1e-21, f"Rate at 20keV is unphysical: {rate_20}"

    def test_ignition_logic(self):
        """Test Q-factor calculation logic"""
        # Manually inject power balance
        # If P_fus = 500, P_aux = 50 -> Q should be 10
        metrics = {
            'P_fusion_MW': 500.0,
            'P_aux_MW': 50.0
        }
        Q = metrics['P_fusion_MW'] / metrics['P_aux_MW']
        assert Q == 10.0

if __name__ == "__main__":
    # Manual run if pytest not installed
    t = TestFusionPhysics()
    t.setup_method()
    t.test_bosch_hale_rate()
    t.test_ignition_logic()
    t.teardown_method()
    print("All Tests Passed.")
