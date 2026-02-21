# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Physics Closure Validation
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ──────────────────────────────────────────────────────────────────────
import unittest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from scpn_fusion.core.eped_pedestal import EpedPedestalModel
from scpn_fusion.core.fusion_ignition_sim import DynamicBurnModel
from scpn_fusion.control.disruption_predictor import simulate_tearing_mode
from scpn_fusion.nuclear.nuclear_wall_interaction import NuclearEngineeringLab

class TestPhysicsClosures(unittest.TestCase):
    """
    Validation suite for v4.0 Hardened Physics Closures.
    Verifies that models produce physically reasonable outputs.
    """
    
    def test_eped_cht_limit(self):
        """Verify Connor-Hastie-Taylor ballooning limit in EPED model."""
        print("\n[Test] EPED CHT Stability Limit")
        
        # SPARC-like parameters (High Field)
        # Replaced 'mass' with 'A_ion'
        model = EpedPedestalModel(
            R0=1.85, a=0.57, B0=12.2, Ip_MA=8.7, kappa=1.97,
            Z_eff=1.5, A_ion=2.5
        )
        
        # Scan densities
        for n_ped in [10.0, 20.0, 30.0]:
            res = model.predict(n_ped_1e19=n_ped)
            print(f"  n_ped={n_ped}e19: T_ped={res.T_ped_keV:.2f} keV, Delta={res.Delta_ped:.3f}")
            
            # Physics Checks
            self.assertGreater(res.T_ped_keV, 1.0, "T_ped should be > 1 keV for SPARC")
            self.assertLess(res.T_ped_keV, 20.0, "T_ped should be < 20 keV (Revised Ideal limit)")
            self.assertGreater(res.Delta_ped, 0.01, "Pedestal width too small")

    def test_ipb98_scaling(self):
        """Verify IPB98(y,2) confinement scaling trends."""
        print("\n[Test] IPB98(y,2) Confinement Scaling")
        
        model = DynamicBurnModel(R0=6.2, a=2.0, B_t=5.3, I_p=15.0)
        
        # Reference confinement at 50MW
        tau_ref = model.iter98y2_tau_e(P_loss_mw=50.0)
        print(f"  Tau_E (50MW): {tau_ref:.2f} s")
        
        # High power degradation check (P^-0.69)
        tau_high = model.iter98y2_tau_e(P_loss_mw=100.0)
        print(f"  Tau_E (100MW): {tau_high:.2f} s")
        
        ratio = tau_high / tau_ref
        expected_ratio = (100.0/50.0)**(-0.69)
        
        self.assertAlmostEqual(ratio, expected_ratio, places=2, msg="Power degradation scaling mismatch")
        
    def test_mre_growth_rates(self):
        """Verify Modified Rutherford Equation (MRE) dynamics."""
        print("\n[Test] Modified Rutherford Equation (NTM)")
        
        # Run simulation with fixed seed for stability
        rng = np.random.default_rng(123) 
        w_history, label, _ = simulate_tearing_mode(steps=500, rng=rng)
        
        # Check saturation
        max_w = np.max(w_history)
        print(f"  Max Island Width: {max_w:.3f}")
        
        if label == 0: # Stable shot
            self.assertLess(max_w, 2.0, "Stable shot shouldn't grow locked mode islands")
        
    def test_sputtering_yield(self):
        """Verify Roth-Bohdansky Sputtering Yields."""
        print("\n[Test] Roth-Bohdansky Sputtering (W)")
        
        # Create temp config
        import json
        import os
        dummy_cfg = "temp_nuclear_config.json"
        with open(dummy_cfg, "w") as f:
            json.dump({
                "reactor_name": "TestReactor",
                "dimensions": {"R_min": 1.0, "R_max": 3.0, "Z_min": -1.0, "Z_max": 1.0},
                "physics": {"plasma_current_target": 10.0, "vacuum_permeability": 1.256e-6},
                "solver": {"nr": 65, "nz": 65},
                "grid_resolution": [65, 65],
                "coils": []
            }, f)
            
        try:
            # Instantiate Nuclear Lab
            pwi = NuclearEngineeringLab(dummy_cfg)
            
            # Threshold check
            Y_low = pwi.calculate_sputtering_yield("Tungsten (W)", E_inc_eV=50.0)
            print(f"  Yield @ 50eV: {Y_low:.2e}")
            self.assertEqual(Y_low, 0.0, "Should be 0 below threshold (~200eV for D on W)")
            
            # Sputtering check
            Y_high = pwi.calculate_sputtering_yield("Tungsten (W)", E_inc_eV=500.0)
            print(f"  Yield @ 500eV: {Y_high:.2e}")
            self.assertGreater(Y_high, 0.0, "Should sputter above threshold")
        finally:
            if os.path.exists(dummy_cfg):
                os.remove(dummy_cfg)

if __name__ == '__main__':
    unittest.main()
