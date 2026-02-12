import unittest
import sys
import os
import json
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from scpn_fusion.core.fusion_kernel import FusionKernel
from scpn_fusion.core.fusion_ignition_sim import FusionBurnPhysics
from scpn_fusion.engineering.balance_of_plant import PowerPlantModel

class TestFusionSystem(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Create a mini config for fast testing
        cls.test_config_path = "test_config_ci.json"
        config = {
            "reactor_name": "Test-Unit",
            "grid_resolution": [30, 30], # Low res for speed
            "dimensions": {"R_min": 1.0, "R_max": 5.0, "Z_min": -2.0, "Z_max": 2.0},
            "physics": {"plasma_current_target": 1.0, "vacuum_permeability": 1.0},
            "coils": [{"name": "PF1", "r": 2.0, "z": 3.0, "current": 1.0}],
            "solver": {"max_iterations": 10, "convergence_threshold": 1e-2, "relaxation_factor": 0.1}
        }
        with open(cls.test_config_path, "w") as f:
            json.dump(config, f)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_config_path):
            os.remove(cls.test_config_path)

    def test_01_kernel_initialization(self):
        """Test if FusionKernel loads and initializes grid correctly."""
        kernel = FusionKernel(self.test_config_path)
        self.assertEqual(kernel.NR, 30)
        self.assertTrue(kernel.Psi.shape == (30, 30))
        print("Kernel Init: PASS")

    def test_02_vacuum_field(self):
        """Test Green's function calculation (Elliptic integrals)."""
        kernel = FusionKernel(self.test_config_path)
        Psi_vac = kernel.calculate_vacuum_field()
        self.assertFalse(np.isnan(Psi_vac).any())
        self.assertNotEqual(np.max(Psi_vac), 0.0)
        print("Vacuum Field: PASS")

    def test_03_solver_stability(self):
        """Test if solver runs without crashing (NaN check)."""
        kernel = FusionKernel(self.test_config_path)
        kernel.solve_equilibrium()
        self.assertFalse(np.isnan(kernel.Psi).any())
        print("Solver Stability: PASS")

    def test_04_physics_functions(self):
        """Test Bosch-Hale and Q calculation."""
        sim = FusionBurnPhysics(self.test_config_path)
        rate = sim.bosch_hale_dt(20.0) # 20 keV
        self.assertGreater(rate, 1e-23)
        print("Nuclear Physics: PASS")

    def test_05_engineering_bop(self):
        """Test Balance of Plant logic."""
        plant = PowerPlantModel()
        res = plant.calculate_plant_performance(500.0, 50.0)
        self.assertAlmostEqual(res['Q_plasma'], 10.0)
        self.assertGreater(res['P_gross'], 0.0)
        print("Engineering BOP: PASS")

if __name__ == '__main__':
    print("--- SCPN CI/CD TEST SUITE ---")
    unittest.main(verbosity=2)
