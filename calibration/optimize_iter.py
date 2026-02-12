# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Optimize Iter
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
import numpy as np
from scipy.optimize import minimize
import sys
import os
import json

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from scpn_fusion.core.fusion_kernel import FusionKernel

def optimize_iter_equilibrium():
    print("--- SCPN EQUILIBRIUM OPTIMIZER ---")
    print("Target: Center Plasma at R=6.2m, Z=0.0m")
    
    # 1. Load Base Config
    # We create a temporary config based on ITER specs
    config_path = "iter_calib.json"
    iter_config = {
        "reactor_name": "ITER-Calibration",
        "grid_resolution": [65, 65],
        "dimensions": {"R_min": 2.0, "R_max": 10.0, "Z_min": -6.0, "Z_max": 6.0},
        "physics": {"plasma_current_target": 15.0, "vacuum_permeability": 1.0},
        "coils": [
            {"name": "PF1", "r": 3.9, "z": 7.6,  "current": 5.0},
            {"name": "PF2", "r": 8.2, "z": 6.7,  "current": -2.0},
            {"name": "PF3", "r": 12.0, "z": 2.7,  "current": -5.0},
            {"name": "PF4", "r": 12.6, "z": -2.3, "current": -5.0},
            {"name": "PF5", "r": 8.4, "z": -6.7, "current": -4.0},
            {"name": "PF6", "r": 4.3, "z": -7.6, "current": 8.0},
            {"name": "CS",  "r": 1.7, "z": 0.0,  "current": 18.0}
        ],
        "solver": {"max_iterations": 200, "convergence_threshold": 1e-3, "relaxation_factor": 0.1}
    }
    
    with open(config_path, "w") as f:
        json.dump(iter_config, f)
        
    kernel = FusionKernel(config_path)
    
    # 2. Define Objective Function
    def objective(currents):
        # Update kernel coils
        for i, I in enumerate(currents):
            kernel.cfg['coils'][i]['current'] = I
            
        # Run Solver (Fast mode)
        # We assume vacuum field + simple source is enough to determine shift
        # Full non-linear solve is expensive, but necessary for Shafranov Shift
        kernel.solve_equilibrium()
        
        # Get Axis
        idx_max = np.argmax(kernel.Psi)
        iz, ir = np.unravel_index(idx_max, kernel.Psi.shape)
        R_ax = kernel.R[ir]
        Z_ax = kernel.Z[iz]
        
        # Cost: Distance from Target (6.2, 0.0)
        cost = (R_ax - 6.2)**2 + (Z_ax - 0.0)**2
        
        print(f"  Eval: R={R_ax:.2f}, Z={Z_ax:.2f} | Cost={cost:.4f}")
        return cost

    # 3. Optimize
    # Initial guess from config
    x0 = [c['current'] for c in kernel.cfg['coils']]
    
    print("Starting Optimization (Powell Method)...")
    # We use Powell because gradients are noisy/unavailable without AD
    res = minimize(objective, x0, method='Powell', tol=0.1, options={'maxiter': 5})
    
    print("\n--- OPTIMIZATION RESULT ---")
    print(f"Success: {res.success}")
    print(f"Best R: {res.fun} (Residual)")
    print("Optimized Currents (MA):")
    
    optimized_coils = []
    for i, val in enumerate(res.x):
        name = kernel.cfg['coils'][i]['name']
        print(f"  {name}: {val:.2f}")
        optimized_coils.append(val)
        
    # 4. Save Validated Config
    # Update config with new currents
    for i, val in enumerate(res.x):
        iter_config['coils'][i]['current'] = val
        
    # Save to validation folder
    with open("../validation/iter_validated_config.json", "w") as f:
        json.dump(iter_config, f, indent=4)
        
    print("Saved optimized configuration to: ../validation/iter_validated_config.json")
    os.remove(config_path)

if __name__ == "__main__":
    optimize_iter_equilibrium()
