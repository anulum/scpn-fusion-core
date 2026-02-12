# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Genetic Optimizer
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
import numpy as np
from scipy.optimize import differential_evolution
import sys
import os
import json
import time

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from scpn_fusion.core.fusion_kernel import FusionKernel

def genetic_calibration():
    print("--- SCPN GENETIC CALIBRATION: DIFFERENTIAL EVOLUTION ---")
    print("Target: Absolute Precision Alignment of ITER Plasma (R=6.2m)")
    
    # 1. Base Config Template
    config_path = "iter_genetic_temp.json"
    base_config = {
        "reactor_name": "ITER-Evol",
        "grid_resolution": [65, 65],
        "dimensions": {"R_min": 2.0, "R_max": 10.0, "Z_min": -6.0, "Z_max": 6.0},
        "physics": {"plasma_current_target": 15.0, "vacuum_permeability": 1.0},
        "coils": [
            {"name": "PF1", "r": 3.9, "z": 7.6,  "current": 0.0},
            {"name": "PF2", "r": 8.2, "z": 6.7,  "current": 0.0},
            {"name": "PF3", "r": 12.0, "z": 2.7,  "current": 0.0},
            {"name": "PF4", "r": 12.6, "z": -2.3, "current": 0.0},
            {"name": "PF5", "r": 8.4, "z": -6.7, "current": 0.0},
            {"name": "PF6", "r": 4.3, "z": -7.6, "current": 0.0},
            {"name": "CS",  "r": 1.7, "z": 0.0,  "current": 0.0}
        ],
        "solver": {"max_iterations": 300, "convergence_threshold": 1e-3, "relaxation_factor": 0.1}
    }
    
    # Save template
    with open(config_path, "w") as f:
        json.dump(base_config, f)
        
    kernel = FusionKernel(config_path)
    
    # 2. Define Bounds for Evolution
    # Allow massive range for currents (-20 MA to +20 MA)
    bounds = [(-20.0, 20.0) for _ in range(7)] 
    
    # 3. Fitness Function
    def fitness(currents):
        # Apply Genotype (Currents) to Phenotype (Coils)
        for i, I in enumerate(currents):
            kernel.cfg['coils'][i]['current'] = I
            
        # Run Simulation
        # Mute output for speed
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            kernel.solve_equilibrium()
        except:
            pass # Ignore divergences
        sys.stdout = original_stdout
        
        # Analyze Result
        idx_max = np.argmax(kernel.Psi)
        iz, ir = np.unravel_index(idx_max, kernel.Psi.shape)
        R_ax = kernel.R[ir]
        Z_ax = kernel.Z[iz]
        
        # Check X-points (We want Single Null Divertor)
        xp, psi_x = kernel.find_x_point(kernel.Psi)
        
        # COST FUNCTION
        # 1. Axis Position Error (Target R=6.2, Z=0.0)
        err_pos = (R_ax - 6.2)**2 + (Z_ax - 0.5)**2 # Slight vertical offset for single null
        
        # 2. X-Point position (Should be at bottom)
        # Target X ~ (5.0, -4.5) approx
        err_x = 0
        if xp[1] > -3.0: # If X-point is too high or non-existent
            err_x = 10.0
            
        # 3. Boundary Flux Error (Limiter vs Divertor)
        # We want Psi_boundary to be defined by X-point, not wall
        
        # 4. Strength Penalty (Prevent zero-current solution)
        idx_max = np.argmax(kernel.Psi)
        iz, ir = np.unravel_index(idx_max, kernel.Psi.shape)
        Psi_ax = kernel.Psi[iz, ir]
        
        err_strength = 0
        if Psi_ax < 5.0: # We expect ~10-30 Wb for ITER
            err_strength = 100.0 * (5.0 - Psi_ax)**2
        
        total_cost = err_pos + err_x + err_strength
        
        # Penalty for extreme currents (Regularization)
        # We prefer smaller currents
        total_cost += 0.001 * np.sum(np.array(currents)**2) # Reduced regularization
        
        return total_cost

    # 4. Run Differential Evolution
    print("Launching Genetic Algorithm (Population=15)...")
    t0 = time.time()
    
    result = differential_evolution(
        fitness, 
        bounds, 
        strategy='best1bin', 
        maxiter=20, 
        popsize=15, 
        tol=0.1,
        disp=True # Show progress
    )
    
    print(f"\n--- EVOLUTION COMPLETE in {time.time()-t0:.1f}s ---")
    print(f"Best Fitness: {result.fun:.4f}")
    print("Optimal DNA (Currents):")
    
    optimized_config = base_config.copy()
    for i, val in enumerate(result.x):
        name = base_config['coils'][i]['name']
        print(f"  {name}: {val:.3f} MA")
        optimized_config['coils'][i]['current'] = val
        
    # Save
    out_path = os.path.join(os.path.dirname(__file__), "../validation/iter_genetic_config.json")
    with open(out_path, "w") as f:
        json.dump(optimized_config, f, indent=4)
    print(f"Saved Genetically Perfected Config: {out_path}")
    
    # Cleanup
    if os.path.exists(config_path):
        os.remove(config_path)

if __name__ == "__main__":
    genetic_calibration()
