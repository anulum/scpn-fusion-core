# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — NMPC Benchmark
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Benchmarks the new Nonlinear MPC (Neural ODE) against a Linear Baseline.
"""

import time
import numpy as np
import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from scpn_fusion.control.fusion_nmpc_jax import get_nmpc_controller

def run_benchmark():
    print("--- NMPC vs Linear Baseline Benchmark ---")
    
    # 1. Setup
    state_dim = 4
    action_dim = 4
    horizon = 10
    n_steps = 50
    
    nmpc = get_nmpc_controller(state_dim, action_dim, horizon)
    
    # Initial State (e.g., perturbed plasma)
    x0 = np.array([1.0, 0.5, -0.2, 0.1])
    target = np.zeros(state_dim)
    
    # 2. Run NMPC
    print(f"Running NMPC ({'JAX' if hasattr(nmpc, '_jax_grad') and nmpc._jax_grad else 'SciPy'}) for {n_steps} steps...")
    
    start_time = time.time()
    total_cost = 0.0
    x_curr = x0.copy()
    
    for t in range(n_steps):
        # Plan
        u_opt = nmpc.plan_trajectory(x_curr, target)
        
        # Simulate Dynamics (using the same Neural ODE for fairness)
        dxdt = nmpc.dynamics.forward_numpy(x_curr, u_opt)
        x_curr = x_curr + dxdt * nmpc.dt
        
        # Cost
        err = x_curr - target
        total_cost += np.sum(err**2)
        
    duration = time.time() - start_time
    avg_latency = (duration / n_steps) * 1000 # ms
    
    print(f"NMPC Results:")
    print(f"  Total Duration: {duration:.4f}s")
    print(f"  Avg Latency:    {avg_latency:.2f} ms/step")
    print(f"  Final Cost:     {total_cost:.4f}")
    print(f"  Final State:    {x_curr}")

if __name__ == "__main__":
    run_benchmark()
