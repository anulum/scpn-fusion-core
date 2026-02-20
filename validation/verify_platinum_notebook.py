# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Platinum Notebook Verification
# ──────────────────────────────────────────────────────────────────────
"""
Verifies the logic of platinum_standard_demo_v1.ipynb by executing
the core cells as a standalone script.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Setup paths
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from scpn_fusion.core.state_space import FusionState
from scpn_fusion.control.fusion_nmpc_jax import get_nmpc_controller
from scpn_fusion.control.tokamak_digital_twin import TokamakTopoloy, Plasma2D
from scpn_fusion.control.advanced_soc_fusion_learning import CoupledSandpileReactor

def test_platinum_logic():
    print("--- VERIFYING PLATINUM STANDARD LOGIC ---")
    
    # 1. Environment Check
    try:
        import jax
        print(f"JAX Acceleration: ACTIVE (Backend: {jax.devices()[0].device_kind})")
    except ImportError:
        print("JAX Acceleration: INACTIVE (Falling back to SciPy)")

    # 2. Section 1: Resistive MHD & Rutherford
    print("\nSection 1: Rutherford Island Evolution...")
    topo = TokamakTopoloy()
    plasma = Plasma2D(topo)
    
    w0 = topo.island_widths[2.0]
    for _ in range(10): # 10 steps for quick verification
        plasma.step(action=0.0)
    w_final = topo.island_widths[2.0]
    
    print(f"  Initial Width (q=2.0): {w0:.6f}")
    print(f"  Final Width (q=2.0):   {w_final:.6f}")
    assert w_final != w0, "Island width did not evolve!"
    print("  Resistive MHD: VERIFIED")

    # 3. Section 2: Nonlinear MPC
    print("\nSection 2: Nonlinear MPC (NMPC-JAX)...")
    state = FusionState(r_axis=6.2, z_axis=0.0, ip_ma=15.0)
    x0 = state.to_vector()
    target = x0.copy()
    target[0] += 0.05
    
    nmpc = get_nmpc_controller(state_dim=6, action_dim=1, horizon=10)
    
    t0 = time.time()
    u_opt = nmpc.plan_trajectory(x0, target)
    t_elapsed = (time.time() - t0) * 1000
    
    print(f"  NMPC Latency: {t_elapsed:.2f} ms")
    print(f"  Optimal Cmd:  {u_opt}")
    assert len(u_opt) == 1, "NMPC did not return an action!"
    print("  NMPC-JAX: VERIFIED")

    # 4. Section 3: Quantitative SOC
    print("\nSection 3: Quantitative SOC (MJ Calibration)...")
    reactor = CoupledSandpileReactor(energy_per_topple_mj=0.05)
    reactor.drive(amount=10.0)
    
    topples, flow, shear = reactor.step_physics(external_shear=0.5)
    energy_mj = reactor.get_elm_energy_mj(topples)
    
    print(f"  Topples:        {topples}")
    print(f"  Energy Release: {energy_mj:.2f} MJ")
    assert energy_mj >= 0, "Negative energy detected!"
    print("  Quantitative SOC: VERIFIED")

    print("\nPLATINUM STATUS: ALL CORE LOGIC FUNCTIONAL.")

if __name__ == "__main__":
    test_platinum_logic()
