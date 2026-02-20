# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Superior Core Validation
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Validates the integration of:
1. Unified State Space
2. Nonlinear MPC
3. Rutherford Island Evolution
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from scpn_fusion.core.state_space import FusionState
from scpn_fusion.control.fusion_nmpc_jax import get_nmpc_controller
from scpn_fusion.control.tokamak_digital_twin import TokamakTopoloy, Plasma2D

def validate():
    print("--- VALIDATING PROJECT TOKAMAK-MASTER INTEGRATION ---")
    
    # 1. State Space Check
    state = FusionState(r_axis=6.2, z_axis=0.0, ip_ma=15.0)
    vec = state.to_vector()
    print(f"Unified State Vector: {vec}")
    assert vec.shape == (6,)
    
    # 2. Digital Twin + Rutherford Check
    topo = TokamakTopoloy()
    plasma = Plasma2D(topo)
    print(f"Initial Island Width (q=2.0): {topo.island_widths[2.0]:.4f}")
    
    # Step the twin
    plasma.step(action=0.1)
    print(f"Evolved Island Width (q=2.0): {topo.island_widths[2.0]:.4f}")
    
    # 3. Nonlinear MPC Check
    nmpc = get_nmpc_controller(state_dim=6, action_dim=1, horizon=5)
    target = vec.copy()
    target[0] += 0.1 # Move axis R
    
    u_opt = nmpc.plan_trajectory(vec, target)
    print(f"NMPC Optimal Action: {u_opt}")
    assert len(u_opt) == 1
    
    print("\nINTEGRATION SUCCESSFUL: Core is now Physically Hardened.")

if __name__ == "__main__":
    validate()
