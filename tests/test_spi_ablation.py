# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — SPI Ablation Tests
# © 1998–2026 Miroslav Šotek. All rights reserved.
# ─────────────────────────────────────────────────────────────────────

import numpy as np
import pytest
from scpn_fusion.control.spi_ablation import SpiAblationSolver

def test_spi_initialization():
    solver = SpiAblationSolver(n_fragments=10)
    assert len(solver.fragments) == 10
    assert solver.fragments[0].active

def test_spi_ablation_step():
    solver = SpiAblationSolver(n_fragments=10, velocity_mps=500.0)
    
    # Mock plasma profiles
    r_grid = np.linspace(0, 2.0, 50) # Minor radius grid
    ne = np.ones(50) * 10.0 # 10^19 m^-3
    te = np.ones(50) * 5.0  # 5 keV
    
    # Run for enough time to reach plasma
    # Injector at 10m, Plasma at 8.2m edge (6.2 + 2.0)
    # Distance = 1.8m. Vel = 500m/s. Time ~ 3.6ms
    
    total_dep = 0.0
    for _ in range(100):
        dep = solver.step(dt=1e-4, plasma_ne_profile=ne, plasma_te_profile=te, r_grid=r_grid)
        total_dep += np.sum(dep)
        
    # Should have some deposition if fragments reached plasma
    # At t=10ms, dist = 5m. They cross edge.
    
    # Check if fragments moved
    assert solver.fragments[0].pos[0] < 10.0
