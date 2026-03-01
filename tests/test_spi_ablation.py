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


# S2-002: Parks SPI Ablation negative density guard


def test_ablation_survives_negative_density_interpolation():
    """ne_20**0.33 must not produce NaN when local n_e goes negative."""
    solver = SpiAblationSolver(
        n_fragments=5,
        total_mass_kg=0.001,
        velocity_mps=200.0,
    )

    nr = 50
    r_grid = np.linspace(0, 1, nr)

    # Craft a density profile with negative values near the edge
    ne_profile = 5.0 * (1 - r_grid ** 2)
    ne_profile[-5:] = -1.0  # Negative densities (unphysical but possible from interp)

    te_profile = 5.0 * (1 - r_grid ** 2)

    # This should not raise or produce NaN
    deposition = solver.step(dt=1e-4, plasma_ne_profile=ne_profile,
                             plasma_te_profile=te_profile, r_grid=r_grid)

    assert np.all(np.isfinite(deposition)), "Deposition has NaN/Inf with negative n_e"
    assert deposition.shape == r_grid.shape


def test_spi_constructor_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="n_fragments"):
        SpiAblationSolver(n_fragments=0)
    with pytest.raises(ValueError, match="total_mass_kg"):
        SpiAblationSolver(total_mass_kg=0.0)
    with pytest.raises(ValueError, match="injector_dir"):
        SpiAblationSolver(injector_dir=np.zeros(3))


def test_spi_initialization_is_deterministic_for_seed():
    a = SpiAblationSolver(n_fragments=8, seed=123)
    b = SpiAblationSolver(n_fragments=8, seed=123)
    assert len(a.fragments) == len(b.fragments)
    for frag_a, frag_b in zip(a.fragments, b.fragments):
        np.testing.assert_allclose(frag_a.pos, frag_b.pos, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(frag_a.vel, frag_b.vel, rtol=0.0, atol=0.0)
