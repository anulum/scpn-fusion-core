"""Phase 1 hardening tests: CFL guard, stiffness bounds, convergence warning."""

from __future__ import annotations

import logging
import numpy as np
import pytest


# ── 1.1 CFL sub-stepping ────────────────────────────────────────────


def test_digital_twin_cfl_substep_high_diffusivity():
    """With D_turb=5.0, CFL requires sub-stepping.  T must stay finite."""
    from scpn_fusion.control.tokamak_digital_twin import (
        Plasma2D, TokamakTopoloy, GRID_SIZE,
    )

    topo = TokamakTopoloy()
    plasma = Plasma2D(topo)

    # Patch danger zones to cover entire grid → D_turb everywhere
    class _AllDanger:
        def get_rational_surfaces(self):
            return np.ones((GRID_SIZE, GRID_SIZE), dtype=bool)
        def update_q_profile(self, _):
            pass
        def step_island_evolution(self):
            pass
        mask = np.ones((GRID_SIZE, GRID_SIZE), dtype=bool)
        q_map = np.ones((GRID_SIZE, GRID_SIZE))

    # Inject high diffusivity by using a gyro_surrogate that returns 10.0
    def high_d(T, q, danger):
        return np.full_like(T, 10.0)

    plasma.topo = _AllDanger()
    plasma._gyro_surrogate = high_d

    # Seed nonzero temperature
    plasma.T[GRID_SIZE // 2, GRID_SIZE // 2] = 50.0

    # Step should not produce NaN or Inf
    _, avg = plasma.step(0.0)
    assert np.all(np.isfinite(plasma.T)), "T has NaN/Inf with high diffusivity"
    assert avg >= 0.0


# ── 1.2 Parks coefficient ───────────────────────────────────────────


def test_parks_coefficient_value():
    """_PARKS_COEFFICIENT should equal 2.0 (Parks NF 57 Eq. 8)."""
    from scpn_fusion.control.spi_ablation import _PARKS_COEFFICIENT
    assert _PARKS_COEFFICIENT == 2.0


def test_spi_ablation_step_unchanged():
    """Existing SPI test must still pass after refactor to named constant."""
    from scpn_fusion.control.spi_ablation import SpiAblationSolver

    solver = SpiAblationSolver(n_fragments=10, velocity_mps=500.0)
    r_grid = np.linspace(0, 2.0, 50)
    ne = np.ones(50) * 10.0
    te = np.ones(50) * 5.0

    for _ in range(100):
        dep = solver.step(dt=1e-4, plasma_ne_profile=ne,
                          plasma_te_profile=te, r_grid=r_grid)
    assert solver.fragments[0].pos[0] < 10.0


# ── 1.3 Stiffness bounds ────────────────────────────────────────────


def test_stiffness_default_in_range():
    from scpn_fusion.core.neural_transport import (
        _STIFFNESS, _STIFFNESS_MIN, _STIFFNESS_MAX,
    )
    assert _STIFFNESS_MIN <= _STIFFNESS <= _STIFFNESS_MAX


def test_stiffness_invalid_raises():
    from scpn_fusion.core.neural_transport import (
        critical_gradient_model, TransportInputs,
    )
    with pytest.raises(ValueError, match="stiffness"):
        critical_gradient_model(TransportInputs(), stiffness=0.5)
    with pytest.raises(ValueError, match="stiffness"):
        critical_gradient_model(TransportInputs(), stiffness=7.0)


def test_stiffness_valid_custom():
    from scpn_fusion.core.neural_transport import (
        critical_gradient_model, TransportInputs,
    )
    result = critical_gradient_model(TransportInputs(grad_ti=8.0), stiffness=3.0)
    assert result.chi_i > 0.0


# ── 1.4 Divertor convergence warning ────────────────────────────────


def test_divertor_converges_normally():
    from scpn_fusion.core.divertor_thermal_sim import DivertorLab

    lab = DivertorLab(P_sol_MW=50.0)
    lab.calculate_heat_load()
    T, q, f = lab.simulate_lithium_vapor()
    assert np.isfinite(T)
    assert np.isfinite(q)


def test_divertor_warns_on_non_convergence(caplog):
    from scpn_fusion.core.divertor_thermal_sim import DivertorLab

    lab = DivertorLab(P_sol_MW=50.0)
    lab.calculate_heat_load()
    # Force non-convergence with 1 iteration and tight tolerance
    with caplog.at_level(logging.WARNING):
        T, q, f = lab.simulate_lithium_vapor(max_iter=1, tol=1e-12)
    assert any("did not converge" in r.message for r in caplog.records)


# ── 1.5 Sawtooth constants ──────────────────────────────────────────


def test_sawtooth_constants_exported():
    from scpn_fusion.core.mhd_sawtooth import (
        _CRASH_THRESHOLD, _CRASH_REDUCTION, _Q_RECOVERY_RATE,
    )
    assert _CRASH_THRESHOLD == 0.1
    assert _CRASH_REDUCTION == 0.01
    assert _Q_RECOVERY_RATE == 0.05


def test_sawtooth_crash_still_occurs():
    """Existing behavior: crash at amplitude > 0.1 after constant extraction."""
    from scpn_fusion.core.mhd_sawtooth import ReducedMHD

    sim = ReducedMHD()
    for _ in range(1000):
        amp, crash = sim.step(dt=0.01)
        if crash:
            return
    pytest.fail("No sawtooth crash occurred in 1000 steps")
