# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Implicit Impurity Transport Solver Tests
from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_fusion.core.impurity_transport_contracts import ImpuritySpecies
from scpn_fusion.core.impurity_transport_solver import ImpurityTransportSolver


def _toroidal_inventory(
    n_z: NDArray[np.float64], rho: NDArray[np.float64], R0: float, a: float
) -> float:
    vol_element = 4.0 * np.pi**2 * R0 * a**2 * rho
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = np.trapz
    return float(trapezoid(n_z * vol_element, rho))


def test_zero_source() -> None:
    species = [ImpuritySpecies("W", 74, 183.8, source_rate=0.0)]
    solver = ImpurityTransportSolver(np.linspace(0, 1, 50), 6.2, 2.0, species)

    res = solver.step(0.1, np.ones(50), np.ones(50), np.ones(50), 1.0, {})
    assert np.allclose(res["W"], 0.0)


def test_steady_source() -> None:
    species = [ImpuritySpecies("W", 74, 183.8, source_rate=1e16)]  # atoms/m2/s
    solver = ImpurityTransportSolver(np.linspace(0, 1, 50), 6.2, 2.0, species)

    res = solver.step(0.1, np.ones(50), np.ones(50), np.ones(50), 1.0, {"W": np.zeros(50)})

    # Should build up at the edge
    assert res["W"][-1] > 0.0


def test_edge_source_is_volume_conservative_and_resolved() -> None:
    rho = np.linspace(0.0, 1.0, 80)
    R0 = 6.2
    a = 2.0
    dt = 0.2
    source_rate = 1.0e16
    species = [ImpuritySpecies("W", 74, 183.8, source_rate=source_rate)]
    solver = ImpurityTransportSolver(rho, R0, a, species)

    res = solver.step(
        dt,
        np.ones_like(rho) * 1.0e20,
        np.ones_like(rho) * 1500.0,
        np.ones_like(rho) * 1500.0,
        0.0,
        {"W": np.zeros_like(rho)},
    )

    expected_particles = source_rate * (4.0 * np.pi**2 * R0 * a) * dt
    actual_particles = _toroidal_inventory(res["W"], rho, R0, a)
    assert actual_particles == pytest.approx(expected_particles, rel=2e-2)
    assert np.count_nonzero(res["W"] > 0.0) > 3
    assert res["W"][-1] > res["W"][len(rho) // 2]


class TestImpurityTransportSolverConstruction:
    def test_rejects_short_rho(self) -> None:
        with pytest.raises(ValueError, match="at least three radial points"):
            ImpurityTransportSolver(np.array([0.0, 1.0]), 6.2, 2.0, [])

    def test_rejects_non_finite_rho(self) -> None:
        with pytest.raises(ValueError, match="only finite values"):
            ImpurityTransportSolver(np.array([0.0, np.nan, 1.0]), 6.2, 2.0, [])

    def test_rejects_non_increasing_rho(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            ImpurityTransportSolver(np.array([0.0, 0.7, 0.3, 1.0]), 6.2, 2.0, [])

    def test_rejects_rho_not_spanning_unit_interval(self) -> None:
        with pytest.raises(ValueError, match=r"normalised interval \[0, 1\]"):
            ImpurityTransportSolver(np.array([0.1, 0.5, 0.9]), 6.2, 2.0, [])

    def test_rejects_bad_major_radius(self) -> None:
        with pytest.raises(ValueError, match="R0 must be finite and positive"):
            ImpurityTransportSolver(np.linspace(0.0, 1.0, 5), 0.0, 2.0, [])

    def test_rejects_bad_minor_radius(self) -> None:
        with pytest.raises(ValueError, match="a must be finite and positive"):
            ImpurityTransportSolver(np.linspace(0.0, 1.0, 5), 6.2, 0.0, [])

    def test_rejects_non_uniform_grid(self) -> None:
        with pytest.raises(ValueError, match="uniformly spaced"):
            ImpurityTransportSolver(np.array([0.0, 0.3, 1.0]), 6.2, 2.0, [])


def test_step_with_outward_pinch_uses_positive_upwind_branch() -> None:
    rho = np.linspace(0.0, 1.0, 5)
    species = ImpuritySpecies("Ar", 18, 39.95, source_rate=1e18)
    solver = ImpurityTransportSolver(rho, R0=6.2, a=2.0, species=[species])
    ne = np.full(5, 1e19)
    Te = np.full(5, 1000.0)
    Ti = np.full(5, 1000.0)
    outward = {"Ar": np.full(5, 5.0)}  # V[i] > 0 -> positive-upwind convection branch
    result = solver.step(0.001, ne, Te, Ti, D_anom=0.5, V_pinch=outward)
    assert np.all(np.isfinite(result["Ar"]))
    assert np.all(result["Ar"] >= 0.0)
