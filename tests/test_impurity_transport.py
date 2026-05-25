# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Impurity Transport Tests
from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.impurity_transport import (
    CoolingCurve,
    ImpuritySpecies,
    ImpurityTransportSolver,
    neoclassical_impurity_pinch,
    total_radiated_power,
    tungsten_accumulation_diagnostic,
)


def _toroidal_inventory(n_z: np.ndarray, rho: np.ndarray, R0: float, a: float) -> float:
    vol_element = 4.0 * np.pi**2 * R0 * a**2 * rho
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = np.trapz
    return float(trapezoid(n_z * vol_element, rho))


def test_cooling_curves():
    c_W = CoolingCurve("W")
    L_W_core = c_W.L_z(np.array([1500.0]))[0]
    L_W_edge = c_W.L_z(np.array([10.0]))[0]

    assert L_W_core > L_W_edge
    assert L_W_core > 1e-32


def test_cooling_curve_returns_zero_for_nonpositive_temperatures_without_warning():
    curve = CoolingCurve("W")

    with np.errstate(all="raise"):
        values = curve.L_z(np.array([-10.0, 0.0, 1500.0]))

    assert values[0] == 0.0
    assert values[1] == 0.0
    assert np.isfinite(values[2])
    assert values[2] > 0.0


def test_impurity_pinch():
    rho = np.linspace(0, 1, 50)
    # Peaked density
    ne = 1e20 * (1.0 - 0.8 * rho**2)
    # Flat Ti
    Ti = 5000.0 * np.ones(50)
    q = np.ones(50)
    eps = 0.2 + 0.2 * rho

    V_W = neoclassical_impurity_pinch(74, ne, 5000.0 * np.ones(50), Ti, q, rho, 6.2, 2.0, eps)

    # With radial coordinate increasing outward, a peaked density has 1/L_n > 0.
    # The trace neoclassical pinch should therefore point inward: V_r < 0.
    assert V_W[25] < 0.0


def test_impurity_pinch_temperature_screening_contract():
    rho = np.linspace(0, 1, 50)
    ne = 1e20 * np.ones(50)
    Ti_hot_core = 5000.0 * (1.0 - 0.6 * rho**2)
    q = 1.0 + rho
    eps = 0.2 + 0.2 * rho

    V_W = neoclassical_impurity_pinch(74, ne, 5000.0 * np.ones(50), Ti_hot_core, q, rho, 6.2, 2.0, eps)

    assert V_W[25] < 0.0
    assert abs(V_W[-2]) > abs(V_W[2])


def test_impurity_pinch_rejects_invalid_domain():
    rho = np.linspace(0, 1, 50)
    ne = np.ones(50) * 1e20
    Ti = np.ones(50) * 5000.0
    q = np.ones(50)
    eps = 0.2 + 0.2 * rho

    with pytest.raises(ValueError, match="Z"):
        neoclassical_impurity_pinch(0, ne, Ti, Ti, q, rho, 6.2, 2.0, eps)
    with pytest.raises(ValueError, match="ne"):
        neoclassical_impurity_pinch(74, np.zeros(50), Ti, Ti, q, rho, 6.2, 2.0, eps)
    with pytest.raises(ValueError, match="matching shapes"):
        neoclassical_impurity_pinch(74, ne[:-1], Ti, Ti, q, rho, 6.2, 2.0, eps)


def test_zero_source():
    species = [ImpuritySpecies("W", 74, 183.8, source_rate=0.0)]
    solver = ImpurityTransportSolver(np.linspace(0, 1, 50), 6.2, 2.0, species)

    res = solver.step(0.1, np.ones(50), np.ones(50), np.ones(50), 1.0, {})
    assert np.allclose(res["W"], 0.0)


def test_steady_source():
    species = [ImpuritySpecies("W", 74, 183.8, source_rate=1e16)]  # atoms/m2/s
    solver = ImpurityTransportSolver(np.linspace(0, 1, 50), 6.2, 2.0, species)

    res = solver.step(0.1, np.ones(50), np.ones(50), np.ones(50), 1.0, {"W": np.zeros(50)})

    # Should build up at the edge
    assert res["W"][-1] > 0.0


def test_edge_source_is_volume_conservative_and_resolved():
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


def test_total_radiated_power():
    rho = np.linspace(0, 1, 50)
    ne = np.ones(50) * 1e20
    Te = np.ones(50) * 1500.0

    # Concentration 1e-4 W
    nW = ne * 1e-4
    n_imp = {"W": nW}

    P_rad = total_radiated_power(ne, n_imp, Te, rho, 6.2, 2.0)
    assert P_rad > 10.0  # Should be substantial (tens of MW)


def test_accumulation_diagnostic():
    ne = np.ones(50) * 1e20
    nW = np.ones(50) * 1e16  # c_W = 1e-4 -> critical
    nW[0] = 1e17  # Core peaked -> peaking factor 10

    diag = tungsten_accumulation_diagnostic(nW, ne)

    assert diag["danger_level"] == "critical"
    assert diag["peaking_factor"] == 10.0
