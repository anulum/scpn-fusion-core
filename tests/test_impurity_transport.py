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
    AuroraStrahlArtifact,
    CoolingCurve,
    ImpuritySpecies,
    ImpurityTransportSolver,
    adas_style_charge_state_coefficients,
    advance_charge_state_collisional_radiative,
    build_aurora_strahl_charge_state_artifact,
    collisional_radiative_source_sink_matrices,
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

    V_W = neoclassical_impurity_pinch(
        74, ne, 5000.0 * np.ones(50), Ti_hot_core, q, rho, 6.2, 2.0, eps
    )

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


def test_charge_state_collisional_radiative_step_conserves_density() -> None:
    radius = np.linspace(0.0, 1.0, 8)
    charge = np.array([0, 1, 2, 3], dtype=float)
    ne = np.ones_like(radius) * 1.0e20
    coeffs = adas_style_charge_state_coefficients("Ar", charge, np.linspace(100.0, 600.0, 4))
    density = np.zeros((radius.size, charge.size))
    density[:, 1] = 1.0e15 * (1.0 - 0.2 * radius)

    updated, ion, rec = advance_charge_state_collisional_radiative(density, ne, coeffs, dt=1.0e-5)

    assert ion.shape == density.shape
    assert rec.shape == density.shape
    assert np.all(updated >= 0.0)
    np.testing.assert_allclose(np.sum(updated, axis=1), np.sum(density, axis=1), rtol=1e-13)


def test_collisional_radiative_source_sink_rejects_invalid_shapes() -> None:
    charge = np.array([0, 1, 2], dtype=float)
    coeffs = adas_style_charge_state_coefficients("Ne", charge, np.array([50.0, 100.0, 200.0]))
    with pytest.raises(ValueError, match="radius"):
        collisional_radiative_source_sink_matrices(
            np.ones((4, 3)),
            np.ones(3) * 1.0e20,
            coeffs,
        )


def test_aurora_strahl_charge_state_artifact_contract() -> None:
    radius = np.linspace(0.0, 1.0, 7)
    time = np.array([0.0, 1.0e-5, 2.0e-5])
    charge = np.array([0, 1, 2, 3], dtype=float)
    ne = np.ones((time.size, radius.size)) * 1.0e20
    te = np.tile(np.linspace(100.0, 500.0, radius.size), (time.size, 1))
    density = np.zeros((radius.size, charge.size))
    density[:, 1] = 1.0e15

    artifact = build_aurora_strahl_charge_state_artifact(
        element="Ar",
        charge_states=charge,
        radius_m=radius,
        time_s=time,
        ne_t_r=ne,
        Te_t_r=te,
        initial_charge_state_density_rz=density,
        major_radius_m=6.2,
    )

    assert isinstance(artifact, AuroraStrahlArtifact)
    payload = artifact.to_dict()
    assert payload["schema"] == "aurora-strahl-charge-state-artifact.v1"
    assert payload["observable_axes"]["charge_state_density_r_t"] == [
        "time_s",
        "radius_m",
        "charge_state",
    ]
    charge_density = np.asarray(payload["observables"]["charge_state_density_r_t"])
    total_density = np.asarray(payload["observables"]["total_impurity_density_r_t"])
    line_power = np.asarray(payload["observables"]["line_radiation_power_t"])
    assert charge_density.shape == (3, 7, 4)
    assert total_density.shape == (3, 7)
    assert line_power.shape == (3,)
    assert np.all(np.isfinite(charge_density))
    assert np.all(charge_density >= 0.0)
    np.testing.assert_allclose(total_density, np.sum(charge_density, axis=2))
    assert artifact.conservation["relative_inventory_error"] <= 1.0e-12
