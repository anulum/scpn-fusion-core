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
    AdasChargeStateCoefficients,
    AuroraParityCase,
    AuroraParityImpuritySolver,
    AuroraStrahlArtifact,
    CoolingCurve,
    ImpuritySpecies,
    ImpurityTransportSolver,
    _source_sink_transfer_matrix,
    _strict_axis,
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


def test_tungsten_cooling_curve_peak_order_of_magnitude():
    """Pin the absolute peak of the tungsten cooling curve.

    Putterich et al. 2010: tungsten coronal Lz peaks at ~1e-31 W m^3 near
    ~1.5 keV. The other ``test_cooling_curves`` checks only an ordering, so a
    wrong peak magnitude would pass it.
    """
    te_eV = np.logspace(1.0, 4.0, 400)  # 10 eV - 10 keV
    cooling = CoolingCurve("W").L_z(te_eV)
    peak = float(np.max(cooling))
    assert 3e-32 < peak < 3e-31
    assert 800.0 < te_eV[int(np.argmax(cooling))] < 2500.0


def test_light_impurity_cooling_curve_peaks_match_verified_data():
    """Pin the C/Ne/Ar cooling-curve peaks to their verified source values.

    Carbon, neon, and argon are the OpenADAS adf11 coronal cooling rates computed
    by tools/compute_coronal_lz_from_adas.py (carbon 5.84e-32 near 7 eV, neon
    5.74e-32 near 30 eV, argon 1.98e-31 near 20 eV). All sit in the coronal
    1e-31 band; guards against the prefactor drifting an order of magnitude low
    (the prior bug) or to the unverified Post & Jensen ~2e-31.
    """
    te_eV = np.logspace(0.0, 3.5, 400)  # 1 eV - ~3 keV
    for element in ("C", "Ne", "Ar"):
        peak = float(np.max(CoolingCurve(element).L_z(te_eV)))
        assert 3e-32 < peak < 3e-31, f"{element} peak {peak:.2e} outside coronal band"

    # Peak magnitudes (evaluated at the peak temperature) and locations.
    assert float(CoolingCurve("C").L_z(np.array([7.0]))[0]) == pytest.approx(5.84e-32, rel=1e-9)
    assert float(CoolingCurve("Ne").L_z(np.array([30.0]))[0]) == pytest.approx(5.74e-32, rel=1e-9)
    assert float(CoolingCurve("Ar").L_z(np.array([20.0]))[0]) == pytest.approx(1.98e-31, rel=1e-9)
    assert 5.0 < te_eV[int(np.argmax(CoolingCurve("C").L_z(te_eV)))] < 10.0
    assert 20.0 < te_eV[int(np.argmax(CoolingCurve("Ne").L_z(te_eV)))] < 45.0
    assert 15.0 < te_eV[int(np.argmax(CoolingCurve("Ar").L_z(te_eV)))] < 30.0


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
    validation = artifact.validate_contract()
    assert validation["passed"] is True
    assert validation["observable_shapes"] == {
        "charge_state_density_r_t": [3, 7, 4],
        "total_impurity_density_r_t": [3, 7],
        "line_radiation_power_t": [3],
        "line_radiation_power_t_r_z": [3, 7, 4],
        "source_sink_matrix_t_r_z_z": [3, 7, 4, 4],
        "total_impurity_inventory_t": [3],
        "ionisation_source_matrix": [7, 4],
        "recombination_sink_matrix": [7, 4],
    }
    source_sink = np.asarray(payload["observables"]["source_sink_matrix_t_r_z_z"])
    np.testing.assert_allclose(np.sum(source_sink, axis=3), 0.0, atol=1.0e-6)
    inventory = np.asarray(payload["observables"]["total_impurity_inventory_t"])
    assert inventory.shape == (3,)
    assert inventory[-1] == pytest.approx(inventory[0], rel=1.0e-12)


# ---------------------------------------------------------------------------
# Native Aurora/STRAHL parity solver and charge-state contract validation.
# ---------------------------------------------------------------------------


def _aurora_parity_case(*, with_optional: bool = False) -> AuroraParityCase:
    charge_states = np.array([0.0, 1.0, 2.0])
    radius_m = np.array([0.0, 0.5, 1.0])
    time_s = np.array([0.0, 0.01, 0.02])
    nt, nr, nz = time_s.size, radius_m.size, charge_states.size
    kwargs: dict = {
        "element": "C",
        "charge_states": charge_states,
        "radius_m": radius_m,
        "time_s": time_s,
        "ne_t_r": np.full((nt, nr), 1e19),
        "Te_t_r": np.full((nt, nr), 100.0),
        "initial_charge_state_density_rz": np.full((nr, nz), 1e15),
        "diffusion_m2_s_r_z": np.full((nr, nz), 0.5),
        "convection_m_s_r_z": np.full((nr, nz), -1.0),
        "major_radius_m": 1.65,
    }
    if with_optional:
        kwargs.update(
            ionisation_m3_s_t_r_z=np.full((nt, nr, nz), 1e-14),
            recombination_m3_s_t_r_z=np.full((nt, nr, nz), 1e-14),
            line_radiation_w_m3_t_r_z=np.full((nt, nr, nz), 1e-32),
            effective_source_m3_s_t_r_z=np.zeros((nt, nr, nz)),
        )
    return AuroraParityCase(**kwargs)


class TestAuroraParitySolver:
    def test_solve_parametric_path_exports_artifact(self):
        artifact = AuroraParityImpuritySolver(_aurora_parity_case()).solve()
        assert isinstance(artifact, AuroraStrahlArtifact)
        payload = artifact.to_dict()
        assert payload["schema"] == "aurora-strahl-charge-state-artifact.v1"
        assert payload["provenance"]["reference_family"] == "Aurora/STRAHL"
        assert np.isfinite(payload["conservation"]["relative_inventory_error"])
        assert len(payload["observables"]["line_radiation_power_t"]) == 3

    def test_solve_with_provided_tables_and_effective_source(self):
        artifact = AuroraParityImpuritySolver(_aurora_parity_case(with_optional=True)).solve()
        power = artifact.observables["line_radiation_power_t"]
        assert len(power) == 3
        assert all(np.isfinite(v) for v in power)

    def test_radial_transport_budget_conserves_inventory(self):
        case = _aurora_parity_case()
        diag = AuroraParityImpuritySolver(case).radial_transport_budget_diagnostic(
            case.initial_charge_state_density_rz, dt_s=1e-4
        )
        assert diag["passed"] is True
        assert diag["relative_inventory_error"] <= 1e-12

    def test_derive_effective_source_closure_shape(self):
        solver = AuroraParityImpuritySolver(_aurora_parity_case())
        closure = solver.derive_effective_source_closure(np.full((3, 3, 3), 1e15))
        assert closure.shape == (3, 3, 3)
        assert np.all(np.isfinite(closure))

    def test_budget_rejects_bad_density_shape(self):
        solver = AuroraParityImpuritySolver(_aurora_parity_case())
        with pytest.raises(ValueError, match="density_r_z must have shape"):
            solver.radial_transport_budget_diagnostic(np.zeros((2, 2)), dt_s=1e-4)

    def test_budget_rejects_bad_dt(self):
        case = _aurora_parity_case()
        with pytest.raises(ValueError, match="dt_s must be finite and positive"):
            AuroraParityImpuritySolver(case).radial_transport_budget_diagnostic(
                case.initial_charge_state_density_rz, dt_s=0.0
            )

    def test_closure_rejects_bad_shape(self):
        solver = AuroraParityImpuritySolver(_aurora_parity_case())
        with pytest.raises(ValueError, match="reference_density_t_r_z must have shape"):
            solver.derive_effective_source_closure(np.zeros((2, 2, 2)))

    def test_closure_rejects_negative(self):
        solver = AuroraParityImpuritySolver(_aurora_parity_case())
        with pytest.raises(ValueError, match="finite and non-negative"):
            solver.derive_effective_source_closure(np.full((3, 3, 3), -1.0))


class TestAuroraParityCaseValidation:
    def test_rejects_negative_radius(self):
        with pytest.raises(ValueError, match="radius_m must be non-negative"):
            _make_case_with(radius_m=np.array([-0.1, 0.5, 1.0]))

    def test_rejects_bad_major_radius(self):
        with pytest.raises(ValueError, match="major_radius_m must be finite and positive"):
            _make_case_with(major_radius_m=0.0)

    def test_rejects_wrong_field_shape(self):
        with pytest.raises(ValueError, match="ne_t_r must have shape"):
            _make_case_with(ne_t_r=np.full((2, 3), 1e19))

    def test_rejects_nonpositive_ne(self):
        with pytest.raises(ValueError, match="ne_t_r must be positive"):
            _make_case_with(ne_t_r=np.zeros((3, 3)))

    def test_rejects_negative_diffusion(self):
        with pytest.raises(ValueError, match="diffusion_m2_s_r_z must be non-negative"):
            _make_case_with(diffusion_m2_s_r_z=np.full((3, 3), -1.0))

    def test_rejects_optional_wrong_shape(self):
        with pytest.raises(ValueError, match="ionisation_m3_s_t_r_z must have shape"):
            _make_case_with(ionisation_m3_s_t_r_z=np.zeros((2, 3, 3)))

    def test_rejects_optional_negative(self):
        with pytest.raises(ValueError, match="recombination_m3_s_t_r_z must be finite and non"):
            _make_case_with(recombination_m3_s_t_r_z=np.full((3, 3, 3), -1.0))

    def test_rejects_effective_source_wrong_shape(self):
        with pytest.raises(ValueError, match="effective_source_m3_s_t_r_z must have shape"):
            _make_case_with(effective_source_m3_s_t_r_z=np.zeros((2, 3, 3)))

    def test_rejects_effective_source_non_finite(self):
        with pytest.raises(ValueError, match="effective_source_m3_s_t_r_z must be finite"):
            _make_case_with(effective_source_m3_s_t_r_z=np.full((3, 3, 3), np.inf))


def _make_case_with(**overrides) -> AuroraParityCase:
    base: dict = {
        "element": "C",
        "charge_states": np.array([0.0, 1.0, 2.0]),
        "radius_m": np.array([0.0, 0.5, 1.0]),
        "time_s": np.array([0.0, 0.01, 0.02]),
        "ne_t_r": np.full((3, 3), 1e19),
        "Te_t_r": np.full((3, 3), 100.0),
        "initial_charge_state_density_rz": np.full((3, 3), 1e15),
        "diffusion_m2_s_r_z": np.full((3, 3), 0.5),
        "convection_m_s_r_z": np.full((3, 3), -1.0),
        "major_radius_m": 1.65,
    }
    base.update(overrides)
    return AuroraParityCase(**base)


class TestImpuritySpeciesValidation:
    @pytest.mark.parametrize(
        "override, match",
        [
            ({"Z_nucleus": 0}, "Z_nucleus must be positive"),
            ({"mass_amu": 0.0}, "mass_amu must be finite and positive"),
            ({"source_rate": -1.0}, "source_rate must be finite and non-negative"),
            ({"source_decay_width_rho": 0.0}, "source_decay_width_rho must be finite and positive"),
        ],
    )
    def test_rejects_invalid_fields(self, override, match):
        base = {"element": "W", "Z_nucleus": 74, "mass_amu": 183.84}
        base.update(override)
        with pytest.raises(ValueError, match=match):
            ImpuritySpecies(**base)


class TestAdasChargeStateCoefficientValidation:
    def test_rejects_short_charge_axis(self):
        with pytest.raises(ValueError, match="at least two states"):
            AdasChargeStateCoefficients(
                np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0])
            )

    def test_rejects_non_increasing_charge(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            AdasChargeStateCoefficients(np.array([1.0, 1.0]), np.zeros(2), np.zeros(2), np.zeros(2))

    def test_rejects_non_integer_charge(self):
        with pytest.raises(ValueError, match="integer charge"):
            AdasChargeStateCoefficients(np.array([0.0, 1.5]), np.zeros(2), np.zeros(2), np.zeros(2))

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="must match charge_states shape"):
            AdasChargeStateCoefficients(
                np.array([0.0, 1.0, 2.0]), np.zeros(2), np.zeros(3), np.zeros(3)
            )

    def test_rejects_negative_values(self):
        with pytest.raises(ValueError, match="finite and non-negative"):
            AdasChargeStateCoefficients(
                np.array([0.0, 1.0, 2.0]), np.array([-1.0, 0.0, 0.0]), np.zeros(3), np.zeros(3)
            )


class TestCoolingCurveBranches:
    def test_all_invalid_temperatures_return_zeros(self):
        out = CoolingCurve("W").L_z(np.array([-1.0, 0.0, np.nan]))
        assert np.allclose(out, 0.0)

    def test_unknown_element_returns_zeros(self):
        out = CoolingCurve("Xe").L_z(np.array([10.0, 100.0]))
        assert np.allclose(out, 0.0)


class TestAdasStyleCoefficientValidation:
    def test_rejects_te_shape_mismatch(self):
        with pytest.raises(ValueError, match="Te_eV must match charge_states shape"):
            adas_style_charge_state_coefficients(
                "C", np.array([0.0, 1.0, 2.0]), np.array([100.0, 100.0])
            )

    def test_rejects_nonpositive_te(self):
        with pytest.raises(ValueError, match="Te_eV must be finite and positive"):
            adas_style_charge_state_coefficients(
                "C", np.array([0.0, 1.0, 2.0]), np.array([100.0, 0.0, 100.0])
            )

    def test_rejects_short_charge_axis(self):
        with pytest.raises(ValueError, match="at least two states"):
            adas_style_charge_state_coefficients("C", np.array([0.0]), np.array([100.0]))

    def test_rejects_non_increasing_charge(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            adas_style_charge_state_coefficients(
                "C", np.array([1.0, 1.0]), np.array([100.0, 100.0])
            )


def _cr_coeffs() -> AdasChargeStateCoefficients:
    return adas_style_charge_state_coefficients(
        "C", np.array([0.0, 1.0, 2.0]), np.array([50.0, 100.0, 200.0])
    )


class TestCollisionalRadiativeValidation:
    def test_rejects_non_2d_density(self):
        with pytest.raises(ValueError, match="radius x charge_state"):
            collisional_radiative_source_sink_matrices(np.zeros(3), np.full(3, 1e19), _cr_coeffs())

    def test_rejects_ne_dimension_mismatch(self):
        with pytest.raises(ValueError, match="ne must match the radius dimension"):
            collisional_radiative_source_sink_matrices(
                np.ones((3, 3)), np.full(2, 1e19), _cr_coeffs()
            )

    def test_rejects_coeff_axis_mismatch(self):
        with pytest.raises(ValueError, match="coefficient charge axis"):
            collisional_radiative_source_sink_matrices(
                np.ones((3, 2)), np.full(3, 1e19), _cr_coeffs()
            )

    def test_rejects_negative_density(self):
        with pytest.raises(ValueError, match="finite and non-negative"):
            collisional_radiative_source_sink_matrices(
                -np.ones((3, 3)), np.full(3, 1e19), _cr_coeffs()
            )

    def test_rejects_nonpositive_ne(self):
        with pytest.raises(ValueError, match="ne must be finite and positive"):
            collisional_radiative_source_sink_matrices(
                np.ones((3, 3)), np.array([1e19, 0.0, 1e19]), _cr_coeffs()
            )

    def test_advance_rejects_bad_dt(self):
        with pytest.raises(ValueError, match="dt must be finite and positive"):
            advance_charge_state_collisional_radiative(
                np.ones((3, 3)), np.full(3, 1e19), _cr_coeffs(), 0.0
            )

    def test_source_sink_transfer_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="matching radius x charge"):
            _source_sink_transfer_matrix(np.ones((3, 2)), np.ones((3, 3)))


def test_strict_axis_rejects_short_and_non_monotonic():
    with pytest.raises(ValueError, match="at least 2 points"):
        _strict_axis("x", np.array([1.0]))
    with pytest.raises(ValueError, match="strictly increasing"):
        _strict_axis("x", np.array([1.0, 0.0]))


def _build_kwargs() -> dict:
    return {
        "element": "C",
        "charge_states": np.array([0.0, 1.0, 2.0]),
        "radius_m": np.array([0.0, 0.5, 1.0]),
        "time_s": np.array([0.0, 0.01, 0.02]),
        "ne_t_r": np.full((3, 3), 1e19),
        "Te_t_r": np.full((3, 3), 100.0),
        "initial_charge_state_density_rz": np.full((3, 3), 1e15),
        "major_radius_m": 1.65,
    }


class TestBuildAuroraArtifactValidation:
    def test_rejects_negative_radius(self):
        kw = _build_kwargs()
        kw["radius_m"] = np.array([-0.1, 0.5, 1.0])
        with pytest.raises(ValueError, match="radius_m must be non-negative"):
            build_aurora_strahl_charge_state_artifact(**kw)

    def test_rejects_bad_major_radius(self):
        kw = _build_kwargs()
        kw["major_radius_m"] = 0.0
        with pytest.raises(ValueError, match="major_radius_m must be finite and positive"):
            build_aurora_strahl_charge_state_artifact(**kw)

    def test_rejects_bad_ne_shape(self):
        kw = _build_kwargs()
        kw["ne_t_r"] = np.full((2, 3), 1e19)
        with pytest.raises(ValueError, match="ne_t_r and Te_t_r must have shape"):
            build_aurora_strahl_charge_state_artifact(**kw)

    def test_rejects_bad_density_shape(self):
        kw = _build_kwargs()
        kw["initial_charge_state_density_rz"] = np.full((3, 2), 1e15)
        with pytest.raises(ValueError, match="initial_charge_state_density_rz must have shape"):
            build_aurora_strahl_charge_state_artifact(**kw)

    def test_rejects_nonpositive_ne(self):
        kw = _build_kwargs()
        kw["ne_t_r"] = np.zeros((3, 3))
        with pytest.raises(ValueError, match="ne_t_r must be finite and positive"):
            build_aurora_strahl_charge_state_artifact(**kw)

    def test_rejects_nonpositive_te(self):
        kw = _build_kwargs()
        kw["Te_t_r"] = np.zeros((3, 3))
        with pytest.raises(ValueError, match="Te_t_r must be finite and positive"):
            build_aurora_strahl_charge_state_artifact(**kw)

    def test_rejects_negative_density(self):
        kw = _build_kwargs()
        kw["initial_charge_state_density_rz"] = -np.ones((3, 3))
        with pytest.raises(
            ValueError, match="initial_charge_state_density_rz must be finite and non"
        ):
            build_aurora_strahl_charge_state_artifact(**kw)


def _pinch_kwargs() -> dict:
    return {
        "Z": 18,
        "ne": np.full(5, 1e19),
        "Te_eV": np.full(5, 1000.0),
        "Ti_eV": np.full(5, 1000.0),
        "q": np.full(5, 2.0),
        "rho": np.linspace(0.1, 1.0, 5),
        "R0": 6.2,
        "a": 2.0,
        "epsilon": np.linspace(0.05, 0.3, 5),
    }


class TestNeoclassicalPinchValidation:
    def test_rejects_short_rho(self):
        kw = _pinch_kwargs()
        for key in ("ne", "Te_eV", "Ti_eV", "q", "rho", "epsilon"):
            kw[key] = kw[key][:2]
        with pytest.raises(ValueError, match="at least three points"):
            neoclassical_impurity_pinch(**kw)

    def test_rejects_non_increasing_rho(self):
        kw = _pinch_kwargs()
        kw["rho"] = np.array([1.0, 0.7, 0.5, 0.3, 0.1])
        with pytest.raises(ValueError, match="rho must be finite and strictly increasing"):
            neoclassical_impurity_pinch(**kw)

    def test_rejects_nonpositive_ti(self):
        kw = _pinch_kwargs()
        kw["Ti_eV"] = np.array([1000.0, 0.0, 1000.0, 1000.0, 1000.0])
        with pytest.raises(ValueError, match="Ti_eV must be finite and positive"):
            neoclassical_impurity_pinch(**kw)

    def test_rejects_nonpositive_q(self):
        kw = _pinch_kwargs()
        kw["q"] = np.array([2.0, 2.0, 0.0, 2.0, 2.0])
        with pytest.raises(ValueError, match="q must be finite and positive"):
            neoclassical_impurity_pinch(**kw)

    def test_rejects_negative_epsilon(self):
        kw = _pinch_kwargs()
        kw["epsilon"] = np.array([0.05, -0.1, 0.1, 0.2, 0.3])
        with pytest.raises(ValueError, match="epsilon must be finite and non-negative"):
            neoclassical_impurity_pinch(**kw)

    def test_rejects_bad_major_radius(self):
        kw = _pinch_kwargs()
        kw["R0"] = 0.0
        with pytest.raises(ValueError, match="R0 must be finite and positive"):
            neoclassical_impurity_pinch(**kw)

    def test_rejects_bad_minor_radius(self):
        kw = _pinch_kwargs()
        kw["a"] = 0.0
        with pytest.raises(ValueError, match="a must be finite and positive"):
            neoclassical_impurity_pinch(**kw)


def test_tungsten_diagnostic_danger_levels():
    ne = np.full(3, 1e20)
    safe = tungsten_accumulation_diagnostic(np.full(3, 1e14), ne)
    assert safe["danger_level"] == "safe"
    warning = tungsten_accumulation_diagnostic(np.full(3, 3e15), ne)
    assert warning["danger_level"] == "warning"


class TestImpurityTransportSolverConstruction:
    def test_rejects_short_rho(self):
        with pytest.raises(ValueError, match="at least three radial points"):
            ImpurityTransportSolver(np.array([0.0, 1.0]), 6.2, 2.0, [])

    def test_rejects_non_finite_rho(self):
        with pytest.raises(ValueError, match="only finite values"):
            ImpurityTransportSolver(np.array([0.0, np.nan, 1.0]), 6.2, 2.0, [])

    def test_rejects_non_increasing_rho(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            ImpurityTransportSolver(np.array([0.0, 0.7, 0.3, 1.0]), 6.2, 2.0, [])

    def test_rejects_rho_not_spanning_unit_interval(self):
        with pytest.raises(ValueError, match=r"normalised interval \[0, 1\]"):
            ImpurityTransportSolver(np.array([0.1, 0.5, 0.9]), 6.2, 2.0, [])

    def test_rejects_bad_major_radius(self):
        with pytest.raises(ValueError, match="R0 must be finite and positive"):
            ImpurityTransportSolver(np.linspace(0.0, 1.0, 5), 0.0, 2.0, [])

    def test_rejects_bad_minor_radius(self):
        with pytest.raises(ValueError, match="a must be finite and positive"):
            ImpurityTransportSolver(np.linspace(0.0, 1.0, 5), 6.2, 0.0, [])

    def test_rejects_non_uniform_grid(self):
        with pytest.raises(ValueError, match="uniformly spaced"):
            ImpurityTransportSolver(np.array([0.0, 0.3, 1.0]), 6.2, 2.0, [])


def test_step_with_outward_pinch_uses_positive_upwind_branch():
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


def test_aurora_parity_case_rejects_non_finite_required_field():
    with pytest.raises(ValueError, match="convection_m_s_r_z must be finite"):
        _make_case_with(convection_m_s_r_z=np.full((3, 3), np.nan))


def test_budget_rejects_negative_density():
    case = _aurora_parity_case()
    with pytest.raises(ValueError, match="density_r_z must be finite and non-negative"):
        AuroraParityImpuritySolver(case).radial_transport_budget_diagnostic(
            np.full((3, 3), -1.0), dt_s=1e-4
        )
