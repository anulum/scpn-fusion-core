# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Aurora/STRAHL Parity Solver Tests
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_fusion.core.impurity_transport_aurora_parity import (
    AuroraParityImpuritySolver,
    build_aurora_strahl_charge_state_artifact,
)
from scpn_fusion.core.impurity_transport_contracts import AuroraParityCase, AuroraStrahlArtifact


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


def _aurora_parity_case(*, with_optional: bool = False) -> AuroraParityCase:
    charge_states = np.array([0.0, 1.0, 2.0])
    radius_m = np.array([0.0, 0.5, 1.0])
    time_s = np.array([0.0, 0.01, 0.02])
    nt, nr, nz = time_s.size, radius_m.size, charge_states.size
    kwargs: dict[str, Any] = {
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
    def test_solve_parametric_path_exports_artifact(self) -> None:
        artifact = AuroraParityImpuritySolver(_aurora_parity_case()).solve()
        assert isinstance(artifact, AuroraStrahlArtifact)
        payload = artifact.to_dict()
        assert payload["schema"] == "aurora-strahl-charge-state-artifact.v1"
        assert payload["provenance"]["reference_family"] == "Aurora/STRAHL"
        assert np.isfinite(payload["conservation"]["relative_inventory_error"])
        assert len(payload["observables"]["line_radiation_power_t"]) == 3

    def test_solve_with_provided_tables_and_effective_source(self) -> None:
        artifact = AuroraParityImpuritySolver(_aurora_parity_case(with_optional=True)).solve()
        power = artifact.observables["line_radiation_power_t"]
        assert len(power) == 3
        assert all(np.isfinite(v) for v in power)

    def test_radial_transport_budget_conserves_inventory(self) -> None:
        case = _aurora_parity_case()
        diag = AuroraParityImpuritySolver(case).radial_transport_budget_diagnostic(
            case.initial_charge_state_density_rz, dt_s=1e-4
        )
        assert diag["passed"] is True
        assert diag["relative_inventory_error"] <= 1e-12

    def test_derive_effective_source_closure_shape(self) -> None:
        solver = AuroraParityImpuritySolver(_aurora_parity_case())
        closure = solver.derive_effective_source_closure(np.full((3, 3, 3), 1e15))
        assert closure.shape == (3, 3, 3)
        assert np.all(np.isfinite(closure))

    def test_budget_rejects_bad_density_shape(self) -> None:
        solver = AuroraParityImpuritySolver(_aurora_parity_case())
        with pytest.raises(ValueError, match="density_r_z must have shape"):
            solver.radial_transport_budget_diagnostic(np.zeros((2, 2)), dt_s=1e-4)

    def test_budget_rejects_bad_dt(self) -> None:
        case = _aurora_parity_case()
        with pytest.raises(ValueError, match="dt_s must be finite and positive"):
            AuroraParityImpuritySolver(case).radial_transport_budget_diagnostic(
                case.initial_charge_state_density_rz, dt_s=0.0
            )

    def test_closure_rejects_bad_shape(self) -> None:
        solver = AuroraParityImpuritySolver(_aurora_parity_case())
        with pytest.raises(ValueError, match="reference_density_t_r_z must have shape"):
            solver.derive_effective_source_closure(np.zeros((2, 2, 2)))

    def test_closure_rejects_negative(self) -> None:
        solver = AuroraParityImpuritySolver(_aurora_parity_case())
        with pytest.raises(ValueError, match="finite and non-negative"):
            solver.derive_effective_source_closure(np.full((3, 3, 3), -1.0))


def _build_kwargs() -> dict[str, Any]:
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
    def test_rejects_negative_radius(self) -> None:
        kw = _build_kwargs()
        kw["radius_m"] = np.array([-0.1, 0.5, 1.0])
        with pytest.raises(ValueError, match="radius_m must be non-negative"):
            build_aurora_strahl_charge_state_artifact(**kw)

    def test_rejects_bad_major_radius(self) -> None:
        kw = _build_kwargs()
        kw["major_radius_m"] = 0.0
        with pytest.raises(ValueError, match="major_radius_m must be finite and positive"):
            build_aurora_strahl_charge_state_artifact(**kw)

    def test_rejects_bad_ne_shape(self) -> None:
        kw = _build_kwargs()
        kw["ne_t_r"] = np.full((2, 3), 1e19)
        with pytest.raises(ValueError, match="ne_t_r and Te_t_r must have shape"):
            build_aurora_strahl_charge_state_artifact(**kw)

    def test_rejects_bad_density_shape(self) -> None:
        kw = _build_kwargs()
        kw["initial_charge_state_density_rz"] = np.full((3, 2), 1e15)
        with pytest.raises(ValueError, match="initial_charge_state_density_rz must have shape"):
            build_aurora_strahl_charge_state_artifact(**kw)

    def test_rejects_nonpositive_ne(self) -> None:
        kw = _build_kwargs()
        kw["ne_t_r"] = np.zeros((3, 3))
        with pytest.raises(ValueError, match="ne_t_r must be finite and positive"):
            build_aurora_strahl_charge_state_artifact(**kw)

    def test_rejects_nonpositive_te(self) -> None:
        kw = _build_kwargs()
        kw["Te_t_r"] = np.zeros((3, 3))
        with pytest.raises(ValueError, match="Te_t_r must be finite and positive"):
            build_aurora_strahl_charge_state_artifact(**kw)

    def test_rejects_negative_density(self) -> None:
        kw = _build_kwargs()
        kw["initial_charge_state_density_rz"] = -np.ones((3, 3))
        with pytest.raises(
            ValueError, match="initial_charge_state_density_rz must be finite and non"
        ):
            build_aurora_strahl_charge_state_artifact(**kw)


def test_budget_rejects_negative_density() -> None:
    case = _aurora_parity_case()
    with pytest.raises(ValueError, match="density_r_z must be finite and non-negative"):
        AuroraParityImpuritySolver(case).radial_transport_budget_diagnostic(
            np.full((3, 3), -1.0), dt_s=1e-4
        )
