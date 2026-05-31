# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Fokker-Planck RE Solver Tests

import numpy as np
import pytest
from scpn_fusion.control.fokker_planck_re import DreamKineticArtifact, FokkerPlanckSolver


def test_fokker_planck_initialization():
    solver = FokkerPlanckSolver(np_grid=50, p_max=10.0)
    assert solver.p.shape == (50,)
    assert solver.f.shape == (50,)
    assert solver.time == 0.0


def test_fokker_planck_coefficients_finite():
    solver = FokkerPlanckSolver()
    A, D, Fc = solver.compute_coefficients(E_field=1.0, n_e=5e19, Z_eff=1.5, T_e_eV=5000.0)
    assert np.all(np.isfinite(A))
    assert np.all(np.isfinite(D))
    assert Fc > 0.0


def test_fokker_planck_step_conserves_positivity():
    solver = FokkerPlanckSolver()
    # Inject some population
    solver.f[10] = 1.0e10

    # Evolve
    # High E field to drive acceleration
    state = solver.step(dt=1e-5, E_field=10.0, n_e=5e19, T_e_eV=5000.0, Z_eff=1.0)

    assert np.all(state.f >= 0.0)
    assert state.n_re > 0.0
    assert state.current_re > 0.0


# S2-001: Fokker-Planck 1/p² divergence guard


def test_drag_finite_near_zero_momentum():
    """With thermal regularization, drag should be finite even near p → 0."""
    solver = FokkerPlanckSolver(np_grid=100, p_max=10.0)
    A, D, Fc = solver.compute_coefficients(E_field=1.0, n_e=5e19, Z_eff=1.5, T_e_eV=5000.0)
    assert np.all(np.isfinite(A)), "Advection has non-finite values near p=0"
    assert np.all(np.isfinite(D)), "Diffusion has non-finite values near p=0"


def test_drag_regularized_at_thermal_speed():
    """Drag at the smallest momentum should be bounded (not divergent)."""
    solver = FokkerPlanckSolver(np_grid=200, p_max=10.0)
    A, D, Fc = solver.compute_coefficients(
        E_field=1.0,
        n_e=5e19,
        Z_eff=1.5,
        T_e_eV=100.0,  # lower Te
    )
    # At the first grid point (smallest p), drag should be large but finite
    assert np.isfinite(A[0])
    # The total advection (drag + synchrotron) should be bounded, not divergent
    # With regularization the value is O(10^5) rather than Inf
    assert abs(A[0]) < 1e8, f"A[0]={A[0]:.2e} is unreasonably large"


def test_fokker_planck_rejects_invalid_constructor_inputs():
    with pytest.raises(ValueError, match="np_grid"):
        FokkerPlanckSolver(np_grid=4, p_max=10.0)
    with pytest.raises(ValueError, match="p_max"):
        FokkerPlanckSolver(np_grid=32, p_max=0.0)


def test_fokker_planck_step_rejects_invalid_dt():
    solver = FokkerPlanckSolver()
    with pytest.raises(ValueError, match="dt"):
        solver.step(dt=0.0, E_field=10.0, n_e=5e19, T_e_eV=5000.0, Z_eff=1.0)


def test_dream_kinetic_artifact_exports_radius_momentum_pitch_contract() -> None:
    solver = FokkerPlanckSolver(np_grid=32, p_max=8.0)
    solver.f[8] = 1.0e10

    artifact = solver.run_dream_kinetic_artifact(
        n_steps=3,
        dt=1.0e-6,
        e_field=10.0,
        n_e=5.0e19,
        t_e_ev=2500.0,
        z_eff=2.0,
        radius_m=[0.0, 0.5, 1.0],
        pitch_cosine=[-1.0, 0.0, 1.0],
    )

    assert isinstance(artifact, DreamKineticArtifact)
    payload = artifact.to_dict()
    assert payload["schema"] == "dream-kinetic-artifact.v1"
    assert payload["coordinate_units"] == {
        "time_s": "s",
        "radius_m": "m",
        "momentum_mec": "m_e_c",
        "pitch_cosine": "dimensionless",
    }
    assert payload["observable_axes"]["f_p_xi_t"] == [
        "time_s",
        "radius_m",
        "momentum_mec",
        "pitch_cosine",
    ]
    f = np.asarray(payload["observables"]["f_p_xi_t"], dtype=np.float64)
    assert f.shape == (3, 3, 32, 3)
    assert np.all(np.isfinite(f))
    assert np.all(f >= 0.0)
    for name in [
        "runaway_current_t",
        "avalanche_growth_rate_t",
        "synchrotron_loss_power_t",
        "partial_screening_drag_t",
        "bremsstrahlung_loss_power_t",
    ]:
        arr = np.asarray(payload["observables"][name], dtype=np.float64)
        assert arr.shape == (3, 3)
        assert np.all(np.isfinite(arr))


def test_dream_kinetic_artifact_rejects_invalid_axes() -> None:
    solver = FokkerPlanckSolver(np_grid=32, p_max=8.0)
    with pytest.raises(ValueError, match="pitch_cosine"):
        solver.run_dream_kinetic_artifact(
            n_steps=2,
            dt=1.0e-6,
            e_field=10.0,
            n_e=5.0e19,
            t_e_ev=2500.0,
            z_eff=2.0,
            radius_m=[0.0, 0.5, 1.0],
            pitch_cosine=[-1.0, -1.0, 1.0],
        )
