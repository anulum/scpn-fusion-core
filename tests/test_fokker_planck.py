# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Fokker-Planck RE Solver Tests

import numpy as np
import pytest
from scpn_fusion.control.fokker_planck_re import (
    _RE_SEED_FLOOR,
    DreamKineticArtifact,
    FokkerPlanckKernel,
    FokkerPlanckSolver,
    create_fokker_planck_kernel,
)


def test_fokker_planck_initialization() -> None:
    solver = FokkerPlanckSolver(np_grid=50, p_max=10.0)
    assert solver.p.shape == (50,)
    assert solver.f.shape == (50,)
    assert solver.time == 0.0


def test_fokker_planck_coefficients_finite() -> None:
    solver = FokkerPlanckSolver()
    A, D, Fc = solver.compute_coefficients(E_field=1.0, n_e=5e19, Z_eff=1.5, T_e_eV=5000.0)
    assert np.all(np.isfinite(A))
    assert np.all(np.isfinite(D))
    assert Fc > 0.0


def test_fokker_planck_step_conserves_positivity() -> None:
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


def test_drag_finite_near_zero_momentum() -> None:
    """With thermal regularization, drag should be finite even near p → 0."""
    solver = FokkerPlanckSolver(np_grid=100, p_max=10.0)
    A, D, Fc = solver.compute_coefficients(E_field=1.0, n_e=5e19, Z_eff=1.5, T_e_eV=5000.0)
    assert np.all(np.isfinite(A)), "Advection has non-finite values near p=0"
    assert np.all(np.isfinite(D)), "Diffusion has non-finite values near p=0"


def test_drag_regularized_at_thermal_speed() -> None:
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


def test_fokker_planck_rejects_invalid_constructor_inputs() -> None:
    with pytest.raises(ValueError, match="np_grid"):
        FokkerPlanckSolver(np_grid=4, p_max=10.0)
    with pytest.raises(ValueError, match="p_max"):
        FokkerPlanckSolver(np_grid=32, p_max=0.0)


def test_fokker_planck_step_rejects_invalid_dt() -> None:
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
    validation = artifact.validate_contract()
    assert validation["passed"] is True
    assert validation["required_axes_present"] is True
    assert validation["required_observables_present"] is True
    assert validation["observable_shapes"] == {
        "f_p_xi_t": [3, 3, 32, 3],
        "runaway_current_t": [3, 3],
        "avalanche_growth_rate_t": [3, 3],
        "synchrotron_loss_power_t": [3, 3],
        "partial_screening_drag_t": [3, 3],
        "bremsstrahlung_loss_power_t": [3, 3],
    }
    assert validation["nonnegative_observables"] is True
    assert validation["finite_observables"] is True


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


class TestStrictAxis:
    """``_strict_axis`` rejects every malformed axis independently of its caller."""

    def test_rejects_non_1d_or_too_short_axis(self) -> None:
        with pytest.raises(ValueError, match="1D axis with at least 2 points"):
            FokkerPlanckSolver._strict_axis("radius_m", [0.5])
        with pytest.raises(ValueError, match="1D axis with at least 2 points"):
            FokkerPlanckSolver._strict_axis("radius_m", np.zeros((2, 2)))

    def test_rejects_non_finite_axis(self) -> None:
        with pytest.raises(ValueError, match="only finite values"):
            FokkerPlanckSolver._strict_axis("radius_m", [0.0, np.nan, 1.0])

    def test_rejects_values_below_lower_bound(self) -> None:
        with pytest.raises(ValueError, match="must be >= 0.0"):
            FokkerPlanckSolver._strict_axis("radius_m", [-1.0, 0.0, 1.0], lower=0.0)

    def test_rejects_values_above_upper_bound(self) -> None:
        with pytest.raises(ValueError, match="must be <= 1.0"):
            FokkerPlanckSolver._strict_axis("pitch_cosine", [0.0, 0.5, 2.0], upper=1.0)


class TestNormalizedAxisWeights:
    """``_normalized_axis_weights`` guards against a zero-mass weight vector."""

    def test_uniform_weights_sum_to_one(self) -> None:
        weights = FokkerPlanckSolver._normalized_axis_weights(np.array([0.0, 1.0, 2.0, 3.0]))
        assert weights == pytest.approx(np.full(4, 0.25))

    def test_rejects_empty_axis_that_cannot_normalise(self) -> None:
        with pytest.raises(ValueError, match="not normalisable"):
            FokkerPlanckSolver._normalized_axis_weights(np.array([], dtype=np.float64))


class TestRunDreamKineticArtifactGuards:
    """The DREAM artifact runner rejects step counts and timesteps below contract."""

    def test_rejects_single_step(self) -> None:
        solver = FokkerPlanckSolver(np_grid=32, p_max=8.0)
        with pytest.raises(ValueError, match="n_steps must be at least 2"):
            solver.run_dream_kinetic_artifact(
                n_steps=1,
                dt=1.0e-6,
                e_field=10.0,
                n_e=5.0e19,
                t_e_ev=2500.0,
                z_eff=2.0,
                radius_m=[0.0, 0.5, 1.0],
                pitch_cosine=[-1.0, 0.0, 1.0],
            )

    def test_rejects_non_positive_dt(self) -> None:
        solver = FokkerPlanckSolver(np_grid=32, p_max=8.0)
        with pytest.raises(ValueError, match="dt must be finite and > 0"):
            solver.run_dream_kinetic_artifact(
                n_steps=2,
                dt=0.0,
                e_field=10.0,
                n_e=5.0e19,
                t_e_ev=2500.0,
                z_eff=2.0,
                radius_m=[0.0, 0.5, 1.0],
                pitch_cosine=[-1.0, 0.0, 1.0],
            )


class TestSeedHottail:
    """``seed_hottail`` injects a Gaussian momentum tail onto the distribution."""

    def test_seed_raises_population_from_zero(self) -> None:
        solver = FokkerPlanckSolver(np_grid=64, p_max=10.0)
        assert float(np.sum(solver.f * solver.dp)) == 0.0
        solver.seed_hottail(T_initial_eV=5000.0, T_final_eV=10.0, t_quench_s=1.0e-3)
        # The low-momentum end is seeded, so total density is now positive.
        assert float(np.sum(solver.f * solver.dp)) > 0.0
        assert np.all(solver.f >= 0.0)
        assert solver.f[0] > solver.f[-1]


class TestExplicitKnockOnSource:
    """The knock-on source validates density and stays silent below the RE seed floor."""

    def test_rejects_non_positive_density(self) -> None:
        solver = FokkerPlanckSolver(np_grid=32, p_max=8.0)
        with pytest.raises(ValueError, match="n_e must be finite and > 0"):
            solver.explicit_knock_on_source(n_e=0.0)

    def test_returns_zeros_below_seed_floor(self) -> None:
        solver = FokkerPlanckSolver(np_grid=32, p_max=8.0)
        # Fresh solver: n_re == 0 < _RE_SEED_FLOOR, so no knock-on avalanche seed.
        assert float(np.sum(solver.f * solver.dp)) < _RE_SEED_FLOOR
        source = solver.explicit_knock_on_source(n_e=5.0e19)
        assert np.all(source == 0.0)

    def test_produces_positive_source_above_seed_floor(self) -> None:
        solver = FokkerPlanckSolver(np_grid=32, p_max=8.0)
        solver.f[:] = 1.0e12  # push n_re well above the seed floor
        source = solver.explicit_knock_on_source(n_e=5.0e19)
        assert np.all(np.isfinite(source))
        assert np.any(source > 0.0)


class TestStepInputGuards:
    """``step`` rejects each non-physical scalar argument with a distinct message."""

    def test_rejects_non_finite_e_field(self) -> None:
        solver = FokkerPlanckSolver(np_grid=32, p_max=8.0)
        with pytest.raises(ValueError, match="E_field must be finite"):
            solver.step(dt=1.0e-5, E_field=np.inf, n_e=5.0e19, T_e_eV=5000.0, Z_eff=1.0)

    def test_rejects_non_positive_density(self) -> None:
        solver = FokkerPlanckSolver(np_grid=32, p_max=8.0)
        with pytest.raises(ValueError, match="n_e must be finite and > 0"):
            solver.step(dt=1.0e-5, E_field=10.0, n_e=0.0, T_e_eV=5000.0, Z_eff=1.0)

    def test_rejects_non_positive_temperature(self) -> None:
        solver = FokkerPlanckSolver(np_grid=32, p_max=8.0)
        with pytest.raises(ValueError, match="T_e_eV must be finite and > 0"):
            solver.step(dt=1.0e-5, E_field=10.0, n_e=5.0e19, T_e_eV=0.0, Z_eff=1.0)

    def test_rejects_non_positive_z_eff(self) -> None:
        solver = FokkerPlanckSolver(np_grid=32, p_max=8.0)
        with pytest.raises(ValueError, match="Z_eff must be finite and > 0"):
            solver.step(dt=1.0e-5, E_field=10.0, n_e=5.0e19, T_e_eV=5000.0, Z_eff=0.0)


# ── Multi-backend dispatch (Rust <-> NumPy Fokker-Planck kernel) ─────


def test_fokker_planck_dispatch_registers_both_tiers() -> None:
    """The class-kernel registry carries RUST and NUMPY Fokker-Planck tiers."""
    from scpn_fusion.core import _multi_compat as multi

    kernels = multi.registered_kernel_classes()
    assert "fokker_planck_re" in kernels
    tiers = [tier.rstrip("*") for tier in kernels["fokker_planck_re"]]
    assert "rust" in tiers
    assert "numpy" in tiers


def test_fokker_planck_numpy_floor_without_rust(monkeypatch: pytest.MonkeyPatch) -> None:
    """The factory resolves to the NumPy kernel when Rust is unavailable."""
    from scpn_fusion.core import _multi_compat as multi

    multi._ensure_probed()
    monkeypatch.setitem(multi._availability, multi.BackendTier.RUST, False)
    monkeypatch.delitem(multi._class_dispatch_cache, "fokker_planck_re", raising=False)
    try:
        kernel = create_fokker_planck_kernel(32, 8.0)
        assert isinstance(kernel, FokkerPlanckKernel)
        n_re, current_re = kernel.step(1.0e-5, 10.0, 5.0e19, 5000.0, 1.0)
        assert np.isfinite(n_re) and np.isfinite(current_re)
    finally:
        multi._class_dispatch_cache.pop("fokker_planck_re", None)


def test_create_fokker_planck_kernel_returns_protocol_surface() -> None:
    """The dispatched kernel exposes the full runaway-electron kernel protocol."""
    kernel = create_fokker_planck_kernel(32, 8.0)
    for attr in ("step", "run", "get_f", "set_f", "get_p", "get_dp"):
        assert callable(getattr(kernel, attr))
    assert isinstance(kernel.time, float)
    assert kernel.get_p().shape == (32,)
    assert kernel.get_dp().shape == (32,)


def test_fokker_planck_kernel_matches_solver() -> None:
    """The NumPy adapter reproduces its wrapped solver exactly (tuple contract)."""
    kernel = FokkerPlanckKernel(64, 10.0)
    solver = FokkerPlanckSolver(64, 10.0)
    seed = np.zeros(64, dtype=np.float64)
    seed[10] = 1.0e10
    kernel.set_f(seed.tolist())
    solver.f = seed.copy()
    params = (1.0e-5, 10.0, 5.0e19, 5000.0, 1.0)
    n_re = current_re = 0.0
    for _ in range(5):
        n_re, current_re = kernel.step(*params)
        state = solver.step(*params)
    assert n_re == state.n_re
    assert current_re == state.current_re
    np.testing.assert_array_equal(kernel.get_f(), solver.f)
    np.testing.assert_array_equal(kernel.get_p(), solver.p)
    np.testing.assert_array_equal(kernel.get_dp(), solver.dp)
    assert kernel.time == solver.time


def test_fokker_planck_kernel_run_matches_stepwise() -> None:
    """``run`` reproduces the same trajectory as repeated ``step`` calls."""
    stepwise = FokkerPlanckKernel(48, 10.0)
    batched = FokkerPlanckKernel(48, 10.0)
    seed = np.zeros(48, dtype=np.float64)
    seed[6] = 1.0e10
    stepwise.set_f(seed.tolist())
    batched.set_f(seed.tolist())
    params = (1.0e-6, 5.0, 5.0e19, 5000.0, 1.0)
    manual = [stepwise.step(*params) for _ in range(4)]
    auto = batched.run(4, *params)
    assert manual == auto


def test_fokker_planck_kernel_set_f_rejects_wrong_shape() -> None:
    """The adapter fails closed on a distribution of the wrong length."""
    kernel = FokkerPlanckKernel(32, 8.0)
    with pytest.raises(ValueError, match="shape"):
        kernel.set_f([1.0, 2.0, 3.0])


def test_fokker_planck_rust_numpy_step_parity() -> None:
    """Rust and NumPy tiers agree on the evolved diagnostics.

    The two backends build the momentum grid independently and implement the
    identical MUSCL-Hancock / central-diffusion / operator-split scheme. A
    single step is bit-tight; a bounded (unforced) trajectory over many steps
    agrees to floating-point summation order. Parity is asserted where the
    solution is bounded — in exponentially growing regimes round-off
    differences amplify for any explicit scheme.
    """
    pytest.importorskip("scpn_fusion_rs")
    from scpn_fusion.core import _multi_compat_providers as providers

    numpy_kernel = providers._load_numpy_fokker_planck()(200, 100.0)
    rust_kernel = providers._load_rust_fokker_planck()(200, 100.0)

    np.testing.assert_allclose(numpy_kernel.get_p(), rust_kernel.get_p(), rtol=0, atol=1e-11)
    np.testing.assert_allclose(numpy_kernel.get_dp(), rust_kernel.get_dp(), rtol=0, atol=1e-11)

    seed = np.zeros(200, dtype=np.float64)
    seed[10] = 1.0e10
    numpy_kernel.set_f(seed.tolist())
    rust_kernel.set_f(seed.tolist())
    n_np, j_np = numpy_kernel.step(1.0e-5, 10.0, 5.0e19, 5000.0, 1.0)
    n_rs, j_rs = rust_kernel.step(1.0e-5, 10.0, 5.0e19, 5000.0, 1.0)
    assert n_np == pytest.approx(n_rs, rel=1e-9)
    assert j_np == pytest.approx(j_rs, rel=1e-9)

    stable_np = providers._load_numpy_fokker_planck()(200, 100.0)
    stable_rs = providers._load_rust_fokker_planck()(200, 100.0)
    p_grid = stable_np.get_p()
    gaussian = (1.0e10 * np.exp(-((p_grid - 5.0) ** 2) / 2.0)).tolist()
    stable_np.set_f(gaussian)
    stable_rs.set_f(gaussian)
    n_stable_np = j_stable_np = n_stable_rs = j_stable_rs = 0.0
    for _ in range(50):
        n_stable_np, j_stable_np = stable_np.step(1.0e-6, 0.0, 1.0e19, 5000.0, 1.0)
        n_stable_rs, j_stable_rs = stable_rs.step(1.0e-6, 0.0, 1.0e19, 5000.0, 1.0)
    assert n_stable_np == pytest.approx(n_stable_rs, rel=1e-8)
    assert j_stable_np == pytest.approx(j_stable_rs, rel=1e-8)
    f_np = stable_np.get_f()
    f_rs = stable_rs.get_f()
    np.testing.assert_allclose(f_np, f_rs, rtol=1e-6, atol=1e-6 * float(np.max(np.abs(f_np))))


def test_fokker_planck_backend_invariant_parity() -> None:
    """Both dispatched backends satisfy the shared physics invariants.

    Diagnostics stay finite and the distribution stays non-negative across a
    multi-step run on either tier.
    """
    from scpn_fusion.core import _multi_compat as multi
    from scpn_fusion.core import _multi_compat_providers as providers

    kernels = [providers._load_numpy_fokker_planck()(64, 10.0)]
    if multi.is_available(multi.BackendTier.RUST):
        kernels.append(providers._load_rust_fokker_planck()(64, 10.0))
    seed = np.zeros(64, dtype=np.float64)
    seed[10] = 1.0e10
    for kernel in kernels:
        kernel.set_f(seed.tolist())
        history = kernel.run(20, 1.0e-6, 10.0, 5.0e19, 5000.0, 1.0)
        assert len(history) == 20
        for n_re, current_re in history:
            assert np.isfinite(n_re) and np.isfinite(current_re)
        assert bool(np.all(kernel.get_f() >= 0.0))
