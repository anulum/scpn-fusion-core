# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Crank-Nicolson Transport Tests
"""Phase 1 verification: implicit Crank-Nicolson transport solver."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import j0

from scpn_fusion.core.integrated_transport_solver import TransportSolver

MOCK_CONFIG = {
    "reactor_name": "CN-Test",
    "grid_resolution": [20, 20],
    "dimensions": {"R_min": 4.0, "R_max": 8.0, "Z_min": -4.0, "Z_max": 4.0},
    "physics": {"plasma_current_target": 15.0, "vacuum_permeability": 1.0},
    "coils": [
        {"name": "CS", "r": 1.7, "z": 0.0, "current": 0.15},
    ],
    "solver": {
        "max_iterations": 10,
        "convergence_threshold": 1e-4,
        "relaxation_factor": 0.1,
    },
}


@pytest.fixture
def solver(tmp_path: Path) -> TransportSolver:
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(MOCK_CONFIG), encoding="utf-8")
    ts = TransportSolver(str(cfg))
    # Give it a reasonable parabolic initial profile
    ts.Ti = 5.0 * (1 - ts.rho**2)
    ts.Te = ts.Ti.copy()
    ts.ne = 5.0 * (1 - ts.rho**2) ** 0.5
    ts.update_transport_model(50.0)
    return ts


def test_impurity_transport_can_be_disabled(tmp_path: Path) -> None:
    """The public runtime can represent a genuinely source-free heat step."""
    config = json.loads(json.dumps(MOCK_CONFIG))
    config["physics"]["transport_backend"] = "fixed_coefficients"
    config["physics"]["impurity_transport_enabled"] = False
    cfg = tmp_path / "source_free.json"
    cfg.write_text(json.dumps(config), encoding="utf-8")

    ts = TransportSolver(str(cfg), nr=33)
    ts.Ti = 0.1 + 0.9 * (1.0 - ts.rho**2)
    ts.Te = ts.Ti.copy()
    ts.ne = np.full(ts.nr, 10.0)
    ts.chi_i = np.ones(ts.nr)
    ts.chi_e = np.ones(ts.nr)
    ts.D_n = np.zeros(ts.nr)
    initial = ts.Ti.copy()

    ts.evolve_profiles(1.0e-3, 0.0)

    np.testing.assert_array_equal(ts.n_impurity, np.zeros(ts.nr))
    assert ts._Z_eff == 1.0
    assert np.all(np.isfinite(ts.Ti))
    assert np.all(ts.Ti > 0.0)
    assert float(np.mean(ts.Ti)) < float(np.mean(initial))


def test_impurity_transport_flag_rejects_non_boolean(tmp_path: Path) -> None:
    """Ambiguous truthy values cannot silently change source physics."""
    config = json.loads(json.dumps(MOCK_CONFIG))
    config["physics"]["impurity_transport_enabled"] = "false"
    cfg = tmp_path / "invalid_impurity_flag.json"
    cfg.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="impurity_transport_enabled must be a boolean"):
        TransportSolver(str(cfg))


def test_impurity_cn_preserves_conservative_axis_solution(solver: TransportSolver) -> None:
    """Impurity transport must retain the solved cylindrical axis row."""
    solver.n_impurity = 1.0 + 0.5 * solver.rho**2

    solver._evolve_impurity(dt=1.0e-3)

    assert solver.n_impurity[0] != pytest.approx(solver.n_impurity[1])


def test_injected_impurity_uses_conservative_axis_solution(solver: TransportSolver) -> None:
    """The public PWI injection path must use the same cylindrical axis row."""
    solver.n_impurity = 1.0 + 0.5 * solver.rho**2

    solver.inject_impurities(flux_from_wall_per_sec=0.0, dt=1.0e-3)

    assert solver.n_impurity[0] != pytest.approx(solver.n_impurity[1])


# ── Thomas solver unit tests ──────────────────────────────────────


def test_thomas_solve_identity():
    """Identity matrix should return RHS unchanged."""
    n = 10
    a = np.zeros(n - 1)
    b = np.ones(n)
    c = np.zeros(n - 1)
    d = np.arange(n, dtype=float)
    x = TransportSolver._thomas_solve(a, b, c, d)
    np.testing.assert_allclose(x, d, atol=1e-12)


def test_thomas_solve_tridiag():
    """Standard [-1, 2, -1] tridiagonal system against known solution."""
    n = 5
    a = -1.0 * np.ones(n - 1)
    b = 2.0 * np.ones(n)
    c = -1.0 * np.ones(n - 1)
    d = np.array([1.0, 0.0, 0.0, 0.0, 1.0])
    x = TransportSolver._thomas_solve(a, b, c, d)
    # Verify A @ x = d
    Ax = np.zeros(n)
    Ax[0] = b[0] * x[0] + c[0] * x[1]
    for i in range(1, n - 1):
        Ax[i] = a[i - 1] * x[i - 1] + b[i] * x[i] + c[i] * x[i + 1]
    Ax[-1] = a[-1] * x[-2] + b[-1] * x[-1]
    np.testing.assert_allclose(Ax, d, atol=1e-10)


# ── CN transport tests ────────────────────────────────────────────


def test_cn_large_dt_no_nan(solver: TransportSolver):
    """dt=1.0 must produce no NaN — the whole point of CN."""
    for _ in range(10):
        T_avg, T_core = solver.evolve_profiles(dt=1.0, P_aux=50.0)
    assert np.all(np.isfinite(solver.Ti)), "CN produced NaN at dt=1.0"
    assert T_core > 0, "Core temperature should be positive"


def test_cn_energy_decreases_no_heating(solver: TransportSolver):
    """Without heating, stored energy should decrease (diffusion only)."""
    W_before = float(np.sum(solver.Ti))
    for _ in range(20):
        solver.evolve_profiles(dt=0.1, P_aux=0.0)
    W_after = float(np.sum(solver.Ti))
    assert W_after < W_before, "Energy should decrease with no heating"


def test_cn_heats_to_steady_state(solver: TransportSolver):
    """50 steps at dt=0.5 with P_aux=50 MW should heat the core above 1 keV.

    The in-step predictor-corrector time-centres the stiff chi(grad T)
    coupling instead of relying on inter-step coefficient relaxation.
    """
    for _ in range(50):
        solver.update_transport_model(50.0)
        solver.evolve_profiles(dt=0.5, P_aux=50.0)
    assert solver.Ti[0] > 1.0, f"Core T = {solver.Ti[0]:.3f} keV, expected > 1"


def test_predictor_corrector_time_centres_transport_coefficients(
    solver: TransportSolver, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The corrector re-solves with the mean of old- and predicted-state chi."""
    chi_i_n = np.full_like(solver.rho, 2.0)
    chi_e_n = np.full_like(solver.rho, 4.0)
    d_n_n = np.full_like(solver.rho, 1.0)
    solver.chi_i = chi_i_n.copy()
    solver.chi_e = chi_e_n.copy()
    solver.D_n = d_n_n.copy()
    built_chi: list[np.ndarray] = []
    original_build = solver._build_cn_tridiag

    def evaluate_trial(_p_aux: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return 2.0 * chi_e_n, 3.0 * chi_i_n, 5.0 * d_n_n

    def record_build(chi: np.ndarray, dt: float):
        built_chi.append(np.asarray(chi).copy())
        return original_build(chi, dt)

    monkeypatch.setattr(solver, "_evaluate_transport_coefficients", evaluate_trial)
    monkeypatch.setattr(solver, "_build_cn_tridiag", record_build)

    solver.evolve_profiles(dt=0.01, P_aux=5.0)

    # The first build belongs to the impurity-density solve. Thermal builds
    # then contain the predictor, nonlinear corrector trial(s), and accepted
    # corrector replay used to record only accepted-state recoveries.
    assert len(built_chi) >= 4
    np.testing.assert_allclose(built_chi[1], chi_i_n)
    for corrected_chi in built_chi[2:]:
        np.testing.assert_allclose(corrected_chi, 2.0 * chi_i_n)
    np.testing.assert_allclose(solver.chi_e, 1.5 * chi_e_n)
    np.testing.assert_allclose(solver.D_n, 3.0 * d_n_n)
    contract = solver._last_transport_predictor_corrector
    assert contract["scheme"] == "picard_predictor_corrector"
    assert contract["theta"] == 0.5
    assert contract["converged"] is True
    assert contract["iterations"] == 2
    assert contract["chi_i_relative_change"] == 2.0
    assert contract["chi_e_relative_change"] == 1.0


def test_cn_matches_euler_small_dt(tmp_path: Path):
    """At tiny dt, CN and forward-Euler should agree within 5%."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(MOCK_CONFIG), encoding="utf-8")

    dt = 0.0001
    P_aux = 20.0

    # CN solver
    ts_cn = TransportSolver(str(cfg))
    ts_cn.Ti = 5.0 * (1 - ts_cn.rho**2)
    ts_cn.Te = ts_cn.Ti.copy()
    ts_cn.ne = 5.0 * (1 - ts_cn.rho**2) ** 0.5
    ts_cn.update_transport_model(P_aux)
    ts_cn.evolve_profiles(dt, P_aux)

    # Manual forward-Euler for comparison
    ts_fe = TransportSolver(str(cfg))
    ts_fe.Ti = 5.0 * (1 - ts_fe.rho**2)
    ts_fe.Te = ts_fe.Ti.copy()
    ts_fe.ne = 5.0 * (1 - ts_fe.rho**2) ** 0.5
    ts_fe.update_transport_model(P_aux)

    # Compute Euler step manually
    Lh = ts_fe._explicit_diffusion_rhs(ts_fe.Ti, ts_fe.chi_i)
    S_heat, _ = ts_fe._compute_aux_heating_sources(P_aux)
    S_rad = 5.0 * ts_fe.ne * ts_fe.n_impurity * np.sqrt(ts_fe.Te + 0.1)
    euler_Ti = ts_fe.Ti + dt * (Lh + S_heat - S_rad)
    euler_Ti[-1] = 0.1
    euler_Ti = np.maximum(0.01, euler_Ti)

    # Interior agreement (exclude boundaries and edge region where
    # the CN half-grid stencil differs from np.gradient)
    core = slice(2, len(ts_cn.rho) * 3 // 4)
    rel_diff = np.abs(ts_cn.Ti[core] - euler_Ti[core]) / (euler_Ti[core] + 1e-10)
    assert np.max(rel_diff) < 0.05, f"Max relative diff {np.max(rel_diff):.4f} > 5%"


def test_cn_cylindrical_axis_operator_uses_half_control_volume(
    solver: TransportSolver,
) -> None:
    """The axis operator must use the cylindrical half-control-volume balance."""
    temperature = 1.4 - 0.9 * solver.rho**2
    diffusivity = 1.0 + 0.5 * solver.rho**2

    diffusion = solver._explicit_diffusion_rhs(temperature, diffusivity)
    face_diffusivity = 0.5 * (diffusivity[0] + diffusivity[1])
    expected_axis = 4.0 * face_diffusivity * (temperature[1] - temperature[0]) / solver.drho**2

    assert diffusion[0] == pytest.approx(expected_axis, rel=1.0e-13, abs=1.0e-13)
    assert diffusion[0] == pytest.approx(-3.6, rel=2.0e-4)


def test_cn_cylindrical_axis_matrix_matches_explicit_operator(
    solver: TransportSolver,
) -> None:
    """The implicit axis row must be the CN counterpart of the explicit row."""
    dt = 0.013
    diffusivity = 0.7 + 0.4 * solver.rho**2
    _, diagonal, upper = solver._build_cn_tridiag(diffusivity, dt)
    axis_coefficient = 4.0 * 0.5 * (diffusivity[0] + diffusivity[1]) / solver.drho**2

    assert diagonal[0] == pytest.approx(1.0 + 0.5 * dt * axis_coefficient)
    assert upper[0] == pytest.approx(-0.5 * dt * axis_coefficient)
    assert diagonal[0] + upper[0] == pytest.approx(1.0)


def test_cn_public_bessel_mode_accuracy(tmp_path: Path) -> None:
    """The public runtime must resolve the frozen cylindrical Bessel mode."""
    config = json.loads(json.dumps(MOCK_CONFIG))
    config["physics"]["transport_backend"] = "fixed_coefficients"
    config["physics"]["impurity_transport_enabled"] = False
    config["solver"]["max_numerical_recoveries_per_step"] = 0
    cfg = tmp_path / "bessel_mode.json"
    cfg.write_text(json.dumps(config), encoding="utf-8")

    transport = TransportSolver(str(cfg), nr=129)
    radial_mode = j0(2.4048255576957728 * transport.rho)
    transport.T_edge_keV = 0.1
    transport.Ti = 0.1 + 0.9 * radial_mode
    transport.Te = transport.Ti.copy()
    transport.ne = np.full(transport.nr, 10.0)
    transport.chi_i = np.ones(transport.nr)
    transport.chi_e = np.ones(transport.nr)
    transport.D_n = np.zeros(transport.nr)

    for _ in range(10):
        transport.evolve_profiles(
            1.0e-3,
            0.0,
            enforce_conservation=True,
            enforce_numerical_recovery=True,
            max_numerical_recoveries=0,
        )

    exact = 0.1 + 0.9 * radial_mode * np.exp(-(2.4048255576957728**2) * 0.01)
    error = transport.Ti - exact
    rmse = float(np.sqrt(np.mean(error**2)))
    max_error = float(np.max(np.abs(error)))

    assert rmse <= 2.2829306504541543e-6
    assert max_error <= 7.870399983275767e-6
    assert transport._last_conservation_error <= 2.564156129034178e-3
    assert transport._last_numerical_recovery_count == 0


def test_cn_public_multi_ion_runtime_solves_both_axis_rows(tmp_path: Path) -> None:
    """Ion and electron public-runtime branches must retain their solved axis values."""
    config = json.loads(json.dumps(MOCK_CONFIG))
    config["physics"]["transport_backend"] = "fixed_coefficients"
    cfg = tmp_path / "multi_ion_axis.json"
    cfg.write_text(json.dumps(config), encoding="utf-8")

    transport = TransportSolver(str(cfg), multi_ion=True, nr=65)
    transport.Ti = 0.2 + 1.1 * (1.0 - transport.rho**2)
    transport.Te = 0.2 + 0.7 * (1.0 - transport.rho**2)
    transport.ne = np.full(transport.nr, 10.0)
    transport.chi_i = np.full(transport.nr, 0.8)
    transport.chi_e = 1.2 + 0.3 * transport.rho**2
    transport.D_n = np.zeros(transport.nr)

    transport.evolve_profiles(1.0e-4, 0.0)

    assert np.all(np.isfinite(transport.Ti))
    assert np.all(np.isfinite(transport.Te))
    assert abs(float(transport.Ti[0] - transport.Ti[1])) > 1.0e-10
    assert abs(float(transport.Te[0] - transport.Te[1])) > 1.0e-10


def test_cn_backward_compatible(solver: TransportSolver):
    """run_to_steady_state() should still work with CN under the hood."""
    result = solver.run_to_steady_state(P_aux=30.0, n_steps=20, dt=0.1)
    assert "T_avg" in result
    assert "T_core" in result
    assert result["T_avg"] > 0
    assert np.all(np.isfinite(result["Ti_profile"]))


def test_cn_boundary_conditions(solver: TransportSolver):
    """Axis finite-volume symmetry and edge Dirichlet rows remain active."""
    for _ in range(5):
        solver.evolve_profiles(dt=0.5, P_aux=40.0)
    assert np.isfinite(solver.Ti[0])
    assert abs(solver.Ti[0] - solver.Ti[1]) > 1.0e-12
    # Dirichlet: T[-1] == 0.1
    assert abs(solver.Ti[-1] - 0.1) < 1e-12, "Edge Dirichlet BC violated"


def test_cn_conservation_error_reported_as_inf_on_degenerate_volume(
    solver: TransportSolver,
) -> None:
    """A non-finite volume element must surface as an ``inf`` conservation error.

    If the radial volume element degenerates to ``inf`` the raw energy balance
    ``dW_actual - dW_source`` evaluates to ``nan`` (``inf - inf``).  The runtime
    rewrites that to ``+inf`` so the downstream ``> 0.01`` gate fails closed —
    ``nan > 0.01`` is ``False`` and would silently pass an unphysical step.
    """
    n = solver.rho.size
    # Shadow the bound method with a degenerate volume element.
    solver._rho_volume_element = lambda: np.full(n, np.inf)  # type: ignore[method-assign]

    # The degenerate inf volume deliberately produces inf/nan in the energy
    # balance; that is the condition under test, so silence the expected
    # numpy invalid-value warnings rather than let them clutter the run.
    with np.errstate(invalid="ignore"):
        solver.evolve_profiles(dt=0.1, P_aux=10.0, enforce_conservation=False)
        assert solver._last_conservation_error == float("inf")

        # With enforcement on, the same degenerate step must fail closed.
        with pytest.raises(Exception, match="Energy conservation violated"):
            solver.evolve_profiles(dt=0.1, P_aux=10.0, enforce_conservation=True)


def test_steady_state_is_dt_independent_and_cycle_free(tmp_path: Path) -> None:
    """BACKLOG 3 regression: the discrete fixed point must not depend on dt.

    The 2026-07-04 real-TORAX comparison found a dt-dependent steady state
    and a period-2 crash-rebuild cycle driven by four numerical defects
    (explicit CFL-violating impurity diffusion, missing impurity sink,
    explicit stiff radiation sink, identity boundary rows leaking dt-scaled
    sources). After the fixes, coarse and fine time steps must relax to the
    same steady state with no alternation in the tail, and the impurity
    content must stay bounded.
    """
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(MOCK_CONFIG), encoding="utf-8")

    finals: dict[float, float] = {}
    for dt, steps in ((0.5, 240), (0.1, 1200)):
        run = TransportSolver(str(cfg))
        run.Ti = 5.0 * (1 - run.rho**2)
        run.Te = run.Ti.copy()
        run.ne = 5.0 * (1 - run.rho**2) ** 0.5
        core: list[float] = []
        for _ in range(steps):
            _, core_ti = run.evolve_profiles(dt, 30.0)
            core.append(core_ti)
        tail = np.asarray(core[-8:], dtype=np.float64)
        assert float(np.max(np.abs(np.diff(tail)))) < 0.05, f"tail alternation at dt={dt}: {tail}"
        assert float(np.max(run.n_impurity)) < 10.0, "impurity content unbounded"
        finals[dt] = core[-1]

    ratio = finals[0.5] / max(finals[0.1], 1e-30)
    assert 0.95 < ratio < 1.05, f"steady state depends on dt: {finals}"
