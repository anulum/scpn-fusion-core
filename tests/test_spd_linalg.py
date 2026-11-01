# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — SPD Solve Contract Tests
"""Factorised SPD solves checked against exact rational arithmetic and their consumers."""

from __future__ import annotations

import logging
from fractions import Fraction
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_fusion.control.fusion_control_room import KalmanObserver
from scpn_fusion.control.mu_synthesis import DScalingError, _reciprocal_scaling
from scpn_fusion.control.rzip_model import RZIPController, RZIPModel
from scpn_fusion.control.state_estimator import ExtendedKalmanFilter
from scpn_fusion.core.spd_linalg import (
    NonSPDMatrixError,
    _usable_rcond,
    factor_spd,
    joseph_update,
    require_psd,
    spd_solve,
    symmetrize,
)
from scpn_fusion.core.vessel_model import VesselElement, VesselModel

EPS = float(np.finfo(np.float64).eps)


def _spd_with_condition(n: int, cond: float, seed: int) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigenvalues = np.logspace(0.0, -np.log10(cond), n)
    a = (q * eigenvalues) @ q.T
    return np.asarray(0.5 * (a + a.T), dtype=np.float64)


def _exact_solve(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Gaussian elimination in exact rationals on the stored binary64 values."""
    n = a.shape[0]
    m = [[Fraction(float(a[i, j])) for j in range(n)] + [Fraction(float(b[i]))] for i in range(n)]
    for col in range(n):
        pivot = next(r for r in range(col, n) if m[r][col] != 0)
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(n):
            if r != col and m[r][col] != 0:
                ratio = m[r][col] / m[col][col]
                m[r] = [x - ratio * y for x, y in zip(m[r], m[col])]
    return np.array([float(m[i][n] / m[i][i]) for i in range(n)], dtype=np.float64)


@pytest.mark.parametrize("cond", [1.0e0, 1.0e3, 1.0e6, 1.0e9, 1.0e12])
@pytest.mark.parametrize("n", [3, 6])
def test_backward_and_forward_error_over_condition_sweep(n: int, cond: float) -> None:
    new_errors: list[float] = []
    old_errors: list[float] = []
    for seed in range(40):
        a = _spd_with_condition(n, cond, seed=seed)
        b = np.random.default_rng(seed + 1000).standard_normal(n)
        x = spd_solve(a, b, name="A")

        backward = np.linalg.norm(a @ x - b) / (np.linalg.norm(a) * np.linalg.norm(x))
        assert backward <= 10.0 * n * EPS

        exact = _exact_solve(a, b)
        forward = float(np.linalg.norm(x - exact) / np.linalg.norm(exact))
        assert forward <= 10.0 * n * cond * EPS
        new_errors.append(forward)
        old_errors.append(
            float(np.linalg.norm(np.linalg.inv(a) @ b - exact) / np.linalg.norm(exact))
        )
    # Not worse than the explicit-inverse path it replaces, judged over the
    # whole sample (single instances scatter both ways).
    assert float(np.median(new_errors)) <= 2.0 * max(float(np.median(old_errors)), n * EPS)


def test_scaling_makes_rcond_unit_free() -> None:
    a = _spd_with_condition(4, 1.0e4, seed=3)
    d = np.diag([1.0e-6, 1.0, 1.0e3, 1.0e8])
    unscaled = factor_spd(a, name="A")
    rescaled = factor_spd(d @ a @ d, name="DAD")
    assert unscaled.rcond == pytest.approx(rescaled.rcond, rel=1.0e-8)
    # (D A D)^-1 1 = D^-1 A^-1 D^-1 1: the scaled solve reproduces the
    # unscaled one to the accuracy allowed by cond(A) = 1e4.
    x = rescaled.solve(np.ones(4))
    reference = unscaled.solve(np.ones(4) / np.diag(d))
    np.testing.assert_allclose(np.diag(d) * x, reference, rtol=1.0e-9)
    inv = rescaled.inverse()
    np.testing.assert_allclose(inv, inv.T, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (np.array([[1.0, 2.0], [2.0, 1.0]]), "not positive definite"),
        (np.array([[1.0, 1.0], [1.0, 1.0]]), "not positive definite|numerically singular"),
        (np.array([[0.0, 0.0], [0.0, 1.0]]), "non-positive diagonal"),
        (np.array([[1.0, 0.5], [0.4, 1.0]]), "not symmetric"),
        (np.array([[1.0, np.nan], [np.nan, 1.0]]), "finite values"),
        (np.ones((2, 3)), "square"),
    ],
)
def test_singular_indefinite_and_malformed_inputs_raise_typed_failure(
    matrix: NDArray[np.float64], message: str
) -> None:
    with pytest.raises(NonSPDMatrixError, match=message):
        spd_solve(matrix, np.ones(matrix.shape[0]), name="A")


def test_solve_rejects_mismatched_right_hand_side() -> None:
    with pytest.raises(ValueError, match="right-hand side"):
        factor_spd(np.eye(3)).solve(np.ones(2))


def test_require_psd_accepts_semidefinite_and_rejects_indefinite() -> None:
    np.testing.assert_array_equal(require_psd(np.zeros((2, 2))), np.zeros((2, 2)))
    with pytest.raises(NonSPDMatrixError, match="positive semi-definite"):
        require_psd(np.diag([1.0, -1.0]))
    np.testing.assert_array_equal(symmetrize(np.eye(2)), np.eye(2))


def test_joseph_form_keeps_ekf_covariance_symmetric_psd_over_long_run() -> None:
    rng = np.random.default_rng(11)
    ekf = ExtendedKalmanFilter(
        np.array([6.2, 0.0, 0.0, 0.0, 15.0, 10.0]),
        np.eye(6) * 0.5,
        np.eye(6) * 1.0e-4,
        np.diag([1.0e-6, 1.0e-6, 1.0e-2, 1.0e-3]),
    )
    for _ in range(10_000):
        ekf.predict(1.0e-3)
        ekf.update(np.array([6.2, 0.0, 15.0, 10.0]) + rng.standard_normal(4) * 1.0e-3)
        p = ekf.P
        norm = np.linalg.norm(p, 2)
        assert np.linalg.norm(p - p.T) <= 1.0e2 * EPS * norm
    assert np.linalg.eigvalsh(ekf.P)[0] >= -1.0e2 * EPS * np.linalg.norm(ekf.P, 2)


def test_joseph_update_matches_standard_form_at_optimal_gain() -> None:
    p = _spd_with_condition(3, 10.0, seed=5)
    h = np.array([[1.0, 0.0, 0.5]])
    r = np.array([[0.2]])
    s = h @ p @ h.T + r
    k = spd_solve(s, h @ p).T
    np.testing.assert_allclose(joseph_update(p, k, h, r), (np.eye(3) - k @ h) @ p, atol=1.0e-14)


def test_ekf_rejects_indefinite_measurement_noise() -> None:
    with pytest.raises(NonSPDMatrixError, match="R_cov"):
        ExtendedKalmanFilter(np.zeros(6), np.eye(6), np.eye(6), -np.eye(4))


def test_control_room_observer_rejects_non_positive_innovation_variance() -> None:
    observer = KalmanObserver()
    observer.R = np.array([[-10.0]])
    with pytest.raises(NonSPDMatrixError, match="innovation variance"):
        observer.update(0.1)


@pytest.mark.parametrize("bad", [[1.0, 0.0], [1.0, -2.0], [1.0, np.inf], [np.nan, 1.0]])
def test_mu_scaling_reciprocal_requires_finite_positive_factors(bad: list[float]) -> None:
    with pytest.raises(DScalingError, match="finite and positive"):
        _reciprocal_scaling(np.array(bad))
    np.testing.assert_array_equal(_reciprocal_scaling(np.array([2.0, 0.5])), [0.5, 2.0])


def _vessel(inductance: float) -> VesselModel:
    return VesselModel(
        [
            VesselElement(R=2.0, Z=0.5, resistance=1e-3, cross_section=0.1, inductance=inductance),
            VesselElement(R=2.0, Z=-0.5, resistance=1e-3, cross_section=0.1, inductance=inductance),
        ]
    )


def test_vessel_inductance_factor_records_rcond_and_inverse() -> None:
    vessel = _vessel(1.0e-5)
    assert vessel.inductance_status == "ok"
    assert vessel.inductance_rcond is not None and 0.0 < vessel.inductance_rcond <= 1.0
    np.testing.assert_allclose(vessel.M @ vessel.M_inv, np.eye(2), atol=1.0e-9)


def test_vessel_non_spd_inductance_is_flagged_not_silent(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.ERROR, logger="scpn_fusion.core.vessel_model"):
        vessel = _vessel(-1.0e-5)
    assert vessel.inductance_status == "zero_inverse_inductance_not_spd"
    assert vessel.inductance_rcond is None
    np.testing.assert_array_equal(vessel.M_inv, np.zeros((2, 2)))
    assert "inverse set to zero" in caplog.text


def test_rzip_state_space_uses_cholesky_and_matches_inverse_path() -> None:
    coils = [VesselElement(R=2.5, Z=1.0, resistance=1e-2, cross_section=0.05, inductance=5e-5)]
    rzip = RZIPModel(
        R0=2.0,
        a=0.5,
        kappa=1.0,
        Ip_MA=1.0,
        B0=1.0,
        n_index=-0.5,
        vessel=_vessel(1.0e-5),
        active_coils=coils,
    )
    a_mat, b_mat, _, _ = rzip.build_state_space()
    assert rzip.inductance_status == "ok"
    assert rzip.inductance_rcond is not None and rzip.inductance_rcond > 0.0
    assert b_mat.shape == (2 + 3, 1)
    assert np.all(np.isfinite(a_mat)) and np.all(np.isfinite(b_mat))


def test_rzip_non_spd_inductance_is_flagged(caplog: pytest.LogCaptureFixture) -> None:
    rzip = RZIPModel(
        R0=2.0, a=0.5, kappa=1.0, Ip_MA=1.0, B0=1.0, n_index=0.5, vessel=_vessel(-1.0e-5)
    )
    with caplog.at_level(logging.ERROR, logger="scpn_fusion.control.rzip_model"):
        rzip.build_state_space()
    assert rzip.inductance_status == "zero_response_inductance_not_spd"
    assert "response set to zero" in caplog.text


class _UnstabilisableModel:
    """Real state-space owner whose unstable mode no input can reach."""

    def build_state_space(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        a = np.diag([1.0, -1.0])
        b = np.array([[0.0], [1.0]])
        return a, b, np.eye(2), np.zeros((2, 1))


def test_rzip_controller_reports_failed_riccati_solve(caplog: pytest.LogCaptureFixture) -> None:
    model: Any = _UnstabilisableModel()
    with caplog.at_level(logging.ERROR, logger="scpn_fusion.control.rzip_model"):
        controller = RZIPController(model, Kp=1.0, Kd=1.0)
    assert controller.gain_status == "zero_gain_care_failed"
    np.testing.assert_array_equal(controller.K_gain, np.zeros((1, 2)))
    assert "Riccati solve failed" in caplog.text


def test_sugama_projection_numpy_and_jax_agree_to_1e12() -> None:
    from scpn_fusion.core.gk_nonlinear import NonlinearGKConfig, NonlinearGKSolver
    from scpn_fusion.core.jax_gk_nonlinear import JaxNonlinearGKSolver, jax_available

    if not jax_available():
        pytest.skip("JAX runtime not installed; NumPy path covered by conservation tests")
    import jax.numpy as jnp

    cfg = NonlinearGKConfig(
        n_kx=4,
        n_ky=4,
        n_theta=8,
        n_vpar=8,
        n_mu=6,
        n_species=2,
        collisions=True,
        collision_model="sugama",
        cfl_adapt=False,
    )
    numpy_solver = NonlinearGKSolver(cfg)
    jax_solver = JaxNonlinearGKSolver(cfg)
    f = numpy_solver.init_state(amplitude=1e-4, seed=13).f[0]
    reference = numpy_solver.collide(f)
    jax_result = np.asarray(jax_solver._jax_collide_sugama(jnp.asarray(f)))
    scale = float(np.max(np.abs(reference)))
    assert float(np.max(np.abs(jax_result - reference))) <= 1.0e-12 * scale


@pytest.mark.parametrize(
    ("rcond", "info"), [(0.0, 0), (float("nan"), 0), (float("inf"), 0), (0.5, 1)]
)
def test_unusable_rcond_estimate_is_refused(rcond: float, info: int) -> None:
    with pytest.raises(NonSPDMatrixError, match="numerically singular"):
        _usable_rcond(rcond, info, "A")
    assert _usable_rcond(0.25, 0, "A") == 0.25


def test_empty_vessel_has_no_inductance_to_factor() -> None:
    vessel = VesselModel([])
    assert vessel.inductance_status == "ok"
    assert vessel.inductance_rcond is None
    assert vessel.M_inv.shape == (0, 0)


def test_rzip_without_circuits_is_rejected() -> None:
    rzip = RZIPModel(
        R0=2.0, a=0.5, kappa=1.0, Ip_MA=1.0, B0=1.0, n_index=0.5, vessel=VesselModel([])
    )
    with pytest.raises(ValueError, match="at least one vessel or coil circuit"):
        rzip.build_state_space()


def test_jax_solver_refuses_degenerate_sugama_grid_outside_jit() -> None:
    from scpn_fusion.core.gk_nonlinear import NonlinearGKConfig
    from scpn_fusion.core.jax_gk_nonlinear import JaxNonlinearGKSolver, jax_available

    if not jax_available():
        pytest.skip("JAX runtime not installed; the check lives in the JAX kernel setup")
    cfg = NonlinearGKConfig(
        n_kx=4,
        n_ky=4,
        n_theta=8,
        n_vpar=1,
        n_mu=1,
        n_species=1,
        collisions=True,
        collision_model="sugama",
        cfl_adapt=False,
    )
    with pytest.raises(NonSPDMatrixError, match="Gram"):
        JaxNonlinearGKSolver(cfg)
