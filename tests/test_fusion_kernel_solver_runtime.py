# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Direct tests for extracted FusionKernel solver runtime helpers."""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import LinearOperator

from scpn_fusion.core.fusion_kernel_solver_runtime import (
    apply_gs_operator,
    compute_gs_residual,
    compute_gs_residual_rms,
    compute_profile_jacobian,
    solve_newton_linear_system,
    solve_via_rust_multigrid,
)


class _KernelStub:
    def __init__(self, nz: int = 6, nr: int = 6) -> None:
        self.Psi = np.zeros((nz, nr), dtype=np.float64)
        self.RR = np.ones((nz, nr), dtype=np.float64) * 6.2
        self.dR = 1.0
        self.dZ = 1.0
        self.dummy_called = False
        self.external_profile_mode = False
        self.J_phi = np.ones((nz, nr), dtype=np.float64)
        self.cfg = {
            "physics": {"plasma_current_target": 1.0, "vacuum_permeability": 1.0},
            "solver": {"solver_method": "rust_multigrid"},
        }

    def solve_equilibrium(self, **kwargs):  # pragma: no cover - trivial dispatch
        self.dummy_called = True
        return {"solver_method": self.cfg["solver"]["solver_method"], "kwargs": kwargs}


def test_gs_residual_and_rms_zero_for_zero_source() -> None:
    k = _KernelStub()
    source = np.zeros_like(k.Psi)
    residual = compute_gs_residual(k, source)
    assert residual.shape == k.Psi.shape
    assert np.allclose(residual, 0.0)
    assert compute_gs_residual_rms(k, source) == 0.0


def test_apply_gs_operator_shape_and_finite() -> None:
    k = _KernelStub()
    vec = np.random.RandomState(0).randn(*k.Psi.shape)
    out = apply_gs_operator(k, vec)
    assert out.shape == vec.shape
    assert np.all(np.isfinite(out))


def test_compute_profile_jacobian_external_profile_mode() -> None:
    k = _KernelStub()
    k.external_profile_mode = True
    k.Psi = np.linspace(0.0, 0.9, k.Psi.size, dtype=np.float64).reshape(k.Psi.shape)
    jac = compute_profile_jacobian(k, Psi_axis=0.0, Psi_boundary=1.0, mu0=1.0)
    assert jac.shape == k.Psi.shape
    assert np.all(np.isfinite(jac))
    assert float(np.min(jac)) <= 0.0


def test_compute_profile_jacobian_stable_with_extreme_profile_inputs() -> None:
    k = _KernelStub()
    k.Psi[1, 1] = np.nan
    k.Psi[2, 2] = np.inf
    k.Psi[3, 3] = -np.inf
    k.RR[1, 2] = np.nan
    k.RR[2, 3] = np.inf
    jac = compute_profile_jacobian(
        k,
        Psi_axis=np.nan,
        Psi_boundary=np.inf,
        mu0=1.0,
    )
    assert jac.shape == k.Psi.shape
    assert np.all(np.isfinite(jac))


def test_compute_profile_jacobian_external_mode_sanitizes_nonfinite_jphi() -> None:
    k = _KernelStub()
    k.external_profile_mode = True
    k.Psi = np.linspace(0.0, 0.9, k.Psi.size, dtype=np.float64).reshape(k.Psi.shape)
    k.J_phi[1, 1] = np.nan
    k.J_phi[2, 2] = np.inf
    jac = compute_profile_jacobian(k, Psi_axis=0.0, Psi_boundary=1.0, mu0=1.0)
    assert np.all(np.isfinite(jac))
    assert np.all(jac <= 0.0)


def test_solve_newton_linear_system_identity_operator() -> None:
    k = _KernelStub(nz=4, nr=4)
    n = 4
    j_op = LinearOperator(shape=(n, n), matvec=lambda x: x, dtype=np.float64)
    rhs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    delta, info = solve_newton_linear_system(
        kernel=k,
        J_op=j_op,
        rhs=rhs,
        diag_term=np.zeros_like(k.Psi),
        n_interior=n,
        nz=4,
        nr=4,
        gmres_preconditioner_mode="none",
        ilu_drop_tol=1e-4,
        ilu_fill_factor=8.0,
        iter_idx=0,
    )
    assert info >= 0
    assert delta.shape == (n,)
    assert np.all(np.isfinite(delta))


def test_solve_via_rust_multigrid_preserve_state_forces_sor_fallback() -> None:
    k = _KernelStub()
    result = solve_via_rust_multigrid(k, preserve_initial_state=True)
    assert k.dummy_called
    assert result["solver_method"] == "sor"
    assert k.cfg["solver"]["solver_method"] == "rust_multigrid"
