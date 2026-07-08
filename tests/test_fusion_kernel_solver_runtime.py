# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Direct tests for extracted FusionKernel solver runtime helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator

from scpn_fusion.core import _rust_compat
from scpn_fusion.core.fusion_kernel_solver_runtime import (
    apply_gs_operator,
    compute_gs_residual,
    compute_gs_residual_rms,
    compute_profile_jacobian,
    solve_newton_linear_system,
    solve_via_rust_multigrid,
)

FloatArray = NDArray[np.float64]


class _KernelStub:
    """Small real-array kernel surface used by extracted runtime helpers."""

    def __init__(self, nz: int = 6, nr: int = 6) -> None:
        self._config_path = "config.json"
        self.Psi: FloatArray = np.zeros((nz, nr), dtype=np.float64)
        self.RR: FloatArray = np.ones((nz, nr), dtype=np.float64) * 6.2
        self.dR = 1.0
        self.dZ = 1.0
        self.dummy_called = False
        self.external_profile_mode = False
        self.J_phi: FloatArray = np.ones((nz, nr), dtype=np.float64)
        self.B_R: FloatArray = np.zeros((nz, nr), dtype=np.float64)
        self.B_Z: FloatArray = np.zeros((nz, nr), dtype=np.float64)
        self.cfg: dict[str, dict[str, Any]] = {
            "physics": {"plasma_current_target": 1.0, "vacuum_permeability": 1.0},
            "solver": {"solver_method": "rust_multigrid", "convergence_threshold": 1.0e-4},
        }

    def solve_equilibrium(self, **kwargs: Any) -> dict[str, Any]:
        """Record fallback delegation and return the active solver method."""
        self.dummy_called = True
        return {"solver_method": self.cfg["solver"]["solver_method"], "kwargs": kwargs}


class _RustResult:
    """Minimal Rust result contract used by the runtime delegation tests."""

    def __init__(self, *, converged: bool, residual: float, iterations: int) -> None:
        self.converged = converged
        self.residual = residual
        self.iterations = iterations


class _RustAcceleratedKernelStub:
    """Fake Rust kernel exposing the runtime helper's synchronization surface."""

    requested_method = ""

    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self.Psi: FloatArray = np.full((6, 6), 0.25, dtype=np.float64)
        self.J_phi: FloatArray = np.full((6, 6), 0.5, dtype=np.float64)
        self.B_R: FloatArray = np.full((6, 6), 0.125, dtype=np.float64)
        self.B_Z: FloatArray = np.full((6, 6), 0.375, dtype=np.float64)

    def set_solver_method(self, method: str) -> None:
        """Capture the requested Rust-side solver method."""
        _RustAcceleratedKernelStub.requested_method = method

    def solve_equilibrium(self) -> _RustResult:
        """Return a result that relies on the runtime practical tolerance gate."""
        return _RustResult(converged=False, residual=1.0e-5, iterations=7)


def _identity_operator(size: int) -> LinearOperator:
    return LinearOperator(shape=(size, size), matvec=lambda x: x, dtype=np.float64)


def test_gs_residual_and_rms_zero_for_zero_source() -> None:
    """Zero flux with zero source yields a zero GS residual and RMS."""
    k = _KernelStub()
    source = np.zeros_like(k.Psi)
    residual = compute_gs_residual(k, source)
    assert residual.shape == k.Psi.shape
    assert np.allclose(residual, 0.0)
    assert compute_gs_residual_rms(k, source) == 0.0


def test_gs_residual_rms_returns_zero_without_interior() -> None:
    """Degenerate two-by-two grids report zero interior residual RMS."""
    k = _KernelStub(nz=2, nr=2)
    source = np.zeros_like(k.Psi)
    assert compute_gs_residual_rms(k, source) == 0.0


def test_apply_gs_operator_shape_and_finite() -> None:
    """The discrete GS operator preserves shape and finite values."""
    k = _KernelStub()
    vec = np.linspace(-1.0, 1.0, k.Psi.size, dtype=np.float64).reshape(k.Psi.shape)
    out = apply_gs_operator(k, vec)
    assert out.shape == vec.shape
    assert np.all(np.isfinite(out))


def test_compute_profile_jacobian_external_profile_mode() -> None:
    """External profile mode returns a finite nonpositive diagonal field."""
    k = _KernelStub()
    k.external_profile_mode = True
    k.Psi = np.linspace(0.0, 0.9, k.Psi.size, dtype=np.float64).reshape(k.Psi.shape)
    jac = compute_profile_jacobian(k, Psi_axis=0.0, Psi_boundary=1.0, mu0=1.0)
    assert jac.shape == k.Psi.shape
    assert np.all(np.isfinite(jac))
    assert float(np.min(jac)) <= 0.0


def test_compute_profile_jacobian_stable_with_extreme_profile_inputs() -> None:
    """Nonfinite profile and radius inputs are sanitized before evaluation."""
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


def test_compute_profile_jacobian_repairs_nonfinite_denominator() -> None:
    """Overflowed axis-boundary differences are repaired to a finite scale."""
    k = _KernelStub()
    k.Psi = np.zeros_like(k.Psi)
    jac = compute_profile_jacobian(
        k,
        Psi_axis=float(np.finfo(np.float64).max),
        Psi_boundary=-float(np.finfo(np.float64).max),
        mu0=1.0,
    )
    assert jac.shape == k.Psi.shape
    assert np.all(np.isfinite(jac))


def test_compute_profile_jacobian_repairs_tiny_negative_denominator() -> None:
    """Near-zero negative profile denominators retain finite Jacobians."""
    k = _KernelStub()
    k.Psi.fill(0.0)
    jac = compute_profile_jacobian(k, Psi_axis=1.0, Psi_boundary=1.0 - 1.0e-12, mu0=1.0)
    assert jac.shape == k.Psi.shape
    assert np.all(np.isfinite(jac))


def test_compute_profile_jacobian_repairs_nonfinite_profile_integral() -> None:
    """Nonfinite profile integrals are guarded before current scaling."""
    k = _KernelStub()
    k.Psi.fill(0.25)
    k.dR = float("inf")
    jac = compute_profile_jacobian(k, Psi_axis=0.0, Psi_boundary=1.0, mu0=1.0)
    assert jac.shape == k.Psi.shape
    assert np.all(np.isfinite(jac))
    assert float(np.max(jac)) <= 0.0


def test_compute_profile_jacobian_external_mode_sanitizes_nonfinite_jphi() -> None:
    """External mode clamps nonfinite current-density samples."""
    k = _KernelStub()
    k.external_profile_mode = True
    k.Psi = np.linspace(0.0, 0.9, k.Psi.size, dtype=np.float64).reshape(k.Psi.shape)
    k.J_phi[1, 1] = np.nan
    k.J_phi[2, 2] = np.inf
    jac = compute_profile_jacobian(k, Psi_axis=0.0, Psi_boundary=1.0, mu0=1.0)
    assert np.all(np.isfinite(jac))
    assert np.all(jac <= 0.0)


def test_solve_newton_linear_system_identity_operator() -> None:
    """GMRES solves the unpreconditioned identity-system path."""
    k = _KernelStub(nz=4, nr=4)
    n = 4
    rhs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    delta, info = solve_newton_linear_system(
        kernel=k,
        J_op=_identity_operator(n),
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


def test_solve_newton_linear_system_with_diagonal_preconditioner() -> None:
    """GMRES accepts the diagonal preconditioner path."""
    k = _KernelStub(nz=4, nr=4)
    n = 4
    rhs = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float64)
    delta, info = solve_newton_linear_system(
        kernel=k,
        J_op=_identity_operator(n),
        rhs=rhs,
        diag_term=np.zeros_like(k.Psi),
        n_interior=n,
        nz=4,
        nr=4,
        gmres_preconditioner_mode="diagonal",
        ilu_drop_tol=1e-4,
        ilu_fill_factor=8.0,
        iter_idx=1,
    )
    assert info >= 0
    assert np.all(np.isfinite(delta))


def test_solve_newton_linear_system_with_ilu_preconditioner() -> None:
    """GMRES accepts the sparse ILU preconditioner path."""
    k = _KernelStub(nz=4, nr=4)
    n = 4
    rhs = np.array([1.0, 2.0, -3.0, -4.0], dtype=np.float64)
    delta, info = solve_newton_linear_system(
        kernel=k,
        J_op=_identity_operator(n),
        rhs=rhs,
        diag_term=np.zeros_like(k.Psi),
        n_interior=n,
        nz=4,
        nr=4,
        gmres_preconditioner_mode="ilu",
        ilu_drop_tol=1e-4,
        ilu_fill_factor=8.0,
        iter_idx=2,
    )
    assert info >= 0
    assert np.all(np.isfinite(delta))


def test_solve_newton_linear_system_falls_back_when_ilu_build_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ILU setup failures fall back to the diagonal preconditioner."""
    import scipy.sparse.linalg

    def fail_spilu(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("forced ilu failure")

    monkeypatch.setattr(scipy.sparse.linalg, "spilu", fail_spilu)
    k = _KernelStub(nz=4, nr=4)
    n = 4
    rhs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    delta, info = solve_newton_linear_system(
        kernel=k,
        J_op=_identity_operator(n),
        rhs=rhs,
        diag_term=np.zeros_like(k.Psi),
        n_interior=n,
        nz=4,
        nr=4,
        gmres_preconditioner_mode="ilu",
        ilu_drop_tol=1e-4,
        ilu_fill_factor=8.0,
        iter_idx=3,
    )
    assert info >= 0
    assert np.all(np.isfinite(delta))


def test_solve_via_rust_multigrid_preserve_state_forces_sor_fallback() -> None:
    """Preserving state delegates to the Python SOR fallback."""
    k = _KernelStub()
    result = solve_via_rust_multigrid(k, preserve_initial_state=True)
    assert k.dummy_called
    assert result["solver_method"] == "sor"
    assert k.cfg["solver"]["solver_method"] == "rust_multigrid"


def test_solve_via_rust_multigrid_boundary_flux_forces_sor_fallback() -> None:
    """Boundary flux constraints delegate to the Python SOR fallback."""
    k = _KernelStub()
    boundary_flux = np.zeros_like(k.Psi)
    result = solve_via_rust_multigrid(k, boundary_flux=boundary_flux)
    assert k.dummy_called
    assert result["kwargs"]["boundary_flux"] is boundary_flux
    assert k.cfg["solver"]["solver_method"] == "rust_multigrid"


def test_solve_via_rust_multigrid_unavailable_backend_falls_back_to_sor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unavailable Rust backend delegates to Python SOR and restores config."""
    monkeypatch.setattr(_rust_compat, "_RUST_AVAILABLE", False)
    k = _KernelStub()
    result = solve_via_rust_multigrid(k)
    assert k.dummy_called
    assert result["solver_method"] == "sor"
    assert k.cfg["solver"]["solver_method"] == "rust_multigrid"


def test_solve_via_rust_multigrid_syncs_successful_rust_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful Rust delegation synchronizes flux, current, and fields."""
    monkeypatch.setattr(_rust_compat, "_RUST_AVAILABLE", True)
    monkeypatch.setattr(_rust_compat, "RustAcceleratedKernel", _RustAcceleratedKernelStub)
    k = _KernelStub()
    result = solve_via_rust_multigrid(k)
    assert result["solver_method"] == "rust_multigrid"
    assert result["converged"] is True
    assert result["iterations"] == 7
    assert np.allclose(k.Psi, 0.25)
    assert np.allclose(k.J_phi, 0.5)
    assert np.allclose(k.B_R, 0.125)
    assert np.allclose(k.B_Z, 0.375)
    assert _RustAcceleratedKernelStub.requested_method == "multigrid"
