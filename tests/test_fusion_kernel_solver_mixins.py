# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Linkage and contract tests for FusionKernel solver mixins."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_fusion.core.fusion_kernel import FusionKernel
from scpn_fusion.core.fusion_kernel_iterative_solver import (
    FusionKernelIterativeSolverMixin,
)
from scpn_fusion.core.fusion_kernel_newton_solver import FusionKernelNewtonSolverMixin
from scpn_fusion.core.fusion_kernel_numerics import (
    NUMERIC_SANITIZE_CAP,
    sanitize_numeric_array,
    stable_rms,
)


def test_fusion_kernel_inherits_solver_mixins() -> None:
    assert issubclass(FusionKernel, FusionKernelIterativeSolverMixin)
    assert issubclass(FusionKernel, FusionKernelNewtonSolverMixin)


def test_numeric_helpers_are_stable_on_non_finite_inputs() -> None:
    arr = np.array([np.nan, np.inf, -np.inf, 1.0, -1.0], dtype=np.float64)
    sanitized = sanitize_numeric_array(arr)
    assert np.all(np.isfinite(sanitized))
    assert float(np.max(np.abs(sanitized))) <= NUMERIC_SANITIZE_CAP
    assert stable_rms(sanitized) >= 0.0


class _IterativeKernelStub(FusionKernelIterativeSolverMixin):
    def __init__(self) -> None:
        self.dR = 0.25
        self.dZ = 0.25
        r = np.linspace(1.0, 2.0, 5)
        z = np.linspace(-0.5, 0.5, 5)
        self.RR, self.ZZ = np.meshgrid(r, z)


@pytest.mark.parametrize("omega", [0.0, 0.999999, 2.0, np.inf, np.nan])
def test_sor_step_rejects_unstable_relaxation_factor(omega: float) -> None:
    """SOR must reject relaxation factors outside the stable GS interval."""
    kernel = _IterativeKernelStub()
    psi = np.zeros((5, 5), dtype=np.float64)
    source = np.zeros_like(psi)

    with pytest.raises(ValueError, match="omega"):
        kernel._sor_step(psi, source, omega=omega)


@pytest.mark.parametrize("omega", [0.0, 0.999999, 2.0, np.inf, np.nan])
def test_multigrid_smoother_rejects_unstable_relaxation_factor(omega: float) -> None:
    """The multigrid smoother shares the same SOR relaxation contract."""
    kernel = _IterativeKernelStub()
    psi = np.zeros((5, 5), dtype=np.float64)
    source = np.zeros_like(psi)

    with pytest.raises(ValueError, match="omega"):
        kernel._mg_smooth(psi, source, kernel.RR, kernel.dR, kernel.dZ, omega, n_sweeps=1)


def _discrete_gs_source(kernel: _IterativeKernelStub, psi: np.ndarray) -> np.ndarray:
    source = np.zeros_like(psi)
    dR2 = kernel.dR**2
    dZ2 = kernel.dZ**2
    r_int = kernel.RR[1:-1, 1:-1]
    d2r = (psi[1:-1, 2:] - 2.0 * psi[1:-1, 1:-1] + psi[1:-1, :-2]) / dR2
    d1r = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2.0 * kernel.dR)
    d2z = (psi[2:, 1:-1] - 2.0 * psi[1:-1, 1:-1] + psi[:-2, 1:-1]) / dZ2
    source[1:-1, 1:-1] = d2r - d1r / r_int + d2z
    return source


def _manufactured_flux(kernel: _IterativeKernelStub) -> np.ndarray:
    rr = kernel.RR
    zz = kernel.ZZ
    return 0.03125 * rr**4 - 0.125 * zz**2 + 0.05 * rr**2 * zz**2


def test_sor_step_preserves_exact_discrete_grad_shafranov_fixed_point() -> None:
    """An exact discrete GS solution must be a fixed point of one SOR sweep."""
    kernel = _IterativeKernelStub()
    psi = _manufactured_flux(kernel)
    source = _discrete_gs_source(kernel, psi)

    updated = kernel._sor_step(psi, source, omega=1.4)

    assert np.max(np.abs(updated - psi)) < 1e-12


def test_multigrid_smoother_preserves_exact_discrete_grad_shafranov_fixed_point() -> None:
    """The multigrid smoother must share the native GS source sign convention."""
    kernel = _IterativeKernelStub()
    psi = _manufactured_flux(kernel)
    source = _discrete_gs_source(kernel, psi)

    updated = kernel._mg_smooth(
        psi.copy(), source, kernel.RR, kernel.dR, kernel.dZ, omega=1.4, n_sweeps=2
    )

    assert np.max(np.abs(updated - psi)) < 1e-12


class _LargeIterativeKernelStub(FusionKernelIterativeSolverMixin):
    """65x65 grid stub for full multigrid V-cycle convergence tests."""

    def __init__(self) -> None:
        nr = nz = 65
        r = np.linspace(1.2, 2.2, nr)
        z = np.linspace(-0.5, 0.5, nz)
        self.dR = float(r[1] - r[0])
        self.dZ = float(z[1] - z[0])
        self.RR, self.ZZ = np.meshgrid(r, z)


def test_multigrid_vcycle_iteration_converges_on_fixed_source() -> None:
    """Iterating the V-cycle on a fixed source drives the GS residual down.

    Guards the coarse-grid defect sign: the error solves L*[e] = -(L*[Psi] -
    Source), so the coarse right-hand side must be the *negated* residual.
    Restricting the raw residual inverts every correction (Psi <- Psi - e) and the
    iteration diverges (residual grows) instead of converging.
    """
    kernel = _LargeIterativeKernelStub()
    r_grid = kernel.RR
    source = -r_grid * np.exp(-((r_grid - 1.7) ** 2 + kernel.ZZ**2) / 0.05)
    psi = np.zeros_like(r_grid)

    def residual_rms(field: np.ndarray) -> float:
        interior = kernel._mg_residual(field, source, r_grid, kernel.dR, kernel.dZ)[1:-1, 1:-1]
        return float(np.sqrt(np.mean(interior**2)))

    initial = residual_rms(psi)
    for _ in range(8):
        psi = kernel._multigrid_vcycle(psi, source, r_grid, kernel.dR, kernel.dZ, omega=1.0)
        # Re-apply the homogeneous Dirichlet boundary the source was posed with.
        psi[0, :] = 0.0
        psi[-1, :] = 0.0
        psi[:, 0] = 0.0
        psi[:, -1] = 0.0

    final = residual_rms(psi)
    assert np.all(np.isfinite(psi))
    assert final < initial * 1e-4, f"V-cycle did not converge: {initial:.3e} -> {final:.3e}"
