# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Runtime contract tests for the FusionKernel iterative solver mixin."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_fusion.core.fusion_kernel_iterative_solver import FusionKernelIterativeSolverMixin


FloatArray = NDArray[np.float64]


class _HpcStub:
    """Small accelerator facade used by elliptic-solver branch tests."""

    def __init__(self, *, available: bool, result: FloatArray | None) -> None:
        self.available = available
        self.result = None if result is None else result.copy()
        self.calls = 0

    def is_available(self) -> bool:
        """Return whether the accelerator should be selected."""
        return self.available

    def solve(self, _j_phi: FloatArray, *, iterations: int) -> FloatArray | None:
        """Return a configured accelerator result while recording the call."""
        assert iterations == 50
        self.calls += 1
        return None if self.result is None else self.result.copy()


class _IterativeKernelStub(FusionKernelIterativeSolverMixin):
    """Minimal FusionKernel-compatible object for isolated mixin tests."""

    def __init__(self, n: int = 5) -> None:
        r = np.linspace(1.2, 2.2, n, dtype=np.float64)
        z = np.linspace(-0.5, 0.5, n, dtype=np.float64)
        self.dR = float(r[1] - r[0])
        self.dZ = float(z[1] - z[0])
        self.RR, self.ZZ = np.meshgrid(r, z)
        self.Psi = np.zeros((n, n), dtype=np.float64)
        self.J_phi = np.zeros_like(self.Psi)
        self.cfg: dict[str, dict[str, Any]] = {
            "physics": {"plasma_current_target": 1.0},
            "dimensions": {"R_min": float(r[0]), "R_max": float(r[-1])},
            "solver": {"solver_method": "multigrid", "sor_omega": 1.1},
        }
        self.hpc = _HpcStub(available=False, result=None)

    def calculate_vacuum_field(self) -> FloatArray:
        """Return a deterministic nonzero vacuum boundary map."""
        vacuum = np.zeros_like(self.Psi)
        vacuum[0, :] = 0.25
        vacuum[-1, :] = 0.5
        vacuum[:, 0] = 0.75
        vacuum[:, -1] = 1.0
        return vacuum


def _boundary_map(shape: tuple[int, ...]) -> FloatArray:
    """Build distinct edge values for boundary-enforcement assertions."""
    if len(shape) != 2:
        raise ValueError("boundary maps require a two-dimensional shape")
    boundary = np.zeros(shape, dtype=np.float64)
    boundary[0, :] = 10.0
    boundary[-1, :] = 20.0
    boundary[:, 0] = 30.0
    boundary[:, -1] = 40.0
    return boundary


def _assert_boundary_matches(field: FloatArray, boundary: FloatArray) -> None:
    """Assert that every edge in *field* equals the boundary source."""
    np.testing.assert_array_equal(field[0, :], boundary[0, :])
    np.testing.assert_array_equal(field[-1, :], boundary[-1, :])
    np.testing.assert_array_equal(field[:, 0], boundary[:, 0])
    np.testing.assert_array_equal(field[:, -1], boundary[:, -1])


def test_jacobi_step_sanitizes_inputs_and_preserves_boundaries() -> None:
    """Jacobi updates finite interior values while leaving Dirichlet edges fixed."""
    kernel = _IterativeKernelStub()
    psi = _boundary_map(kernel.Psi.shape)
    psi[2, 2] = np.nan
    source = np.ones_like(psi)
    source[1, 1] = np.inf

    updated = kernel._jacobi_step(psi, source)

    _assert_boundary_matches(updated, psi)
    assert np.all(np.isfinite(updated))
    assert updated.shape == psi.shape


def test_multigrid_wrapper_methods_preserve_shape_contracts() -> None:
    """Mixin wrapper methods delegate restriction and prolongation without shape drift."""
    kernel = _IterativeKernelStub()
    fine = np.arange(81, dtype=np.float64).reshape(9, 9)

    coarse = kernel._restrict_full_weight(fine)
    restored = kernel._prolongate_bilinear(coarse, 9, 9)

    assert coarse.shape == (5, 5)
    assert restored.shape == fine.shape
    assert np.all(np.isfinite(restored))


def test_anderson_step_with_single_residual_returns_latest_copy() -> None:
    """Anderson acceleration falls back when there is not enough history."""
    kernel = _IterativeKernelStub()
    latest = np.ones((2, 2), dtype=np.float64)

    mixed = kernel._anderson_step([np.zeros_like(latest), latest], [latest], m=5)

    np.testing.assert_array_equal(mixed, latest)
    assert mixed is not latest


def test_anderson_step_mixes_independent_history() -> None:
    """Independent residual history produces a finite mixed iterate."""
    kernel = _IterativeKernelStub()
    psi_history = [
        np.zeros((2, 2), dtype=np.float64),
        np.ones((2, 2), dtype=np.float64),
        np.full((2, 2), 2.0, dtype=np.float64),
    ]
    res_history = [
        np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64),
        np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float64),
        np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
    ]

    mixed = kernel._anderson_step(psi_history, res_history, m=3)

    assert mixed.shape == psi_history[-1].shape
    assert np.all(np.isfinite(mixed))


def test_anderson_step_falls_back_on_singular_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed least-squares solve returns the latest iterate copy."""
    kernel = _IterativeKernelStub()
    latest = np.full((2, 2), 3.0, dtype=np.float64)

    def raise_linalg_error(_gram: FloatArray, _rhs: FloatArray) -> FloatArray:
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(np.linalg, "solve", raise_linalg_error)

    mixed = kernel._anderson_step(
        [np.zeros_like(latest), latest],
        [np.zeros_like(latest), np.ones_like(latest)],
        m=2,
    )

    np.testing.assert_array_equal(mixed, latest)
    assert mixed is not latest


def test_anderson_step_falls_back_when_alpha_sum_collapses() -> None:
    """Near-opposite residuals with huge norm trigger the alpha-sum guard."""
    kernel = _IterativeKernelStub()
    latest = np.ones((2, 2), dtype=np.float64)
    large = np.full((2, 2), 1.0e10, dtype=np.float64)

    mixed = kernel._anderson_step(
        [np.zeros_like(latest), latest],
        [-large, large],
        m=2,
    )

    np.testing.assert_array_equal(mixed, latest)
    assert mixed is not latest


def test_apply_boundary_conditions_copies_all_edges() -> None:
    """Boundary enforcement copies every edge from the supplied map in place."""
    kernel = _IterativeKernelStub()
    field = np.zeros_like(kernel.Psi)
    boundary = _boundary_map(field.shape)

    kernel._apply_boundary_conditions(field, boundary)

    _assert_boundary_matches(field, boundary)


def test_elliptic_solve_uses_accelerator_result_when_available() -> None:
    """The accelerator path still enforces Python-side boundary conditions."""
    kernel = _IterativeKernelStub()
    accelerated = np.full_like(kernel.Psi, 2.0)
    kernel.hpc = _HpcStub(available=True, result=accelerated)
    boundary = _boundary_map(kernel.Psi.shape)

    solved = kernel._elliptic_solve(np.zeros_like(kernel.Psi), boundary)

    assert kernel.hpc.calls == 1
    _assert_boundary_matches(solved, boundary)
    assert solved[2, 2] == 2.0


@pytest.mark.parametrize("method", ["jacobi", "multigrid", "sor", "anderson"])
def test_elliptic_solve_python_methods_enforce_boundary(method: str) -> None:
    """Every Python elliptic backend applies the configured boundary map."""
    kernel = _IterativeKernelStub()
    kernel.cfg["solver"]["solver_method"] = method
    kernel.cfg["solver"]["sor_omega"] = 1.1
    source = -0.01 * kernel.RR
    boundary = _boundary_map(kernel.Psi.shape)

    solved = kernel._elliptic_solve(source, boundary)

    _assert_boundary_matches(solved, boundary)
    assert np.all(np.isfinite(solved))


def test_seed_plasma_with_zero_target_clears_current_without_flux_drift() -> None:
    """A zero-current seed leaves the existing vacuum flux untouched."""
    kernel = _IterativeKernelStub()
    kernel.Psi = kernel.calculate_vacuum_field()
    expected_psi = kernel.Psi.copy()
    kernel.cfg["physics"]["plasma_current_target"] = 0.0

    kernel._seed_plasma(mu0=1.0)

    np.testing.assert_array_equal(kernel.J_phi, np.zeros_like(kernel.Psi))
    np.testing.assert_array_equal(kernel.Psi, expected_psi)


def test_seed_plasma_normalizes_gaussian_current_seed() -> None:
    """A nonzero target creates a finite Gaussian seed with the requested current."""
    kernel = _IterativeKernelStub()
    kernel.cfg["physics"]["plasma_current_target"] = 2.5

    kernel._seed_plasma(mu0=1.0)

    seeded_current = float(np.sum(kernel.J_phi)) * kernel.dR * kernel.dZ
    assert seeded_current == pytest.approx(2.5)
    assert np.all(np.isfinite(kernel.J_phi))
    assert np.all(np.isfinite(kernel.Psi))


def test_prepare_initial_flux_rejects_shape_mismatch() -> None:
    """Explicit boundary maps must match the solver grid shape."""
    kernel = _IterativeKernelStub()

    with pytest.raises(ValueError, match="boundary_flux shape"):
        kernel._prepare_initial_flux(False, np.zeros((3, 3), dtype=np.float64))


def test_prepare_initial_flux_accepts_explicit_boundary_copy() -> None:
    """An explicit boundary map becomes the solver state when not preserving."""
    kernel = _IterativeKernelStub()
    boundary = _boundary_map(kernel.Psi.shape)

    prepared = kernel._prepare_initial_flux(False, boundary)

    np.testing.assert_array_equal(prepared, boundary)
    np.testing.assert_array_equal(kernel.Psi, boundary)
    assert prepared is not boundary


def test_prepare_initial_flux_preserves_existing_state_edges() -> None:
    """Preserve mode keeps the current interior and reapplies existing edges."""
    kernel = _IterativeKernelStub()
    kernel.Psi = np.arange(kernel.Psi.size, dtype=np.float64).reshape(kernel.Psi.shape)
    expected = kernel.Psi.copy()

    prepared = kernel._prepare_initial_flux(True, None)

    np.testing.assert_array_equal(prepared, expected)
    np.testing.assert_array_equal(kernel.Psi, expected)


def test_prepare_initial_flux_uses_vacuum_field_when_not_preserving() -> None:
    """Fresh iterative solves start from the calculated vacuum field."""
    kernel = _IterativeKernelStub()

    prepared = kernel._prepare_initial_flux(False, None)

    np.testing.assert_array_equal(prepared, kernel.calculate_vacuum_field())
    np.testing.assert_array_equal(kernel.Psi, prepared)
