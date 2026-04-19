# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Direct tests for extracted free-boundary FusionKernel helpers."""

from __future__ import annotations

import numpy as np

from scpn_fusion.core.fusion_kernel import CoilSet
from scpn_fusion.core.fusion_kernel_free_boundary import (
    green_function,
    interp_psi,
    optimize_coil_currents,
    resolve_shape_target_flux,
    solve_free_boundary,
)


class _KernelStub:
    def __init__(self) -> None:
        self.R = np.linspace(5.0, 6.0, 5, dtype=np.float64)
        self.Z = np.linspace(-0.5, 0.5, 5, dtype=np.float64)
        self.NR = len(self.R)
        self.NZ = len(self.Z)
        self.dR = float(self.R[1] - self.R[0])
        self.dZ = float(self.Z[1] - self.Z[0])
        self.Psi = np.zeros((self.NZ, self.NR), dtype=np.float64)
        self.solve_calls = 0

    def solve_equilibrium(self, **kwargs) -> dict[str, object]:
        self.solve_calls += 1
        return {"converged": True, "kwargs": kwargs}


def test_green_function_returns_finite_value() -> None:
    value = green_function(5.2, 0.1, 5.5, -0.2)
    assert np.isfinite(value)


def test_interp_psi_bilinear_center() -> None:
    k = _KernelStub()
    # Set a simple 2x2 cell in the lower-left corner.
    k.Psi[0, 0] = 0.0
    k.Psi[0, 1] = 1.0
    k.Psi[1, 0] = 2.0
    k.Psi[1, 1] = 3.0
    value = interp_psi(k, float(k.R[0] + 0.5 * k.dR), float(k.Z[0] + 0.5 * k.dZ))
    assert abs(value - 1.5) < 1e-12


def test_resolve_shape_target_flux_prefers_explicit_values() -> None:
    k = _KernelStub()
    coils = CoilSet(
        positions=[(5.2, 0.0)],
        currents=np.array([1.0], dtype=np.float64),
        target_flux_points=np.array([[5.2, 0.0], [5.4, 0.1]], dtype=np.float64),
        target_flux_values=np.array([0.1, 0.2], dtype=np.float64),
    )
    target = resolve_shape_target_flux(k, coils)
    np.testing.assert_allclose(target, np.array([0.1, 0.2], dtype=np.float64))


def test_optimize_coil_currents_returns_finite_vector() -> None:
    k = _KernelStub()
    coils = CoilSet(
        positions=[(5.3, 0.0)],
        currents=np.array([0.0], dtype=np.float64),
        turns=[10],
        current_limits=np.array([2.0], dtype=np.float64),
        target_flux_points=np.array([[5.35, 0.0]], dtype=np.float64),
    )
    solution = optimize_coil_currents(
        k,
        coils,
        target_flux=np.array([0.02], dtype=np.float64),
        tikhonov_alpha=1e-4,
    )
    assert solution.shape == (1,)
    assert np.all(np.isfinite(solution))


def test_solve_free_boundary_returns_contract() -> None:
    k = _KernelStub()
    coils = CoilSet(
        positions=[(5.2, 0.0)],
        currents=np.array([0.0], dtype=np.float64),
    )
    result = solve_free_boundary(k, coils, max_outer_iter=1, tol=0.0)
    assert result["outer_iterations"] == 1
    assert "final_diff" in result
    assert "coil_currents" in result
    assert k.solve_calls == 1
