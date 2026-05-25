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
    build_magnetic_probe_response_matrix,
    build_mutual_inductance_matrix,
    green_function,
    interp_psi,
    optimize_coil_currents,
    reconstruct_boundary_flux_from_coils,
    reconstruct_coil_currents_from_magnetic_probes,
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


def test_inverse_magnetic_probe_reconstruction_recovers_synthetic_currents() -> None:
    k = _KernelStub()
    coils = CoilSet(
        positions=[(5.1, -0.35), (5.9, 0.35)],
        currents=np.array([0.0, 0.0], dtype=np.float64),
        turns=[6, 8],
        current_limits=np.array([2.0, 2.0], dtype=np.float64),
    )
    flux_points = np.array([[5.25, -0.15], [5.75, 0.15]], dtype=np.float64)
    b_probe_points = np.array([[5.35, 0.0], [5.65, 0.0]], dtype=np.float64)
    b_probe_directions = ["R", "Z"]
    true_currents = np.array([0.55, -0.35], dtype=np.float64)
    response = build_magnetic_probe_response_matrix(
        k,
        coils,
        flux_points=flux_points,
        b_probe_points=b_probe_points,
        b_probe_directions=b_probe_directions,
    )
    measurements = response @ true_currents

    result = reconstruct_coil_currents_from_magnetic_probes(
        k,
        coils,
        flux_points=flux_points,
        flux_measurements=measurements[: flux_points.shape[0]],
        b_probe_points=b_probe_points,
        b_probe_directions=b_probe_directions,
        b_probe_measurements=measurements[flux_points.shape[0] :],
        tikhonov_alpha=0.0,
    )

    np.testing.assert_allclose(result["coil_currents"], true_currents, atol=1e-9)
    assert result["residual_rms"] < 1e-12
    assert result["response_rank"] == 2


def test_boundary_flux_reconstruction_uses_coil_green_functions() -> None:
    k = _KernelStub()
    coils = CoilSet(
        positions=[(5.1, -0.35), (5.9, 0.35)],
        currents=np.array([0.55, -0.35], dtype=np.float64),
        turns=[6, 8],
    )
    boundary_points = np.array(
        [[5.15, -0.45], [5.45, -0.48], [5.85, 0.42], [5.55, 0.47]], dtype=np.float64
    )
    response = build_mutual_inductance_matrix(k, coils, boundary_points)
    target_flux = response.T @ coils.currents

    result = reconstruct_boundary_flux_from_coils(
        k, coils, boundary_points=boundary_points, target_flux=target_flux
    )

    np.testing.assert_allclose(result["reconstructed_flux"], target_flux, atol=1e-13)
    assert result["rmse"] < 1e-13
    assert result["max_abs_error"] < 1e-13
    assert result["response_rank"] == 2
    assert result["point_count"] == boundary_points.shape[0]


def test_boundary_flux_reconstruction_reports_limiter_and_topology_metadata() -> None:
    k = _KernelStub()
    coils = CoilSet(
        positions=[(5.1, -0.35), (5.9, 0.35)],
        currents=np.array([0.55, -0.35], dtype=np.float64),
        turns=[6, 8],
    )
    boundary_points = np.array([[5.2, -0.4], [5.8, 0.4]], dtype=np.float64)
    limiter_points = np.array([[5.0, -0.5], [6.0, -0.5], [6.0, 0.5], [5.0, 0.5]], dtype=np.float64)
    axis_point = np.array([5.5, 0.0], dtype=np.float64)
    x_points = np.array([[5.25, -0.45], [5.75, 0.45]], dtype=np.float64)

    result = reconstruct_boundary_flux_from_coils(
        k,
        coils,
        boundary_points=boundary_points,
        limiter_points=limiter_points,
        axis_point=axis_point,
        x_points=x_points,
    )

    assert result["limiter_point_count"] == limiter_points.shape[0]
    assert result["x_point_count"] == x_points.shape[0]
    assert result["axis_flux"] is not None
    assert result["limiter_flux"].shape == (limiter_points.shape[0],)
    assert result["x_point_flux"].shape == (x_points.shape[0],)
    assert result["min_limiter_distance_m"] > 0.0
    assert np.all(np.isfinite(result["limiter_flux"]))
    assert np.all(np.isfinite(result["x_point_flux"]))


def test_inverse_magnetic_probe_reconstruction_respects_current_limits() -> None:
    k = _KernelStub()
    coils = CoilSet(
        positions=[(5.1, -0.35)],
        currents=np.array([0.0], dtype=np.float64),
        current_limits=np.array([0.2], dtype=np.float64),
    )
    flux_points = np.array([[5.25, -0.15]], dtype=np.float64)
    result = reconstruct_coil_currents_from_magnetic_probes(
        k,
        coils,
        flux_points=flux_points,
        flux_measurements=np.array([1.0], dtype=np.float64),
        tikhonov_alpha=0.0,
    )

    assert np.all(np.abs(result["coil_currents"]) <= coils.current_limits + 1e-12)
    assert result["active_bounds"] >= 1


def test_inverse_magnetic_probe_reconstruction_rejects_bad_probe_contract() -> None:
    k = _KernelStub()
    coils = CoilSet(
        positions=[(5.1, -0.35)],
        currents=np.array([0.0], dtype=np.float64),
    )

    with np.testing.assert_raises(ValueError):
        reconstruct_coil_currents_from_magnetic_probes(
            k,
            coils,
            flux_points=np.array([[5.25, -0.15]], dtype=np.float64),
            flux_measurements=np.array([0.0, 1.0], dtype=np.float64),
        )

    with np.testing.assert_raises(ValueError):
        reconstruct_coil_currents_from_magnetic_probes(
            k,
            coils,
            b_probe_points=np.array([[5.25, -0.15]], dtype=np.float64),
            b_probe_directions=["phi"],
            b_probe_measurements=np.array([0.0], dtype=np.float64),
        )


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
