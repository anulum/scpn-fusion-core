# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Free-boundary adapter methods mixed into :class:`FusionKernel`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scpn_fusion.core.fusion_kernel_free_boundary import (
    build_coilset_from_config as _build_coilset_from_config_runtime,
    build_magnetic_probe_response_matrix as _build_magnetic_probe_response_matrix_runtime,
    build_mutual_inductance_matrix as _build_mutual_inductance_matrix_runtime,
    compute_external_flux as _compute_external_flux_runtime,
    green_function as _green_function_runtime,
    interp_psi as _interp_psi_runtime,
    optimize_coil_currents as _optimize_coil_currents_runtime,
    reconstruct_coil_currents_from_magnetic_probes as _reconstruct_coil_currents_from_magnetic_probes_runtime,
    resolve_shape_target_flux as _resolve_shape_target_flux_runtime,
    sample_flux_at_points as _sample_flux_at_points_runtime,
    solve_free_boundary as _solve_free_boundary_runtime,
)
from scpn_fusion.core.fusion_kernel_numerics import FloatArray

if TYPE_CHECKING:
    from scpn_fusion.core.fusion_kernel import CoilSet


class FusionKernelFreeBoundaryMixin:
    """Attach free-boundary operations to ``FusionKernel``.

    The implementation lives in :mod:`scpn_fusion.core.fusion_kernel_free_boundary`
    so the core kernel remains focused on grid state and equilibrium solving.
    This mixin preserves the existing ``FusionKernel`` method API while routing
    each operation to the validated adapter function used by tests and tooling.
    """

    @staticmethod
    def _green_function(R_src: float, Z_src: float, R_obs: float, Z_obs: float) -> float:
        """Evaluate the scalar circular-filament flux Green's function."""
        return _green_function_runtime(R_src, Z_src, R_obs, Z_obs)

    def _compute_external_flux(self, coils: CoilSet) -> FloatArray:
        """Sum external coil flux contributions on the kernel grid."""
        return _compute_external_flux_runtime(self, coils)

    def build_coilset_from_config(self) -> CoilSet:
        """Build a validated free-boundary coil set from ``self.cfg``."""
        return _build_coilset_from_config_runtime(self)

    def _build_mutual_inductance_matrix(
        self,
        coils: CoilSet,
        obs_points: FloatArray,
    ) -> FloatArray:
        """Build the coil-to-control-point response matrix."""
        return _build_mutual_inductance_matrix_runtime(self, coils, obs_points)

    def _build_magnetic_probe_response_matrix(
        self,
        coils: CoilSet,
        *,
        flux_points: FloatArray | None = None,
        b_probe_points: FloatArray | None = None,
        b_probe_directions: list[str] | tuple[str, ...] | None = None,
    ) -> FloatArray:
        """Build coil-current response rows for flux loops and magnetic probes."""
        return _build_magnetic_probe_response_matrix_runtime(
            self,
            coils,
            flux_points=flux_points,
            b_probe_points=b_probe_points,
            b_probe_directions=b_probe_directions,
        )

    def reconstruct_coil_currents_from_magnetic_probes(
        self,
        coils: CoilSet,
        *,
        flux_points: FloatArray | None = None,
        flux_measurements: FloatArray | None = None,
        b_probe_points: FloatArray | None = None,
        b_probe_directions: list[str] | tuple[str, ...] | None = None,
        b_probe_measurements: FloatArray | None = None,
        measurement_sigma: FloatArray | None = None,
        tikhonov_alpha: float = 1.0e-6,
    ) -> dict[str, Any]:
        """Fit free-boundary coil currents from magnetic diagnostic measurements."""
        return _reconstruct_coil_currents_from_magnetic_probes_runtime(
            self,
            coils,
            flux_points=flux_points,
            flux_measurements=flux_measurements,
            b_probe_points=b_probe_points,
            b_probe_directions=b_probe_directions,
            b_probe_measurements=b_probe_measurements,
            measurement_sigma=measurement_sigma,
            tikhonov_alpha=tikhonov_alpha,
        )

    def optimize_coil_currents(
        self,
        coils: CoilSet,
        target_flux: FloatArray,
        tikhonov_alpha: float = 1e-4,
    ) -> FloatArray:
        """Fit bounded coil currents for the requested target flux vector."""
        return _optimize_coil_currents_runtime(
            self,
            coils,
            target_flux,
            tikhonov_alpha=tikhonov_alpha,
        )

    def _resolve_shape_target_flux(self, coils: CoilSet) -> FloatArray:
        """Resolve target flux vector for shape optimisation control points."""
        return _resolve_shape_target_flux_runtime(self, coils)

    def solve_free_boundary(
        self,
        coils: CoilSet,
        max_outer_iter: int = 20,
        tol: float = 1e-4,
        optimize_shape: bool = False,
        tikhonov_alpha: float = 1e-4,
    ) -> dict[str, Any]:
        """Run the free-boundary outer loop with optional coil optimisation."""
        return _solve_free_boundary_runtime(
            self,
            coils,
            max_outer_iter=max_outer_iter,
            tol=tol,
            optimize_shape=optimize_shape,
            tikhonov_alpha=tikhonov_alpha,
        )

    def _interp_psi(self, R_pt: float, Z_pt: float) -> float:
        """Bilinear interpolation of Psi at an arbitrary (R, Z) point."""
        return _interp_psi_runtime(self, R_pt, Z_pt)

    def _sample_flux_at_points(self, points: FloatArray) -> FloatArray:
        """Sample poloidal flux at validated ``(R, Z)`` points."""
        return _sample_flux_at_points_runtime(self, points)
