# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — AMR Convergence Tests
"""Contract tests for the validation AMR equilibrium utility."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_fusion.core.amr_patch import (
    AMRPatch,
    _find_patch_bounds,
    _jacobi_smooth,
    flag_refinement_cells,
    gradient_magnitude,
    prolongate,
    restrict,
    solve_amr,
)

FloatArray = NDArray[np.float64]
AMRInputMutator = Callable[
    [FloatArray, FloatArray, FloatArray, FloatArray],
    tuple[FloatArray, FloatArray, FloatArray, FloatArray],
]


def _solovev_psi(
    R: FloatArray,
    Z: FloatArray,
    R0: float = 6.2,
    a: float = 2.0,
    kappa: float = 1.7,
) -> FloatArray:
    """Return a compact Solovev-like normalized flux surface."""
    RR, ZZ = np.meshgrid(R, Z)
    eps_s = ((R0 + a) ** 2 - R0**2) / R0**2
    u = (RR**2 - R0**2) / (eps_s * R0**2)
    v = ZZ / (kappa * a)
    return np.asarray(np.maximum(0.0, 1.0 - u**2 - v**2), dtype=np.float64)


def _nrmse(y_true: FloatArray, y_pred: FloatArray) -> float:
    """Return normalized root-mean-square error for two flux grids."""
    rng = float(np.max(y_true) - np.min(y_true))
    if rng < 1e-30:
        return 0.0
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)) / rng)


def _base_grid(size: int = 17) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Build a square finite AMR input grid with zero source."""
    R = np.linspace(1.0, 2.0, size, dtype=np.float64)
    Z = np.linspace(-0.5, 0.5, size, dtype=np.float64)
    psi = _solovev_psi(R, Z, R0=1.5, a=0.3, kappa=1.5)
    source = np.zeros_like(psi)
    return R, Z, psi, source


class TestAMRPatch:
    """AMRPatch metadata accessors should expose physical patch coordinates."""

    def test_patch_accessors_return_spacing_and_coordinates(self) -> None:
        """Patch properties should derive spacing and coordinate grids from bounds."""
        patch = AMRPatch(
            r_lo=1.0,
            r_hi=2.0,
            z_lo=-0.5,
            z_hi=0.5,
            level=1,
            nr=5,
            nz=3,
            psi=np.zeros((3, 5), dtype=np.float64),
        )

        assert patch.dr == pytest.approx(0.25)
        assert patch.dz == pytest.approx(0.5)
        np.testing.assert_allclose(patch.r_grid, np.linspace(1.0, 2.0, 5))
        np.testing.assert_allclose(patch.z_grid, np.linspace(-0.5, 0.5, 3))


class TestGradientMagnitude:
    """Gradient helper tests for the AMR refinement indicator."""

    def test_uniform_field_zero_gradient(self) -> None:
        """Uniform fields should have zero discrete gradient."""
        psi = np.ones((32, 32), dtype=np.float64)
        grad = gradient_magnitude(psi, dr=0.1, dz=0.1)
        assert np.allclose(grad, 0.0, atol=1e-12)

    def test_linear_field_constant_gradient(self) -> None:
        """Linear radial fields should recover the imposed radial slope inside."""
        R = np.linspace(0.0, 1.0, 32, dtype=np.float64)
        Z = np.linspace(0.0, 1.0, 32, dtype=np.float64)
        RR, _ = np.meshgrid(R, Z)
        psi = np.asarray(3.0 * RR, dtype=np.float64)
        dr = float(R[1] - R[0])
        dz = float(Z[1] - Z[0])
        grad = gradient_magnitude(psi, dr, dz)
        assert np.allclose(grad[1:-1, 1:-1], 3.0, atol=0.1)

    def test_one_cell_field_uses_zero_edge_gradient(self) -> None:
        """Degenerate one-cell inputs should remain finite and zero-gradient."""
        psi = np.ones((1, 1), dtype=np.float64)
        grad = gradient_magnitude(psi, dr=1.0, dz=1.0)
        np.testing.assert_allclose(grad, np.zeros((1, 1), dtype=np.float64))


class TestProlongateRestrict:
    """Interpolation helpers should preserve shapes and scalar ranges."""

    def test_prolongate_preserves_range(self) -> None:
        """Bilinear prolongation should not overshoot a bounded coarse field."""
        coarse = np.random.default_rng(42).uniform(0.0, 1.0, (16, 16))
        fine = prolongate(np.asarray(coarse, dtype=np.float64), (32, 32))
        assert fine.shape == (32, 32)
        assert float(np.min(fine)) >= float(np.min(coarse)) - 1e-10
        assert float(np.max(fine)) <= float(np.max(coarse)) + 1e-10

    def test_restrict_then_prolongate_idempotent_shape(self) -> None:
        """Restricting and prolongating should preserve the requested final shape."""
        fine = np.random.default_rng(7).uniform(0.0, 1.0, (32, 32))
        coarse = restrict(np.asarray(fine, dtype=np.float64), (16, 16))
        restored = prolongate(coarse, (32, 32))
        assert restored.shape == fine.shape


class TestFlagRefinement:
    """Refinement flagging should isolate steep but finite gradients."""

    def test_flags_steep_gradient(self) -> None:
        """Percentile thresholding should flag a subset of steep-gradient cells."""
        R = np.linspace(0.0, 1.0, 64, dtype=np.float64)
        Z = np.linspace(0.0, 1.0, 64, dtype=np.float64)
        psi = _solovev_psi(R, Z, R0=0.5, a=0.3, kappa=1.5)
        dr = float(R[1] - R[0])
        dz = float(Z[1] - Z[0])
        flagged = flag_refinement_cells(psi, dr, dz)
        assert flagged.any(), "some cells should be flagged"
        assert not flagged.all(), "not all cells should be flagged"

    def test_small_flagged_component_is_rejected(self) -> None:
        """Patch detection should reject components below the minimum patch extent."""
        R = np.linspace(1.0, 2.0, 8, dtype=np.float64)
        Z = np.linspace(-1.0, 1.0, 8, dtype=np.float64)
        flagged = np.zeros((8, 8), dtype=np.bool_)
        flagged[3, 3] = True

        assert _find_patch_bounds(flagged, R, Z, pad_cells=0) == []


class TestJacobiSmooth:
    """Jacobi smoother contracts for degenerate and cylindrical stencils."""

    def test_nearly_zero_coefficients_return_original_field(self) -> None:
        """Huge grid spacing should trigger the zero-coefficient guard."""
        psi = np.arange(9, dtype=np.float64).reshape(3, 3)
        source = np.zeros_like(psi)
        smoothed = _jacobi_smooth(psi, source, dr=1e200, dz=1e200)
        np.testing.assert_allclose(smoothed, psi)


class TestSolveAMR:
    """End-to-end AMR solve and validation-boundary tests."""

    @pytest.mark.parametrize(
        ("mutator", "message"),
        [
            (lambda R, Z, psi, source: (R, Z, psi[0], source), "psi_base must be a 2D array"),
            (
                lambda R, Z, psi, source: (R, Z, psi, source[:-1]),
                "source shape must match psi_base",
            ),
            (
                lambda R, Z, psi, source: (R.reshape(1, -1), Z, psi, source),
                "R and Z must be 1D coordinate arrays",
            ),
            (
                lambda R, Z, psi, source: (R[:-1], Z, psi, source),
                "R/Z lengths must match psi_base dimensions",
            ),
            (
                lambda R, Z, psi, source: (R[:1], Z, psi[:, :1], source[:, :1]),
                "AMR requires at least two R and Z coordinates",
            ),
            (
                lambda R, Z, psi, source: (R, Z, psi * np.nan, source),
                "AMR inputs must be finite",
            ),
            (
                lambda R, Z, psi, source: (R[::-1], Z, psi, source),
                "R and Z coordinates must be strictly increasing",
            ),
        ],
    )
    def test_solve_amr_rejects_invalid_inputs(
        self,
        mutator: AMRInputMutator,
        message: str,
    ) -> None:
        """Invalid grids and fields should fail before smoothing or patch creation."""
        R, Z, psi, source = _base_grid()
        candidate = mutator(R, Z, psi, source)
        bad_R, bad_Z, bad_psi, bad_source = candidate

        with pytest.raises(ValueError, match=message):
            solve_amr(
                bad_psi,
                bad_R,
                bad_Z,
                bad_source,
                smooth_iters=1,
                refine_smooth_iters=1,
            )

    def test_solve_amr_rejects_negative_iteration_counts(self) -> None:
        """Negative smoother budgets should be rejected as invalid AMR input."""
        R, Z, psi, source = _base_grid()

        with pytest.raises(ValueError, match="iteration counts"):
            solve_amr(psi, R, Z, source, smooth_iters=-1, refine_smooth_iters=1)
        with pytest.raises(ValueError, match="iteration counts"):
            solve_amr(psi, R, Z, source, smooth_iters=1, refine_smooth_iters=-1)

    def test_amr_returns_finite_field_and_patches(self) -> None:
        """AMR should run on a Solovev-like field and return finite patches."""
        nr_base, nz_base = 65, 65
        R0, a, kappa = 6.2, 2.0, 1.7

        R_base = np.linspace(R0 - 1.5 * a, R0 + 1.5 * a, nr_base, dtype=np.float64)
        Z_base = np.linspace(-kappa * 1.5 * a, kappa * 1.5 * a, nz_base, dtype=np.float64)
        psi_init = _solovev_psi(R_base, Z_base, R0, a, kappa)
        source = np.zeros_like(psi_init)

        psi_amr, patches = solve_amr(
            psi_init,
            R_base,
            Z_base,
            source,
            smooth_iters=5,
            refine_smooth_iters=10,
        )

        assert psi_amr.shape == psi_init.shape
        assert all(np.all(np.isfinite(patch.psi)) for patch in patches)

    def test_amr_nrmse_below_threshold(self) -> None:
        """Jacobi-smoothed AMR should preserve a Solovev-like field within tolerance."""
        nr, nz = 65, 65
        R0, a, kappa = 1.85, 0.57, 1.97

        R = np.linspace(R0 - 1.5 * a, R0 + 1.5 * a, nr, dtype=np.float64)
        Z = np.linspace(-kappa * 1.5 * a, kappa * 1.5 * a, nz, dtype=np.float64)
        psi_true = _solovev_psi(R, Z, R0, a, kappa)
        source = np.zeros_like(psi_true)

        psi_amr, _ = solve_amr(
            psi_true,
            R,
            Z,
            source,
            smooth_iters=5,
            refine_smooth_iters=10,
        )
        err = _nrmse(psi_true, psi_amr)
        assert err < 0.02, f"NRMSE={err:.4f} exceeds 2% threshold"
