# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Jacobi/SOR Stencil Parity Regression Test
"""
Regression tests for the cylindrical GS* operator stencil.

Bug fixed in commit 490a314: _jacobi_step used a flat Cartesian
0.25 * (4-neighbor) stencil instead of the correct cylindrical
a_E/a_W coefficients with 1/(2R·dR) correction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scpn_fusion.core.fusion_kernel import FusionKernel


def _make_kernel(tmp_path: Path, NR: int = 17, NZ: int = 17) -> FusionKernel:
    """Build a minimal FusionKernel for stencil tests."""
    cfg = {
        "reactor_name": "Stencil-Test",
        "grid_resolution": [NR, NZ],
        "dimensions": {"R_min": 0.5, "R_max": 1.5, "Z_min": -0.5, "Z_max": 0.5},
        "physics": {"plasma_current_target": 1.0, "vacuum_permeability": 1.0},
        "coils": [],
        "solver": {
            "max_iterations": 3,
            "convergence_threshold": 1e-6,
            "relaxation_factor": 0.1,
        },
    }
    path = tmp_path / "stencil_test.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return FusionKernel(str(path))


class TestStencilParity:
    """Verify Jacobi and SOR use identical cylindrical stencils."""

    def test_single_sweep_ordering_bounded(self, tmp_path: Path) -> None:
        """Jacobi vs Gauss-Seidel(omega=1) differ only by ordering effects.

        SOR with omega=1 is Red-Black Gauss-Seidel (in-place updates),
        so a single sweep differs from Jacobi (out-of-place) by O(h²).
        Both must use the same cylindrical a_E/a_W coefficients; the
        ordering effect alone accounts for the difference.
        """
        kernel = _make_kernel(tmp_path)

        psi_init = (kernel.RR - 1.0) ** 2 + kernel.ZZ**2
        source = -0.1 * kernel.RR

        psi_jacobi = kernel._jacobi_step(psi_init, source)
        psi_sor = kernel._sor_step(psi_init, source, omega=1.0)

        interior_j = psi_jacobi[1:-1, 1:-1]
        interior_s = psi_sor[1:-1, 1:-1]
        max_diff = float(np.max(np.abs(interior_j - interior_s)))
        # Ordering effect is O(h²) ~ O(1/N²) ~ 0.004 for N=17
        assert max_diff < 0.05, f"Jacobi vs GS ordering diff unexpectedly large: {max_diff}"
        # Both must produce nontrivial updates (not zero)
        assert float(np.max(np.abs(interior_j - psi_init[1:-1, 1:-1]))) > 1e-6
        assert float(np.max(np.abs(interior_s - psi_init[1:-1, 1:-1]))) > 1e-6

    def test_cylindrical_stencil_r_dependence(self, tmp_path: Path) -> None:
        """The stencil must produce R-dependent update for psi=R.

        The GS* operator has a_E = 1/dR^2 - 1/(2R·dR) and
        a_W = 1/dR^2 + 1/(2R·dR).  For psi = R (linear in R) with
        zero source, the update should differ from psi_init because
        d^2(R)/dR^2 = 0 but -(1/R) dR/dR = -1/R != 0.

        A flat Cartesian stencil (a_E = a_W = 1/dR^2) would give back
        psi = R exactly, which is wrong.
        """
        kernel = _make_kernel(tmp_path)
        psi = kernel.RR.copy()
        source = np.zeros_like(psi)

        result = kernel._jacobi_step(psi, source)

        interior_diff = result[1:-1, 1:-1] - psi[1:-1, 1:-1]
        max_correction = float(np.max(np.abs(interior_diff)))
        assert max_correction > 1e-6, "Jacobi stencil appears Cartesian (no 1/R correction)"

    def test_convergence_to_same_solution(self, tmp_path: Path) -> None:
        """Jacobi and SOR must converge to the same equilibrium."""
        kernel = _make_kernel(tmp_path)
        source = -0.01 * kernel.RR

        psi_j = np.zeros_like(kernel.RR)
        psi_s = np.zeros_like(kernel.RR)

        for _ in range(200):
            psi_j = kernel._jacobi_step(psi_j, source)
        for _ in range(200):
            psi_s = kernel._sor_step(psi_s, source, omega=1.5)

        rel_l2 = float(np.linalg.norm(psi_j[1:-1, 1:-1] - psi_s[1:-1, 1:-1])) / (
            float(np.linalg.norm(psi_s[1:-1, 1:-1])) + 1e-30
        )
        assert rel_l2 < 0.05, f"Jacobi/SOR solution mismatch: rel_L2 = {rel_l2:.4f}"

    def test_amr_jacobi_uses_cylindrical_stencil(self) -> None:
        """AMR patch smoother must also use cylindrical coefficients."""
        from scpn_fusion.core.amr_patch import _jacobi_smooth

        NR, NZ = 20, 20
        R_1d = np.linspace(0.5, 1.5, NR)
        dr = float(R_1d[1] - R_1d[0])
        dz = 0.05
        psi = np.tile(R_1d, (NZ, 1))
        source = np.zeros_like(psi)

        result_cyl = _jacobi_smooth(psi, source, dr, dz, R_1d=R_1d, iterations=1, omega=1.0)
        result_cart = _jacobi_smooth(psi, source, dr, dz, R_1d=None, iterations=1, omega=1.0)

        diff = float(np.max(np.abs(result_cyl - result_cart)))
        assert diff > 1e-6, "AMR Jacobi smoother ignores R_1d parameter"
