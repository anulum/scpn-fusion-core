# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Geometric Multigrid Solver
"""Free-function geometric multigrid solver for the Grad-Shafranov GS* operator.

The V-cycle and its smoother, residual, restriction and prolongation operators are
pure functions of their arguments. They live here as free functions so they can be
driven both by the kernel's iterative-solver mixin (which delegates to them) and by
the canonical :func:`multigrid_solve` full-solve loop registered as the NumPy tier
of the ``multigrid_solve`` dispatch kernel (:mod:`scpn_fusion.core._multi_compat`).

The full solve is the cross-tier contract: given a source term and a
boundary-valued initial flux on an ``nr x nz`` ``R-Z`` grid, repeat V-cycles until
the GS* residual (L-infinity over the interior, matching the Rust tier) falls to
``tol`` (or ``max_cycles`` is reached). It is algorithm-parity with the Rust tier
(``scpn_fusion_rs.multigrid_vcycle``): both relax the identical toroidal GS*
operator to the same fixed point, so the converged flux maps agree to a tight
relative tolerance even though the per-cycle counts and exact residual paths differ.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def validate_sor_omega(omega: float) -> float:
    """Validate the SOR relaxation factor for elliptic GS solves.

    Parameters
    ----------
    omega : float
        Over-relaxation factor.

    Returns
    -------
    float
        The validated factor.

    Raises
    ------
    ValueError
        If ``omega`` is non-finite or outside ``[1.0, 2.0)``.
    """
    omega_value = float(omega)
    if not np.isfinite(omega_value) or omega_value < 1.0 or omega_value >= 2.0:
        raise ValueError("omega must be finite and satisfy 1.0 <= omega < 2.0")
    return omega_value


def restrict_full_weight(fine: FloatArray) -> FloatArray:
    """Full-weighting restriction operator (fine → coarse, 9-point stencil).

    Parameters
    ----------
    fine : FloatArray
        Fine-grid field.

    Returns
    -------
    FloatArray
        Coarse-grid field.
    """
    nz_f, nr_f = fine.shape
    nz_c = (nz_f + 1) // 2
    nr_c = (nr_f + 1) // 2
    coarse = np.zeros((nz_c, nr_c))

    # Interior: vectorised 9-point stencil via even-index slicing
    coarse[1:-1, 1:-1] = (
        4.0 * fine[2:-2:2, 2:-2:2]
        + 2.0
        * (
            fine[1:-3:2, 2:-2:2]
            + fine[3:-1:2, 2:-2:2]
            + fine[2:-2:2, 1:-3:2]
            + fine[2:-2:2, 3:-1:2]
        )
        + (
            fine[1:-3:2, 1:-3:2]
            + fine[1:-3:2, 3:-1:2]
            + fine[3:-1:2, 1:-3:2]
            + fine[3:-1:2, 3:-1:2]
        )
    ) / 16.0

    # Boundary: inject directly
    coarse[0, :] = fine[0, ::2][:nr_c]
    coarse[-1, :] = fine[-1, ::2][:nr_c]
    coarse[:, 0] = fine[::2, 0][:nz_c]
    coarse[:, -1] = fine[::2, -1][:nz_c]

    return coarse


def prolongate_bilinear(coarse: FloatArray, nz_f: int, nr_f: int) -> FloatArray:
    """Bilinear prolongation operator (coarse → fine).

    Parameters
    ----------
    coarse : FloatArray
        Coarse-grid field.
    nz_f, nr_f : int
        Target fine-grid dimensions.

    Returns
    -------
    FloatArray
        Fine-grid field.
    """
    nz_c, nr_c = coarse.shape
    fine = np.zeros((nz_f, nr_f))

    # Coincident points (even rows, even cols)
    nz_used = min(nz_c, (nz_f + 1) // 2)
    nr_used = min(nr_c, (nr_f + 1) // 2)
    fine[: 2 * nz_used - 1 : 2, : 2 * nr_used - 1 : 2] = coarse[:nz_used, :nr_used]

    # Horizontal midpoints (even rows, odd cols)
    h_end = min(2 * (nr_c - 1), nr_f - 1)
    fine[: 2 * nz_used - 1 : 2, 1:h_end:2] = (
        0.5 * (coarse[:nz_used, :-1] + coarse[:nz_used, 1:])[:, : (h_end - 1) // 2 + 1]
    )

    # Vertical midpoints (odd rows, even cols)
    v_end = min(2 * (nz_c - 1), nz_f - 1)
    fine[1:v_end:2, : 2 * nr_used - 1 : 2] = (
        0.5 * (coarse[:-1, :nr_used] + coarse[1:, :nr_used])[: ((v_end - 1) // 2 + 1), :]
    )

    # Centre points (odd rows, odd cols)
    fine[1:v_end:2, 1:h_end:2] = (
        0.25
        * (coarse[:-1, :-1] + coarse[1:, :-1] + coarse[:-1, 1:] + coarse[1:, 1:])[
            : ((v_end - 1) // 2 + 1), : (h_end - 1) // 2 + 1
        ]
    )

    return fine


def mg_smooth(
    psi: FloatArray,
    source: FloatArray,
    r_grid: FloatArray,
    dr: float,
    dz: float,
    omega: float,
    n_sweeps: int,
) -> FloatArray:
    """Red-Black SOR smoother with the toroidal ``1/R`` stencil for multigrid.

    Parameters
    ----------
    psi : FloatArray
        Current solution estimate (mutated in place and returned).
    source : FloatArray
        Right-hand-side source term.
    r_grid : FloatArray
        ``R``-coordinate meshgrid matching ``psi`` shape.
    dr, dz : float
        Grid spacings.
    omega : float
        SOR over-relaxation factor.
    n_sweeps : int
        Number of Red-Black sweeps.

    Returns
    -------
    FloatArray
        The smoothed estimate.
    """
    omega = validate_sor_omega(omega)
    nz, nr = psi.shape
    dr2 = dr**2
    dz2 = dz**2

    r_int = r_grid[1:-1, 1:-1]
    r_safe = np.maximum(r_int, 1e-10)
    a_e = 1.0 / dr2 - 1.0 / (2.0 * r_safe * dr)
    a_w = 1.0 / dr2 + 1.0 / (2.0 * r_safe * dr)
    a_ns = 1.0 / dz2
    a_c = 2.0 / dr2 + 2.0 / dz2

    ii, jj = np.mgrid[1 : nz - 1, 1 : nr - 1]

    for _ in range(n_sweeps):
        for parity in (0, 1):
            mask = ((ii + jj) % 2) == parity
            gs_update = (
                a_e[mask] * psi[1:-1, 2:][mask]
                + a_w[mask] * psi[1:-1, 0:-2][mask]
                + a_ns * psi[0:-2, 1:-1][mask]
                + a_ns * psi[2:, 1:-1][mask]
                - source[1:-1, 1:-1][mask]
            ) / a_c
            old_vals = psi[1:-1, 1:-1][mask]
            interior = psi[1:-1, 1:-1]
            interior[mask] = (1.0 - omega) * old_vals + omega * gs_update
            psi[1:-1, 1:-1] = interior

    return psi


def mg_residual(
    psi: FloatArray,
    source: FloatArray,
    r_grid: FloatArray,
    dr: float,
    dz: float,
) -> FloatArray:
    """Compute the GS* residual ``r = L*[psi] - source`` on the given grid.

    Parameters
    ----------
    psi : FloatArray
        Current solution estimate.
    source : FloatArray
        Right-hand-side source term.
    r_grid : FloatArray
        ``R``-coordinate meshgrid matching ``psi`` shape.
    dr, dz : float
        Grid spacings.

    Returns
    -------
    FloatArray
        The residual, zero on the boundary.
    """
    dr2 = dr**2
    dz2 = dz**2

    residual = np.zeros_like(psi)
    r_int = r_grid[1:-1, 1:-1]
    r_safe = np.maximum(r_int, 1e-10)

    d2r = (psi[1:-1, 2:] - 2.0 * psi[1:-1, 1:-1] + psi[1:-1, 0:-2]) / dr2
    d1r = (psi[1:-1, 2:] - psi[1:-1, 0:-2]) / (2.0 * dr)
    d2z = (psi[2:, 1:-1] - 2.0 * psi[1:-1, 1:-1] + psi[0:-2, 1:-1]) / dz2

    lpsi = d2r - d1r / r_safe + d2z
    residual[1:-1, 1:-1] = lpsi - source[1:-1, 1:-1]
    return residual


def multigrid_vcycle(
    psi: FloatArray,
    source: FloatArray,
    r_grid: FloatArray,
    dr: float,
    dz: float,
    *,
    omega: float = 1.0,
    pre_smooth: int = 3,
    post_smooth: int = 3,
    min_grid: int = 5,
) -> FloatArray:
    """One V-cycle of geometric multigrid for the GS* operator.

    Parameters
    ----------
    psi : FloatArray
        Current solution estimate.
    source : FloatArray
        Right-hand-side source term.
    r_grid : FloatArray
        ``R``-coordinate meshgrid matching ``psi`` shape.
    dr, dz : float
        Grid spacings.
    omega : float, optional
        SOR over-relaxation factor for the smoother, by default 1.0 (Red-Black
        Gauss-Seidel, the best multigrid smoother; over-relaxation smooths poorly).
    pre_smooth, post_smooth : int, optional
        Smoothing sweeps before/after the coarse correction, by default 3.
    min_grid : int, optional
        Minimum grid dimension before switching to a direct solve, by default 5.

    Returns
    -------
    FloatArray
        The improved solution estimate.
    """
    nz, nr = psi.shape

    # Base case: grid too coarse — solve directly with many SOR sweeps
    if min_grid >= nz or min_grid >= nr:
        return mg_smooth(psi.copy(), source, r_grid, dr, dz, omega, n_sweeps=50)

    # 1. Pre-smooth
    psi = mg_smooth(psi.copy(), source, r_grid, dr, dz, omega, pre_smooth)

    # 2. Compute the defect (negative residual). The error e satisfies
    #    L*[e] = source - L*[psi] = -(L*[psi] - source), so the coarse-grid
    #    right-hand side is the *negated* residual. Restricting the raw residual
    #    instead solves L*[e] = +r, which inverts every correction (psi <- psi - e)
    #    and stalls/diverges the solve.
    defect = -mg_residual(psi, source, r_grid, dr, dz)

    # 3. Restrict the defect and R-grid to the coarse level
    d_coarse = restrict_full_weight(defect)
    rgrid_coarse = restrict_full_weight(r_grid)
    nz_c, nr_c = d_coarse.shape

    # Coarse grid spacings (doubled)
    dr_c = dr * 2.0
    dz_c = dz * 2.0

    # 4. Solve the coarse-grid correction: L*[e] = defect
    e_coarse: FloatArray = np.zeros((nz_c, nr_c))
    e_coarse = multigrid_vcycle(
        e_coarse,
        d_coarse,
        rgrid_coarse,
        dr_c,
        dz_c,
        omega=omega,
        pre_smooth=pre_smooth,
        post_smooth=post_smooth,
        min_grid=min_grid,
    )

    # 5. Prolongate the correction and apply
    correction = prolongate_bilinear(e_coarse, nz, nr)
    psi = psi + correction

    # 6. Post-smooth
    psi = mg_smooth(psi, source, r_grid, dr, dz, omega, post_smooth)

    return psi


def residual_linf(
    psi: FloatArray,
    source: FloatArray,
    r_grid: FloatArray,
    dr: float,
    dz: float,
) -> float:
    """Return the L-infinity GS* residual over the interior (matches the Rust tier)."""
    interior = mg_residual(psi, source, r_grid, dr, dz)[1:-1, 1:-1]
    if interior.size == 0:
        return 0.0
    return float(np.max(np.abs(interior)))


def multigrid_solve(
    source: FloatArray,
    psi_bc: FloatArray,
    r_min: float,
    r_max: float,
    z_min: float,
    z_max: float,
    nr: int,
    nz: int,
    *,
    tol: float = 1e-6,
    max_cycles: int = 500,
    omega: float = 1.0,
    pre_smooth: int = 3,
    post_smooth: int = 3,
    min_grid: int = 5,
) -> tuple[FloatArray, float, int, bool]:
    """Full geometric-multigrid solve of the GS* operator on an ``R-Z`` grid.

    Canonical NumPy tier of the ``multigrid_solve`` dispatch kernel; algorithm-parity
    with the Rust tier (``scpn_fusion_rs.multigrid_vcycle``): both relax the same
    elliptic operator to the same fixed point, so the converged flux maps agree to a
    tight relative tolerance.

    Parameters
    ----------
    source : FloatArray
        Right-hand-side source term, shape ``(nz, nr)``.
    psi_bc : FloatArray
        Initial flux carrying the Dirichlet boundary values, shape ``(nz, nr)``.
        The boundary ring is preserved (the bilinear prolongation of the coarse
        correction writes onto the boundary, so it is re-applied after each cycle).
    r_min, r_max, z_min, z_max : float
        Grid extent [m].
    nr, nz : int
        Grid dimensions.
    tol : float, optional
        Target L-infinity residual, by default 1e-6.
    max_cycles : int, optional
        Maximum number of V-cycles, by default 500.
    omega : float, optional
        Smoother relaxation factor, by default 1.0.
    pre_smooth, post_smooth : int, optional
        Smoothing sweeps before/after the coarse correction, by default 3.
    min_grid : int, optional
        Minimum grid dimension before a direct solve, by default 5.

    Returns
    -------
    psi : FloatArray
        The converged flux map.
    residual : float
        The final L-infinity residual over the interior.
    n_cycles : int
        The number of V-cycles performed.
    converged : bool
        Whether the residual reached ``tol`` within ``max_cycles``.

    Raises
    ------
    ValueError
        If the grid dimensions are inconsistent with the array shapes, or if
        ``tol``/``max_cycles`` are not strictly positive.
    """
    source_arr = np.asarray(source, dtype=np.float64)
    psi = np.asarray(psi_bc, dtype=np.float64).copy()
    if source_arr.shape != (nz, nr) or psi.shape != (nz, nr):
        raise ValueError(
            f"source and psi_bc must have shape (nz, nr) = ({nz}, {nr}); "
            f"got source={source_arr.shape}, psi_bc={psi.shape}."
        )
    if not (np.isfinite(tol) and tol > 0.0):
        raise ValueError("tol must be finite and > 0.")
    if max_cycles < 1:
        raise ValueError("max_cycles must be >= 1.")

    r_axis = np.linspace(r_min, r_max, nr)
    z_axis = np.linspace(z_min, z_max, nz)
    r_grid, _z_grid = np.meshgrid(r_axis, z_axis)
    dr = float(r_axis[1] - r_axis[0]) if nr > 1 else 1.0
    dz = float(z_axis[1] - z_axis[0]) if nz > 1 else 1.0

    # Dirichlet boundary ring captured from psi_bc, re-applied after each cycle.
    psi_boundary = psi.copy()

    def _enforce_boundary(field: FloatArray) -> None:
        field[0, :] = psi_boundary[0, :]
        field[-1, :] = psi_boundary[-1, :]
        field[:, 0] = psi_boundary[:, 0]
        field[:, -1] = psi_boundary[:, -1]

    n_cycles = 0
    residual = residual_linf(psi, source_arr, r_grid, dr, dz)
    converged = residual < tol
    while not converged and n_cycles < max_cycles:
        psi = multigrid_vcycle(
            psi,
            source_arr,
            r_grid,
            dr,
            dz,
            omega=omega,
            pre_smooth=pre_smooth,
            post_smooth=post_smooth,
            min_grid=min_grid,
        )
        _enforce_boundary(psi)
        n_cycles += 1
        residual = residual_linf(psi, source_arr, r_grid, dr, dz)
        converged = residual < tol

    return psi, residual, n_cycles, converged
