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

from scpn_fusion.core.array_contract import (
    array_input,
    scalar_kind,
    integer_value,
    real_value,
    checked_grid,
    grid_spacing,
    finite_array,
)

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
    *,
    radius_floor: float | None = None,
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
    radius_floor : float or None, optional
        Explicit legacy caller policy; None uses actual radii.

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
    r_safe = r_int if radius_floor is None else np.maximum(r_int, radius_floor)
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
    *,
    radius_floor: float | None = None,
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
    radius_floor : float or None, optional
        Explicit legacy caller policy; None uses actual radii.

    Returns
    -------
    FloatArray
        The residual, zero on the boundary.
    """
    dr2 = dr**2
    dz2 = dz**2

    residual = np.zeros_like(psi)
    r_int = r_grid[1:-1, 1:-1]
    r_safe = r_int if radius_floor is None else np.maximum(r_int, radius_floor)

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
    radius_floor: float | None = None,
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
    radius_floor : float or None, optional
        Explicit legacy caller policy applied independently at each mesh level.

    Returns
    -------
    FloatArray
        The improved solution estimate.
    """
    nz, nr = psi.shape

    # Base case: grid too coarse — solve directly with many SOR sweeps
    if min_grid >= nz or min_grid >= nr:
        return mg_smooth(
            psi.copy(), source, r_grid, dr, dz, omega, n_sweeps=50, radius_floor=radius_floor
        )

    # 1. Pre-smooth
    psi = mg_smooth(
        psi.copy(), source, r_grid, dr, dz, omega, pre_smooth, radius_floor=radius_floor
    )

    # 2. Compute the defect (negative residual). The error e satisfies
    #    L*[e] = source - L*[psi] = -(L*[psi] - source), so the coarse-grid
    #    right-hand side is the *negated* residual. Restricting the raw residual
    #    instead solves L*[e] = +r, which inverts every correction (psi <- psi - e)
    #    and stalls/diverges the solve.
    defect = -mg_residual(psi, source, r_grid, dr, dz, radius_floor=radius_floor)

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
        radius_floor=radius_floor,
    )

    # 5. Prolongate the correction and apply
    correction = prolongate_bilinear(e_coarse, nz, nr)
    psi = psi + correction

    # 6. Post-smooth
    psi = mg_smooth(psi, source, r_grid, dr, dz, omega, post_smooth, radius_floor=radius_floor)

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
    if not np.all(np.isfinite(interior)):
        raise RuntimeError("multigrid residual arithmetic became nonfinite")
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
        controls, finite inputs or fine/coarse operator geometry are invalid.
    TypeError
        If inputs violate the native float64 ndarray or scalar kind contract.
    RuntimeError
        If accepted inputs cause nonfinite solver arithmetic.

    Notes
    -----
    Conversions are explicit caller responsibilities. All valid ndarray layouts,
    including readonly views, are accepted. Output is fresh, native float64 and
    C contiguous. The positive radial coordinates are used directly in the operator.
    """
    source_arr = array_input(source, "source")
    initial = array_input(psi_bc, "psi_bc")
    for name, value, integer in (
        ("r_min", r_min, False),
        ("r_max", r_max, False),
        ("z_min", z_min, False),
        ("z_max", z_max, False),
        ("nr", nr, True),
        ("nz", nz, True),
        ("tol", tol, False),
        ("max_cycles", max_cycles, True),
        ("omega", omega, False),
        ("pre_smooth", pre_smooth, True),
        ("post_smooth", post_smooth, True),
        ("min_grid", min_grid, True),
    ):
        scalar_kind(value, name, integer=integer)
    nr, nz = integer_value(nr, "nr", 3), integer_value(nz, "nz", 3)
    max_cycles = integer_value(max_cycles, "max_cycles", 1)
    pre_smooth = integer_value(pre_smooth, "pre_smooth", 0)
    post_smooth = integer_value(post_smooth, "post_smooth", 0)
    min_grid = integer_value(min_grid, "min_grid", 3)
    checked_grid(nr, nz)
    if source_arr.shape != (nz, nr) or initial.shape != (nz, nr):
        raise ValueError(
            f"source and psi_bc must have shape (nz, nr) = ({nz}, {nr}); "
            f"got source={source_arr.shape}, psi_bc={initial.shape}."
        )
    r_min, r_max = real_value(r_min, "r_min"), real_value(r_max, "r_max")
    z_min, z_max = real_value(z_min, "z_min"), real_value(z_max, "z_max")
    tol = real_value(tol, "tol")
    omega = validate_sor_omega(real_value(omega, "omega"))
    if tol <= 0.0:
        raise ValueError("tol must be finite and > 0.")
    if r_min <= 0.0:
        raise ValueError("r_min must be positive for the GS operator")
    grid_spacing(r_min, r_max, nr, "R")
    grid_spacing(z_min, z_max, nz, "Z")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            r_axis = np.linspace(r_min, r_max, nr)
            z_axis = np.linspace(z_min, z_max, nz)
            for name, axis in (("R", r_axis), ("Z", z_axis)):
                if not np.all(np.isfinite(axis)) or not np.all(np.diff(axis) > 0.0):
                    raise ValueError(f"{name} actual coordinates must be finite and increasing")
            dr, dz = float(r_axis[1] - r_axis[0]), float(z_axis[1] - z_axis[0])
            r_grid, _ = np.meshgrid(r_axis, z_axis)
            _validate_operator_levels(r_grid, dr, dz, min_grid, (r_min, r_max, z_min, z_max))
    except (FloatingPointError, OverflowError, ZeroDivisionError) as error:
        raise ValueError("multigrid operator geometry is not representable") from error
    finite_array(source_arr, "source")
    finite_array(initial, "psi_bc")
    source_arr = np.array(source_arr, dtype=np.float64, order="C", copy=True)
    psi = np.array(initial, dtype=np.float64, order="C", copy=True)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            return _solve_validated(
                source_arr,
                psi,
                r_grid,
                dr,
                dz,
                tol,
                max_cycles,
                omega,
                pre_smooth,
                post_smooth,
                min_grid,
            )
    except (FloatingPointError, OverflowError, ZeroDivisionError) as error:
        raise RuntimeError("multigrid arithmetic became nonfinite") from error


def _validate_operator_levels(
    r_grid: FloatArray,
    dr: float,
    dz: float,
    min_grid: int,
    bounds: tuple[float, float, float, float],
) -> None:
    """Check the actual fine and restricted coarse geometry before solver work."""
    native_nr, native_nz = r_grid.shape[1], r_grid.shape[0]
    native_r_min, native_r_max, native_z_min, native_z_max = bounds
    while True:
        native_r = native_r_min + np.arange(native_nr) * (
            (native_r_max - native_r_min) / (native_nr - 1)
        )
        native_z = native_z_min + np.arange(native_nz) * (
            (native_z_max - native_z_min) / (native_nz - 1)
        )
        for axis in (native_r, native_z):
            if not np.all(np.isfinite(axis)) or not np.all(np.diff(axis) > 0.0):
                raise ValueError("native coarse coordinates must be finite and increasing")
        mesh, _ = np.meshgrid(native_r, native_z)
        _validate_operator_level(
            mesh, float(native_r[1] - native_r[0]), float(native_z[1] - native_z[0])
        )
        if min(native_nr, native_nz) <= min_grid:
            break
        native_r_min, native_r_max = float(native_r[0]), float(native_r[-1])
        native_z_min, native_z_max = float(native_z[0]), float(native_z[-1])
        native_nr, native_nz = (native_nr + 1) // 2, (native_nz + 1) // 2
    while True:
        _validate_operator_level(r_grid, dr, dz)
        if min(r_grid.shape) <= min_grid:
            return
        r_grid = restrict_full_weight(r_grid)
        dr, dz = dr * 2.0, dz * 2.0


def _validate_operator_level(r_grid: FloatArray, dr: float, dz: float) -> None:
    """Require every denominator, reciprocal and coefficient at one actual mesh level."""
    dr2, dz2 = dr * dr, dz * dz
    radius = r_grid[1:-1, 1:-1]
    radial = 2.0 * radius * dr
    inverse_radius = 1.0 / radius
    inverses = (1.0 / dr2, 1.0 / dz2, 1.0 / (2.0 * dr))
    if (
        not all(np.isfinite(x) and x > 0.0 for x in (dr2, dz2, *inverses))
        or not np.all(np.isfinite(radial))
        or not np.all(radial > 0.0)
        or not np.all(np.isfinite(inverse_radius))
        or not np.all(inverse_radius > 0.0)
    ):
        raise ValueError("multigrid stencil denominators are not representable")
    a_e = inverses[0] - 1.0 / radial
    a_w = inverses[0] + 1.0 / radial
    a_c = 2.0 * inverses[0] + 2.0 * inverses[1]
    if (
        not np.all(np.isfinite(a_e))
        or not np.all(np.isfinite(a_w))
        or not np.isfinite(a_c)
        or a_c <= 0.0
    ):
        raise ValueError("multigrid stencil coefficients are not representable")


def _solve_validated(
    source_arr: FloatArray,
    psi: FloatArray,
    r_grid: FloatArray,
    dr: float,
    dz: float,
    tol: float,
    max_cycles: int,
    omega: float,
    pre_smooth: int,
    post_smooth: int,
    min_grid: int,
) -> tuple[FloatArray, float, int, bool]:
    """Run checked V-cycles after strict admission while preserving boundary values."""
    # Dirichlet boundary ring captured from psi_bc, re-applied after each cycle.
    psi_boundary = psi.copy()

    def _enforce_boundary(field: FloatArray) -> None:
        """Restore all original Dirichlet boundary cells after a correction."""
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
        if not np.all(np.isfinite(psi)):
            raise RuntimeError("multigrid flux arithmetic became nonfinite")
        _enforce_boundary(psi)
        n_cycles += 1
        residual = residual_linf(psi, source_arr, r_grid, dr, dz)
        converged = residual < tol

    return psi, residual, n_cycles, converged
