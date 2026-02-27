# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Adaptive Mesh Refinement (Patch-Based)
# © 1998–2026 Miroslav Šotek. All rights reserved.
# License: GNU AGPL v3
# ──────────────────────────────────────────────────────────────────────
"""Two-level patch-based AMR for X-point and pedestal resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# Refinement ratio between levels
REFINE_FACTOR = 2
# Minimum patch size (cells) in each direction
MIN_PATCH_CELLS = 4


@dataclass
class AMRPatch:
    """Sub-grid refined patch on the equilibrium domain.

    Parameters
    ----------
    r_lo, r_hi, z_lo, z_hi : float
        Physical bounds of the patch [m].
    level : int
        Refinement level (0 = base grid).
    nr, nz : int
        Number of cells in each direction at this level.
    psi : FloatArray
        Poloidal flux on this patch grid, shape ``(nz, nr)``.
    """
    r_lo: float
    r_hi: float
    z_lo: float
    z_hi: float
    level: int
    nr: int
    nz: int
    psi: FloatArray = field(repr=False)

    @property
    def dr(self) -> float:
        return (self.r_hi - self.r_lo) / max(self.nr - 1, 1)

    @property
    def dz(self) -> float:
        return (self.z_hi - self.z_lo) / max(self.nz - 1, 1)

    @property
    def r_grid(self) -> FloatArray:
        return np.linspace(self.r_lo, self.r_hi, self.nr)

    @property
    def z_grid(self) -> FloatArray:
        return np.linspace(self.z_lo, self.z_hi, self.nz)


def gradient_magnitude(psi: FloatArray, dr: float, dz: float) -> FloatArray:
    """Compute |grad psi| on the interior using central differences."""
    dpsi_dr = np.zeros_like(psi)
    dpsi_dz = np.zeros_like(psi)
    if psi.shape[1] > 2:
        dpsi_dr[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / (2.0 * dr)
    if psi.shape[0] > 2:
        dpsi_dz[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2.0 * dz)
    return np.sqrt(dpsi_dr**2 + dpsi_dz**2)


def flag_refinement_cells(
    psi: FloatArray,
    dr: float,
    dz: float,
    threshold: float | None = None,
) -> FloatArray:
    """Return boolean mask of cells where |grad psi| exceeds ``threshold``.

    If ``threshold`` is None, uses the 90th percentile of |grad psi|.
    """
    grad = gradient_magnitude(psi, dr, dz)
    if threshold is None:
        threshold = float(np.percentile(grad, 90))
    return grad > threshold


def _find_patch_bounds(
    flagged: FloatArray,
    R: FloatArray,
    Z: FloatArray,
    pad_cells: int = 2,
) -> list[tuple[float, float, float, float]]:
    """Find rectangular bounding boxes around flagged regions.

    Uses connected-component labelling via flood fill on the flag mask.
    Returns list of (r_lo, r_hi, z_lo, z_hi) tuples.
    """
    visited = np.zeros_like(flagged, dtype=bool)
    patches: list[tuple[float, float, float, float]] = []
    nz, nr = flagged.shape

    for j in range(nz):
        for i in range(nr):
            if flagged[j, i] and not visited[j, i]:
                stack = [(j, i)]
                cells = []
                while stack:
                    cj, ci = stack.pop()
                    if visited[cj, ci]:
                        continue
                    visited[cj, ci] = True
                    cells.append((cj, ci))
                    for dj, di in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nj, ni = cj + dj, ci + di
                        if 0 <= nj < nz and 0 <= ni < nr and flagged[nj, ni] and not visited[nj, ni]:
                            stack.append((nj, ni))

                if not cells:
                    continue
                js = [c[0] for c in cells]
                is_ = [c[1] for c in cells]
                j_lo = max(min(js) - pad_cells, 0)
                j_hi = min(max(js) + pad_cells, nz - 1)
                i_lo = max(min(is_) - pad_cells, 0)
                i_hi = min(max(is_) + pad_cells, nr - 1)

                if (i_hi - i_lo) < MIN_PATCH_CELLS or (j_hi - j_lo) < MIN_PATCH_CELLS:
                    continue

                patches.append((
                    float(R[i_lo]),
                    float(R[i_hi]),
                    float(Z[j_lo]),
                    float(Z[j_hi]),
                ))

    return patches


def prolongate(coarse: FloatArray, fine_shape: tuple[int, int]) -> FloatArray:
    """Bilinear interpolation from coarse grid to fine grid."""
    nz_c, nr_c = coarse.shape
    nz_f, nr_f = fine_shape
    r_c = np.linspace(0.0, 1.0, nr_c)
    z_c = np.linspace(0.0, 1.0, nz_c)
    r_f = np.linspace(0.0, 1.0, nr_f)
    z_f = np.linspace(0.0, 1.0, nz_f)

    tmp = np.empty((nz_c, nr_f), dtype=np.float64)
    for j in range(nz_c):
        tmp[j, :] = np.interp(r_f, r_c, coarse[j, :])
    fine = np.empty((nz_f, nr_f), dtype=np.float64)
    for i in range(nr_f):
        fine[:, i] = np.interp(z_f, z_c, tmp[:, i])
    return fine


def restrict(fine: FloatArray, coarse_shape: tuple[int, int]) -> FloatArray:
    """Restriction (injection + averaging) from fine grid to coarse grid."""
    return prolongate(fine, coarse_shape)


def _extract_subgrid(
    psi: FloatArray,
    R: FloatArray,
    Z: FloatArray,
    r_lo: float,
    r_hi: float,
    z_lo: float,
    z_hi: float,
) -> tuple[FloatArray, int, int, int, int]:
    """Extract the sub-region of psi corresponding to the patch bounds.

    Returns (sub_psi, i_lo, i_hi, j_lo, j_hi) indices into the base grid.
    """
    i_lo = int(np.searchsorted(R, r_lo, side="left"))
    i_hi = int(np.searchsorted(R, r_hi, side="right")) - 1
    j_lo = int(np.searchsorted(Z, z_lo, side="left"))
    j_hi = int(np.searchsorted(Z, z_hi, side="right")) - 1
    i_hi = max(i_hi, i_lo + 1)
    j_hi = max(j_hi, j_lo + 1)
    i_hi = min(i_hi, len(R) - 1)
    j_hi = min(j_hi, len(Z) - 1)
    return psi[j_lo : j_hi + 1, i_lo : i_hi + 1], i_lo, i_hi, j_lo, j_hi


def _jacobi_smooth(
    psi: FloatArray,
    source: FloatArray,
    dr: float,
    dz: float,
    iterations: int = 5,
    omega: float = 0.8,
) -> FloatArray:
    """Weighted Jacobi smoothing for the GS elliptic operator on a sub-patch."""
    u = psi.copy()
    dr2 = dr * dr
    dz2 = dz * dz
    coeff = 2.0 / dr2 + 2.0 / dz2
    if coeff < 1e-30:
        return u
    inv_coeff = 1.0 / coeff
    for _ in range(iterations):
        u_new = u.copy()
        u_new[1:-1, 1:-1] = inv_coeff * (
            (u[1:-1, 2:] + u[1:-1, :-2]) / dr2
            + (u[2:, 1:-1] + u[:-2, 1:-1]) / dz2
            - source[1:-1, 1:-1]
        )
        u[1:-1, 1:-1] = (1.0 - omega) * u[1:-1, 1:-1] + omega * u_new[1:-1, 1:-1]
    return u


def solve_amr(
    psi_base: FloatArray,
    R: FloatArray,
    Z: FloatArray,
    source: FloatArray,
    *,
    grad_threshold: float | None = None,
    smooth_iters: int = 10,
    refine_smooth_iters: int = 20,
) -> tuple[FloatArray, list[AMRPatch]]:
    """Two-level AMR solve: base grid + refined patches.

    Parameters
    ----------
    psi_base : FloatArray
        Initial psi on the base grid, shape ``(NZ, NR)``.
    R, Z : FloatArray
        1D coordinate arrays for the base grid.
    source : FloatArray
        RHS source term (mu_0 R J_phi), shape ``(NZ, NR)``.
    grad_threshold : float or None
        |grad psi| threshold for flagging cells. None = 90th percentile.
    smooth_iters : int
        Jacobi iterations on the base grid.
    refine_smooth_iters : int
        Jacobi iterations on each refined patch.

    Returns
    -------
    psi_out : FloatArray
        Updated psi on the base grid after AMR correction.
    patches : list[AMRPatch]
        Refined patches with their local solutions.
    """
    nr = len(R)
    nz = len(Z)
    dr = float(R[1] - R[0]) if nr > 1 else 1.0
    dz = float(Z[1] - Z[0]) if nz > 1 else 1.0

    psi_out = _jacobi_smooth(psi_base, source, dr, dz, iterations=smooth_iters)

    flagged = flag_refinement_cells(psi_out, dr, dz, threshold=grad_threshold)
    patch_bounds = _find_patch_bounds(flagged, R, Z)

    patches: list[AMRPatch] = []
    for r_lo, r_hi, z_lo, z_hi in patch_bounds:
        sub_psi, i_lo, i_hi, j_lo, j_hi = _extract_subgrid(
            psi_out, R, Z, r_lo, r_hi, z_lo, z_hi,
        )
        sub_src = source[j_lo : j_hi + 1, i_lo : i_hi + 1]

        fine_nr = (i_hi - i_lo + 1) * REFINE_FACTOR
        fine_nz = (j_hi - j_lo + 1) * REFINE_FACTOR
        fine_psi = prolongate(sub_psi, (fine_nz, fine_nr))
        fine_src = prolongate(sub_src, (fine_nz, fine_nr))

        fine_dr = dr / REFINE_FACTOR
        fine_dz = dz / REFINE_FACTOR

        fine_psi = _jacobi_smooth(
            fine_psi, fine_src, fine_dr, fine_dz,
            iterations=refine_smooth_iters,
        )

        correction = restrict(fine_psi, sub_psi.shape) - sub_psi
        psi_out[j_lo : j_hi + 1, i_lo : i_hi + 1] += correction

        patches.append(AMRPatch(
            r_lo=r_lo, r_hi=r_hi, z_lo=z_lo, z_hi=z_hi,
            level=1, nr=fine_nr, nz=fine_nz, psi=fine_psi,
        ))

    return psi_out, patches
