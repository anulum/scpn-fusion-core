# CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Equilibrium Reconstruction Comparison
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Compare two equilibrium reconstructions quantitatively.

All functions take our psi(R,Z) and a reference psi(R,Z) on the same grid
and return scalar error metrics.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import directed_hausdorff


# ── Normalised psi RMSE ─────────────────────────────────────────────


def normalized_psi_rmse(psi_ours: np.ndarray, psi_ref: np.ndarray) -> float:
    """Normalized RMSE: ||psi_ours - psi_ref||_2 / ||psi_ref||_2.

    Returns value in [0, inf).  Good: < 0.05.  Acceptable: < 0.10.
    If *psi_ref* is identically zero the denominator would vanish;
    in that case we return 0.0 when both arrays are identical, or inf
    otherwise.
    """
    psi_ours = np.asarray(psi_ours, dtype=np.float64)
    psi_ref = np.asarray(psi_ref, dtype=np.float64)

    diff_norm = np.linalg.norm(psi_ours - psi_ref)
    ref_norm = np.linalg.norm(psi_ref)

    if ref_norm == 0.0:
        return 0.0 if diff_norm == 0.0 else float("inf")

    return float(diff_norm / ref_norm)


# ── Magnetic-axis position error ─────────────────────────────────────


def axis_position_error(
    psi: np.ndarray,
    r_grid: np.ndarray,
    z_grid: np.ndarray,
    r_axis_ref: float,
    z_axis_ref: float,
) -> dict:
    """Find the magnetic axis (psi extremum) and compare to a reference.

    The magnetic axis is the location of the *minimum* of psi on the
    (R, Z) grid (convention: psi is smallest at the core).  If psi is
    constant the flat-array argmin (top-left corner) is returned, but
    the resulting error is still well-defined.

    Parameters
    ----------
    psi : (nR, nZ) or (nZ, nR) array — flux on a 2-D grid.
    r_grid, z_grid : 1-D arrays — R and Z coordinates of the grid.
    r_axis_ref, z_axis_ref : reference axis position [m].

    Returns
    -------
    dict with keys ``dr_m``, ``dz_m``, ``total_m`` (all in metres).
    """
    psi = np.asarray(psi, dtype=np.float64)
    r_grid = np.asarray(r_grid, dtype=np.float64).ravel()
    z_grid = np.asarray(z_grid, dtype=np.float64).ravel()

    # psi may be shaped (nR, nZ) or (nZ, nR).  We adopt the convention
    # that axis-0 corresponds to the first coordinate array (r_grid)
    # and axis-1 to the second (z_grid).  If the shape is transposed,
    # swap the grids to stay consistent.
    if psi.shape == (len(z_grid), len(r_grid)):
        # (nZ, nR) layout — transpose so axis-0 = R, axis-1 = Z
        psi = psi.T
    elif psi.shape != (len(r_grid), len(z_grid)):
        # Ambiguous shape: fall back to flat argmin without transposing
        pass

    idx_flat = int(np.argmin(psi))
    if psi.ndim == 2:
        ir, iz = np.unravel_index(idx_flat, psi.shape)
        r_axis = float(r_grid[ir]) if ir < len(r_grid) else float(r_grid[-1])
        z_axis = float(z_grid[iz]) if iz < len(z_grid) else float(z_grid[-1])
    else:
        # 1-D fallback (shouldn't happen, but be safe)
        r_axis = float(r_grid[0])
        z_axis = float(z_grid[0])

    dr = r_axis - float(r_axis_ref)
    dz = z_axis - float(z_axis_ref)
    total = float(np.hypot(dr, dz))

    return {"dr_m": dr, "dz_m": dz, "total_m": total}


# ── Boundary Hausdorff distance ──────────────────────────────────────


def boundary_hausdorff(
    r_bdry_ours: np.ndarray,
    z_bdry_ours: np.ndarray,
    r_bdry_ref: np.ndarray,
    z_bdry_ref: np.ndarray,
) -> float:
    """Hausdorff distance between two LCFS contours [m].

    Each boundary is given as matched (R, Z) coordinate arrays of
    arbitrary length.  The *symmetric* Hausdorff distance is returned:
      H = max( h(A,B), h(B,A) )
    where h(A,B) = max_{a in A} min_{b in B} ||a - b||.
    """
    pts_ours = np.column_stack(
        (np.asarray(r_bdry_ours, dtype=np.float64).ravel(),
         np.asarray(z_bdry_ours, dtype=np.float64).ravel())
    )
    pts_ref = np.column_stack(
        (np.asarray(r_bdry_ref, dtype=np.float64).ravel(),
         np.asarray(z_bdry_ref, dtype=np.float64).ravel())
    )

    h_forward = directed_hausdorff(pts_ours, pts_ref)[0]
    h_backward = directed_hausdorff(pts_ref, pts_ours)[0]

    return float(max(h_forward, h_backward))


# ── q-profile RMSE ───────────────────────────────────────────────────


def q_profile_rmse(
    q_ours: np.ndarray,
    q_ref: np.ndarray,
    psi_n_points: int = 10,
) -> float:
    """RMSE of q(psi_N) at equispaced normalised-flux points.

    Both *q_ours* and *q_ref* are 1-D arrays that are assumed to be
    sampled uniformly in normalised poloidal flux psi_N in [0, 1].
    They are linearly interpolated onto *psi_n_points* equispaced
    locations before computing the RMSE.

    If either profile has fewer than 2 points, only the first value is
    compared (single-point RMSE).
    """
    q_ours = np.asarray(q_ours, dtype=np.float64).ravel()
    q_ref = np.asarray(q_ref, dtype=np.float64).ravel()

    if q_ours.size == 0 or q_ref.size == 0:
        return 0.0

    # Build normalised-flux coordinates for each profile
    psi_n_ours = np.linspace(0.0, 1.0, len(q_ours))
    psi_n_ref = np.linspace(0.0, 1.0, len(q_ref))

    # Common evaluation points
    psi_n_eval = np.linspace(0.0, 1.0, max(psi_n_points, 1))

    q_ours_interp = np.interp(psi_n_eval, psi_n_ours, q_ours)
    q_ref_interp = np.interp(psi_n_eval, psi_n_ref, q_ref)

    diff = q_ours_interp - q_ref_interp
    return float(np.sqrt(np.mean(diff ** 2)))


# ── Stored-energy relative error ─────────────────────────────────────


def stored_energy_error(
    psi_ours: np.ndarray,
    psi_ref: np.ndarray,
    pressure_ours: np.ndarray,
    pressure_ref: np.ndarray,
    r_grid: np.ndarray,
    z_grid: np.ndarray,
) -> float:
    """Relative error in stored energy W = (3/2) int p dV.

    The volume element in cylindrical (R, phi, Z) geometry — assuming
    axisymmetry — is  dV = 2*pi*R * dR * dZ.  The integral is
    approximated as a 2-D Riemann sum.

    Returns |W_ours - W_ref| / |W_ref|.  If W_ref == 0 the function
    returns 0.0 when W_ours is also zero, inf otherwise.
    """
    pressure_ours = np.asarray(pressure_ours, dtype=np.float64)
    pressure_ref = np.asarray(pressure_ref, dtype=np.float64)
    r_grid = np.asarray(r_grid, dtype=np.float64).ravel()
    z_grid = np.asarray(z_grid, dtype=np.float64).ravel()

    dr = np.abs(r_grid[1] - r_grid[0]) if len(r_grid) > 1 else 1.0
    dz = np.abs(z_grid[1] - z_grid[0]) if len(z_grid) > 1 else 1.0

    # Build 2-D R mesh for the volume element  dV = 2*pi*R*dR*dZ
    R2d = r_grid[:, np.newaxis] * np.ones((1, len(z_grid)))

    def _stored_energy(p: np.ndarray) -> float:
        """W = (3/2) * 2*pi * sum_ij p_ij * R_ij * dR * dZ."""
        # Ensure p has shape (nR, nZ)
        if p.shape == (len(z_grid), len(r_grid)):
            p = p.T
        return float(1.5 * 2.0 * np.pi * np.sum(p * R2d) * dr * dz)

    w_ours = _stored_energy(pressure_ours)
    w_ref = _stored_energy(pressure_ref)

    if w_ref == 0.0:
        return 0.0 if w_ours == 0.0 else float("inf")

    return float(abs(w_ours - w_ref) / abs(w_ref))


# ── Full comparison suite ─────────────────────────────────────────────


def full_comparison(ours: dict, ref: dict) -> dict:
    """Run all comparison metrics.

    *ours* and *ref* are dicts that must contain the keys:

        psi_rz       — 2-D array of psi(R, Z)
        r_grid       — 1-D R coordinate array [m]
        z_grid       — 1-D Z coordinate array [m]
        q_profile    — 1-D safety-factor profile
        r_axis       — scalar magnetic-axis R [m]
        z_axis       — scalar magnetic-axis Z [m]
        r_boundary   — 1-D LCFS R coordinates [m]
        z_boundary   — 1-D LCFS Z coordinates [m]
        pressure     — 2-D pressure array (same shape as psi_rz) [Pa]

    Returns a dict with keys:
        normalized_psi_rmse, axis_error, boundary_hausdorff_m,
        q_profile_rmse, stored_energy_relative_error
    """
    psi_rmse = normalized_psi_rmse(ours["psi_rz"], ref["psi_rz"])

    axis_err = axis_position_error(
        ours["psi_rz"],
        ours["r_grid"],
        ours["z_grid"],
        ref["r_axis"],
        ref["z_axis"],
    )

    bdry_h = boundary_hausdorff(
        ours["r_boundary"],
        ours["z_boundary"],
        ref["r_boundary"],
        ref["z_boundary"],
    )

    q_rmse = q_profile_rmse(ours["q_profile"], ref["q_profile"])

    w_err = stored_energy_error(
        ours["psi_rz"],
        ref["psi_rz"],
        ours["pressure"],
        ref["pressure"],
        ours["r_grid"],
        ours["z_grid"],
    )

    return {
        "normalized_psi_rmse": psi_rmse,
        "axis_error": axis_err,
        "boundary_hausdorff_m": bdry_h,
        "q_profile_rmse": q_rmse,
        "stored_energy_relative_error": w_err,
    }
