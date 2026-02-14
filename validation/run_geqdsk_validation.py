# CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
# ----------------------------------------------------------------------
# SCPN Fusion Core -- GEQDSK Real-Data Validation
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ----------------------------------------------------------------------
"""Validate the Picard+SOR Grad-Shafranov solver against SPARC GEQDSK
equilibrium files from the CFS SPARCPublic repository.

For each GEQDSK file the script:

1. Loads the equilibrium using ``scpn_fusion.core.eqdsk.read_geqdsk``.
2. Extracts the GS source from the 1-D profiles::

       source(R,Z) = -mu0 * R^2 * p'(psi_N) - FF'(psi_N)

   where *p'* and *FF'* are interpolated onto the 2-D grid via the
   normalised flux psi_N(R,Z).
3. Runs Picard iterations (re-evaluating the nonlinear source from the
   current psi estimate) with Red-Black SOR inner sweeps.
4. Compares the converged psi against the GEQDSK reference via:
   - Normalised psi RMSE
   - Magnetic-axis position error (with quadratic sub-grid interpolation)
   - Boundary-contour Hausdorff distance

Usage::

    python validation/run_geqdsk_validation.py [--max-picard 20] [--max-sor 6000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Path setup -- make scpn_fusion importable without installation
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
_SRC = str(ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from scpn_fusion.core.eqdsk import read_geqdsk, GEqdsk

# Import the vectorised SOR sweep from the forward-validation module.
# validation/ is not a Python package, so we load it via importlib and
# register it in sys.modules (required for @dataclass introspection).
import importlib.util as _ilu

_FWD_PATH = ROOT / "validation" / "run_forward_validation.py"
_spec = _ilu.spec_from_file_location("run_forward_validation", _FWD_PATH)
_fwd = _ilu.module_from_spec(_spec)
sys.modules["run_forward_validation"] = _fwd
_spec.loader.exec_module(_fwd)

_sor_sweep_vectorised = _fwd._sor_sweep_vectorised
_apply_bc = _fwd._apply_bc
_build_initial_guess = _fwd._build_initial_guess

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
MU_0 = 4.0 * np.pi * 1e-7  # vacuum permeability [H/m]


# ---------------------------------------------------------------------------
# Sub-grid axis finder (quadratic interpolation around argmin pixel)
# ---------------------------------------------------------------------------

def _find_axis_subgrid(
    psi: NDArray,
    r_grid: NDArray,
    z_grid: NDArray,
    psi_axis_hint: Optional[float] = None,
) -> Tuple[float, float]:
    """Find magnetic axis with quadratic sub-grid interpolation.

    Parameters
    ----------
    psi : (nz, nr) array
        Poloidal flux on the (Z, R) grid.
    r_grid : (nr,) array
        R coordinate values.
    z_grid : (nz,) array
        Z coordinate values.
    psi_axis_hint : float or None
        If given, search for the grid point closest to this psi value
        (useful for GEQDSK data where simag may not be the global
        min or max).  If None, find the point closest to the global
        minimum (standard Solov'ev convention).

    Returns
    -------
    (r_axis, z_axis) : floats
        Sub-pixel magnetic axis position.
    """
    if psi_axis_hint is not None:
        # Find the grid point whose psi is closest to the hint value,
        # restricting to interior points (exclude 2-cell boundary band)
        margin = 2
        nz, nr = psi.shape
        interior = psi[margin:nz - margin, margin:nr - margin]
        idx_flat = int(np.argmin(np.abs(interior - psi_axis_hint)))
        iz_local, ir_local = np.unravel_index(idx_flat, interior.shape)
        iz0 = iz_local + margin
        ir0 = ir_local + margin
    else:
        # Default: find global minimum (works for simple/synthetic cases)
        idx = np.unravel_index(np.argmin(psi), psi.shape)
        iz0, ir0 = idx

    dr = float(r_grid[1] - r_grid[0])
    dz = float(z_grid[1] - z_grid[0])

    # Quadratic sub-pixel refinement using 3-point parabola fit
    if 1 <= ir0 < psi.shape[1] - 1 and 1 <= iz0 < psi.shape[0] - 1:
        # Along R (axis-1)
        fL = psi[iz0, ir0 - 1]
        fC = psi[iz0, ir0]
        fR = psi[iz0, ir0 + 1]
        denom_r = fR - 2.0 * fC + fL
        if abs(denom_r) > 1e-30:
            shift_r = -0.5 * (fR - fL) / denom_r
        else:
            shift_r = 0.0
        shift_r = np.clip(shift_r, -0.5, 0.5)
        r_axis = r_grid[ir0] + shift_r * dr

        # Along Z (axis-0)
        fD = psi[iz0 - 1, ir0]
        fU = psi[iz0 + 1, ir0]
        denom_z = fU - 2.0 * fC + fD
        if abs(denom_z) > 1e-30:
            shift_z = -0.5 * (fU - fD) / denom_z
        else:
            shift_z = 0.0
        shift_z = np.clip(shift_z, -0.5, 0.5)
        z_axis = z_grid[iz0] + shift_z * dz
    else:
        r_axis = float(r_grid[ir0])
        z_axis = float(z_grid[iz0])

    return float(r_axis), float(z_axis)


# ---------------------------------------------------------------------------
# GS source from GEQDSK profiles
# ---------------------------------------------------------------------------

def _compute_gs_source_from_profiles(
    psi: NDArray,
    RR: NDArray,
    pprime: NDArray,
    ffprime: NDArray,
    psi_axis: float,
    psi_boundary: float,
) -> NDArray:
    """Compute the GS source from 1-D p' and FF' profiles.

    source(R,Z) = -mu0 * R^2 * p'(psi_N) - FF'(psi_N)

    The profiles are given on a uniform psi_N grid [0, 1] of length
    ``len(pprime)``.  The normalised flux at each grid point is::

        psi_N(R,Z) = (psi(R,Z) - psi_axis) / (psi_boundary - psi_axis)

    Points outside [0, 1] are clamped.

    Parameters
    ----------
    psi : (nz, nr) array
        Current psi estimate.
    RR : (nz, nr) array
        2-D R coordinate mesh.
    pprime, ffprime : 1-D arrays (length nw)
        dp/dpsi and F*dF/dpsi profiles.
    psi_axis, psi_boundary : float
        Reference psi at magnetic axis and separatrix.

    Returns
    -------
    source : (nz, nr) array
    """
    denom = psi_boundary - psi_axis
    if abs(denom) < 1e-30:
        denom = 1e-30
    psi_n = np.clip((psi - psi_axis) / denom, 0.0, 1.0)

    n_prof = len(pprime)
    psi_n_prof = np.linspace(0.0, 1.0, n_prof)

    pp_2d = np.interp(psi_n.ravel(), psi_n_prof, pprime).reshape(psi.shape)
    ff_2d = np.interp(psi_n.ravel(), psi_n_prof, ffprime).reshape(psi.shape)

    return -MU_0 * RR ** 2 * pp_2d - ff_2d


# ---------------------------------------------------------------------------
# Boundary contour extraction via sign-change interpolation
# ---------------------------------------------------------------------------

def _extract_psi_contour(
    psi: NDArray,
    r_grid: NDArray,
    z_grid: NDArray,
    psi_level: float,
) -> Tuple[NDArray, NDArray]:
    """Extract an approximate contour at *psi_level* from psi(nz, nr).

    Uses linear interpolation of sign changes along both grid directions.
    """
    shifted = psi - psi_level
    nz, nr = shifted.shape
    r_pts: list = []
    z_pts: list = []

    # Horizontal scan (along R for each Z row)
    for iz in range(nz):
        for ir in range(nr - 1):
            if shifted[iz, ir] * shifted[iz, ir + 1] < 0:
                denom = abs(shifted[iz, ir]) + abs(shifted[iz, ir + 1])
                if denom > 0:
                    frac = abs(shifted[iz, ir]) / denom
                    r_pts.append(r_grid[ir] + frac * (r_grid[ir + 1] - r_grid[ir]))
                    z_pts.append(z_grid[iz])

    # Vertical scan (along Z for each R column)
    for ir in range(nr):
        for iz in range(nz - 1):
            if shifted[iz, ir] * shifted[iz + 1, ir] < 0:
                denom = abs(shifted[iz, ir]) + abs(shifted[iz + 1, ir])
                if denom > 0:
                    frac = abs(shifted[iz, ir]) / denom
                    r_pts.append(r_grid[ir])
                    z_pts.append(z_grid[iz] + frac * (z_grid[iz + 1] - z_grid[iz]))

    return np.array(r_pts), np.array(z_pts)


# ---------------------------------------------------------------------------
# Hausdorff distance (no scipy dependency required)
# ---------------------------------------------------------------------------

def _hausdorff_distance(
    r1: NDArray, z1: NDArray,
    r2: NDArray, z2: NDArray,
) -> float:
    """Symmetric Hausdorff distance between two 2-D point sets.

    Falls back to a pure-NumPy implementation if scipy is not available.
    """
    if len(r1) == 0 or len(r2) == 0:
        return float("nan")

    try:
        from scipy.spatial.distance import directed_hausdorff
        pts1 = np.column_stack((r1, z1))
        pts2 = np.column_stack((r2, z2))
        return float(max(directed_hausdorff(pts1, pts2)[0],
                         directed_hausdorff(pts2, pts1)[0]))
    except ImportError:
        pass

    # Pure-NumPy fallback
    pts1 = np.column_stack((r1, z1))  # (N1, 2)
    pts2 = np.column_stack((r2, z2))  # (N2, 2)
    # h(A, B) = max_a min_b ||a - b||
    diff1 = pts1[:, np.newaxis, :] - pts2[np.newaxis, :, :]  # (N1, N2, 2)
    dist1 = np.sqrt(np.sum(diff1 ** 2, axis=-1))  # (N1, N2)
    h_forward = float(np.max(np.min(dist1, axis=1)))
    h_backward = float(np.max(np.min(dist1, axis=0)))
    return max(h_forward, h_backward)


# ---------------------------------------------------------------------------
# Picard + SOR solver for GEQDSK profiles
# ---------------------------------------------------------------------------

def _compute_discrete_gs_source(
    psi_ref: NDArray,
    RR: NDArray,
    dr: float,
    dz: float,
) -> NDArray:
    """Compute the discrete GS source by applying the stencil to psi_ref.

    This is the "manufactured solution" approach: the source is defined
    as Delta*_h(psi_ref), so the exact discrete solution of the linear
    system is psi_ref itself.  This tests the SOR solver in isolation,
    eliminating discretisation-vs-profile inconsistencies.

    Parameters
    ----------
    psi_ref : (nz, nr) array
    RR : (nz, nr) array  -- R coordinate mesh
    dr, dz : grid spacing

    Returns
    -------
    source : (nz, nr) array
    """
    nz, nr = psi_ref.shape
    dr_sq = dr * dr
    dz_sq = dz * dz

    R_inner = RR[1:-1, 1:-1]
    c_r_plus = 1.0 / dr_sq - 1.0 / (2.0 * R_inner * dr)
    c_r_minus = 1.0 / dr_sq + 1.0 / (2.0 * R_inner * dr)
    c_z = 1.0 / dz_sq
    center = 2.0 / dr_sq + 2.0 / dz_sq

    source = np.zeros_like(psi_ref)
    source[1:-1, 1:-1] = (
        c_r_plus * psi_ref[1:-1, 2:]
        + c_r_minus * psi_ref[1:-1, :-2]
        + c_z * (psi_ref[2:, 1:-1] + psi_ref[:-2, 1:-1])
        - center * psi_ref[1:-1, 1:-1]
    )
    return source


def picard_sor_geqdsk(
    eq: GEqdsk,
    *,
    max_picard: int = 20,
    max_sor: int = 6000,
    omega: float = 1.65,
    tol_sor: float = 1e-8,
    tol_picard: float = 1e-6,
    check_every: int = 200,
    source_mode: str = "manufactured",
) -> dict:
    """Picard + SOR Grad-Shafranov solver for GEQDSK validation.

    Two source modes are supported:

    * ``"manufactured"`` (default): The source is computed by applying the
      discrete GS stencil to the reference psirz.  This tests the SOR
      solver in isolation: the exact discrete solution IS the reference,
      so any remaining error is pure solver residual.

    * ``"profiles"``: The source is computed from the GEQDSK p' and FF'
      profiles.  This tests the full Picard+SOR pipeline including the
      profile-to-source mapping.  The NRMSE will be higher because the
      stored profiles are only approximately consistent with the stored
      psi on the discrete grid.

    Parameters
    ----------
    eq : GEqdsk
        Parsed GEQDSK equilibrium.
    max_picard : int
        Maximum Picard (source-update) iterations.
    max_sor : int
        Maximum SOR sweeps per Picard iteration.
    omega : float
        SOR relaxation factor.
    tol_sor : float
        SOR convergence tolerance (max absolute change per check).
    tol_picard : float
        Picard convergence tolerance (normalised psi change).
    check_every : int
        SOR convergence check interval.
    source_mode : str
        ``"manufactured"`` or ``"profiles"``.

    Returns
    -------
    dict with keys: psi, r_grid, z_grid, r_axis, z_axis, picard_iters,
                    total_sor_iters, converged
    """
    r_grid = eq.r
    z_grid = eq.z
    nz, nr = eq.nh, eq.nw
    dr = float(r_grid[1] - r_grid[0])
    dz = float(z_grid[1] - z_grid[0])

    # 2-D R mesh in (nz, nr) layout
    RR = np.ones((nz, 1)) * r_grid[np.newaxis, :]

    # Reference psi in (nz, nr) layout
    psi_ref = eq.psirz.copy()

    # Dirichlet BCs from the edges of the reference psirz
    bc = np.zeros((nz, nr))
    bc[0, :] = psi_ref[0, :]
    bc[-1, :] = psi_ref[-1, :]
    bc[:, 0] = psi_ref[:, 0]
    bc[:, -1] = psi_ref[:, -1]

    # Initial guess: smooth blend from boundary values
    psi = _build_initial_guess(bc, nz, nr)
    _apply_bc(psi, bc)

    psi_axis = eq.simag
    psi_bdy = eq.sibry

    total_sor_iters = 0
    converged = False

    # In manufactured mode, compute the source once from the reference
    if source_mode == "manufactured":
        source = _compute_discrete_gs_source(psi_ref, RR, dr, dz)
        source[0, :] = 0.0
        source[-1, :] = 0.0
        source[:, 0] = 0.0
        source[:, -1] = 0.0

    for picard in range(1, max_picard + 1):
        if source_mode == "profiles":
            # Evaluate source from current psi estimate
            source = _compute_gs_source_from_profiles(
                psi, RR, eq.pprime, eq.ffprime, psi_axis, psi_bdy,
            )
            source[0, :] = 0.0
            source[-1, :] = 0.0
            source[:, 0] = 0.0
            source[:, -1] = 0.0

        # SOR inner loop
        psi_snapshot = psi.copy()
        sor_converged = False
        for sor_it in range(1, max_sor + 1):
            _sor_sweep_vectorised(psi, source, RR, dr, dz, omega)
            _apply_bc(psi, bc)

            if sor_it % check_every == 0:
                diff = float(np.max(np.abs(psi - psi_snapshot)))
                if diff < tol_sor:
                    sor_converged = True
                    total_sor_iters += sor_it
                    break
                psi_snapshot = psi.copy()
        else:
            total_sor_iters += max_sor

        if source_mode == "profiles":
            # Update axis psi from current solution for next source eval
            r_ax, z_ax = _find_axis_subgrid(
                psi, r_grid, z_grid, psi_axis_hint=psi_axis,
            )
            ir_ax = np.argmin(np.abs(r_grid - r_ax))
            iz_ax = np.argmin(np.abs(z_grid - z_ax))
            psi_axis = float(psi[iz_ax, ir_ax])

        # Picard convergence check: normalised change in psi
        if picard > 1:
            psi_norm_ref = np.linalg.norm(psi)
            if psi_norm_ref > 0:
                picard_change = np.linalg.norm(psi - psi_snapshot) / psi_norm_ref
            else:
                picard_change = 0.0
            if picard_change < tol_picard and sor_converged:
                converged = True
                break

        # In manufactured mode, one Picard iteration suffices (linear problem)
        if source_mode == "manufactured" and sor_converged:
            converged = True
            break

    # Final axis position
    r_axis, z_axis = _find_axis_subgrid(
        psi, r_grid, z_grid, psi_axis_hint=eq.simag,
    )

    return {
        "psi": psi,
        "r_grid": r_grid,
        "z_grid": z_grid,
        "r_axis": r_axis,
        "z_axis": z_axis,
        "picard_iters": picard,
        "total_sor_iters": total_sor_iters,
        "converged": converged,
    }


# ---------------------------------------------------------------------------
# Single-file validation
# ---------------------------------------------------------------------------

@dataclass
class GeqdskResult:
    """Container for per-file validation results."""
    filename: str
    grid_nz: int
    grid_nr: int
    axis_dr_mm: float
    axis_dz_mm: float
    psi_rmse: float
    hausdorff_mm: float
    picard_iters: int
    total_sor_iters: int
    converged: bool
    solve_time_s: float
    status: str


def validate_one_geqdsk(
    geqdsk_path: Path,
    *,
    max_picard: int = 20,
    max_sor: int = 6000,
    omega: float = 1.65,
    source_mode: str = "manufactured",
) -> GeqdskResult:
    """Run the full validation pipeline on a single GEQDSK file.

    Parameters
    ----------
    geqdsk_path : Path
        Path to the GEQDSK file.
    max_picard, max_sor, omega : solver parameters.
    source_mode : str
        ``"manufactured"`` or ``"profiles"``.

    Returns
    -------
    GeqdskResult
    """
    eq = read_geqdsk(geqdsk_path)

    t0 = time.time()
    result = picard_sor_geqdsk(
        eq,
        max_picard=max_picard,
        max_sor=max_sor,
        omega=omega,
        source_mode=source_mode,
    )
    elapsed = time.time() - t0

    psi_ours = result["psi"]
    psi_ref = eq.psirz

    # 1. Normalised psi RMSE
    diff_norm = np.linalg.norm(psi_ours - psi_ref)
    ref_norm = np.linalg.norm(psi_ref)
    psi_rmse = float(diff_norm / ref_norm) if ref_norm > 0 else 0.0

    # 2. Magnetic axis position error (sub-grid)
    r_axis_ours = result["r_axis"]
    z_axis_ours = result["z_axis"]
    r_axis_ref, z_axis_ref = _find_axis_subgrid(psi_ref, eq.r, eq.z, psi_axis_hint=eq.simag)

    axis_dr_mm = abs(r_axis_ours - r_axis_ref) * 1000.0
    axis_dz_mm = abs(z_axis_ours - z_axis_ref) * 1000.0

    # 3. Boundary contour Hausdorff distance
    #    Extract the psi = sibry contour from our solution and compare
    #    against the GEQDSK boundary points.
    #    Filter: keep only contour points near the reference boundary
    #    centroid to exclude external/domain-edge contour segments.
    hausdorff_mm = float("nan")
    if len(eq.rbdry) >= 3:
        our_r, our_z = _extract_psi_contour(psi_ours, eq.r, eq.z, eq.sibry)
        if len(our_r) >= 3:
            # Filter: keep contour points within 1.5x the reference
            # boundary extent from the boundary centroid.
            r_ctr = np.mean(eq.rbdry)
            z_ctr = np.mean(eq.zbdry)
            r_extent = np.max(eq.rbdry) - np.min(eq.rbdry)
            z_extent = np.max(eq.zbdry) - np.min(eq.zbdry)
            margin = 1.5
            keep = (
                (np.abs(our_r - r_ctr) < margin * r_extent / 2)
                & (np.abs(our_z - z_ctr) < margin * z_extent / 2)
            )
            if np.sum(keep) >= 3:
                hausdorff_mm = _hausdorff_distance(
                    our_r[keep], our_z[keep], eq.rbdry, eq.zbdry,
                ) * 1000.0  # convert m -> mm

    # Status thresholds.
    # In "manufactured" mode (default), the discrete source is derived from
    # the reference psi so the SOR solver should reproduce it near-exactly.
    # In "profiles" mode, inherent discretization differences between the
    # stored profiles and the discrete GS stencil widen the expected gap.
    if psi_rmse < 0.05:
        status = "OK"
    elif psi_rmse < 0.10:
        status = "WARN"
    else:
        status = "FAIL"

    return GeqdskResult(
        filename=geqdsk_path.name,
        grid_nz=eq.nh,
        grid_nr=eq.nw,
        axis_dr_mm=axis_dr_mm,
        axis_dz_mm=axis_dz_mm,
        psi_rmse=psi_rmse,
        hausdorff_mm=hausdorff_mm,
        picard_iters=result["picard_iters"],
        total_sor_iters=result["total_sor_iters"],
        converged=result["converged"],
        solve_time_s=elapsed,
        status=status,
    )


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _sanitise_for_json(obj):
    """Recursively convert numpy types to native Python for JSON."""
    if isinstance(obj, dict):
        return {k: _sanitise_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitise_for_json(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


# ---------------------------------------------------------------------------
# Main validation loop
# ---------------------------------------------------------------------------

def run_geqdsk_validation(
    sparc_dir: Path,
    results_dir: Path,
    *,
    max_picard: int = 20,
    max_sor: int = 6000,
    omega: float = 1.65,
    filenames: Optional[List[str]] = None,
    source_mode: str = "manufactured",
) -> List[GeqdskResult]:
    """Run validation on all (or selected) GEQDSK files in *sparc_dir*.

    Parameters
    ----------
    sparc_dir : Path
        Directory containing .geqdsk files.
    results_dir : Path
        Directory to write per-file JSON results and summary.
    max_picard, max_sor, omega : solver tuning parameters.
    filenames : optional list of str
        If given, only validate files whose name appears in this list.
    source_mode : str
        ``"manufactured"`` or ``"profiles"``.

    Returns
    -------
    List of GeqdskResult.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    geqdsk_files = sorted(sparc_dir.glob("*.geqdsk"))
    if filenames:
        geqdsk_files = [f for f in geqdsk_files if f.name in filenames]

    if not geqdsk_files:
        print(f"No GEQDSK files found in {sparc_dir}")
        return []

    n_files = len(geqdsk_files)
    print("GEQDSK Real-Data Validation")
    print("=" * 90)
    print(
        f"{'File':<24s} {'Grid':>10s} {'Axis dR(mm)':>12s} "
        f"{'Axis dZ(mm)':>12s} {'Psi RMSE':>10s} "
        f"{'Hausdorff':>10s} {'Status':>8s}"
    )
    print("-" * 90)

    all_results: List[GeqdskResult] = []

    for idx, gpath in enumerate(geqdsk_files):
        res = validate_one_geqdsk(
            gpath,
            max_picard=max_picard,
            max_sor=max_sor,
            omega=omega,
            source_mode=source_mode,
        )
        all_results.append(res)

        hausdorff_str = (
            f"{res.hausdorff_mm:.1f}mm" if not np.isnan(res.hausdorff_mm) else "N/A"
        )
        print(
            f"{res.filename:<24s} "
            f"{res.grid_nz}x{res.grid_nr:>3d}     "
            f"{res.axis_dr_mm:>10.1f}  "
            f"{res.axis_dz_mm:>10.1f}  "
            f"{res.psi_rmse:>10.4f} "
            f"{hausdorff_str:>10s} "
            f"{res.status:>8s}"
        )

        # Save per-file JSON
        result_dict = {
            "filename": res.filename,
            "grid": f"{res.grid_nz}x{res.grid_nr}",
            "axis_dr_mm": res.axis_dr_mm,
            "axis_dz_mm": res.axis_dz_mm,
            "psi_rmse": res.psi_rmse,
            "hausdorff_mm": res.hausdorff_mm,
            "picard_iterations": res.picard_iters,
            "total_sor_iterations": res.total_sor_iters,
            "converged": res.converged,
            "solve_time_s": res.solve_time_s,
            "status": res.status,
        }
        json_path = results_dir / f"{gpath.stem}.json"
        with open(json_path, "w") as f:
            json.dump(_sanitise_for_json(result_dict), f, indent=2)

    print("-" * 90)

    # Summary statistics
    rmses = [r.psi_rmse for r in all_results]
    dr_errs = [r.axis_dr_mm for r in all_results]
    dz_errs = [r.axis_dz_mm for r in all_results]
    n_ok = sum(1 for r in all_results if r.status == "OK")
    n_warn = sum(1 for r in all_results if r.status == "WARN")
    n_fail = sum(1 for r in all_results if r.status == "FAIL")

    print(f"\nSummary: {n_ok} OK / {n_warn} WARN / {n_fail} FAIL  "
          f"(out of {n_files} files)")
    print(f"  Mean psi RMSE:      {np.mean(rmses):.4f}")
    print(f"  Max  psi RMSE:      {np.max(rmses):.4f}")
    print(f"  Mean axis dR:       {np.mean(dr_errs):.1f} mm")
    print(f"  Mean axis dZ:       {np.mean(dz_errs):.1f} mm")
    total_time = sum(r.solve_time_s for r in all_results)
    print(f"  Total solve time:   {total_time:.1f} s")

    # Write aggregate summary JSON
    summary = {
        "n_files": n_files,
        "n_ok": n_ok,
        "n_warn": n_warn,
        "n_fail": n_fail,
        "mean_psi_rmse": float(np.mean(rmses)),
        "max_psi_rmse": float(np.max(rmses)),
        "mean_axis_dr_mm": float(np.mean(dr_errs)),
        "mean_axis_dz_mm": float(np.mean(dz_errs)),
        "total_solve_time_s": total_time,
        "files": [
            _sanitise_for_json({
                "filename": r.filename,
                "psi_rmse": r.psi_rmse,
                "axis_dr_mm": r.axis_dr_mm,
                "axis_dz_mm": r.axis_dz_mm,
                "hausdorff_mm": r.hausdorff_mm,
                "status": r.status,
            })
            for r in all_results
        ],
    }
    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(_sanitise_for_json(summary), f, indent=2)
    print(f"\nResults saved to {results_dir}")

    return all_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Picard+SOR GS solver against SPARC GEQDSK "
                    "reference equilibria.",
    )
    parser.add_argument(
        "--sparc-dir", type=str, default=None,
        help="Directory containing SPARC .geqdsk files. "
             "Defaults to validation/reference_data/sparc/.",
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Directory to write results. "
             "Defaults to validation/results/geqdsk/.",
    )
    parser.add_argument(
        "--max-picard", type=int, default=20,
        help="Maximum Picard iterations (default: 20).",
    )
    parser.add_argument(
        "--max-sor", type=int, default=6000,
        help="Maximum SOR sweeps per Picard iteration (default: 6000).",
    )
    parser.add_argument(
        "--omega", type=float, default=1.65,
        help="SOR relaxation factor (default: 1.65).",
    )
    parser.add_argument(
        "--files", nargs="*", default=None,
        help="Specific GEQDSK filenames to validate (default: all).",
    )
    parser.add_argument(
        "--source-mode", type=str, default="manufactured",
        choices=["manufactured", "profiles"],
        help="Source term mode: 'manufactured' (tests SOR solver using "
             "discrete stencil applied to reference psi) or 'profiles' "
             "(tests full Picard+SOR with GEQDSK p'/FF' profiles). "
             "Default: manufactured.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    sparc_dir = (
        Path(args.sparc_dir)
        if args.sparc_dir
        else script_dir / "reference_data" / "sparc"
    )
    results_dir = (
        Path(args.results_dir)
        if args.results_dir
        else script_dir / "results" / "geqdsk"
    )

    results = run_geqdsk_validation(
        sparc_dir=sparc_dir,
        results_dir=results_dir,
        max_picard=args.max_picard,
        max_sor=args.max_sor,
        omega=args.omega,
        filenames=args.files,
        source_mode=args.source_mode,
    )

    if not results:
        print("No results produced.")
        sys.exit(1)

    # Non-zero exit code if any file FAILed
    n_fail = sum(1 for r in results if r.status == "FAIL")
    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
