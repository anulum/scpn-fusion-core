# CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
# ----------------------------------------------------------------------
# SCPN Fusion Core -- Inverse Reconstruction Validation
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ----------------------------------------------------------------------
"""Inverse reconstruction validation on synthetic shots.

For each synthetic shot:
1. Load probe measurements (boundary psi values) as "observations"
2. Start with perturbed source profiles (+/-20% of true p', FF')
3. Run iterative reconstruction:
   a. Forward solve with current profiles -> psi(R,Z)
   b. Compare psi at probe locations to observations
   c. Update profiles using least-squares fit
   d. Repeat until convergence or max_iterations
4. Compare final reconstruction to ground truth

Algorithm
---------
The Solov'ev equilibrium depends on a single shape-source parameter A
(Cerfon-Freidberg 2010) that controls the p' vs FF' balance.  The
inverse problem is: given probe measurements psi_obs at N boundary
locations, find the value of A that minimises

    ||psi_computed(probes; A) - psi_obs||^2

We use a Newton/Gauss-Newton iteration on the scalar parameter A:

    A_{k+1} = A_k - (J^T J)^{-1} J^T r_k

where r_k = psi_computed(probes; A_k) - psi_obs is the residual vector
and J = dr/dA is estimated by finite difference.

This is a simplified inverse solver for the CPC paper.  A production
inverse solver would use Levenberg-Marquardt with Tikhonov regularisation,
multi-parameter profiles (p'(psi_n), FF'(psi_n) as spline coefficients),
and measured external magnetics rather than boundary psi values.

Limitations
-----------
- Source profiles are parameterised by a single scalar A rather than
  independent p'(psi) and FF'(psi) radial profiles.
- The forward model is the same Solov'ev analytical solution used to
  generate the ground truth, so the "inverse crime" is present.  The
  validation is intended to demonstrate that the iterative reconstruction
  loop converges, not to prove model-independent accuracy.
- No regularisation is applied because the single-parameter problem is
  well-posed.  Multi-parameter inversions require regularisation.

Usage:
    python validation/run_inverse_validation.py [--shots-dir validation/synthetic_shots]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# ── Local imports ────────────────────────────────────────────────────

# We import from siblings in the validation/ directory.
# Ensure the parent is on sys.path for direct script execution.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from run_forward_validation import (
    ShotData,
    load_shot,
    discover_shots,
    _sanitise_for_json,
    _extract_zero_contour,
)
from generate_synthetic_shots import (
    SolovevEquilibrium,
    _HOMOGENEOUS,
    _psi_particular_p,
    _psi_particular_ff,
)


# ── Physical constants ────────────────────────────────────────────────

MU_0 = 4.0 * np.pi * 1e-7  # vacuum permeability [H/m]


# =====================================================================
# Inverse Reconstruction Engine
# =====================================================================


@dataclass
class InverseConfig:
    """Configuration for the inverse reconstruction loop."""

    max_iterations: int = 20
    tolerance: float = 1e-6
    perturbation_range: float = 0.20
    fd_step: float = 1e-4
    damping: float = 0.8
    min_damping: float = 0.1
    seed: int = 20260214


@dataclass
class InverseResult:
    """Container for a single shot's inverse reconstruction result."""

    shot_id: str
    category: str
    A_true: float
    A_initial: float
    A_recovered: float
    perturbation_applied: float
    iterations: int
    converged: bool
    residual_initial: float
    residual_final: float
    psi_rmse_initial: float
    psi_rmse_final: float
    axis_error_m: float
    boundary_hausdorff_m: float
    wall_time_s: float
    A_relative_error: float
    improvement_ratio: float

    def to_dict(self) -> dict:
        """Convert to JSON-serialisable dict."""
        return _sanitise_for_json({
            "shot_id": self.shot_id,
            "category": self.category,
            "A_true": self.A_true,
            "A_initial": self.A_initial,
            "A_recovered": self.A_recovered,
            "perturbation_applied": self.perturbation_applied,
            "iterations": self.iterations,
            "converged": self.converged,
            "residual_initial": self.residual_initial,
            "residual_final": self.residual_final,
            "psi_rmse_initial": self.psi_rmse_initial,
            "psi_rmse_final": self.psi_rmse_final,
            "axis_error_m": self.axis_error_m,
            "boundary_hausdorff_m": self.boundary_hausdorff_m,
            "wall_time_s": self.wall_time_s,
            "A_relative_error": self.A_relative_error,
            "improvement_ratio": self.improvement_ratio,
        })


def _solve_solovev_with_A(
    shot: ShotData,
    A_param: float,
) -> SolovevEquilibrium:
    """Forward-solve a Solov'ev equilibrium with a given A parameter.

    Re-uses the shot's geometry (R0, a, kappa, delta, B0, Ip) but
    overrides the Cerfon-Freidberg A parameter.

    Parameters
    ----------
    shot : ShotData
        Reference shot for geometry and plasma parameters.
    A_param : float
        Cerfon-Freidberg A parameter (controls p'/FF' balance).

    Returns
    -------
    SolovevEquilibrium
        Solved equilibrium with the given A parameter.
    """
    eq = SolovevEquilibrium(
        R0=shot.R0,
        a=shot.a,
        B0=shot.B0,
        Ip=shot.Ip,
        kappa=shot.kappa,
        delta=shot.delta,
        nr=len(shot.r_grid),
        nz=len(shot.z_grid),
        n_probes=len(shot.probe_r),
    )
    # Override the A parameter before solving.
    # We need to modify the solve() method's behaviour, so we replicate
    # the key steps with the overridden A.
    eq.A_param = A_param

    # Boundary shape (same as original)
    theta_b = np.linspace(0.0, 2.0 * np.pi, eq.n_boundary, endpoint=False)
    from generate_synthetic_shots import shaped_boundary
    Rb, Zb = shaped_boundary(theta_b, eq.R0, eq.a, eq.kappa, eq.delta)
    eq.boundary_r = Rb
    eq.boundary_z = Zb
    xb = Rb / eq.R0
    yb = Zb / eq.R0

    # Solve for coefficients
    n_terms = len(_HOMOGENEOUS)
    M = np.zeros((eq.n_boundary, n_terms))
    for k, psi_hk in enumerate(_HOMOGENEOUS):
        M[:, k] = psi_hk(xb, yb)
    rhs = -(_psi_particular_p(xb, yb) + A_param * _psi_particular_ff(xb, yb))
    eq.coefficients, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)

    # Grid (match shot's grid exactly)
    eq.r_grid = shot.r_grid.copy()
    eq.z_grid = shot.z_grid.copy()
    eq.nr = len(eq.r_grid)
    eq.nz = len(eq.z_grid)

    # Evaluate psi on grid
    x2d = np.zeros((eq.nr, eq.nz))
    y2d = np.zeros((eq.nr, eq.nz))
    for i, R_val in enumerate(eq.r_grid):
        for j, Z_val in enumerate(eq.z_grid):
            x2d[i, j] = R_val / eq.R0
            y2d[i, j] = Z_val / eq.R0

    psi_raw = _psi_particular_p(x2d, y2d) + A_param * _psi_particular_ff(x2d, y2d)
    for k, psi_hk in enumerate(_HOMOGENEOUS):
        psi_raw += eq.coefficients[k] * psi_hk(x2d, y2d)

    # Normalisation: shift so boundary ~= 0, flip if needed
    psi_on_bdry = _psi_particular_p(xb, yb) + A_param * _psi_particular_ff(xb, yb)
    for k, psi_hk in enumerate(_HOMOGENEOUS):
        psi_on_bdry += eq.coefficients[k] * psi_hk(xb, yb)
    psi_raw -= np.mean(psi_on_bdry)

    i_axis = np.argmin(np.abs(eq.r_grid - eq.R0))
    j_axis = np.argmin(np.abs(eq.z_grid - 0.0))
    if psi_raw[i_axis, j_axis] > 0:
        psi_raw = -psi_raw

    # Scale to physical units
    idx_min = np.unravel_index(np.argmin(psi_raw), psi_raw.shape)
    eq.r_axis = float(eq.r_grid[idx_min[0]])
    eq.z_axis = float(eq.z_grid[idx_min[1]])
    eq.psi_axis = float(psi_raw[idx_min])
    eq.psi_boundary = 0.0

    Ip_A = eq.Ip * 1e6
    psi_scale = MU_0 * eq.R0 * Ip_A / (2.0 * np.pi)
    if abs(eq.psi_axis) > 1e-30:
        scale_factor = psi_scale / abs(eq.psi_axis)
    else:
        scale_factor = psi_scale
    psi_raw *= scale_factor
    eq.psi_axis *= scale_factor
    eq.psi_rz = psi_raw

    # Probes: same locations as the shot
    eq.probe_r = shot.probe_r.copy()
    eq.probe_z = shot.probe_z.copy()
    eq.probe_psi = eq._interpolate_psi(eq.probe_r, eq.probe_z)

    return eq


def _compute_probe_residual(
    shot: ShotData,
    A_param: float,
    psi_obs: NDArray,
) -> Tuple[NDArray, SolovevEquilibrium]:
    """Compute the residual vector at probe locations for a given A.

    Parameters
    ----------
    shot : ShotData
        Reference shot with geometry and probe locations.
    A_param : float
        Current estimate of the A parameter.
    psi_obs : (n_probes,) array
        Observed psi values at probe locations.

    Returns
    -------
    residual : (n_probes,) array
        psi_computed(probes) - psi_obs
    eq : SolovevEquilibrium
        The solved equilibrium (for post-analysis).
    """
    eq = _solve_solovev_with_A(shot, A_param)
    psi_computed = eq.probe_psi
    residual = psi_computed - psi_obs
    return residual, eq


def _normalised_psi_rmse(psi_ours: NDArray, psi_ref: NDArray) -> float:
    """Normalised RMSE: ||psi_ours - psi_ref||_2 / ||psi_ref||_2."""
    diff_norm = np.linalg.norm(psi_ours - psi_ref)
    ref_norm = np.linalg.norm(psi_ref)
    if ref_norm == 0.0:
        return 0.0 if diff_norm == 0.0 else float("inf")
    return float(diff_norm / ref_norm)


def _axis_error_total_m(
    psi: NDArray,
    r_grid: NDArray,
    z_grid: NDArray,
    r_axis_ref: float,
    z_axis_ref: float,
) -> float:
    """Return the total magnetic axis position error [m]."""
    idx_min = np.unravel_index(np.argmin(psi), psi.shape)
    ir_ax, iz_ax = idx_min
    r_axis = float(r_grid[ir_ax]) if ir_ax < len(r_grid) else float(r_grid[-1])
    z_axis = float(z_grid[iz_ax]) if iz_ax < len(z_grid) else float(z_grid[-1])
    return float(np.hypot(r_axis - r_axis_ref, z_axis - z_axis_ref))


def _boundary_hausdorff(
    psi_ours: NDArray,
    r_grid: NDArray,
    z_grid: NDArray,
    bdry_r_ref: NDArray,
    bdry_z_ref: NDArray,
) -> float:
    """Compute boundary Hausdorff distance, with fallback."""
    try:
        our_bdry_r, our_bdry_z = _extract_zero_contour(psi_ours, r_grid, z_grid)
        if len(our_bdry_r) < 3:
            return float("nan")
        from scipy.spatial.distance import directed_hausdorff
        pts_ours = np.column_stack((our_bdry_r, our_bdry_z))
        pts_ref = np.column_stack((bdry_r_ref, bdry_z_ref))
        h_fwd = directed_hausdorff(pts_ours, pts_ref)[0]
        h_bwd = directed_hausdorff(pts_ref, pts_ours)[0]
        return float(max(h_fwd, h_bwd))
    except Exception:
        return float("nan")


def run_inverse_reconstruction(
    shot: ShotData,
    config: InverseConfig,
    rng: np.random.Generator,
) -> InverseResult:
    """Run the inverse reconstruction for a single shot.

    Parameters
    ----------
    shot : ShotData
        Loaded synthetic shot with ground-truth data.
    config : InverseConfig
        Reconstruction settings.
    rng : numpy Generator
        For reproducible perturbations.

    Returns
    -------
    InverseResult
        Comprehensive result container.
    """
    t0 = time.time()

    # ── 1. Observed data: probe psi values from ground truth ──────
    psi_obs = shot.probe_psi.copy()
    A_true = shot.A_param

    # ── 2. Perturbed initial guess ────────────────────────────────
    perturbation = rng.uniform(-config.perturbation_range, config.perturbation_range)
    A_current = A_true * (1.0 + perturbation)

    A_initial = A_current

    # ── 3. Compute initial-guess psi field for RMSE baseline ──────
    residual_init, eq_init = _compute_probe_residual(shot, A_current, psi_obs)
    residual_norm_init = float(np.linalg.norm(residual_init))
    psi_rmse_initial = _normalised_psi_rmse(eq_init.psi_rz, shot.psi_rz)

    # ── 4. Newton iteration loop ──────────────────────────────────
    converged = False
    n_iter = 0
    residual_norm = residual_norm_init
    damping = config.damping

    for it in range(1, config.max_iterations + 1):
        n_iter = it

        # Current residual
        residual, eq_current = _compute_probe_residual(shot, A_current, psi_obs)
        residual_norm = float(np.linalg.norm(residual))

        # Check convergence
        if residual_norm < config.tolerance:
            converged = True
            break

        # Jacobian by finite difference: dr/dA
        A_pert = A_current + config.fd_step
        residual_pert, _ = _compute_probe_residual(shot, A_pert, psi_obs)
        J = (residual_pert - residual) / config.fd_step  # (n_probes,)

        # Gauss-Newton step for scalar parameter:
        # delta_A = -(J^T J)^{-1} J^T r
        JtJ = float(np.dot(J, J))
        if JtJ < 1e-30:
            # Jacobian is effectively zero; cannot update
            break
        Jtr = float(np.dot(J, residual))
        delta_A = -Jtr / JtJ

        # Damped update
        A_candidate = A_current + damping * delta_A

        # Evaluate candidate
        residual_candidate, _ = _compute_probe_residual(shot, A_candidate, psi_obs)
        residual_norm_candidate = float(np.linalg.norm(residual_candidate))

        # Simple line-search: halve damping if no improvement
        ls_iter = 0
        while residual_norm_candidate > residual_norm and ls_iter < 5:
            damping = max(damping * 0.5, config.min_damping)
            A_candidate = A_current + damping * delta_A
            residual_candidate, _ = _compute_probe_residual(
                shot, A_candidate, psi_obs,
            )
            residual_norm_candidate = float(np.linalg.norm(residual_candidate))
            ls_iter += 1

        A_current = A_candidate
        residual_norm = residual_norm_candidate

        # Restore damping towards nominal for next iteration
        damping = min(damping * 1.2, config.damping)

        if residual_norm < config.tolerance:
            converged = True
            break

    # ── 5. Final evaluation ───────────────────────────────────────
    _, eq_final = _compute_probe_residual(shot, A_current, psi_obs)

    psi_rmse_final = _normalised_psi_rmse(eq_final.psi_rz, shot.psi_rz)

    axis_err = _axis_error_total_m(
        eq_final.psi_rz, eq_final.r_grid, eq_final.z_grid,
        shot.r_axis, shot.z_axis,
    )

    bdry_hausdorff = _boundary_hausdorff(
        eq_final.psi_rz, eq_final.r_grid, eq_final.z_grid,
        shot.boundary_r, shot.boundary_z,
    )

    # A-parameter relative error
    if abs(A_true) > 1e-30:
        A_rel_err = abs(A_current - A_true) / abs(A_true)
    else:
        A_rel_err = abs(A_current - A_true)

    # Improvement ratio: how much better is final vs initial RMSE
    if psi_rmse_initial > 1e-30:
        improvement = psi_rmse_initial / max(psi_rmse_final, 1e-30)
    else:
        improvement = 1.0

    wall_time = time.time() - t0

    return InverseResult(
        shot_id=shot.shot_id,
        category=shot.category,
        A_true=A_true,
        A_initial=A_initial,
        A_recovered=A_current,
        perturbation_applied=perturbation,
        iterations=n_iter,
        converged=converged,
        residual_initial=residual_norm_init,
        residual_final=residual_norm,
        psi_rmse_initial=psi_rmse_initial,
        psi_rmse_final=psi_rmse_final,
        axis_error_m=axis_err,
        boundary_hausdorff_m=bdry_hausdorff,
        wall_time_s=wall_time,
        A_relative_error=A_rel_err,
        improvement_ratio=improvement,
    )


# =====================================================================
# Main validation loop
# =====================================================================


def run_inverse_validation(
    shots_dir: Path,
    results_dir: Path,
    config: InverseConfig,
) -> List[dict]:
    """Run inverse reconstruction validation on all synthetic shots.

    Parameters
    ----------
    shots_dir : Path
        Directory containing synthetic shot NPZ/JSON files.
    results_dir : Path
        Directory to write per-shot results and summary.
    config : InverseConfig
        Reconstruction settings.

    Returns
    -------
    list of dict
        Per-shot result dictionaries.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    shot_pairs = discover_shots(shots_dir)
    if not shot_pairs:
        print(f"No shots found in {shots_dir}")
        return []

    n_shots = len(shot_pairs)
    rng = np.random.default_rng(config.seed)

    print(f"Inverse Reconstruction Validation")
    print(f"=" * 78)
    print(f"Found {n_shots} synthetic shots in {shots_dir}")
    print(f"Results will be saved to {results_dir}")
    print(f"Settings: max_iter={config.max_iterations}, "
          f"tol={config.tolerance:.1e}, "
          f"perturbation=+/-{config.perturbation_range*100:.0f}%")
    print(f"-" * 78)

    all_results: List[dict] = []

    for idx, (npz_path, json_path) in enumerate(shot_pairs):
        shot = load_shot(npz_path, json_path)

        result = run_inverse_reconstruction(shot, config, rng)

        # Status classification
        if result.converged and result.psi_rmse_final < 0.01:
            status = "OK"
        elif result.psi_rmse_final < 0.05:
            status = "WARN"
        else:
            status = "FAIL"

        print(
            f"[{idx + 1:2d}/{n_shots}] {shot.shot_id} "
            f"({shot.category:>20s}): "
            f"RMSE {result.psi_rmse_initial:.4f} -> {result.psi_rmse_final:.6f}  "
            f"A_err={result.A_relative_error:.2e}  "
            f"iter={result.iterations:2d}  "
            f"[{status}]  ({result.wall_time_s:.2f}s)"
        )

        result_dict = result.to_dict()
        result_dict["status"] = status

        # Save per-shot result
        shot_result_path = results_dir / f"{shot.shot_id}.json"
        with open(shot_result_path, "w") as f:
            json.dump(result_dict, f, indent=2)

        all_results.append(result_dict)

    return all_results


# =====================================================================
# Summary generation
# =====================================================================


def generate_summary_csv(results: List[dict], csv_path: Path) -> None:
    """Write summary CSV from per-shot results."""
    if not results:
        return

    fieldnames = [
        "shot_id", "category",
        "A_true", "A_initial", "A_recovered", "perturbation_applied",
        "iterations", "converged",
        "residual_initial", "residual_final",
        "psi_rmse_initial", "psi_rmse_final",
        "axis_error_m", "boundary_hausdorff_m",
        "A_relative_error", "improvement_ratio",
        "wall_time_s", "status",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(_sanitise_for_json(r))

    print(f"\nSummary CSV written: {csv_path}")


def print_aggregate_statistics(results: List[dict]) -> None:
    """Print aggregate statistics grouped by category."""
    if not results:
        print("No results to summarise.")
        return

    categories: Dict[str, List[dict]] = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    print("\n" + "=" * 90)
    print("Inverse Reconstruction Validation Summary")
    print("=" * 90)
    print(
        f"{'Category':<25s} {'Shots':>5s} "
        f"{'Mean RMSE_f':>11s} {'Max RMSE_f':>11s} "
        f"{'Mean A_err':>11s} "
        f"{'Mean Improv':>11s} "
        f"{'Converged':>10s}"
    )
    print("-" * 90)

    total_converged = 0
    total_shots = 0

    for cat in sorted(categories.keys()):
        shots = categories[cat]
        rmses_final = [s["psi_rmse_final"] for s in shots]
        a_errs = [s["A_relative_error"] for s in shots]
        improvements = [s["improvement_ratio"] for s in shots]
        n_conv = sum(1 for s in shots if s["converged"])

        total_converged += n_conv
        total_shots += len(shots)

        print(
            f"{cat:<25s} {len(shots):>5d} "
            f"{float(np.mean(rmses_final)):>11.6f} "
            f"{float(np.max(rmses_final)):>11.6f} "
            f"{float(np.mean(a_errs)):>11.2e} "
            f"{float(np.mean(improvements)):>11.1f}x "
            f"{n_conv:>4d}/{len(shots):<4d}"
        )

    print("-" * 90)

    all_rmses_init = [r["psi_rmse_initial"] for r in results]
    all_rmses_final = [r["psi_rmse_final"] for r in results]
    all_a_errs = [r["A_relative_error"] for r in results]
    all_improvements = [r["improvement_ratio"] for r in results]

    print(
        f"{'OVERALL':<25s} {total_shots:>5d} "
        f"{float(np.mean(all_rmses_final)):>11.6f} "
        f"{float(np.max(all_rmses_final)):>11.6f} "
        f"{float(np.mean(all_a_errs)):>11.2e} "
        f"{float(np.mean(all_improvements)):>11.1f}x "
        f"{total_converged:>4d}/{total_shots:<4d}"
    )
    print("=" * 90)

    # Additional statistics
    total_time = sum(r["wall_time_s"] for r in results)
    mean_time = total_time / len(results)
    mean_iter = np.mean([r["iterations"] for r in results])

    print(f"\nTotal wall time:   {total_time:.1f}s  "
          f"(mean {mean_time:.2f}s/shot)")
    print(f"Mean iterations:   {mean_iter:.1f}")
    print(f"Converged:         {total_converged}/{total_shots}")
    print(f"Mean initial RMSE: {float(np.mean(all_rmses_init)):.6f}")
    print(f"Mean final RMSE:   {float(np.mean(all_rmses_final)):.6f}")
    print(f"Mean improvement:  {float(np.mean(all_improvements)):.1f}x")

    # Print note about limitations
    print()
    print("NOTE: This is a simplified single-parameter (A) inverse solver.")
    print("The inverse crime is present: the same Solov'ev forward model is")
    print("used for both data generation and reconstruction.  Production")
    print("inverse solvers would use multi-parameter profiles, independent")
    print("forward models, and Tikhonov regularisation.")


# =====================================================================
# CLI entry point
# =====================================================================


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Inverse reconstruction validation: recover Solov'ev "
                    "equilibrium parameters from synthetic probe measurements.",
    )
    parser.add_argument(
        "--shots-dir", type=str, default=None,
        help="Directory containing synthetic shot files. "
             "Defaults to validation/synthetic_shots/ relative to this script.",
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Directory to write results. "
             "Defaults to validation/results/inverse/ relative to this script.",
    )
    parser.add_argument(
        "--max-iter", type=int, default=20,
        help="Maximum Newton iterations per shot (default: 20).",
    )
    parser.add_argument(
        "--tol", type=float, default=1e-6,
        help="Convergence tolerance on ||residual|| (default: 1e-6).",
    )
    parser.add_argument(
        "--perturbation", type=float, default=0.20,
        help="Initial perturbation range for A parameter (default: 0.20 = +/-20%%).",
    )
    parser.add_argument(
        "--seed", type=int, default=20260214,
        help="RNG seed for reproducible perturbations (default: 20260214).",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    shots_dir = (
        Path(args.shots_dir) if args.shots_dir
        else script_dir / "synthetic_shots"
    )
    results_dir = (
        Path(args.results_dir) if args.results_dir
        else script_dir / "results" / "inverse"
    )

    # Generate synthetic shots if they don't exist
    if not shots_dir.exists() or not list(shots_dir.glob("*.npz")):
        print(f"Synthetic shots not found at {shots_dir}")
        print("Generating synthetic shots...")
        print()

        import json as _json

        class _NumpyEncoder(_json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.bool_,)):
                    return bool(obj)
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        _original_dump = _json.dump
        def _patched_dump(obj, fp, **kwargs):
            kwargs.setdefault("cls", _NumpyEncoder)
            return _original_dump(obj, fp, **kwargs)
        _json.dump = _patched_dump

        from generate_synthetic_shots import generate_all_shots
        generate_all_shots(seed=args.seed, output_dir=shots_dir, save=True)

        _json.dump = _original_dump
        print()

    config = InverseConfig(
        max_iterations=args.max_iter,
        tolerance=args.tol,
        perturbation_range=args.perturbation,
        seed=args.seed,
    )

    # Run validation
    results = run_inverse_validation(
        shots_dir=shots_dir,
        results_dir=results_dir,
        config=config,
    )

    if not results:
        print("No results produced. Check that synthetic shots exist.")
        sys.exit(1)

    csv_path = results_dir / "summary.csv"
    generate_summary_csv(results, csv_path)
    print_aggregate_statistics(results)


if __name__ == "__main__":
    main()
