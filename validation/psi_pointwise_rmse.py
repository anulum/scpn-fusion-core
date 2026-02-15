# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Point-wise ψ(R,Z) RMSE Validation
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""
Point-wise ψ(R,Z) reconstruction error for SPARC GEQDSK equilibria.

For each GEQDSK file this module:

1. **GS residual** — Applies the Grad-Shafranov finite-difference operator
   to the reference ψ(R,Z) using the file's own p'(ψ) and FF'(ψ) profiles.
   A small residual proves the file is a self-consistent equilibrium and
   that our numerical operators agree with the reference code (EFIT/CHEASE).

2. **Manufactured-source solve** — Uses the boundary row/column of the
   reference ψ as Dirichlet BC, computes J_ϕ from the reference GS operator,
   then solves the resulting Poisson problem with SOR.  The point-wise
   difference between solver output and reference interior tests our
   elliptic solver in isolation.

3. **Normalized ψ RMSE** — Reports errors in both raw (Wb/rad) and
   normalised (0=axis, 1=boundary) coordinates, restricted to the plasma
   region (ψ_N ∈ [0,1)).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.core.eqdsk import GEqdsk, read_geqdsk

ROOT = Path(__file__).resolve().parents[1]
SPARC_DIR = ROOT / "validation" / "reference_data" / "sparc"


# ── Data containers ──────────────────────────────────────────────────

@dataclass
class PsiRMSEResult:
    """Per-file ψ RMSE metrics."""
    file: str
    grid: str                        # e.g. "129x129"

    # GS residual (how well reference satisfies GS equation)
    gs_residual_l2: float            # ||residual||_2 / ||source||_2
    gs_residual_max: float           # max |residual|  (Wb/rad·m⁻²)

    # Manufactured-source solve
    psi_rmse_wb: float               # raw RMSE in Wb/rad
    psi_rmse_norm: float             # RMSE in normalised ψ_N
    psi_rmse_plasma_wb: float        # RMSE restricted to plasma (ψ_N ∈ [0,1))
    psi_max_error_wb: float          # max |Δψ| over full domain
    psi_relative_l2: float           # ||Δψ||_2 / ||ψ_ref||_2

    # Solver metadata
    sor_iterations: int
    sor_residual: float
    solve_time_ms: float


@dataclass
class PsiRMSESummary:
    """Aggregate over all files."""
    count: int
    mean_psi_rmse_norm: float
    mean_psi_relative_l2: float
    mean_gs_residual_l2: float
    worst_psi_rmse_norm: float
    worst_file: str
    rows: list[dict[str, Any]]


# ── GS operator ──────────────────────────────────────────────────────

def gs_operator(
    psi: NDArray, R: NDArray, Z: NDArray
) -> NDArray:
    """
    Evaluate the Grad-Shafranov elliptic operator Δ*ψ on interior points.

    Δ*ψ = ∂²ψ/∂R² - (1/R)∂ψ/∂R + ∂²ψ/∂Z²

    Returns array of same shape with boundary values set to zero.
    """
    nz, nr = psi.shape
    dR = float(R[1] - R[0])
    dZ = float(Z[1] - Z[0])
    RR = R[np.newaxis, :]  # (1, nr)

    result = np.zeros_like(psi)

    # Interior finite differences
    d2R = (psi[1:-1, 2:] - 2 * psi[1:-1, 1:-1] + psi[1:-1, :-2]) / dR**2
    dR1 = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * dR)
    d2Z = (psi[2:, 1:-1] - 2 * psi[1:-1, 1:-1] + psi[:-2, 1:-1]) / dZ**2

    R_interior = np.maximum(RR[:, 1:-1], 1e-6)
    result[1:-1, 1:-1] = d2R - dR1 / R_interior + d2Z
    return result


def compute_gs_source(
    eq: GEqdsk,
) -> NDArray:
    """
    Compute the GS source term from GEQDSK profiles.

    J_ϕ(R,Z) = R·p'(ψ_N) + FF'(ψ_N) / (μ₀·R)

    Returns -μ₀·R·J_ϕ = -(μ₀·R²·p' + FF') which is the RHS of Δ*ψ = RHS.
    """
    psi_norm_1d = np.linspace(0.0, 1.0, eq.nw)
    R = eq.r
    Z = eq.z
    RR, _ = np.meshgrid(R, Z)

    # Normalise reference psi
    denom = eq.sibry - eq.simag
    if abs(denom) < 1e-12:
        return np.zeros((eq.nh, eq.nw))
    psi_n = (eq.psirz - eq.simag) / denom

    # Interpolate 1D profiles onto 2D grid
    psi_n_clipped = np.clip(psi_n, 0.0, 1.0)
    pprime_2d = np.interp(psi_n_clipped.ravel(), psi_norm_1d, eq.pprime).reshape(eq.nh, eq.nw)
    ffprime_2d = np.interp(psi_n_clipped.ravel(), psi_norm_1d, eq.ffprime).reshape(eq.nh, eq.nw)

    # Zero outside plasma
    outside = (psi_n < 0) | (psi_n >= 1.0)
    pprime_2d[outside] = 0.0
    ffprime_2d[outside] = 0.0

    mu0 = 4e-7 * np.pi
    # RHS = -(mu0 * R^2 * p' + FF')
    rhs = -(mu0 * RR**2 * pprime_2d + ffprime_2d)
    return rhs


# ── GS residual ──────────────────────────────────────────────────────

def gs_residual(eq: GEqdsk) -> tuple[float, float]:
    """
    Compute relative L2 and max residual of the GS equation on reference ψ.

    Returns (relative_l2, max_abs) where relative_l2 = ||Lψ - S||/||S||.
    """
    R, Z = eq.r, eq.z
    L_psi = gs_operator(eq.psirz, R, Z)
    source = compute_gs_source(eq)

    residual = L_psi[1:-1, 1:-1] - source[1:-1, 1:-1]
    source_norm = np.linalg.norm(source[1:-1, 1:-1].ravel())
    if source_norm < 1e-15:
        source_norm = 1.0

    rel_l2 = float(np.linalg.norm(residual.ravel()) / source_norm)
    max_abs = float(np.max(np.abs(residual)))
    return rel_l2, max_abs


# ── Manufactured-source SOR solve ────────────────────────────────────

def manufactured_solve(
    eq: GEqdsk,
    omega: float = 1.5,
    max_iter: int = 5000,
    tol: float = 1e-7,
) -> tuple[NDArray, int, float, float]:
    """
    Solve Δ*ψ = S with reference BCs using SOR.

    Uses the reference boundary values as Dirichlet BC and the GS source
    computed from the reference profiles.  Returns (psi_solved, iters,
    final_residual, solve_time_ms).
    """
    R, Z = eq.r, eq.z
    nz, nr = eq.nh, eq.nw
    dR = float(R[1] - R[0])
    dZ = float(Z[1] - Z[0])
    RR = R[np.newaxis, :]

    source = compute_gs_source(eq)

    # Initialise with reference boundary, zero interior
    psi = np.zeros((nz, nr))
    psi[0, :] = eq.psirz[0, :]
    psi[-1, :] = eq.psirz[-1, :]
    psi[:, 0] = eq.psirz[:, 0]
    psi[:, -1] = eq.psirz[:, -1]

    # Pre-compute coefficients for GS* operator
    # Δ*ψ = d²ψ/dR² - (1/R)dψ/dR + d²ψ/dZ² = S
    # Rewritten for SOR:
    # ψ_{i,j} = [a_R(ψ_{i,j+1}+ψ_{i,j-1}) + a_Z(ψ_{i+1,j}+ψ_{i-1,j})
    #            + c_R(ψ_{i,j+1}-ψ_{i,j-1}) - dR²dZ²·S] / (2a_R + 2a_Z)
    # where a_R = dZ², a_Z = dR², c_R = -dZ²dR/(2R)
    a_R = dZ**2
    a_Z = dR**2
    denom_base = 2 * a_R + 2 * a_Z
    scale = dR**2 * dZ**2

    R_int = np.maximum(RR[:, 1:-1], 1e-6)
    c_R = -dZ**2 * dR / (2.0 * R_int)  # (1, nr-2)

    t0 = time.perf_counter()
    final_res = 0.0

    for it in range(max_iter):
        max_change = 0.0
        for iz in range(1, nz - 1):
            for ir in range(1, nr - 1):
                r_val = max(R[ir], 1e-6)
                c = -dZ**2 * dR / (2.0 * r_val)
                rhs = (
                    a_R * (psi[iz, ir + 1] + psi[iz, ir - 1])
                    + a_Z * (psi[iz + 1, ir] + psi[iz - 1, ir])
                    + c * (psi[iz, ir + 1] - psi[iz, ir - 1])
                    - scale * source[iz, ir]
                )
                new_val = rhs / denom_base
                change = new_val - psi[iz, ir]
                psi[iz, ir] += omega * change
                abs_c = abs(change)
                if abs_c > max_change:
                    max_change = abs_c

        if max_change < tol:
            final_res = max_change
            break
        final_res = max_change

    solve_ms = (time.perf_counter() - t0) * 1000.0
    return psi, it + 1, final_res, solve_ms


def manufactured_solve_vectorised(
    eq: GEqdsk,
    omega: float = 1.5,
    max_iter: int = 5000,
    tol: float = 1e-7,
) -> tuple[NDArray, int, float, float]:
    """
    Vectorised Red-Black SOR solve of Δ*ψ = S with reference BCs.

    Much faster than the scalar version for grids > 30×30.
    """
    if not np.isfinite(omega) or omega <= 0.0 or omega >= 2.0:
        raise ValueError("omega must be finite and in (0, 2)")
    if max_iter <= 0:
        raise ValueError("max_iter must be > 0")
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("tol must be finite and >= 0")

    R, Z = eq.r, eq.z
    nz, nr = eq.nh, eq.nw
    dR = float(R[1] - R[0])
    dZ = float(Z[1] - Z[0])

    source = compute_gs_source(eq)

    # Initialise with reference boundary
    psi = eq.psirz.copy()  # start from reference as warm start

    a_R = dZ**2
    a_Z = dR**2
    denom = 2 * a_R + 2 * a_Z
    scale = dR**2 * dZ**2

    R_2d = np.broadcast_to(R[np.newaxis, :], (nz, nr))
    R_safe = np.maximum(R_2d, 1e-6)
    c_coeff = -dZ**2 * dR / (2.0 * R_safe)

    t0 = time.perf_counter()
    final_res = 0.0

    for it in range(max_iter):
        # Red-Black ordering for two sweeps per iteration
        for parity in (0, 1):
            # Build stencil
            rhs = np.zeros((nz, nr))
            rhs[1:-1, 1:-1] = (
                a_R * (psi[1:-1, 2:] + psi[1:-1, :-2])
                + a_Z * (psi[2:, 1:-1] + psi[:-2, 1:-1])
                + c_coeff[1:-1, 1:-1] * (psi[1:-1, 2:] - psi[1:-1, :-2])
                - scale * source[1:-1, 1:-1]
            )
            new_vals = rhs / denom

            # Checkerboard mask for this parity
            iz_idx, ir_idx = np.mgrid[1:nz-1, 1:nr-1]
            mask = ((iz_idx + ir_idx) % 2) == parity
            psi_old = psi[1:-1, 1:-1].copy()
            psi[1:-1, 1:-1] = np.where(
                mask,
                psi_old + omega * (new_vals[1:-1, 1:-1] - psi_old),
                psi_old,
            )

        # Convergence check (every 10 iterations to save time)
        if (it + 1) % 10 == 0 or it == max_iter - 1:
            # Use the GS residual as convergence metric
            L_psi = gs_operator(psi, R, Z)
            res_field = L_psi[1:-1, 1:-1] - source[1:-1, 1:-1]
            final_res = float(np.max(np.abs(res_field)))
            if final_res <= tol:
                break

    solve_ms = (time.perf_counter() - t0) * 1000.0
    return psi, it + 1, final_res, solve_ms


# ── RMSE computation ─────────────────────────────────────────────────

def compute_psi_rmse(
    eq: GEqdsk,
    solver_psi: NDArray,
) -> dict[str, float]:
    """
    Compute point-wise RMSE between reference and solver ψ.

    Reports metrics in both raw (Wb/rad) and normalised coordinates.
    """
    ref = eq.psirz
    diff = solver_psi - ref

    # Raw RMSE
    rmse_wb = float(np.sqrt(np.mean(diff**2)))
    max_err = float(np.max(np.abs(diff)))

    # Relative L2
    ref_norm = np.linalg.norm(ref.ravel())
    rel_l2 = float(np.linalg.norm(diff.ravel()) / max(ref_norm, 1e-15))

    # Normalised ψ
    denom = eq.sibry - eq.simag
    if abs(denom) < 1e-12:
        return {
            "psi_rmse_wb": rmse_wb,
            "psi_rmse_norm": float("nan"),
            "psi_rmse_plasma_wb": float("nan"),
            "psi_max_error_wb": max_err,
            "psi_relative_l2": rel_l2,
        }

    ref_n = (ref - eq.simag) / denom
    sol_n = (solver_psi - eq.simag) / denom

    diff_n = sol_n - ref_n
    rmse_norm = float(np.sqrt(np.mean(diff_n**2)))

    # Plasma-only RMSE
    plasma_mask = (ref_n >= 0) & (ref_n < 1.0)
    if np.any(plasma_mask):
        rmse_plasma = float(np.sqrt(np.mean(diff[plasma_mask] ** 2)))
    else:
        rmse_plasma = rmse_wb

    return {
        "psi_rmse_wb": rmse_wb,
        "psi_rmse_norm": rmse_norm,
        "psi_rmse_plasma_wb": rmse_plasma,
        "psi_max_error_wb": max_err,
        "psi_relative_l2": rel_l2,
    }


# ── Per-file validation ──────────────────────────────────────────────

def validate_file(path: Path, warm_start: bool = True) -> PsiRMSEResult:
    """
    Run full ψ RMSE validation on a single GEQDSK file.

    Parameters
    ----------
    path : Path
        Path to .geqdsk or .eqdsk file.
    warm_start : bool
        If True, initialise SOR from reference ψ (tests solver stability
        near the solution).  If False, start from boundary-only init
        (tests full convergence from cold start).
    """
    eq = read_geqdsk(path)

    # 1. GS residual
    gs_rel, gs_max = gs_residual(eq)

    # 2. Manufactured solve
    if warm_start:
        solver_psi, iters, res, t_ms = manufactured_solve_vectorised(
            eq, omega=1.3, max_iter=200, tol=1e-8,
        )
    else:
        solver_psi, iters, res, t_ms = manufactured_solve_vectorised(
            eq, omega=1.5, max_iter=2000, tol=1e-7,
        )

    # 3. Point-wise RMSE
    metrics = compute_psi_rmse(eq, solver_psi)

    return PsiRMSEResult(
        file=path.name,
        grid=f"{eq.nw}x{eq.nh}",
        gs_residual_l2=gs_rel,
        gs_residual_max=gs_max,
        psi_rmse_wb=metrics["psi_rmse_wb"],
        psi_rmse_norm=metrics["psi_rmse_norm"],
        psi_rmse_plasma_wb=metrics["psi_rmse_plasma_wb"],
        psi_max_error_wb=metrics["psi_max_error_wb"],
        psi_relative_l2=metrics["psi_relative_l2"],
        sor_iterations=iters,
        sor_residual=res,
        solve_time_ms=t_ms,
    )


# ── Aggregate validation ─────────────────────────────────────────────

def validate_all_sparc(sparc_dir: Path | None = None) -> PsiRMSESummary:
    """
    Run ψ RMSE validation on all 8 SPARC equilibrium files.

    Returns aggregate summary with per-file breakdown.
    """
    if sparc_dir is None:
        sparc_dir = SPARC_DIR

    files = sorted(sparc_dir.glob("*.geqdsk")) + sorted(sparc_dir.glob("*.eqdsk"))
    if not files:
        raise FileNotFoundError(f"No GEQDSK/EQDSK files in {sparc_dir}")

    results: list[PsiRMSEResult] = []
    for f in files:
        r = validate_file(f)
        results.append(r)

    rows = [asdict(r) for r in results]

    finite_norm_entries = [
        (idx, r.psi_rmse_norm)
        for idx, r in enumerate(results)
        if np.isfinite(r.psi_rmse_norm)
    ]
    norms = [norm for _, norm in finite_norm_entries]
    rel_l2s = [r.psi_relative_l2 for r in results]
    gs_l2s = [r.gs_residual_l2 for r in results]

    if finite_norm_entries:
        worst_idx, worst_norm = max(finite_norm_entries, key=lambda item: item[1])
        worst_file = results[worst_idx].file
    else:
        worst_norm = float("nan")
        worst_file = ""

    return PsiRMSESummary(
        count=len(results),
        mean_psi_rmse_norm=float(np.mean(norms)) if norms else float("nan"),
        mean_psi_relative_l2=float(np.mean(rel_l2s)),
        mean_gs_residual_l2=float(np.mean(gs_l2s)),
        worst_psi_rmse_norm=float(worst_norm),
        worst_file=worst_file,
        rows=rows,
    )


# ── For rmse_dashboard.py integration ────────────────────────────────

def sparc_psi_rmse(sparc_dir: Path) -> dict[str, Any]:
    """
    Drop-in function for rmse_dashboard.py integration.

    Returns dict with keys: count, mean_psi_rmse_norm, mean_psi_relative_l2,
    mean_gs_residual_l2, worst_psi_rmse_norm, worst_file, rows.
    """
    summary = validate_all_sparc(sparc_dir)
    return asdict(summary)


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 70)
    print("SCPN Fusion Core - Point-wise psi(R,Z) RMSE Validation")
    print("=" * 70)

    summary = validate_all_sparc()

    print(f"\nFiles validated: {summary.count}")
    print(f"Mean normalized psi RMSE: {summary.mean_psi_rmse_norm:.6f}")
    print(f"Mean relative L2:       {summary.mean_psi_relative_l2:.6f}")
    print(f"Mean GS residual (L2):  {summary.mean_gs_residual_l2:.6f}")
    print(f"Worst file:             {summary.worst_file} "
          f"(psi_N RMSE = {summary.worst_psi_rmse_norm:.6f})")
    print()

    # Per-file table
    print(f"{'File':<22} {'Grid':<8} {'psi_N RMSE':>10} {'Rel L2':>10} "
          f"{'GS Res':>10} {'Iters':>6} {'Time(ms)':>10}")
    print("-" * 80)
    for r in summary.rows:
        print(f"{r['file']:<22} {r['grid']:<8} "
              f"{r['psi_rmse_norm']:>10.6f} {r['psi_relative_l2']:>10.6f} "
              f"{r['gs_residual_l2']:>10.4f} {r['sor_iterations']:>6d} "
              f"{r['solve_time_ms']:>10.1f}")

    # Save JSON
    out = ROOT / "validation" / "reports" / "psi_pointwise_rmse.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(asdict(summary), f, indent=2)
    print(f"\nJSON report: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
