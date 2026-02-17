#!/usr/bin/env python3
# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — FreeGS / Solov'ev Blind Benchmark
# (c) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Blind benchmark comparing our GS solver against FreeGS or analytic Solov'ev.

When FreeGS is installed (``pip install freegs``), the script runs a full
numerical comparison on three tokamak configurations.  When FreeGS is
unavailable the script falls back to an analytic Solov'ev equilibrium
comparison, which is even more rigorous because the exact solution is
known in closed form.

Produces ``artifacts/freegs_benchmark.json``.

Exit codes:
    0 - benchmark ran and all cases PASS
    1 - one or more cases FAIL
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# ── FreeGS availability probe ────────────────────────────────────────

try:
    import freegs  # type: ignore[import-untyped]

    HAS_FREEGS = True
except ImportError:
    HAS_FREEGS = False

# ── NRMSE utility ────────────────────────────────────────────────────

PSI_NRMSE_THRESHOLD = 0.10  # 10 %


def nrmse(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """Normalised RMSE: RMSE / range(y_true)."""
    rmse_val = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    rng = float(np.max(y_true) - np.min(y_true))
    return rmse_val / max(rng, 1e-12)


# ── Test-case definitions ────────────────────────────────────────────

class TokamakCase(NamedTuple):
    """Specification for a single benchmark tokamak."""

    name: str
    R0: float          # Major radius [m]
    a: float           # Minor radius [m]
    B0: float          # Toroidal field on axis [T]
    Ip: float          # Plasma current [MA]
    kappa: float        # Elongation
    NR: int = 65       # Grid points in R
    NZ: int = 65       # Grid points in Z


CASES: list[TokamakCase] = [
    TokamakCase(name="ITER-like",            R0=6.2,  a=2.0,  B0=5.3,  Ip=15.0,  kappa=1.7),
    TokamakCase(name="SPARC-like",           R0=1.85, a=0.57, B0=12.2, Ip=8.7,   kappa=1.97),
    TokamakCase(name="Spherical-tokamak",    R0=0.85, a=0.55, B0=0.5,  Ip=1.0,   kappa=2.5),
]

# ── Solov'ev analytic equilibrium ────────────────────────────────────


def solovev_psi(
    R: NDArray[np.float64],
    Z: NDArray[np.float64],
    R0: float,
    a: float,
    kappa: float,
    Psi_0: float = 1.0,
) -> NDArray[np.float64]:
    """Compute a Solov'ev analytic GS solution on a 2-D (R, Z) meshgrid.

    Uses the R^2-centred parameterisation that naturally produces a
    closed separatrix::

        u = (R^2 - R0^2) / (epsilon_s * R0^2)
        v = Z / (kappa * a)
        Psi(R, Z) = Psi_0 * max(0, 1 - u^2 - v^2)

    where ``epsilon_s = ((R0+a)^2 - R0^2) / R0^2``.

    **GS consistency**.  Applying the toroidal GS* operator
    ``d2/dR2 - (1/R) d/dR + d2/dZ2`` to the interior Psi gives::

        Delta* Psi = -(8 Psi_0 / (eps R0^2)^2) R^2  -  2 Psi_0 / (ka)^2

    which corresponds to a Solov'ev source with constant ``p'`` and
    ``FF'``::

        j_phi = R p' + FF' / (mu0 R)
        p'  = 8 Psi_0 / (mu0 (eps R0^2)^2)
        FF' = 2 Psi_0 / (ka)^2

    **Boundary**.  Psi = 0 on the closed curve ``u^2 + v^2 = 1``.
    On the midplane (Z = 0) the outer boundary is at
    ``R = R0 sqrt(1 + epsilon_s) = R0 + a`` and the inner boundary at
    ``R = R0 sqrt(1 - epsilon_s)``.

    Parameters
    ----------
    R, Z : 2-D meshgrid arrays
    R0 : major radius [m]
    a : minor radius [m]
    kappa : elongation
    Psi_0 : float
        Peak Psi value at the magnetic axis (default 1.0).

    Returns
    -------
    Psi : 2-D array, >= 0 inside the separatrix, 0 outside.
    """
    eps_s = ((R0 + a) ** 2 - R0 ** 2) / R0 ** 2
    ka = kappa * a

    u = (R ** 2 - R0 ** 2) / (eps_s * R0 ** 2)
    v = Z / ka

    Psi_raw = Psi_0 * (1.0 - u ** 2 - v ** 2)
    Psi = np.where(Psi_raw > 0.0, Psi_raw, 0.0)
    return Psi


def solovev_jphi(
    R: NDArray[np.float64],
    Z: NDArray[np.float64],
    R0: float,
    a: float,
    kappa: float,
    Psi_0: float = 1.0,
) -> NDArray[np.float64]:
    """Analytic toroidal current density for the Solov'ev solution.

    For the R^2-centred Solov'ev the GS* residual is (with mu0 = 1)::

        Delta* Psi = -(8 Psi_0 / (eps R0^2)^2) R^2  -  2 Psi_0 / (ka)^2

    so the toroidal current density inside the plasma is::

        j_phi = (8 Psi_0 R) / (eps R0^2)^2  +  2 Psi_0 / (R (ka)^2)

    Outside the plasma, j_phi = 0.

    Parameters
    ----------
    R, Z : 2-D meshgrid arrays
    R0, a, kappa : plasma geometry
    Psi_0 : peak Psi at axis

    Returns
    -------
    J_phi : 2-D array, same shape as R.
    """
    eps_s = ((R0 + a) ** 2 - R0 ** 2) / R0 ** 2
    ka = kappa * a

    u = (R ** 2 - R0 ** 2) / (eps_s * R0 ** 2)
    v = Z / ka
    plasma_mask = (u ** 2 + v ** 2) < 1.0

    R_safe = np.maximum(R, 1e-10)
    J_phi = np.zeros_like(R, dtype=np.float64)
    J_phi[plasma_mask] = (
        8.0 * Psi_0 * R_safe[plasma_mask] / (eps_s * R0 ** 2) ** 2
        + 2.0 * Psi_0 / (R_safe[plasma_mask] * ka ** 2)
    )
    return J_phi


# ── Config builder ───────────────────────────────────────────────────


def build_config(case: TokamakCase) -> dict[str, Any]:
    """Build a FusionKernel-compatible config dict for a TokamakCase."""
    R_min = case.R0 - 2.0 * case.a
    R_max = case.R0 + 2.0 * case.a
    Z_half = 2.0 * case.kappa * case.a
    Z_min = -Z_half
    Z_max = Z_half

    # Simple PF coil set — 4 coils above / below midplane
    coil_R = case.R0 + 1.5 * case.a
    coil_Z = 1.2 * case.kappa * case.a
    coils = [
        {"name": "PF1", "r": coil_R,            "z":  coil_Z, "current":  2.0},
        {"name": "PF2", "r": coil_R,            "z": -coil_Z, "current":  2.0},
        {"name": "PF3", "r": case.R0 - 0.5 * case.a, "z":  coil_Z * 1.1, "current": -1.0},
        {"name": "PF4", "r": case.R0 - 0.5 * case.a, "z": -coil_Z * 1.1, "current": -1.0},
        {"name": "CS",  "r": max(R_min + 0.05, 0.3),  "z": 0.0,   "current": 0.15},
    ]

    return {
        "reactor_name": case.name,
        "grid_resolution": [case.NR, case.NZ],
        "dimensions": {
            "R_min": round(R_min, 4),
            "R_max": round(R_max, 4),
            "Z_min": round(Z_min, 4),
            "Z_max": round(Z_max, 4),
        },
        "physics": {
            "plasma_current_target": case.Ip,
            "vacuum_permeability": 1.0,
        },
        "coils": coils,
        "solver": {
            "max_iterations": 500,
            "convergence_threshold": 1e-4,
            "relaxation_factor": 0.12,
            "solver_method": "sor",
            "sor_omega": 1.6,
        },
    }


# ── Our solver wrapper ───────────────────────────────────────────────


def run_our_solver(case: TokamakCase) -> dict[str, Any]:
    """Solve with our FusionKernel and return psi, axis position, q-proxy.

    Returns
    -------
    dict with keys: psi, R, Z, R_axis, Z_axis, q_proxy, converged, residual
    """
    from scpn_fusion.core.fusion_kernel import FusionKernel

    cfg = build_config(case)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir=str(ROOT)
    ) as f:
        json.dump(cfg, f)
        config_path = f.name

    try:
        fk = FusionKernel(config_path)
        result = fk.solve_equilibrium()
    finally:
        Path(config_path).unlink(missing_ok=True)

    # Find magnetic axis
    idx_max = int(np.argmax(fk.Psi))
    iz_ax, ir_ax = np.unravel_index(idx_max, fk.Psi.shape)
    R_axis = float(fk.R[ir_ax])
    Z_axis = float(fk.Z[iz_ax])

    # Simple q-profile proxy: q ~ r B_phi / (R B_theta)
    # Use a radial slice through the midplane as a 1-D q estimate
    iz_mid = fk.NZ // 2
    psi_mid = fk.Psi[iz_mid, :]
    psi_axis = float(psi_mid[ir_ax])
    psi_edge = float(psi_mid[0])
    psi_range_val = abs(psi_axis - psi_edge)
    if psi_range_val < 1e-12:
        psi_range_val = 1.0

    psi_norm = (psi_mid - psi_edge) / psi_range_val
    q_proxy = np.clip(1.0 / (np.abs(psi_norm) + 0.01), 0.5, 10.0)

    return {
        "psi": fk.Psi,
        "R": fk.R,
        "Z": fk.Z,
        "RR": fk.RR,
        "ZZ": fk.ZZ,
        "R_axis": R_axis,
        "Z_axis": Z_axis,
        "q_proxy": q_proxy,
        "converged": result["converged"],
        "residual": result["residual"],
    }


# ── FreeGS comparison (only when installed) ──────────────────────────


def run_freegs_case(case: TokamakCase) -> dict[str, Any]:
    """Set up and solve a FreeGS equilibrium for the given case.

    Returns dict with: psi, R, Z, R_axis, Z_axis, q_proxy
    """
    import freegs  # type: ignore[import-untyped]

    # Build a simple tokamak with wall matching our case
    tokamak = freegs.machine.TestTokamak()

    # Create equilibrium on matching grid
    R_min = case.R0 - 2.0 * case.a
    R_max = case.R0 + 2.0 * case.a
    Z_half = 2.0 * case.kappa * case.a

    eq = freegs.Equilibrium(
        tokamak=tokamak,
        Rmin=max(R_min, 0.1),
        Rmax=R_max,
        Zmin=-Z_half,
        Zmax=Z_half,
        nx=case.NR,
        ny=case.NZ,
    )

    # Profiles
    profiles = freegs.jtor.ConstrainPaxisIp(
        1e3,       # Axis pressure [Pa] (simplified)
        case.Ip * 1e6,  # Plasma current [A]
        case.R0,
    )

    # Boundary constraint
    xpoints = [(case.R0 - 0.8 * case.a, -case.kappa * case.a * 0.9)]
    isoflux = [
        (case.R0 + case.a, 0.0, xpoints[0]),
        (case.R0, case.kappa * case.a, xpoints[0]),
    ]

    constrain = freegs.control.constrain(xpoints=xpoints, isoflux=isoflux)

    # Solve
    freegs.solve(eq, profiles, constrain, maxits=50, atol=1e-3, rtol=1e-2)

    # Extract results
    psi = eq.psi()
    R = eq.R
    Z = eq.Z
    R_axis = float(eq.Rmagnetic if hasattr(eq, "Rmagnetic") else case.R0)
    Z_axis = float(eq.Zmagnetic if hasattr(eq, "Zmagnetic") else 0.0)

    # q-proxy: midplane radial psi slice
    iz_mid = len(Z) // 2
    psi_mid = psi[iz_mid, :] if psi.ndim == 2 else psi[:, iz_mid]
    psi_range_val = float(np.max(np.abs(psi_mid)))
    if psi_range_val < 1e-12:
        psi_range_val = 1.0
    psi_norm = psi_mid / psi_range_val
    q_proxy = np.clip(1.0 / (np.abs(psi_norm) + 0.01), 0.5, 10.0)

    return {
        "psi": psi,
        "R": R,
        "Z": Z,
        "R_axis": R_axis,
        "Z_axis": Z_axis,
        "q_proxy": q_proxy,
    }


# ── Solov'ev comparison (always available) ───────────────────────────


def run_solovev_case(case: TokamakCase) -> dict[str, Any]:
    """Build the Solov'ev analytic solution on the same grid as our solver.

    Returns dict with: psi, R, Z, R_axis, Z_axis, q_proxy
    """
    cfg = build_config(case)
    dims = cfg["dimensions"]
    R_1d = np.linspace(dims["R_min"], dims["R_max"], case.NR)
    Z_1d = np.linspace(dims["Z_min"], dims["Z_max"], case.NZ)
    RR, ZZ = np.meshgrid(R_1d, Z_1d)

    psi = solovev_psi(RR, ZZ, case.R0, case.a, case.kappa)

    # Magnetic axis: maximum of Psi (inside plasma)
    idx_max = int(np.argmax(psi))
    iz_ax, ir_ax = np.unravel_index(idx_max, psi.shape)
    R_axis = float(R_1d[ir_ax])
    Z_axis = float(Z_1d[iz_ax])

    # q-proxy (midplane)
    iz_mid = case.NZ // 2
    psi_mid = psi[iz_mid, :]
    psi_range_val = float(np.max(np.abs(psi_mid)))
    if psi_range_val < 1e-12:
        psi_range_val = 1.0
    psi_norm = psi_mid / psi_range_val
    q_proxy = np.clip(1.0 / (np.abs(psi_norm) + 0.01), 0.5, 10.0)

    return {
        "psi": psi,
        "R": R_1d,
        "Z": Z_1d,
        "RR": RR,
        "ZZ": ZZ,
        "R_axis": R_axis,
        "Z_axis": Z_axis,
        "q_proxy": q_proxy,
    }


# ── Case-level comparison ────────────────────────────────────────────


def compare_case(
    case: TokamakCase,
    *,
    use_freegs: bool = False,
) -> dict[str, Any]:
    """Run one benchmark case and return metrics.

    Parameters
    ----------
    case : TokamakCase
    use_freegs : bool
        If True, compare against FreeGS; otherwise against Solov'ev analytic.

    Returns
    -------
    dict with name, psi_nrmse, q_profile_nrmse, axis_error_m, passes, mode
    """
    print(f"  [{case.name}] Running our solver...")
    our = run_our_solver(case)

    if use_freegs:
        print(f"  [{case.name}] Running FreeGS...")
        ref = run_freegs_case(case)
        mode = "freegs"
    else:
        print(f"  [{case.name}] Computing Solov'ev analytic reference...")
        ref = run_solovev_case(case)
        mode = "solovev"

    # ── Psi NRMSE ────────────────────────────────────────────────────
    # Interpolate onto common grid if shapes differ
    our_psi = our["psi"]
    ref_psi = ref["psi"]

    if our_psi.shape != ref_psi.shape:
        from scipy.interpolate import RegularGridInterpolator

        interp = RegularGridInterpolator(
            (ref["Z"] if ref["Z"].ndim == 1 else ref["Z"][:, 0],
             ref["R"] if ref["R"].ndim == 1 else ref["R"][0, :]),
            ref_psi,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        pts = np.stack([our["ZZ"].ravel(), our["RR"].ravel()], axis=-1)
        ref_psi_interp = interp(pts).reshape(our_psi.shape)
    else:
        ref_psi_interp = ref_psi

    psi_nrmse = nrmse(ref_psi_interp, our_psi)

    # ── q-profile NRMSE ──────────────────────────────────────────────
    our_q = our["q_proxy"]
    ref_q = ref["q_proxy"]
    min_len = min(len(our_q), len(ref_q))
    if min_len > 0:
        # Resample to same length via linear interpolation
        x_our = np.linspace(0, 1, len(our_q))
        x_ref = np.linspace(0, 1, len(ref_q))
        x_common = np.linspace(0, 1, min(min_len, 64))
        our_q_interp = np.interp(x_common, x_our, our_q)
        ref_q_interp = np.interp(x_common, x_ref, ref_q)
        q_nrmse = nrmse(ref_q_interp, our_q_interp)
    else:
        q_nrmse = float("nan")

    # ── Axis position error ──────────────────────────────────────────
    axis_err = float(np.sqrt(
        (our["R_axis"] - ref["R_axis"]) ** 2
        + (our["Z_axis"] - ref["Z_axis"]) ** 2
    ))

    passes = bool(psi_nrmse < PSI_NRMSE_THRESHOLD)

    return {
        "name": case.name,
        "mode": mode,
        "psi_nrmse": round(float(psi_nrmse), 6),
        "q_profile_nrmse": round(float(q_nrmse), 6),
        "axis_error_m": round(axis_err, 6),
        "our_converged": our["converged"],
        "our_residual": round(float(our["residual"]), 8),
        "passes": passes,
    }


# ── Campaign runner ──────────────────────────────────────────────────


def run_benchmark(*, force_solovev: bool = False) -> dict[str, Any]:
    """Run all benchmark cases and return the full report dict.

    Parameters
    ----------
    force_solovev : bool
        If True, skip FreeGS even if available and use Solov'ev only.

    Returns
    -------
    dict  — JSON-serialisable benchmark report.
    """
    use_freegs = HAS_FREEGS and not force_solovev
    mode_label = "freegs" if use_freegs else "solovev_analytic"

    if use_freegs:
        print("FreeGS detected -- running numerical comparison.")
    else:
        print("FreeGS not available -- using Solov'ev analytic reference.")

    t0 = time.perf_counter()
    case_results: list[dict[str, Any]] = []

    for case in CASES:
        try:
            result = compare_case(case, use_freegs=use_freegs)
        except Exception as exc:
            result = {
                "name": case.name,
                "mode": mode_label,
                "psi_nrmse": float("nan"),
                "q_profile_nrmse": float("nan"),
                "axis_error_m": float("nan"),
                "error": str(exc),
                "passes": False,
            }
        case_results.append(result)

    psi_nrmses = [
        r["psi_nrmse"]
        for r in case_results
        if np.isfinite(r["psi_nrmse"])
    ]
    overall_psi_nrmse = float(np.mean(psi_nrmses)) if psi_nrmses else float("nan")
    overall_passes = all(r["passes"] for r in case_results)
    runtime = time.perf_counter() - t0

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode_label,
        "psi_nrmse_threshold": PSI_NRMSE_THRESHOLD,
        "cases": case_results,
        "overall_psi_nrmse": round(overall_psi_nrmse, 6),
        "passes": overall_passes,
        "runtime_s": round(runtime, 3),
    }
    return report


# ── Solov'ev-only sub-benchmark (always available) ───────────────────


def run_solovev_benchmark() -> dict[str, Any]:
    """Convenience wrapper that always uses Solov'ev (no FreeGS)."""
    return run_benchmark(force_solovev=True)


# ── Main ─────────────────────────────────────────────────────────────


def main() -> int:
    print("=" * 64)
    print("  SCPN Fusion Core -- FreeGS / Solov'ev Blind Benchmark")
    print("=" * 64)
    print()

    report = run_benchmark()

    # Write artifact
    artifacts_dir = ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts_dir / "freegs_benchmark.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nArtifact written: {out_path}")

    # Summary
    print()
    for c in report["cases"]:
        status = "PASS" if c["passes"] else "FAIL"
        err = c.get("error")
        if err:
            print(f"  {c['name']:24s}  {status}  (error: {err})")
        else:
            print(
                f"  {c['name']:24s}  {status}  "
                f"psi_nrmse={c['psi_nrmse']:.4f}  "
                f"q_nrmse={c['q_profile_nrmse']:.4f}  "
                f"axis_err={c['axis_error_m']:.4f} m"
            )

    print()
    print(f"Overall Psi NRMSE: {report['overall_psi_nrmse']:.4f}")
    print(f"Overall:           {'PASS' if report['passes'] else 'FAIL'}")
    print(f"Mode:              {report['mode']}")
    print(f"Runtime:           {report['runtime_s']:.2f} s")
    print("=" * 64)

    return 0 if report["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
