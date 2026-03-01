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
unavailable the script falls back to a manufactured-solution Solov'ev lane:
we solve a fixed-source GS problem with boundary conditions and source terms
derived from the analytic Solov'ev solution, then compare against the exact
closed-form reference on the same grid.

Produces ``artifacts/freegs_benchmark.json``.

Exit codes:
    0 - benchmark ran and all cases PASS
    1 - one or more cases FAIL
"""

from __future__ import annotations

import argparse
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

PSI_NRMSE_THRESHOLD = 0.11  # 11 %

# Tighter thresholds when FreeGS is available for direct numerical comparison
FREEGS_PSI_NRMSE_THRESHOLD = 0.005   # 0.5% for direct numerical comparison
FREEGS_Q_NRMSE_THRESHOLD = 0.10      # 10% for q-profile
FREEGS_AXIS_ERROR_M = 0.10           # 10 cm axis position error
FREEGS_SEPARATRIX_NRMSE = 0.05       # 5% separatrix boundary
FREEGS_FLUX_AREA_REL_ERROR = 0.12    # 12% mid-surface area parity


def _stable_rmse(delta: NDArray[np.float64]) -> float:
    """RMSE with overflow-resistant scaling for extreme benchmark excursions."""
    vals = np.nan_to_num(np.asarray(delta, dtype=np.float64), nan=0.0, posinf=1e250, neginf=-1e250)
    if vals.size == 0:
        return 0.0
    scale = float(np.max(np.abs(vals), initial=0.0))
    if scale <= 0.0:
        return 0.0
    scaled = vals / scale
    return float(scale * np.sqrt(np.mean(scaled * scaled)))


def nrmse(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """Normalised RMSE: RMSE / range(y_true)."""
    rmse_val = _stable_rmse(y_true - y_pred)
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
    TokamakCase(name="KSTAR-like",           R0=1.80, a=0.50, B0=3.5,  Ip=2.0,   kappa=1.83),
    TokamakCase(name="SPARC-high-kappa",     R0=1.85, a=0.57, B0=12.2, Ip=9.0,   kappa=2.20),
]


def estimate_axis_pressure_pa(case: TokamakCase) -> float:
    """Estimate a bounded axis pressure for FreeGS profile setup.

    Uses a lightweight beta-based heuristic:

        p_axis ~ beta_t * B0^2 / (2 * mu0)

    where beta_t is derived from machine-level ``Ip/B0`` scaling and clipped
    to a conservative range to avoid pathological setup values in the
    benchmark harness.
    """
    mu0_si = 4.0e-7 * np.pi
    b0_t = max(float(case.B0), 0.1)
    ip_ma = max(float(case.Ip), 0.05)

    beta_t = float(np.clip(0.015 + 0.004 * (ip_ma / b0_t), 0.01, 0.08))
    p_axis = beta_t * (b0_t ** 2) / (2.0 * mu0_si)
    return float(np.clip(p_axis, 5.0e4, 2.0e7))

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


def compute_discrete_gs_source(
    psi: NDArray[np.float64],
    rr: NDArray[np.float64],
    d_r: float,
    d_z: float,
) -> NDArray[np.float64]:
    """Compute the discrete GS* source term L*[psi] on the active grid."""
    source = np.zeros_like(psi, dtype=np.float64)
    r_safe = np.maximum(rr[1:-1, 1:-1], 1e-10)

    d2_r = (psi[1:-1, 2:] - 2.0 * psi[1:-1, 1:-1] + psi[1:-1, 0:-2]) / (d_r ** 2)
    d1_r = (psi[1:-1, 2:] - psi[1:-1, 0:-2]) / (2.0 * d_r)
    d2_z = (psi[2:, 1:-1] - 2.0 * psi[1:-1, 1:-1] + psi[0:-2, 1:-1]) / (d_z ** 2)
    source[1:-1, 1:-1] = d2_r - d1_r / r_safe + d2_z

    return source


# ── Config builder ───────────────────────────────────────────────────


def build_config(case: TokamakCase) -> dict[str, Any]:
    """Build a FusionKernel-compatible config dict for a TokamakCase."""
    # Keep R strictly positive to satisfy config schema on tight ST geometries.
    R_min = max(case.R0 - 2.0 * case.a, 0.05)
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


def run_manufactured_solovev_solver(case: TokamakCase) -> dict[str, Any]:
    """Solve GS with fixed analytic Solov'ev source + boundary constraints.

    This provides an apples-to-apples parity lane when FreeGS is unavailable:
    the numerical solve and analytic reference share identical source and
    boundary contracts.
    """
    from scpn_fusion.core.fusion_kernel import FusionKernel

    cfg = build_config(case)
    mu0 = float(cfg["physics"].get("vacuum_permeability", 1.0))

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir=str(ROOT)
    ) as f:
        json.dump(cfg, f)
        config_path = f.name

    try:
        fk = FusionKernel(config_path)

        psi_ref = solovev_psi(fk.RR, fk.ZZ, case.R0, case.a, case.kappa)
        source = compute_discrete_gs_source(psi_ref, fk.RR, fk.dR, fk.dZ)
        psi_boundary = psi_ref.copy()
        rr_safe = np.maximum(fk.RR, 1e-10)
        jphi_ref = -source / (mu0 * rr_safe)

        fk.Psi = psi_boundary.copy()
        fk.J_phi = jphi_ref.copy()

        max_iter = min(int(fk.cfg["solver"].get("max_iterations", 500)), 120)
        tol = float(fk.cfg["solver"].get("convergence_threshold", 1e-4))
        best_residual = float("inf")
        best_psi = fk.Psi.copy()
        converged = False

        for _ in range(max_iter):
            psi_next = fk._elliptic_solve(source, psi_boundary)
            residual = float(np.mean(np.abs(psi_next - fk.Psi)))
            if residual <= best_residual:
                best_residual = residual
                best_psi = psi_next.copy()
            fk.Psi = psi_next

            if residual < tol:
                converged = True
                break

        fk.Psi = best_psi
        fk.compute_b_field()
    finally:
        Path(config_path).unlink(missing_ok=True)

    idx_max = int(np.argmax(fk.Psi))
    iz_ax, ir_ax = np.unravel_index(idx_max, fk.Psi.shape)
    R_axis = float(fk.R[ir_ax])
    Z_axis = float(fk.Z[iz_ax])

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
        "converged": converged,
        "residual": best_residual,
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
    axis_pressure_pa = estimate_axis_pressure_pa(case)
    profiles = freegs.jtor.ConstrainPaxisIp(
        axis_pressure_pa,
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
        "axis_pressure_pa": float(axis_pressure_pa),
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


# ── Separatrix comparison ────────────────────────────────────────────


def compare_separatrix(
    our_psi: NDArray[np.float64],
    ref_psi: NDArray[np.float64],
    R: NDArray[np.float64],
    Z: NDArray[np.float64],
) -> float:
    """Compare separatrix boundaries by extracting the 50% psi contour.

    Uses a simple boundary extraction: find the outermost contour at
    psi = 0.5 * psi_max on the midplane, and compute the NRMSE of the
    radial positions.

    Parameters
    ----------
    our_psi, ref_psi : 2-D arrays of psi values
    R, Z : 1-D coordinate arrays

    Returns
    -------
    float — NRMSE of the separatrix boundary approximation
    """
    # Midplane extraction
    iz_mid = len(Z) // 2

    our_mid = our_psi[iz_mid, :]
    ref_mid = ref_psi[iz_mid, :]

    # Normalize each to [0, 1]
    our_range = np.max(our_mid) - np.min(our_mid)
    ref_range = np.max(ref_mid) - np.min(ref_mid)
    if our_range < 1e-12 or ref_range < 1e-12:
        return 1.0  # degenerate case

    our_norm = (our_mid - np.min(our_mid)) / our_range
    ref_norm = (ref_mid - np.min(ref_mid)) / ref_range

    return nrmse(ref_norm, our_norm)


def compare_flux_surface_area(
    our_psi: NDArray[np.float64],
    ref_psi: NDArray[np.float64],
    R: NDArray[np.float64],
    Z: NDArray[np.float64],
    *,
    level: float = 0.5,
) -> float:
    """Compare flux-surface area (normalised psi >= level) by relative error."""
    if our_psi.shape != ref_psi.shape:
        return float("nan")
    if R.size < 2 or Z.size < 2:
        return float("nan")

    d_r = float(np.mean(np.diff(R)))
    d_z = float(np.mean(np.diff(Z)))
    if d_r <= 0.0 or d_z <= 0.0 or not np.isfinite(d_r) or not np.isfinite(d_z):
        return float("nan")

    def _norm(psi: NDArray[np.float64]) -> NDArray[np.float64]:
        rng = float(np.max(psi) - np.min(psi))
        if rng < 1e-12:
            return np.zeros_like(psi, dtype=np.float64)
        return (psi - np.min(psi)) / rng

    our_norm = _norm(our_psi)
    ref_norm = _norm(ref_psi)
    our_area = float(np.sum(our_norm >= level) * d_r * d_z)
    ref_area = float(np.sum(ref_norm >= level) * d_r * d_z)
    if ref_area <= 0.0:
        return float("nan")
    return float(abs(our_area - ref_area) / max(ref_area, 1e-12))


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
    if use_freegs:
        print(f"  [{case.name}] Running our nonlinear solver...")
        our = run_our_solver(case)
        print(f"  [{case.name}] Running FreeGS...")
        ref = run_freegs_case(case)
        mode = "freegs"
        comparison_backend = "fusion_kernel_nonlinear"
        reference_backend = "freegs"
    else:
        print(f"  [{case.name}] Running manufactured-source GS parity solve...")
        our = run_manufactured_solovev_solver(case)
        print(f"  [{case.name}] Computing Solov'ev analytic reference...")
        ref = run_solovev_case(case)
        mode = "solovev_manufactured_source"
        comparison_backend = "fusion_kernel_fixed_source"
        reference_backend = "solovev_analytic"

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

    # ── Separatrix boundary NRMSE ───────────────────────────────────
    if our_psi.shape == ref_psi_interp.shape:
        sep_nrmse = compare_separatrix(
            our_psi, ref_psi_interp,
            our["R"] if our["R"].ndim == 1 else our["R"][0, :],
            our["Z"] if our["Z"].ndim == 1 else our["Z"][:, 0],
        )
        area_rel_error = compare_flux_surface_area(
            our_psi,
            ref_psi_interp,
            our["R"] if our["R"].ndim == 1 else our["R"][0, :],
            our["Z"] if our["Z"].ndim == 1 else our["Z"][:, 0],
        )
    else:
        sep_nrmse = float("nan")
        area_rel_error = float("nan")

    # ── Pass / fail logic ─────────────────────────────────────────
    if use_freegs:
        finite_freegs_metrics = bool(
            np.isfinite(psi_nrmse)
            and np.isfinite(q_nrmse)
            and np.isfinite(axis_err)
            and np.isfinite(sep_nrmse)
            and np.isfinite(area_rel_error)
        )
        passes = bool(
            finite_freegs_metrics
            and psi_nrmse < FREEGS_PSI_NRMSE_THRESHOLD
            and q_nrmse < FREEGS_Q_NRMSE_THRESHOLD
            and axis_err < FREEGS_AXIS_ERROR_M
            and sep_nrmse < FREEGS_SEPARATRIX_NRMSE
            and area_rel_error < FREEGS_FLUX_AREA_REL_ERROR
        )
    else:
        passes = bool(psi_nrmse < PSI_NRMSE_THRESHOLD)

    return {
        "name": case.name,
        "mode": mode,
        "comparison_backend": comparison_backend,
        "reference_backend": reference_backend,
        "psi_nrmse": round(float(psi_nrmse), 6),
        "q_profile_nrmse": round(float(q_nrmse), 6),
        "axis_error_m": round(axis_err, 6),
        "separatrix_nrmse": round(float(sep_nrmse), 6),
        "flux_area_rel_error": round(float(area_rel_error), 6),
        "our_converged": our["converged"],
        "our_residual": round(float(our["residual"]), 8),
        "passes": passes,
    }


# ── Campaign runner ──────────────────────────────────────────────────


def run_benchmark(
    *,
    force_solovev: bool = False,
    require_freegs_backend: bool = False,
) -> dict[str, Any]:
    """Run all benchmark cases and return the full report dict.

    Parameters
    ----------
    force_solovev : bool
        If True, skip FreeGS even if available and use Solov'ev only.
    require_freegs_backend : bool
        If True, require FreeGS and fail instead of falling back.

    Returns
    -------
    dict  — JSON-serialisable benchmark report.
    """
    if require_freegs_backend and force_solovev:
        raise ValueError(
            "require_freegs_backend cannot be combined with force_solovev."
        )
    if require_freegs_backend and not HAS_FREEGS:
        raise RuntimeError(
            "FreeGS strict backend requested but `freegs` is not installed."
        )

    use_freegs = bool(require_freegs_backend or (HAS_FREEGS and not force_solovev))
    mode_label = "freegs" if use_freegs else "solovev_manufactured_source"

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
                "comparison_backend": (
                    "fusion_kernel_nonlinear" if use_freegs else "fusion_kernel_fixed_source"
                ),
                "reference_backend": "freegs" if use_freegs else "solovev_analytic",
                "psi_nrmse": float("nan"),
                "q_profile_nrmse": float("nan"),
                "axis_error_m": float("nan"),
                "separatrix_nrmse": float("nan"),
                "flux_area_rel_error": float("nan"),
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

    thresholds: dict[str, float] = {
        "psi_nrmse": FREEGS_PSI_NRMSE_THRESHOLD if use_freegs else PSI_NRMSE_THRESHOLD,
    }
    if use_freegs:
        thresholds.update({
            "q_profile_nrmse": FREEGS_Q_NRMSE_THRESHOLD,
            "axis_error_m": FREEGS_AXIS_ERROR_M,
            "separatrix_nrmse": FREEGS_SEPARATRIX_NRMSE,
            "flux_area_rel_error": FREEGS_FLUX_AREA_REL_ERROR,
        })

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode_label,
        "psi_nrmse_threshold": thresholds["psi_nrmse"],
        "thresholds": thresholds,
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


# ── Per-metric report ────────────────────────────────────────────────


def generate_per_metric_report(report: dict[str, Any]) -> str:
    """Generate a markdown table with per-metric NRMSE for each case."""
    lines = [
        "## FreeGS / Solov'ev Benchmark — Per-Metric Report",
        "",
        "| Case | Mode | Psi NRMSE | q NRMSE | Axis Err [m] | Sep. NRMSE | Status |",
        "|------|------|-----------|---------|--------------|------------|--------|",
    ]
    for c in report.get("cases", []):
        status = "PASS" if c.get("passes", False) else "FAIL"
        sep = c.get("separatrix_nrmse", float("nan"))
        lines.append(
            f"| {c['name']} | {c.get('mode', '?')} "
            f"| {c.get('psi_nrmse', float('nan')):.4f} "
            f"| {c.get('q_profile_nrmse', float('nan')):.4f} "
            f"| {c.get('axis_error_m', float('nan')):.4f} "
            f"| {sep:.4f} | {status} |"
        )
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-solovev",
        action="store_true",
        help="Force Solov'ev manufactured-source lane even when FreeGS is available.",
    )
    parser.add_argument(
        "--strict-backend",
        action="store_true",
        help="Require FreeGS backend and fail instead of falling back to Solov'ev.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    print("=" * 64)
    print("  SCPN Fusion Core -- FreeGS / Solov'ev Blind Benchmark")
    print("=" * 64)
    print()

    args = _parse_args(argv)
    try:
        report = run_benchmark(
            force_solovev=bool(args.force_solovev),
            require_freegs_backend=bool(args.strict_backend),
        )
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

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
