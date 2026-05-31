# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Point-wise ψ(R,Z) RMSE Validation
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
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.core.eqdsk import GEqdsk, read_geqdsk

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas"
SPARC_DIR = ROOT / "validation" / "reference_data" / "sparc"
REFERENCE_DATA_DIR = ROOT / "validation" / "reference_data"
EFIT_NRMSE_BENCHMARK_SCHEMA_VERSION = "efit-nrmse-benchmark.v2"
EFIT_BENCHMARK_SCOPE = "profile_source_fixed_boundary_reconstruction"
EFIT_BENCHMARK_CONTRACT = (
    "Strict raw GEQDSK profile-source fixed-boundary reconstruction gate with separate "
    "operator-source solver and named-adapter profile-source diagnostics."
)
RAW_PROFILE_SOLVER_MODE = "raw_geqdsk_profile_source_fixed_boundary"
OPERATOR_SOURCE_SOLVER_MODE = "operator_source_fixed_boundary_delta_star_psi"
ADAPTED_PROFILE_SOLVER_MODE = "adapted_geqdsk_profile_source_fixed_boundary"
OPERATOR_SOURCE_RMSE_THRESHOLD = 1e-6
OPERATOR_CURRENT_CLOSURE_THRESHOLD = 0.05
SOURCE_CONVENTION_DISTANCE_THRESHOLD = 0.15
SOURCE_CONVENTION_ADAPTER_RESIDUAL_THRESHOLD = 0.15
ADAPTED_PROFILE_RMSE_THRESHOLD = 0.05
EFIT_BENCHMARK_MACHINE_PROVENANCE = {
    "sparc": "real_public_design_reference",
    "diiid": "synthetic_proxy_reference",
    "jet": "synthetic_proxy_reference",
}
EFIT_REFERENCE_MACHINE_CONTRACTS = {
    "sparc": {
        "reference_class": "public_efit_reference",
        "reference_role": "gate",
        "reference_dataset_id": "sparc_reference_bundle",
        "reference_expected_contract": "public_efit_geqdsk_operator_and_profile_source_contract",
        "reference_expected_convention": "raw_canonical_strict_unless_named_adapter_passes",
    },
    "diiid": {
        "reference_class": "synthetic_proxy_reference",
        "reference_role": "diagnostic",
        "reference_dataset_id": "diiid_reference_equilibria",
        "reference_expected_contract": "synthetic_solovev_geqdsk_diagnostic_contract",
        "reference_expected_convention": "synthetic_proxy_profile_source",
    },
    "jet": {
        "reference_class": "synthetic_proxy_reference",
        "reference_role": "diagnostic",
        "reference_dataset_id": "jet_reference_equilibria",
        "reference_expected_contract": "synthetic_solovev_geqdsk_diagnostic_contract",
        "reference_expected_convention": "synthetic_proxy_profile_source",
    },
}
EFIT_REFERENCE_CASE_CONVENTIONS = {
    "sparc/sparc_1305.eqdsk": "scaled_by_2pi_adapter_gate",
    "sparc/sparc_1310.eqdsk": "scaled_by_2pi_adapter_gate",
    "sparc/sparc_1315.eqdsk": "scaled_by_2pi_adapter_gate",
    "sparc/sparc_1349.eqdsk": "scaled_by_2pi_adapter_gate",
}


# ── Data containers ──────────────────────────────────────────────────


@dataclass
class PsiRMSEResult:
    """Per-file ψ RMSE metrics."""

    file: str
    grid: str  # e.g. "129x129"

    # GS residual (how well reference satisfies GS equation)
    gs_residual_l2: float  # ||residual||_2 / ||source||_2
    gs_residual_max: float  # max |residual|  (Wb/rad·m⁻²)

    # Manufactured-source solve
    psi_rmse_wb: float  # raw RMSE in Wb/rad
    psi_rmse_norm: float  # RMSE in normalised ψ_N
    psi_rmse_plasma_wb: float  # RMSE restricted to plasma (ψ_N ∈ [0,1))
    psi_max_error_wb: float  # max |Δψ| over full domain
    psi_relative_l2: float  # ||Δψ||_2 / ||ψ_ref||_2

    # Solver metadata
    sor_iterations: int
    sor_residual: float
    solve_time_ms: float
    operator_source_psi_rmse_norm: float = float("nan")
    operator_source_sor_iterations: int = 0
    operator_source_sor_residual: float = float("nan")
    source_consistency_class: str = "not_evaluated"
    operator_source_norm: float = float("nan")
    profile_source_norm: float = float("nan")
    source_residual_l2: float = float("nan")
    source_correlation: float = float("nan")
    source_best_fit_scale: float = float("nan")
    source_best_fit_offset: float = float("nan")
    source_best_fit_relative_l2: float = float("nan")
    source_best_fit_convention: str = "not_evaluated"
    source_convention_adapter: str = "not_evaluated"
    source_convention_adapter_residual_l2: float = float("nan")
    source_convention_adapter_pass: bool = False
    adapted_profile_psi_rmse_norm: float = float("nan")
    adapted_profile_sor_iterations: int = 0
    adapted_profile_sor_residual: float = float("nan")
    adapted_profile_axis_error_m: float = float("nan")
    adapted_profile_boundary_containment_fraction: float = float("nan")
    adapted_profile_boundary_psi_rmse_norm: float = float("nan")
    q_profile_sanity_pass: bool = False
    q_profile_finite_fraction: float = float("nan")
    q_profile_min_abs: float = float("nan")
    q_profile_sign_changes: int = 0
    q_profile_monotonic_fraction: float = float("nan")
    adapted_profile_pass: bool = False
    plasma_mask_fraction: float = float("nan")
    pressure_source_norm: float = float("nan")
    ffprime_source_norm: float = float("nan")
    total_source_norm: float = float("nan")
    pressure_source_sum: float = float("nan")
    ffprime_source_sum: float = float("nan")
    total_source_sum: float = float("nan")
    pressure_source_fraction: float = float("nan")
    ffprime_source_fraction: float = float("nan")
    source_plasma_residual_l2: float = float("nan")
    source_vacuum_residual_l2: float = float("nan")
    source_plasma_operator_norm: float = float("nan")
    source_vacuum_operator_norm: float = float("nan")
    source_plasma_point_count: float = float("nan")
    source_vacuum_point_count: float = float("nan")
    best_source_candidate: str = ""
    best_source_candidate_residual_l2: float = float("nan")
    profile_source_candidate_rank: int = 0
    best_operator_candidate: str = ""
    best_operator_candidate_residual_l2: float = float("nan")
    delta_star_psi_candidate_rank: int = 0
    declared_toroidal_current_A: float = float("nan")
    operator_toroidal_current_A: float = float("nan")
    profile_toroidal_current_A: float = float("nan")
    operator_current_relative_error: float = float("nan")
    profile_current_relative_error: float = float("nan")
    operator_current_closure_pass: bool = False


@dataclass
class PsiRMSESummary:
    """Aggregate over all files."""

    benchmark_id: str
    benchmark_scope: str
    benchmark_contract: str
    solver_mode: str
    count: int
    mean_psi_rmse_norm: float
    mean_psi_relative_l2: float
    mean_gs_residual_l2: float
    worst_psi_rmse_norm: float
    worst_file: str
    rows: list[dict[str, Any]]


@dataclass
class EfitNRMSEBenchmarkGate:
    """Strict 10+ file ψ_N RMSE benchmark gate over EFIT/GEQDSK references."""

    schema_version: str
    benchmark_id: str
    benchmark_scope: str
    benchmark_contract: str
    raw_profile_solver_mode: str
    operator_source_solver_mode: str
    adapted_profile_solver_mode: str
    count: int
    min_required_files: int
    threshold: float
    pass_count: int
    gate_row_count: int
    gate_pass_count: int
    gate_worst_psi_rmse_norm: float
    gate_worst_file: str
    passes: bool
    mean_psi_rmse_norm: float
    worst_psi_rmse_norm: float
    worst_file: str
    count_by_machine: dict[str, int]
    provenance_by_machine: dict[str, str]
    reference_role_counts: dict[str, int]
    reference_class_counts: dict[str, int]
    solver_mode_counts: dict[str, int]
    source_consistency_counts: dict[str, int]
    operator_source_threshold: float
    operator_source_pass_count: int
    gate_operator_source_pass_count: int
    gate_operator_source_worst_psi_rmse_norm: float
    gate_operator_source_worst_file: str
    operator_source_worst_psi_rmse_norm: float
    operator_source_worst_file: str
    source_convention_adapter_threshold: float
    source_convention_adapter_pass_count: int
    gate_source_convention_adapter_pass_count: int
    source_convention_adapter_counts: dict[str, int]
    adapted_profile_threshold: float
    adapted_profile_pass_count: int
    gate_adapted_profile_pass_count: int
    adapted_profile_worst_psi_rmse_norm: float
    adapted_profile_worst_file: str
    worst_source_residual_l2: float
    worst_source_alignment_file: str
    gate_worst_source_residual_l2: float
    gate_worst_source_alignment_file: str
    source_sum_identity_max_abs_error: float
    source_sum_identity_pass: bool
    operator_current_closure_pass_count: int
    gate_operator_current_closure_pass_count: int
    failure_reasons: list[str]
    rows: list[dict[str, Any]]


# ── GS operator ──────────────────────────────────────────────────────


def gs_operator(psi: NDArray, R: NDArray, Z: NDArray) -> NDArray:
    """
    Evaluate the Grad-Shafranov elliptic operator Δ*ψ on interior points.

    Δ*ψ = ∂²ψ/∂R² - (1/R)∂ψ/∂R + ∂²ψ/∂Z²

    Returns array of same shape with boundary values set to zero.
    """
    psi_arr = np.asarray(psi, dtype=np.float64)
    r_arr = np.asarray(R, dtype=np.float64)
    z_arr = np.asarray(Z, dtype=np.float64)

    if psi_arr.ndim != 2:
        raise ValueError("psi must be a 2D array")
    if r_arr.ndim != 1 or z_arr.ndim != 1:
        raise ValueError("R and Z must be 1D arrays")

    nz, nr = psi_arr.shape
    if nz < 3 or nr < 3:
        raise ValueError("psi grid must be at least 3x3")
    if r_arr.size != nr or z_arr.size != nz:
        raise ValueError(
            f"R/Z axis lengths must match psi shape: got R={r_arr.size}, Z={z_arr.size}, "
            f"psi={psi_arr.shape}"
        )

    if not np.all(np.isfinite(psi_arr)):
        raise ValueError("psi must contain only finite values")
    if not np.all(np.isfinite(r_arr)) or not np.all(np.isfinite(z_arr)):
        raise ValueError("R and Z axes must contain only finite values")
    if np.any(np.diff(r_arr) <= 0.0):
        raise ValueError("R axis must be strictly increasing")
    if np.any(np.diff(z_arr) <= 0.0):
        raise ValueError("Z axis must be strictly increasing")

    dR = float(r_arr[1] - r_arr[0])
    dZ = float(z_arr[1] - z_arr[0])
    RR = r_arr[np.newaxis, :]  # (1, nr)

    result = np.zeros_like(psi_arr)

    # Interior finite differences
    d2R = (psi_arr[1:-1, 2:] - 2 * psi_arr[1:-1, 1:-1] + psi_arr[1:-1, :-2]) / dR**2
    dR1 = (psi_arr[1:-1, 2:] - psi_arr[1:-1, :-2]) / (2 * dR)
    d2Z = (psi_arr[2:, 1:-1] - 2 * psi_arr[1:-1, 1:-1] + psi_arr[:-2, 1:-1]) / dZ**2

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
    return compute_source_components(eq)["total_source"]


def interpolate_flux_profile_second_order(
    psi_norm: NDArray,
    profile: NDArray,
) -> NDArray:
    """
    Interpolate a GEQDSK flux profile with local quadratic Lagrange stencils.

    GEQDSK pprime/ffprime profiles are sampled on a uniform normalised-flux
    axis.  This routine is exact for quadratic profile data on that axis and
    uses one-sided quadratic stencils near the magnetic axis and separatrix.
    """
    psi_arr = np.asarray(psi_norm, dtype=np.float64)
    profile_arr = np.asarray(profile, dtype=np.float64)
    if profile_arr.ndim != 1:
        raise ValueError("profile must be 1D")
    if profile_arr.size < 3:
        raise ValueError("profile must contain at least three flux samples")
    if not np.all(np.isfinite(psi_arr)) or not np.all(np.isfinite(profile_arr)):
        raise ValueError("psi_norm and profile must contain only finite values")

    axis = np.linspace(0.0, 1.0, profile_arr.size, dtype=np.float64)
    flat = np.clip(psi_arr, 0.0, 1.0).ravel()
    idx = np.searchsorted(axis, flat, side="right") - 1
    idx = np.clip(idx, 1, profile_arr.size - 2)

    x0 = axis[idx - 1]
    x1 = axis[idx]
    x2 = axis[idx + 1]
    y0 = profile_arr[idx - 1]
    y1 = profile_arr[idx]
    y2 = profile_arr[idx + 1]

    term0 = y0 * (flat - x1) * (flat - x2) / ((x0 - x1) * (x0 - x2))
    term1 = y1 * (flat - x0) * (flat - x2) / ((x1 - x0) * (x1 - x2))
    term2 = y2 * (flat - x0) * (flat - x1) / ((x2 - x0) * (x2 - x1))
    return (term0 + term1 + term2).reshape(psi_arr.shape)


def interpolate_flux_profile_current_conserving(
    psi_norm: NDArray,
    profile: NDArray,
    weights: NDArray,
    mask: NDArray,
) -> NDArray:
    """
    Interpolate a flux profile with local quadratic accuracy while preserving
    its masked weighted integral from the linear GEQDSK profile contract.

    The weighted integral is the discrete current contribution used by the
    Grad-Shafranov source components: p' uses an R weight and FF' uses a 1/R
    weight after division by mu0 R. Preserving that integral prevents the
    higher-order interpolation stencil from adding hidden net-current drift.
    """
    psi_arr = np.asarray(psi_norm, dtype=np.float64)
    profile_arr = np.asarray(profile, dtype=np.float64)
    weights_arr = np.asarray(weights, dtype=np.float64)
    mask_raw = np.asarray(mask)
    if mask_raw.dtype != np.bool_:
        raise ValueError("mask must be boolean")
    mask_arr = mask_raw.astype(bool, copy=False)
    if weights_arr.shape != psi_arr.shape:
        raise ValueError("weights shape must match psi_norm shape")
    if mask_arr.shape != psi_arr.shape:
        raise ValueError("mask shape must match psi_norm shape")
    if not np.all(np.isfinite(weights_arr)):
        raise ValueError("weights must contain only finite values")
    if np.any(weights_arr[mask_arr] < 0.0):
        raise ValueError("weights must be non-negative on the masked domain")

    quadratic = interpolate_flux_profile_second_order(psi_arr, profile_arr)
    if not np.any(mask_arr):
        return quadratic

    linear = np.interp(
        np.clip(psi_arr, 0.0, 1.0).ravel(),
        np.linspace(0.0, 1.0, profile_arr.size, dtype=np.float64),
        profile_arr,
    ).reshape(psi_arr.shape)
    target_integral = float(np.sum(linear[mask_arr] * weights_arr[mask_arr]))
    observed_integral = float(np.sum(quadratic[mask_arr] * weights_arr[mask_arr]))
    scale = max(abs(target_integral), 1.0)
    if abs(target_integral) <= 1.0e-15 and abs(observed_integral) <= 1.0e-15:
        return quadratic
    if abs(observed_integral) <= 1.0e-15 * scale:
        raise ValueError("quadratic profile interpolation has zero weighted integral")
    return quadratic * (target_integral / observed_integral)


def compute_source_components(eq: GEqdsk) -> dict[str, Any]:
    """Split the GEQDSK-derived Grad-Shafranov source into profile components."""
    if eq.nw <= 0 or eq.nh <= 0:
        raise ValueError("eq.nw and eq.nh must be positive")
    if not np.isfinite(eq.simag) or not np.isfinite(eq.sibry):
        raise ValueError("eq.simag and eq.sibry must be finite")

    psirz = np.asarray(eq.psirz, dtype=np.float64)
    pprime = np.asarray(eq.pprime, dtype=np.float64)
    ffprime = np.asarray(eq.ffprime, dtype=np.float64)
    R = np.asarray(eq.r, dtype=np.float64)
    Z = np.asarray(eq.z, dtype=np.float64)

    if psirz.shape != (eq.nh, eq.nw):
        raise ValueError(f"eq.psirz shape must be ({eq.nh}, {eq.nw}), got {psirz.shape}")
    if pprime.shape != (eq.nw,):
        raise ValueError(f"eq.pprime length must be {eq.nw}, got {pprime.shape}")
    if ffprime.shape != (eq.nw,):
        raise ValueError(f"eq.ffprime length must be {eq.nw}, got {ffprime.shape}")
    if R.shape != (eq.nw,) or Z.shape != (eq.nh,):
        raise ValueError(
            f"R/Z axis lengths must match nw/nh ({eq.nw}, {eq.nh}), got R={R.shape}, Z={Z.shape}"
        )

    if not np.all(np.isfinite(psirz)):
        raise ValueError("eq.psirz must contain only finite values")
    if not np.all(np.isfinite(pprime)) or not np.all(np.isfinite(ffprime)):
        raise ValueError("eq.pprime and eq.ffprime must contain only finite values")
    if not np.all(np.isfinite(R)) or not np.all(np.isfinite(Z)):
        raise ValueError("R and Z axes must contain only finite values")
    if np.any(np.diff(R) <= 0.0):
        raise ValueError("R axis must be strictly increasing")
    if np.any(np.diff(Z) <= 0.0):
        raise ValueError("Z axis must be strictly increasing")

    RR, _ = np.meshgrid(R, Z)

    # Normalise reference psi
    denom = eq.sibry - eq.simag
    if abs(denom) < 1e-12:
        zero = np.zeros((eq.nh, eq.nw), dtype=np.float64)
        return {
            "pressure_source": zero.copy(),
            "ffprime_source": zero.copy(),
            "total_source": zero.copy(),
            "plasma_mask": np.zeros((eq.nh, eq.nw), dtype=bool),
            "plasma_mask_fraction": 0.0,
            "pressure_source_norm": 0.0,
            "ffprime_source_norm": 0.0,
            "total_source_norm": 0.0,
            "pressure_source_sum": 0.0,
            "ffprime_source_sum": 0.0,
            "total_source_sum": 0.0,
        }
    psi_n = (psirz - eq.simag) / denom

    psi_n_clipped = np.clip(psi_n, 0.0, 1.0)
    plasma_mask = (psi_n >= 0) & (psi_n < 1.0)
    source_mask = plasma_mask.copy()
    source_mask[[0, -1], :] = False
    source_mask[:, [0, -1]] = False

    # Interpolate 1D profiles onto the 2D grid while preserving the masked
    # current-relevant weighted integral of the established linear profile
    # contract.
    rr_safe = np.maximum(RR, 1.0e-12)
    pprime_2d = interpolate_flux_profile_current_conserving(
        psi_n_clipped, pprime, rr_safe, source_mask
    )
    ffprime_2d = interpolate_flux_profile_current_conserving(
        psi_n_clipped, ffprime, 1.0 / rr_safe, source_mask
    )

    # Zero outside the physical plasma source domain and on Dirichlet boundary
    # rows/columns.
    pprime_2d[~source_mask] = 0.0
    ffprime_2d[~source_mask] = 0.0

    mu0 = 4e-7 * np.pi
    pressure_source = -(mu0 * RR**2 * pprime_2d)
    ffprime_source = -ffprime_2d
    total_source = pressure_source + ffprime_source
    interior = (slice(1, -1), slice(1, -1))
    return {
        "pressure_source": pressure_source,
        "ffprime_source": ffprime_source,
        "total_source": total_source,
        "plasma_mask": plasma_mask,
        "plasma_mask_fraction": float(np.mean(plasma_mask[interior])),
        "pressure_source_norm": float(np.linalg.norm(pressure_source[interior].ravel())),
        "ffprime_source_norm": float(np.linalg.norm(ffprime_source[interior].ravel())),
        "total_source_norm": float(np.linalg.norm(total_source[interior].ravel())),
        "pressure_source_sum": float(np.sum(pressure_source[interior])),
        "ffprime_source_sum": float(np.sum(ffprime_source[interior])),
        "total_source_sum": float(np.sum(total_source[interior])),
    }


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
    source_norm = float(np.linalg.norm(source[1:-1, 1:-1].ravel()))
    if source_norm < 1e-15:
        source_norm = 1.0

    rel_l2 = float(np.linalg.norm(residual.ravel()) / source_norm)
    max_abs = float(np.max(np.abs(residual)))
    return rel_l2, max_abs


def classify_source_scale_convention(scale: float) -> str:
    """Classify a global source scale against common GEQDSK convention factors."""
    if not np.isfinite(scale):
        return "non_finite_scale"
    candidates = {
        "canonical": 1.0,
        "negated": -1.0,
        "scaled_by_2pi": 2.0 * np.pi,
        "scaled_by_minus_2pi": -2.0 * np.pi,
        "scaled_by_inv_2pi": 1.0 / (2.0 * np.pi),
        "scaled_by_minus_inv_2pi": -1.0 / (2.0 * np.pi),
    }
    convention = min(candidates, key=lambda name: abs(scale - candidates[name]))
    relative_distance = abs(scale - candidates[convention]) / max(abs(candidates[convention]), 1.0)
    if relative_distance > SOURCE_CONVENTION_DISTANCE_THRESHOLD:
        return "unclassified_global_scale"
    return convention


def source_convention_multiplier(convention: str, flux_span: float) -> float:
    """Return the explicit multiplier for a named GEQDSK source convention."""
    if convention == "canonical":
        return 1.0
    if convention == "negated":
        return -1.0
    if convention == "scaled_by_2pi":
        return 2.0 * np.pi
    if convention == "scaled_by_minus_2pi":
        return -2.0 * np.pi
    if convention == "scaled_by_inv_2pi":
        return 1.0 / (2.0 * np.pi)
    if convention == "scaled_by_minus_inv_2pi":
        return -1.0 / (2.0 * np.pi)
    if not np.isfinite(flux_span) or abs(flux_span) < 1e-15:
        raise ValueError("flux-span source conventions require a finite non-zero flux span")
    if convention == "times_flux_span":
        return flux_span
    if convention == "over_flux_span":
        return 1.0 / flux_span
    if convention == "negated_times_flux_span":
        return -flux_span
    if convention == "negated_over_flux_span":
        return -1.0 / flux_span
    raise ValueError(f"unsupported GEQDSK source convention: {convention}")


def apply_source_convention(
    source: NDArray,
    *,
    convention: str,
    flux_span: float,
) -> NDArray:
    """Apply an explicit, documented GEQDSK source convention transform."""
    source_arr = np.asarray(source, dtype=np.float64)
    if not np.all(np.isfinite(source_arr)):
        raise ValueError("source must contain only finite values")
    multiplier = source_convention_multiplier(convention, flux_span)
    return multiplier * source_arr


def compute_source_alignment(
    eq: GEqdsk,
    source: NDArray | None = None,
) -> tuple[dict[str, float], str]:
    """
    Compare the profile-derived GS source against the discrete operator source.

    The operator source is Δ*ψ_ref on the same grid.  If the profile source is
    compatible with the reference flux and stencil, these fields should be
    strongly correlated and the residual should be small.
    """
    operator_source = gs_operator(eq.psirz, eq.r, eq.z)
    source_components = compute_source_components(eq)
    plasma_mask_2d = np.asarray(source_components["plasma_mask"], dtype=bool)
    if source is None:
        profile_source = np.asarray(source_components["total_source"], dtype=np.float64)
    else:
        profile_source = np.asarray(source, dtype=np.float64)
        if profile_source.shape != (eq.nh, eq.nw):
            raise ValueError(f"source shape must be ({eq.nh}, {eq.nw}), got {profile_source.shape}")
        if not np.all(np.isfinite(profile_source)):
            raise ValueError("source must contain only finite values")

    operator_interior = operator_source[1:-1, 1:-1]
    profile_interior = profile_source[1:-1, 1:-1]
    plasma_mask = plasma_mask_2d[1:-1, 1:-1]
    operator_flat = operator_interior.ravel()
    profile_flat = profile_interior.ravel()
    finite_mask = np.isfinite(operator_flat) & np.isfinite(profile_flat)
    operator_flat = operator_flat[finite_mask]
    profile_flat = profile_flat[finite_mask]
    if operator_flat.size == 0:
        raise ValueError("source alignment requires at least one finite interior point")

    operator_norm = float(np.linalg.norm(operator_flat))
    profile_norm = float(np.linalg.norm(profile_flat))
    residual_l2 = float(np.linalg.norm(operator_flat - profile_flat) / max(profile_norm, 1e-15))

    operator_std = float(np.std(operator_flat))
    profile_std = float(np.std(profile_flat))
    if operator_std <= 1e-15 or profile_std <= 1e-15:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(operator_flat, profile_flat)[0, 1])

    design = np.vstack([profile_flat, np.ones_like(profile_flat)]).T
    scale, offset = np.linalg.lstsq(design, operator_flat, rcond=None)[0]
    fitted = scale * profile_flat + offset
    best_fit_relative_l2 = float(np.linalg.norm(operator_flat - fitted) / max(operator_norm, 1e-15))
    best_fit_convention = classify_source_scale_convention(float(scale))

    def _masked_metrics(mask: NDArray[np.bool_]) -> tuple[float, float, int]:
        finite = np.isfinite(operator_interior) & np.isfinite(profile_interior) & mask
        count = int(np.count_nonzero(finite))
        if count == 0:
            return float("nan"), float("nan"), 0
        op = operator_interior[finite]
        prof = profile_interior[finite]
        norm = float(np.linalg.norm(op))
        profile_norm = float(np.linalg.norm(prof))
        residual = float(np.linalg.norm(op - prof) / max(profile_norm, norm, 1e-15))
        return residual, norm, count

    plasma_residual, plasma_operator_norm, plasma_count = _masked_metrics(plasma_mask)
    vacuum_residual, vacuum_operator_norm, vacuum_count = _masked_metrics(~plasma_mask)

    metrics: dict[str, float] = {
        "operator_source_norm": operator_norm,
        "profile_source_norm": profile_norm,
        "source_residual_l2": residual_l2,
        "source_correlation": correlation,
        "source_best_fit_scale": float(scale),
        "source_best_fit_offset": float(offset),
        "source_best_fit_relative_l2": best_fit_relative_l2,
        "source_plasma_residual_l2": plasma_residual,
        "source_vacuum_residual_l2": vacuum_residual,
        "source_plasma_operator_norm": plasma_operator_norm,
        "source_vacuum_operator_norm": vacuum_operator_norm,
        "source_plasma_point_count": float(plasma_count),
        "source_vacuum_point_count": float(vacuum_count),
    }
    return metrics, best_fit_convention


def compute_toroidal_current_consistency(eq: GEqdsk) -> dict[str, float | bool]:
    """Compare declared plasma current with operator- and profile-derived currents."""
    if eq.nw < 3 or eq.nh < 3:
        raise ValueError("toroidal current consistency requires a grid of at least 3x3")
    r = np.asarray(eq.r, dtype=np.float64)
    z = np.asarray(eq.z, dtype=np.float64)
    if np.any(np.diff(r) <= 0.0) or np.any(np.diff(z) <= 0.0):
        raise ValueError("R and Z axes must be strictly increasing")

    mu0 = 4e-7 * np.pi
    d_r = float(r[1] - r[0])
    d_z = float(z[1] - z[0])
    rr = np.maximum(r[np.newaxis, :], 1e-12)
    cell_area = d_r * d_z

    operator_source = gs_operator(eq.psirz, r, z)
    operator_jphi = -operator_source / (mu0 * rr)
    operator_current = float(np.sum(operator_jphi[1:-1, 1:-1]) * cell_area)

    source_components = compute_source_components(eq)
    profile_source = np.asarray(source_components["total_source"], dtype=np.float64)
    plasma_mask = np.asarray(source_components["plasma_mask"], dtype=bool)
    profile_jphi = -profile_source / (mu0 * rr)
    profile_current = float(np.sum(profile_jphi[plasma_mask]) * cell_area)

    declared_current = float(eq.current)
    scale = max(abs(declared_current), 1.0)
    operator_error = abs(abs(operator_current) - abs(declared_current)) / scale
    profile_error = abs(abs(profile_current) - abs(declared_current)) / scale
    operator_pass = bool(
        np.isfinite(operator_error) and operator_error <= OPERATOR_CURRENT_CLOSURE_THRESHOLD
    )
    return {
        "declared_toroidal_current_A": declared_current,
        "operator_toroidal_current_A": operator_current,
        "profile_toroidal_current_A": profile_current,
        "operator_current_relative_error": float(operator_error),
        "profile_current_relative_error": float(profile_error),
        "operator_current_closure_pass": operator_pass,
    }


def compute_source_candidate_rankings(eq: GEqdsk) -> list[dict[str, float | str]]:
    """
    Rank explicit source-convention candidates against Δ*ψ_ref.

    This is diagnostic only.  It does not change the benchmark's canonical
    profile-source solve or pass/fail gate.
    """
    components = compute_source_components(eq)
    pressure = np.asarray(components["pressure_source"], dtype=np.float64)
    ffprime = np.asarray(components["ffprime_source"], dtype=np.float64)
    total = np.asarray(components["total_source"], dtype=np.float64)
    flux_span = float(eq.sibry - eq.simag)

    source_conventions = {
        "profile_source": "canonical",
        "negated_profile_source": "negated",
        "profile_source_scaled_by_2pi": "scaled_by_2pi",
        "profile_source_scaled_by_minus_2pi": "scaled_by_minus_2pi",
        "profile_source_scaled_by_inv_2pi": "scaled_by_inv_2pi",
        "profile_source_scaled_by_minus_inv_2pi": "scaled_by_minus_inv_2pi",
    }
    if abs(flux_span) >= 1e-15:
        source_conventions.update(
            {
                "profile_source_times_flux_span": "times_flux_span",
                "profile_source_over_flux_span": "over_flux_span",
                "negated_profile_source_times_flux_span": "negated_times_flux_span",
                "negated_profile_source_over_flux_span": "negated_over_flux_span",
            }
        )

    candidates = {
        name: apply_source_convention(total, convention=convention, flux_span=flux_span)
        for name, convention in source_conventions.items()
    }
    candidates.update(
        {
            "pressure_only": pressure,
            "negated_pressure_only": -pressure,
            "ffprime_only": ffprime,
            "negated_ffprime_only": -ffprime,
            "pressure_plus_negated_ffprime": pressure - ffprime,
            "negated_pressure_plus_ffprime": -pressure + ffprime,
        }
    )

    rows: list[dict[str, float | str]] = []
    for name, source in candidates.items():
        metrics, best_fit_convention = compute_source_alignment(eq, source=source)
        rows.append(
            {
                "candidate": name,
                "source_convention": source_conventions.get(name, "not_profile_source_convention"),
                "source_residual_l2": metrics["source_residual_l2"],
                "source_correlation": metrics["source_correlation"],
                "source_best_fit_scale": metrics["source_best_fit_scale"],
                "source_best_fit_relative_l2": metrics["source_best_fit_relative_l2"],
                "source_best_fit_convention": best_fit_convention,
                "source_plasma_residual_l2": metrics["source_plasma_residual_l2"],
                "source_vacuum_residual_l2": metrics["source_vacuum_residual_l2"],
            }
        )

    return sorted(rows, key=lambda row: float(row["source_residual_l2"]))


def select_source_convention_adapter(eq: GEqdsk) -> dict[str, float | bool | str]:
    """
    Select the best explicit profile-source convention without accepting fitted scales.

    The raw canonical source contract remains strict elsewhere.  This adapter
    contract only reports whether a named, reproducible GEQDSK transform explains
    the operator source within the declared residual threshold.
    """
    flux_span = float(eq.sibry - eq.simag)
    candidates: list[dict[str, float | str]] = []
    for row in compute_source_candidate_rankings(eq):
        convention = str(row["source_convention"])
        if convention == "not_profile_source_convention":
            continue
        try:
            source_convention_multiplier(convention, flux_span)
        except ValueError:
            continue
        candidates.append(row)
    if not candidates:
        return {
            "source_convention_adapter": "not_evaluated",
            "source_convention_adapter_residual_l2": float("nan"),
            "source_convention_adapter_pass": False,
        }
    best = candidates[0]
    residual = float(best["source_residual_l2"])
    convention = str(best["source_convention"])
    return {
        "source_convention_adapter": convention,
        "source_convention_adapter_residual_l2": residual,
        "source_convention_adapter_pass": bool(
            np.isfinite(residual) and residual <= SOURCE_CONVENTION_ADAPTER_RESIDUAL_THRESHOLD
        ),
    }


def _laplacian_operator_variant(
    psi: NDArray,
    R: NDArray,
    Z: NDArray,
    *,
    curvature_sign: float,
) -> NDArray:
    """Evaluate d2R psi + sign*(1/R)dR psi + d2Z psi on the GEQDSK grid."""
    psi_arr = np.asarray(psi, dtype=np.float64)
    r_arr = np.asarray(R, dtype=np.float64)
    z_arr = np.asarray(Z, dtype=np.float64)
    if psi_arr.ndim != 2 or r_arr.ndim != 1 or z_arr.ndim != 1:
        raise ValueError("psi must be 2D and R/Z must be 1D")
    nz, nr = psi_arr.shape
    if nz < 3 or nr < 3 or r_arr.size != nr or z_arr.size != nz:
        raise ValueError("R/Z axis lengths must match a psi grid of at least 3x3")
    if not np.all(np.isfinite(psi_arr)):
        raise ValueError("psi must contain only finite values")
    if not np.all(np.isfinite(r_arr)) or not np.all(np.isfinite(z_arr)):
        raise ValueError("R and Z axes must contain only finite values")
    if np.any(np.diff(r_arr) <= 0.0) or np.any(np.diff(z_arr) <= 0.0):
        raise ValueError("R and Z axes must be strictly increasing")

    dR = float(r_arr[1] - r_arr[0])
    dZ = float(z_arr[1] - z_arr[0])
    RR = r_arr[np.newaxis, :]
    result = np.zeros_like(psi_arr)
    d2R = (psi_arr[1:-1, 2:] - 2 * psi_arr[1:-1, 1:-1] + psi_arr[1:-1, :-2]) / dR**2
    dR1 = (psi_arr[1:-1, 2:] - psi_arr[1:-1, :-2]) / (2 * dR)
    d2Z = (psi_arr[2:, 1:-1] - 2 * psi_arr[1:-1, 1:-1] + psi_arr[:-2, 1:-1]) / dZ**2
    R_interior = np.maximum(RR[:, 1:-1], 1e-6)
    result[1:-1, 1:-1] = d2R + curvature_sign * dR1 / R_interior + d2Z
    return result


def compute_operator_candidate_rankings(eq: GEqdsk) -> list[dict[str, float | str]]:
    """
    Rank explicit operator and flux-normalisation candidates against profile source.

    This diagnostic tests whether the profile-source mismatch is explained by
    a simple Δ* sign, flux-span scaling, normalised-flux operator, or curvature
    sign convention.  It does not alter the benchmark pass/fail gate.
    """
    components = compute_source_components(eq)
    profile_source = np.asarray(components["total_source"], dtype=np.float64)
    flux_span = float(eq.sibry - eq.simag)
    psi_norm = (
        np.zeros_like(eq.psirz, dtype=np.float64)
        if abs(flux_span) < 1e-15
        else (np.asarray(eq.psirz, dtype=np.float64) - float(eq.simag)) / flux_span
    )

    canonical = gs_operator(eq.psirz, eq.r, eq.z)
    normalised_operator = gs_operator(psi_norm, eq.r, eq.z)
    candidates = {
        "delta_star_psi": canonical,
        "negated_delta_star_psi": -canonical,
        "delta_star_psi_over_flux_span": canonical / flux_span
        if abs(flux_span) >= 1e-15
        else np.zeros_like(canonical),
        "delta_star_psi_times_flux_span": canonical * flux_span,
        "delta_star_psi_norm": normalised_operator,
        "negated_delta_star_psi_norm": -normalised_operator,
        "operator_without_curvature": _laplacian_operator_variant(
            eq.psirz,
            eq.r,
            eq.z,
            curvature_sign=0.0,
        ),
        "operator_positive_curvature": _laplacian_operator_variant(
            eq.psirz,
            eq.r,
            eq.z,
            curvature_sign=1.0,
        ),
    }

    profile_interior = profile_source[1:-1, 1:-1]
    profile_flat = profile_interior.ravel()
    rows: list[dict[str, float | str]] = []
    for name, operator_source in candidates.items():
        operator_interior = np.asarray(operator_source, dtype=np.float64)[1:-1, 1:-1]
        operator_flat = operator_interior.ravel()
        finite = np.isfinite(operator_flat) & np.isfinite(profile_flat)
        if not np.any(finite):
            raise ValueError("operator candidate ranking requires finite interior points")
        op = operator_flat[finite]
        profile = profile_flat[finite]
        profile_norm = float(np.linalg.norm(profile))
        operator_norm = float(np.linalg.norm(op))
        residual_l2 = float(np.linalg.norm(op - profile) / max(profile_norm, 1e-15))
        design = np.vstack([profile, np.ones_like(profile)]).T
        scale, offset = np.linalg.lstsq(design, op, rcond=None)[0]
        fitted = scale * profile + offset
        best_fit_relative_l2 = float(np.linalg.norm(op - fitted) / max(operator_norm, 1e-15))
        rows.append(
            {
                "candidate": name,
                "profile_residual_l2": residual_l2,
                "operator_norm": operator_norm,
                "profile_norm": profile_norm,
                "profile_best_fit_scale": float(scale),
                "profile_best_fit_offset": float(offset),
                "profile_best_fit_relative_l2": best_fit_relative_l2,
            }
        )

    return sorted(rows, key=lambda row: float(row["profile_residual_l2"]))


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
    c_R = -(dZ**2) * dR / (2.0 * R_int)  # (1, nr-2)

    t0 = time.perf_counter()
    final_res = 0.0

    for it in range(max_iter):
        max_change = 0.0
        for iz in range(1, nz - 1):
            for ir in range(1, nr - 1):
                r_val = max(R[ir], 1e-6)
                c = -(dZ**2) * dR / (2.0 * r_val)
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
    source_override: NDArray | None = None,
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

    if source_override is None:
        source = compute_gs_source(eq)
    else:
        source = np.asarray(source_override, dtype=np.float64)
        if source.shape != (eq.nh, eq.nw):
            raise ValueError(
                f"source_override shape must be ({eq.nh}, {eq.nw}), got {source.shape}"
            )
        if not np.all(np.isfinite(source)):
            raise ValueError("source_override must contain only finite values")

    # Initialise with reference boundary
    psi = eq.psirz.copy()  # start from reference as warm start

    a_R = dZ**2
    a_Z = dR**2
    denom = 2 * a_R + 2 * a_Z
    scale = dR**2 * dZ**2

    R_2d = np.broadcast_to(R[np.newaxis, :], (nz, nr))
    R_safe = np.maximum(R_2d, 1e-6)
    c_coeff = -(dZ**2) * dR / (2.0 * R_safe)

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
            iz_idx, ir_idx = np.mgrid[1 : nz - 1, 1 : nr - 1]
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
    if solver_psi.shape != ref.shape:
        raise ValueError(
            f"solver_psi shape {solver_psi.shape} does not match reference {ref.shape}"
        )
    if not np.all(np.isfinite(solver_psi)):
        raise ValueError("solver_psi must contain only finite values")
    if not np.all(np.isfinite(ref)):
        raise ValueError("reference psi contains non-finite values")

    diff = solver_psi - ref

    # Raw RMSE
    rmse_wb = float(np.sqrt(np.mean(diff**2)))
    max_err = float(np.max(np.abs(diff)))

    # Relative L2
    ref_norm = float(np.linalg.norm(ref.ravel()))
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


def _bilinear_sample_grid(
    r_axis: NDArray,
    z_axis: NDArray,
    values: NDArray,
    points: NDArray,
) -> tuple[NDArray, NDArray]:
    """Sample a rectilinear ``values[Z, R]`` grid at ``(R, Z)`` points."""
    r_arr = np.asarray(r_axis, dtype=np.float64)
    z_arr = np.asarray(z_axis, dtype=np.float64)
    value_arr = np.asarray(values, dtype=np.float64)
    point_arr = np.asarray(points, dtype=np.float64)
    if value_arr.shape != (z_arr.size, r_arr.size):
        raise ValueError("values must have shape (len(z_axis), len(r_axis))")
    if point_arr.ndim != 2 or point_arr.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if r_arr.size < 2 or z_arr.size < 2:
        raise ValueError("grid axes must contain at least two points")
    if np.any(np.diff(r_arr) <= 0.0) or np.any(np.diff(z_arr) <= 0.0):
        raise ValueError("grid axes must be strictly increasing")

    r_pts = point_arr[:, 0]
    z_pts = point_arr[:, 1]
    inside = (r_pts >= r_arr[0]) & (r_pts <= r_arr[-1]) & (z_pts >= z_arr[0]) & (z_pts <= z_arr[-1])
    samples = np.full(point_arr.shape[0], np.nan, dtype=np.float64)
    if not np.any(inside):
        return samples, inside

    r_inside = np.clip(r_pts[inside], r_arr[0], r_arr[-1])
    z_inside = np.clip(z_pts[inside], z_arr[0], z_arr[-1])
    ir = np.clip(np.searchsorted(r_arr, r_inside, side="right") - 1, 0, r_arr.size - 2)
    iz = np.clip(np.searchsorted(z_arr, z_inside, side="right") - 1, 0, z_arr.size - 2)
    r0 = r_arr[ir]
    r1 = r_arr[ir + 1]
    z0 = z_arr[iz]
    z1 = z_arr[iz + 1]
    tr = (r_inside - r0) / np.maximum(r1 - r0, 1e-15)
    tz = (z_inside - z0) / np.maximum(z1 - z0, 1e-15)
    v00 = value_arr[iz, ir]
    v10 = value_arr[iz, ir + 1]
    v01 = value_arr[iz + 1, ir]
    v11 = value_arr[iz + 1, ir + 1]
    samples[inside] = (
        (1.0 - tr) * (1.0 - tz) * v00
        + tr * (1.0 - tz) * v10
        + (1.0 - tr) * tz * v01
        + tr * tz * v11
    )
    return samples, inside


def compute_reconstruction_geometry_contracts(
    eq: GEqdsk, solver_psi: NDArray
) -> dict[str, float | bool]:
    """Report axis, boundary-containment, boundary-flux, and q-profile sanity checks."""
    if solver_psi.shape != eq.psirz.shape:
        raise ValueError("solver_psi shape must match the GEQDSK psi grid")
    if not np.all(np.isfinite(solver_psi)):
        raise ValueError("solver_psi must contain only finite values")

    axis_index = np.unravel_index(
        int(np.argmin(np.abs(solver_psi - float(eq.simag)))),
        solver_psi.shape,
    )
    axis_r = float(eq.r[axis_index[1]])
    axis_z = float(eq.z[axis_index[0]])
    axis_error_m = float(np.hypot(axis_r - float(eq.rmaxis), axis_z - float(eq.zmaxis)))

    boundary_points = np.column_stack(
        [np.asarray(eq.rbdry, dtype=np.float64), np.asarray(eq.zbdry, dtype=np.float64)]
    )
    if boundary_points.size:
        boundary_samples, boundary_inside = _bilinear_sample_grid(
            eq.r,
            eq.z,
            solver_psi,
            boundary_points,
        )
        containment = float(np.count_nonzero(boundary_inside) / boundary_inside.size)
        finite_boundary = boundary_inside & np.isfinite(boundary_samples)
        if np.any(finite_boundary):
            boundary_error = boundary_samples[finite_boundary] - float(eq.sibry)
            boundary_rmse = float(np.sqrt(np.mean(boundary_error**2)))
            boundary_rmse_norm = boundary_rmse / max(abs(float(eq.sibry) - float(eq.simag)), 1e-15)
        else:
            boundary_rmse_norm = float("nan")
    else:
        containment = 0.0
        boundary_rmse_norm = float("nan")

    q = np.asarray(eq.qpsi, dtype=np.float64)
    finite_mask = np.isfinite(q)
    finite_q = q[finite_mask]
    finite_fraction = float(np.count_nonzero(finite_mask) / q.size) if q.size else 0.0
    if finite_q.size:
        q_min_abs = float(np.min(np.abs(finite_q)))
    else:
        q_min_abs = float("nan")
    sign_changes = 0
    monotonic_fraction = 0.0
    if finite_q.size > 1:
        q_signs = np.sign(finite_q)
        nonzero_signs = q_signs[q_signs != 0.0]
        if nonzero_signs.size > 1:
            sign_changes = int(np.count_nonzero(nonzero_signs[1:] != nonzero_signs[:-1]))
        dq = np.diff(finite_q)
        monotonic_fraction = float(
            max(
                np.count_nonzero(dq >= -1.0e-12),
                np.count_nonzero(dq <= 1.0e-12),
            )
            / dq.size
        )
    q_profile_sanity_pass = bool(
        finite_q.size == q.size
        and finite_q.size > 1
        and np.all(np.abs(finite_q) > 1e-12)
        and (np.all(finite_q > 0.0) or np.all(finite_q < 0.0))
        and sign_changes == 0
    )

    return {
        "adapted_profile_axis_error_m": axis_error_m,
        "adapted_profile_boundary_containment_fraction": containment,
        "adapted_profile_boundary_psi_rmse_norm": float(boundary_rmse_norm),
        "q_profile_sanity_pass": q_profile_sanity_pass,
        "q_profile_finite_fraction": finite_fraction,
        "q_profile_min_abs": q_min_abs,
        "q_profile_sign_changes": sign_changes,
        "q_profile_monotonic_fraction": monotonic_fraction,
    }


def compute_adapted_profile_reconstruction(
    eq: GEqdsk,
    *,
    omega: float,
    adapter: Mapping[str, float | bool | str] | None = None,
) -> dict[str, float | bool | int]:
    """
    Reconstruct ψ from physical profiles after a passing named GEQDSK convention adapter.

    Non-passing adapters are diagnostic-only and intentionally return non-finite
    reconstruction metrics so raw canonical failures cannot be silently relaxed.
    """
    source_adapter = adapter if adapter is not None else select_source_convention_adapter(eq)
    if not bool(source_adapter["source_convention_adapter_pass"]):
        return {
            "adapted_profile_psi_rmse_norm": float("nan"),
            "adapted_profile_sor_iterations": 0,
            "adapted_profile_sor_residual": float("nan"),
            "adapted_profile_axis_error_m": float("nan"),
            "adapted_profile_boundary_containment_fraction": float("nan"),
            "adapted_profile_boundary_psi_rmse_norm": float("nan"),
            "q_profile_sanity_pass": False,
            "q_profile_finite_fraction": 0.0,
            "q_profile_min_abs": float("nan"),
            "q_profile_sign_changes": 0,
            "q_profile_monotonic_fraction": 0.0,
            "adapted_profile_pass": False,
        }

    adapted_source = apply_source_convention(
        compute_gs_source(eq),
        convention=str(source_adapter["source_convention_adapter"]),
        flux_span=float(eq.sibry - eq.simag),
    )
    adapted_psi, adapted_iters, adapted_res, _ = manufactured_solve_vectorised(
        eq,
        omega=omega,
        max_iter=5000,
        tol=1e-8,
        source_override=adapted_source,
    )
    adapted_metrics = compute_psi_rmse(eq, adapted_psi)
    geometry = compute_reconstruction_geometry_contracts(eq, adapted_psi)
    adapted_rmse = float(adapted_metrics["psi_rmse_norm"])
    boundary_rmse = float(geometry["adapted_profile_boundary_psi_rmse_norm"])
    containment = float(geometry["adapted_profile_boundary_containment_fraction"])
    adapted_pass = bool(
        np.isfinite(adapted_rmse)
        and adapted_rmse <= ADAPTED_PROFILE_RMSE_THRESHOLD
        and np.isfinite(boundary_rmse)
        and boundary_rmse <= ADAPTED_PROFILE_RMSE_THRESHOLD
        and containment >= 0.999
        and bool(geometry["q_profile_sanity_pass"])
    )
    return {
        "adapted_profile_psi_rmse_norm": adapted_rmse,
        "adapted_profile_sor_iterations": int(adapted_iters),
        "adapted_profile_sor_residual": float(adapted_res),
        "adapted_profile_axis_error_m": float(geometry["adapted_profile_axis_error_m"]),
        "adapted_profile_boundary_containment_fraction": containment,
        "adapted_profile_boundary_psi_rmse_norm": boundary_rmse,
        "q_profile_sanity_pass": bool(geometry["q_profile_sanity_pass"]),
        "q_profile_finite_fraction": float(geometry["q_profile_finite_fraction"]),
        "q_profile_min_abs": float(geometry["q_profile_min_abs"]),
        "q_profile_sign_changes": int(geometry["q_profile_sign_changes"]),
        "q_profile_monotonic_fraction": float(geometry["q_profile_monotonic_fraction"]),
        "adapted_profile_pass": adapted_pass,
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
    # Optimal SOR omega for NxN Laplacian: 2/(1 + sin(pi/N))
    n_eff = max(eq.nw, eq.nh)
    omega_opt = 2.0 / (1.0 + np.sin(np.pi / n_eff))
    if warm_start:
        solver_psi, iters, res, t_ms = manufactured_solve_vectorised(
            eq,
            omega=omega_opt,
            max_iter=5000,
            tol=1e-8,
        )
    else:
        solver_psi, iters, res, t_ms = manufactured_solve_vectorised(
            eq,
            omega=omega_opt,
            max_iter=10000,
            tol=1e-7,
        )

    # 3. Point-wise RMSE
    metrics = compute_psi_rmse(eq, solver_psi)
    source_components = compute_source_components(eq)
    source_alignment, source_alignment_best_fit_convention = compute_source_alignment(eq)
    current_consistency = compute_toroidal_current_consistency(eq)
    source_candidates = compute_source_candidate_rankings(eq)
    source_convention_adapter = select_source_convention_adapter(eq)
    adapted_profile_reconstruction = compute_adapted_profile_reconstruction(
        eq,
        omega=omega_opt,
        adapter=source_convention_adapter,
    )
    best_source_candidate = str(source_candidates[0]["candidate"])
    best_source_candidate_residual = float(source_candidates[0]["source_residual_l2"])
    profile_source_rank = next(
        index
        for index, candidate in enumerate(source_candidates, start=1)
        if candidate["candidate"] == "profile_source"
    )
    operator_candidates = compute_operator_candidate_rankings(eq)
    best_operator_candidate = str(operator_candidates[0]["candidate"])
    best_operator_candidate_residual = float(operator_candidates[0]["profile_residual_l2"])
    delta_star_psi_rank = next(
        index
        for index, candidate in enumerate(operator_candidates, start=1)
        if candidate["candidate"] == "delta_star_psi"
    )

    operator_source = gs_operator(eq.psirz, eq.r, eq.z)
    operator_psi, operator_iters, operator_res, _ = manufactured_solve_vectorised(
        eq,
        omega=omega_opt,
        max_iter=50,
        tol=1e-10,
        source_override=operator_source,
    )
    operator_metrics = compute_psi_rmse(eq, operator_psi)
    operator_rmse_norm = operator_metrics["psi_rmse_norm"]
    if np.isfinite(operator_rmse_norm) and operator_rmse_norm <= OPERATOR_SOURCE_RMSE_THRESHOLD:
        if np.isfinite(metrics["psi_rmse_norm"]) and metrics["psi_rmse_norm"] <= 0.05:
            source_consistency_class = "profile_source_consistent"
        else:
            source_consistency_class = "profile_source_mismatch"
    else:
        source_consistency_class = "solver_consistency_failure"

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
        operator_source_psi_rmse_norm=operator_rmse_norm,
        operator_source_sor_iterations=operator_iters,
        operator_source_sor_residual=operator_res,
        source_consistency_class=source_consistency_class,
        operator_source_norm=source_alignment["operator_source_norm"],
        profile_source_norm=source_alignment["profile_source_norm"],
        source_residual_l2=source_alignment["source_residual_l2"],
        source_correlation=source_alignment["source_correlation"],
        source_best_fit_scale=source_alignment["source_best_fit_scale"],
        source_best_fit_offset=source_alignment["source_best_fit_offset"],
        source_best_fit_relative_l2=source_alignment["source_best_fit_relative_l2"],
        source_best_fit_convention=source_alignment_best_fit_convention,
        source_convention_adapter=str(source_convention_adapter["source_convention_adapter"]),
        source_convention_adapter_residual_l2=float(
            source_convention_adapter["source_convention_adapter_residual_l2"]
        ),
        source_convention_adapter_pass=bool(
            source_convention_adapter["source_convention_adapter_pass"]
        ),
        adapted_profile_psi_rmse_norm=float(
            adapted_profile_reconstruction["adapted_profile_psi_rmse_norm"]
        ),
        adapted_profile_sor_iterations=int(
            adapted_profile_reconstruction["adapted_profile_sor_iterations"]
        ),
        adapted_profile_sor_residual=float(
            adapted_profile_reconstruction["adapted_profile_sor_residual"]
        ),
        adapted_profile_axis_error_m=float(
            adapted_profile_reconstruction["adapted_profile_axis_error_m"]
        ),
        adapted_profile_boundary_containment_fraction=float(
            adapted_profile_reconstruction["adapted_profile_boundary_containment_fraction"]
        ),
        adapted_profile_boundary_psi_rmse_norm=float(
            adapted_profile_reconstruction["adapted_profile_boundary_psi_rmse_norm"]
        ),
        q_profile_sanity_pass=bool(adapted_profile_reconstruction["q_profile_sanity_pass"]),
        q_profile_finite_fraction=float(
            adapted_profile_reconstruction["q_profile_finite_fraction"]
        ),
        q_profile_min_abs=float(adapted_profile_reconstruction["q_profile_min_abs"]),
        q_profile_sign_changes=int(adapted_profile_reconstruction["q_profile_sign_changes"]),
        q_profile_monotonic_fraction=float(
            adapted_profile_reconstruction["q_profile_monotonic_fraction"]
        ),
        adapted_profile_pass=bool(adapted_profile_reconstruction["adapted_profile_pass"]),
        plasma_mask_fraction=float(source_components["plasma_mask_fraction"]),
        pressure_source_norm=float(source_components["pressure_source_norm"]),
        ffprime_source_norm=float(source_components["ffprime_source_norm"]),
        total_source_norm=float(source_components["total_source_norm"]),
        pressure_source_sum=float(source_components["pressure_source_sum"]),
        ffprime_source_sum=float(source_components["ffprime_source_sum"]),
        total_source_sum=float(source_components["total_source_sum"]),
        pressure_source_fraction=float(
            source_components["pressure_source_norm"]
            / max(source_components["total_source_norm"], 1e-15)
        ),
        ffprime_source_fraction=float(
            source_components["ffprime_source_norm"]
            / max(source_components["total_source_norm"], 1e-15)
        ),
        source_plasma_residual_l2=source_alignment["source_plasma_residual_l2"],
        source_vacuum_residual_l2=source_alignment["source_vacuum_residual_l2"],
        source_plasma_operator_norm=source_alignment["source_plasma_operator_norm"],
        source_vacuum_operator_norm=source_alignment["source_vacuum_operator_norm"],
        source_plasma_point_count=source_alignment["source_plasma_point_count"],
        source_vacuum_point_count=source_alignment["source_vacuum_point_count"],
        best_source_candidate=best_source_candidate,
        best_source_candidate_residual_l2=best_source_candidate_residual,
        profile_source_candidate_rank=profile_source_rank,
        best_operator_candidate=best_operator_candidate,
        best_operator_candidate_residual_l2=best_operator_candidate_residual,
        delta_star_psi_candidate_rank=delta_star_psi_rank,
        declared_toroidal_current_A=float(current_consistency["declared_toroidal_current_A"]),
        operator_toroidal_current_A=float(current_consistency["operator_toroidal_current_A"]),
        profile_toroidal_current_A=float(current_consistency["profile_toroidal_current_A"]),
        operator_current_relative_error=float(
            current_consistency["operator_current_relative_error"]
        ),
        profile_current_relative_error=float(current_consistency["profile_current_relative_error"]),
        operator_current_closure_pass=bool(current_consistency["operator_current_closure_pass"]),
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
    for row in rows:
        row["benchmark_scope"] = EFIT_BENCHMARK_SCOPE
        row["solver_mode"] = RAW_PROFILE_SOLVER_MODE

    finite_norm_entries = [
        (idx, r.psi_rmse_norm) for idx, r in enumerate(results) if np.isfinite(r.psi_rmse_norm)
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
        benchmark_id="sparc-pointwise-rmse",
        benchmark_scope=EFIT_BENCHMARK_SCOPE,
        benchmark_contract=(
            "SPARC-only raw GEQDSK profile-source fixed-boundary pointwise psi(R,Z) "
            "RMSE diagnostic; not an operator-source solve, free-boundary reconstruction, "
            "or reduced-order surrogate."
        ),
        solver_mode=RAW_PROFILE_SOLVER_MODE,
        count=len(results),
        mean_psi_rmse_norm=float(np.mean(norms)) if norms else float("nan"),
        mean_psi_relative_l2=float(np.mean(rel_l2s)),
        mean_gs_residual_l2=float(np.mean(gs_l2s)),
        worst_psi_rmse_norm=float(worst_norm),
        worst_file=worst_file,
        rows=rows,
    )


def _benchmark_reference_files(
    reference_root: Path,
    *,
    include_proxy: bool,
) -> list[tuple[str, Path]]:
    machines = ("sparc", "diiid", "jet") if include_proxy else ("sparc",)
    files: list[tuple[str, Path]] = []
    for machine in machines:
        machine_dir = reference_root / machine
        if not machine_dir.exists():
            continue
        machine_files = sorted(machine_dir.glob("*.geqdsk")) + sorted(machine_dir.glob("*.eqdsk"))
        files.extend((machine, path) for path in machine_files)
    return files


def _reference_case_contract(machine: str, rel_path: str) -> dict[str, str]:
    """Return machine-readable provenance and gate-role metadata for a benchmark case."""
    try:
        contract = dict(EFIT_REFERENCE_MACHINE_CONTRACTS[machine])
    except KeyError as exc:
        raise ValueError(f"unknown EFIT reference machine: {machine}") from exc
    contract["reference_expected_convention"] = EFIT_REFERENCE_CASE_CONVENTIONS.get(
        rel_path,
        contract["reference_expected_convention"],
    )
    return contract


def validate_efit_nrmse_benchmark(
    reference_root: Path | None = None,
    *,
    min_files: int = 10,
    max_nrmse: float = 0.05,
    include_proxy: bool = True,
) -> EfitNRMSEBenchmarkGate:
    """
    Run a strict aggregate ψ_N RMSE gate over EFIT/GEQDSK reference equilibria.

    SPARC entries are public design references.  Bundled DIII-D/JET entries are
    synthetic proxy references, so the returned provenance must travel with any
    public benchmark claim.
    """
    if min_files <= 0:
        raise ValueError("min_files must be > 0")
    if not np.isfinite(max_nrmse) or max_nrmse <= 0.0:
        raise ValueError("max_nrmse must be finite and > 0")

    if reference_root is None:
        reference_root = REFERENCE_DATA_DIR

    files = _benchmark_reference_files(reference_root, include_proxy=include_proxy)
    if not files:
        raise FileNotFoundError(f"No GEQDSK/EQDSK files in {reference_root}")

    rows: list[dict[str, Any]] = []
    count_by_machine: dict[str, int] = {}
    finite_entries: list[tuple[str, float]] = []
    gate_entries: list[tuple[str, float]] = []
    operator_source_entries: list[tuple[str, float]] = []
    gate_operator_source_entries: list[tuple[str, float]] = []
    adapted_profile_entries: list[tuple[str, float]] = []
    pass_count = 0
    operator_source_pass_count = 0
    gate_operator_source_pass_count = 0
    source_convention_adapter_pass_count = 0
    adapted_profile_pass_count = 0
    gate_source_convention_adapter_pass_count = 0
    gate_adapted_profile_pass_count = 0
    operator_current_closure_pass_count = 0
    gate_operator_current_closure_pass_count = 0
    gate_row_count = 0
    gate_pass_count = 0
    failure_reasons: list[str] = []

    for machine, path in files:
        result = validate_file(path)
        rel_path = f"{machine}/{path.name}"
        rmse_norm = float(result.psi_rmse_norm)
        provenance = EFIT_BENCHMARK_MACHINE_PROVENANCE[machine]
        reference_contract = _reference_case_contract(machine, rel_path)
        is_gate_row = reference_contract["reference_role"] == "gate"
        if is_gate_row:
            gate_row_count += 1

        count_by_machine[machine] = count_by_machine.get(machine, 0) + 1
        if np.isfinite(rmse_norm):
            finite_entries.append((rel_path, rmse_norm))
            if is_gate_row:
                gate_entries.append((rel_path, rmse_norm))
            if rmse_norm <= max_nrmse:
                pass_count += 1
                if is_gate_row:
                    gate_pass_count += 1
        else:
            failure_reasons.append(f"non-finite psi_rmse_norm in {rel_path}")

        operator_source_rmse = float(result.operator_source_psi_rmse_norm)
        if np.isfinite(operator_source_rmse):
            operator_source_entries.append((rel_path, operator_source_rmse))
            if is_gate_row:
                gate_operator_source_entries.append((rel_path, operator_source_rmse))
            if operator_source_rmse <= OPERATOR_SOURCE_RMSE_THRESHOLD:
                operator_source_pass_count += 1
                if is_gate_row:
                    gate_operator_source_pass_count += 1
        else:
            failure_reasons.append(f"non-finite operator_source_psi_rmse_norm in {rel_path}")

        if result.source_convention_adapter_pass:
            source_convention_adapter_pass_count += 1
            if is_gate_row:
                gate_source_convention_adapter_pass_count += 1
            adapted_profile_rmse = float(result.adapted_profile_psi_rmse_norm)
            if np.isfinite(adapted_profile_rmse):
                adapted_profile_entries.append((rel_path, adapted_profile_rmse))
                if result.adapted_profile_pass:
                    adapted_profile_pass_count += 1
                    if is_gate_row:
                        gate_adapted_profile_pass_count += 1
            else:
                failure_reasons.append(f"non-finite adapted_profile_psi_rmse_norm in {rel_path}")

        row = asdict(result)
        row["file"] = rel_path
        row["machine"] = machine
        row["provenance"] = provenance
        row["raw_profile_solver_mode"] = RAW_PROFILE_SOLVER_MODE
        row["operator_source_solver_mode"] = OPERATOR_SOURCE_SOLVER_MODE
        row["adapted_profile_solver_mode"] = ADAPTED_PROFILE_SOLVER_MODE
        row.update(reference_contract)
        row["threshold"] = max_nrmse
        row["passes_threshold"] = bool(np.isfinite(rmse_norm) and rmse_norm <= max_nrmse)
        if bool(row["operator_current_closure_pass"]):
            operator_current_closure_pass_count += 1
            if is_gate_row:
                gate_operator_current_closure_pass_count += 1
        rows.append(row)

    if finite_entries:
        worst_file, worst_norm = max(finite_entries, key=lambda item: item[1])
        mean_norm = float(np.mean([norm for _, norm in finite_entries]))
    else:
        worst_file = ""
        worst_norm = float("nan")
        mean_norm = float("nan")

    if operator_source_entries:
        operator_source_worst_file, operator_source_worst_norm = max(
            operator_source_entries,
            key=lambda item: item[1],
        )
    else:
        operator_source_worst_file = ""
        operator_source_worst_norm = float("nan")

    if gate_operator_source_entries:
        gate_operator_source_worst_file, gate_operator_source_worst_norm = max(
            gate_operator_source_entries,
            key=lambda item: item[1],
        )
    else:
        gate_operator_source_worst_file = ""
        gate_operator_source_worst_norm = float("nan")

    if gate_entries:
        gate_worst_file, gate_worst_norm = max(gate_entries, key=lambda item: item[1])
    else:
        gate_worst_file = ""
        gate_worst_norm = float("nan")

    if adapted_profile_entries:
        adapted_profile_worst_file, adapted_profile_worst_norm = max(
            adapted_profile_entries,
            key=lambda item: item[1],
        )
    else:
        adapted_profile_worst_file = ""
        adapted_profile_worst_norm = float("nan")

    source_consistency_counts: dict[str, int] = {}
    source_convention_adapter_counts: dict[str, int] = {}
    reference_role_counts: dict[str, int] = {}
    reference_class_counts: dict[str, int] = {}
    solver_mode_counts: dict[str, int] = {}
    source_residual_entries: list[tuple[str, float]] = []
    gate_source_residual_entries: list[tuple[str, float]] = []
    source_sum_identity_errors: list[float] = []
    for row in rows:
        for key in (
            "raw_profile_solver_mode",
            "operator_source_solver_mode",
            "adapted_profile_solver_mode",
        ):
            solver_mode = str(row[key])
            solver_mode_counts[solver_mode] = solver_mode_counts.get(solver_mode, 0) + 1
        source_class = str(row["source_consistency_class"])
        source_consistency_counts[source_class] = source_consistency_counts.get(source_class, 0) + 1
        source_adapter = str(row["source_convention_adapter"])
        source_convention_adapter_counts[source_adapter] = (
            source_convention_adapter_counts.get(source_adapter, 0) + 1
        )
        reference_role = str(row["reference_role"])
        reference_role_counts[reference_role] = reference_role_counts.get(reference_role, 0) + 1
        reference_class = str(row["reference_class"])
        reference_class_counts[reference_class] = reference_class_counts.get(reference_class, 0) + 1
        source_residual = float(row["source_residual_l2"])
        if np.isfinite(source_residual):
            source_residual_entries.append((str(row["file"]), source_residual))
            if row["reference_role"] == "gate":
                gate_source_residual_entries.append((str(row["file"]), source_residual))
        source_sum_error = abs(
            float(row["total_source_sum"])
            - float(row["pressure_source_sum"])
            - float(row["ffprime_source_sum"])
        )
        if np.isfinite(source_sum_error):
            source_sum_identity_errors.append(source_sum_error)
        else:
            failure_reasons.append(f"non-finite source sum identity error in {row['file']}")

    if source_residual_entries:
        worst_source_alignment_file, worst_source_residual = max(
            source_residual_entries,
            key=lambda item: item[1],
        )
    else:
        worst_source_alignment_file = ""
        worst_source_residual = float("nan")

    if gate_source_residual_entries:
        gate_worst_source_alignment_file, gate_worst_source_residual = max(
            gate_source_residual_entries,
            key=lambda item: item[1],
        )
    else:
        gate_worst_source_alignment_file = ""
        gate_worst_source_residual = float("nan")

    if len(files) < min_files:
        failure_reasons.append(f"count {len(files)} < required {min_files}")
    if np.isfinite(worst_norm) and worst_norm > max_nrmse:
        failure_reasons.append(f"worst psi_rmse_norm {worst_norm:.6g} > threshold {max_nrmse:.6g}")
    if len(operator_source_entries) != len(files):
        missing = len(files) - len(operator_source_entries)
        failure_reasons.append(f"operator-source solver gate missing finite RMSE in {missing} rows")
    if np.isfinite(operator_source_worst_norm) and (
        operator_source_worst_norm > OPERATOR_SOURCE_RMSE_THRESHOLD
    ):
        failure_reasons.append(
            "operator-source psi_rmse_norm "
            f"{operator_source_worst_norm:.6g} > threshold {OPERATOR_SOURCE_RMSE_THRESHOLD:.6g}"
        )
    source_mismatch_count = sum(
        1 for row in rows if row["source_consistency_class"] == "profile_source_mismatch"
    )
    solver_failure_count = sum(
        1 for row in rows if row["source_consistency_class"] == "solver_consistency_failure"
    )
    if source_mismatch_count:
        failure_reasons.append(
            f"profile-source mismatch attribution in {source_mismatch_count} rows"
        )
    if solver_failure_count:
        failure_reasons.append(
            f"operator-source solver consistency failure in {solver_failure_count} rows"
        )
    source_sum_identity_max_abs_error = (
        max(source_sum_identity_errors) if source_sum_identity_errors else float("nan")
    )
    source_sum_identity_pass = bool(
        source_sum_identity_errors
        and len(source_sum_identity_errors) == len(rows)
        and source_sum_identity_max_abs_error <= 1.0e-9
    )
    if not source_sum_identity_pass:
        failure_reasons.append(
            "source-sum identity gate failed: max abs error "
            f"{source_sum_identity_max_abs_error:.6g} > 1e-09"
        )
    if operator_current_closure_pass_count != len(rows):
        failure_reasons.append(
            "operator-current closure gate failed in "
            f"{len(rows) - operator_current_closure_pass_count}/{len(rows)} rows"
        )
    if adapted_profile_entries and adapted_profile_pass_count != len(adapted_profile_entries):
        failure_reasons.append(
            "adapted-profile reconstruction gate failed in "
            f"{len(adapted_profile_entries) - adapted_profile_pass_count}/"
            f"{len(adapted_profile_entries)} accepted adapter rows"
        )

    provenance_by_machine = {
        machine: EFIT_BENCHMARK_MACHINE_PROVENANCE[machine] for machine in count_by_machine
    }

    return EfitNRMSEBenchmarkGate(
        schema_version=EFIT_NRMSE_BENCHMARK_SCHEMA_VERSION,
        benchmark_id="efit-nrmse-benchmark",
        benchmark_scope=EFIT_BENCHMARK_SCOPE,
        benchmark_contract=EFIT_BENCHMARK_CONTRACT,
        raw_profile_solver_mode=RAW_PROFILE_SOLVER_MODE,
        operator_source_solver_mode=OPERATOR_SOURCE_SOLVER_MODE,
        adapted_profile_solver_mode=ADAPTED_PROFILE_SOLVER_MODE,
        count=len(files),
        min_required_files=min_files,
        threshold=max_nrmse,
        pass_count=pass_count,
        gate_row_count=gate_row_count,
        gate_pass_count=gate_pass_count,
        gate_worst_psi_rmse_norm=float(gate_worst_norm),
        gate_worst_file=gate_worst_file,
        passes=not failure_reasons,
        mean_psi_rmse_norm=mean_norm,
        worst_psi_rmse_norm=float(worst_norm),
        worst_file=worst_file,
        count_by_machine=count_by_machine,
        provenance_by_machine=provenance_by_machine,
        reference_role_counts=reference_role_counts,
        reference_class_counts=reference_class_counts,
        solver_mode_counts=solver_mode_counts,
        source_consistency_counts=source_consistency_counts,
        operator_source_threshold=OPERATOR_SOURCE_RMSE_THRESHOLD,
        operator_source_pass_count=operator_source_pass_count,
        gate_operator_source_pass_count=gate_operator_source_pass_count,
        gate_operator_source_worst_psi_rmse_norm=float(gate_operator_source_worst_norm),
        gate_operator_source_worst_file=gate_operator_source_worst_file,
        operator_source_worst_psi_rmse_norm=float(operator_source_worst_norm),
        operator_source_worst_file=operator_source_worst_file,
        source_convention_adapter_threshold=SOURCE_CONVENTION_ADAPTER_RESIDUAL_THRESHOLD,
        source_convention_adapter_pass_count=source_convention_adapter_pass_count,
        gate_source_convention_adapter_pass_count=gate_source_convention_adapter_pass_count,
        source_convention_adapter_counts=source_convention_adapter_counts,
        adapted_profile_threshold=ADAPTED_PROFILE_RMSE_THRESHOLD,
        adapted_profile_pass_count=adapted_profile_pass_count,
        gate_adapted_profile_pass_count=gate_adapted_profile_pass_count,
        adapted_profile_worst_psi_rmse_norm=float(adapted_profile_worst_norm),
        adapted_profile_worst_file=adapted_profile_worst_file,
        worst_source_residual_l2=float(worst_source_residual),
        worst_source_alignment_file=worst_source_alignment_file,
        gate_worst_source_residual_l2=float(gate_worst_source_residual),
        gate_worst_source_alignment_file=gate_worst_source_alignment_file,
        source_sum_identity_max_abs_error=float(source_sum_identity_max_abs_error),
        source_sum_identity_pass=source_sum_identity_pass,
        operator_current_closure_pass_count=operator_current_closure_pass_count,
        gate_operator_current_closure_pass_count=gate_operator_current_closure_pass_count,
        failure_reasons=failure_reasons,
        rows=rows,
    )


def _contract_path(parent: str, key: str) -> str:
    return key if not parent else f"{parent}.{key}"


def _validate_schema_contract(
    payload: Any,
    schema: Mapping[str, Any],
    *,
    path: str,
    label: str,
) -> None:
    """Enforce the required/no-extra-fields subset of local JSON-schema contracts."""
    if schema.get("type") == "object":
        if not isinstance(payload, Mapping):
            return
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            return
        for key in schema.get("required", []):
            if key not in payload:
                raise ValueError(f"missing required {label} key: {_contract_path(path, str(key))}")
        if schema.get("additionalProperties") is False:
            for key in payload:
                if key not in properties:
                    raise ValueError(f"unexpected {label} key: {_contract_path(path, str(key))}")
        for key, child_schema in properties.items():
            if key in payload and isinstance(child_schema, Mapping):
                _validate_schema_contract(
                    payload[key],
                    child_schema,
                    path=_contract_path(path, str(key)),
                    label=label,
                )
    elif schema.get("type") == "array":
        if not isinstance(payload, list):
            return
        item_schema = schema.get("items", {})
        if not isinstance(item_schema, Mapping):
            return
        for index, item in enumerate(payload):
            _validate_schema_contract(
                item,
                item_schema,
                path=f"{path}[{index}]",
                label=label,
            )


def load_efit_nrmse_benchmark_schema() -> dict[str, Any]:
    """Load the public EFIT/GEQDSK ψ_N RMSE benchmark report schema."""
    schema_path = SCHEMA_DIR / "efit_nrmse_benchmark.schema.json"
    return json.loads(schema_path.read_text(encoding="utf-8"))


def validate_efit_nrmse_benchmark_report(
    report: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> None:
    """Validate the EFIT/GEQDSK benchmark report against the local schema contract."""
    root_properties = schema["properties"]
    for key in report:
        if key not in root_properties:
            raise ValueError(f"unexpected benchmark report key: {key}")
    _validate_schema_contract(report, schema, path="", label="benchmark report")
    for key in schema["required"]:
        if key not in report:
            raise ValueError(f"missing required benchmark report key: {key}")
    expected_version = root_properties["schema_version"]["const"]
    if report["schema_version"] != expected_version:
        raise ValueError("unexpected schema_version")
    if report["benchmark_id"] != root_properties["benchmark_id"]["const"]:
        raise ValueError("unexpected benchmark_id")
    if not isinstance(report["rows"], list) or not report["rows"]:
        raise ValueError("rows must be a non-empty list")


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
    """Run full point-wise psi(R,Z) RMSE and EFIT/GEQDSK benchmark validation.

    Returns
    -------
    int
        Zero when validation contracts pass, one otherwise.
    """
    print("=" * 70)
    print("SCPN Fusion Core - Point-wise psi(R,Z) RMSE Validation")
    print("=" * 70)

    summary = validate_all_sparc()

    print(f"\nFiles validated: {summary.count}")
    print(f"Mean normalized psi RMSE: {summary.mean_psi_rmse_norm:.6f}")
    print(f"Mean relative L2:       {summary.mean_psi_relative_l2:.6f}")
    print(f"Mean GS residual (L2):  {summary.mean_gs_residual_l2:.6f}")
    print(
        f"Worst file:             {summary.worst_file} "
        f"(psi_N RMSE = {summary.worst_psi_rmse_norm:.6f})"
    )
    print()

    # Per-file table
    print(
        f"{'File':<22} {'Grid':<8} {'psi_N RMSE':>10} {'Rel L2':>10} "
        f"{'GS Res':>10} {'Iters':>6} {'Time(ms)':>10}"
    )
    print("-" * 80)
    for r in summary.rows:
        print(
            f"{r['file']:<22} {r['grid']:<8} "
            f"{r['psi_rmse_norm']:>10.6f} {r['psi_relative_l2']:>10.6f} "
            f"{r['gs_residual_l2']:>10.4f} {r['sor_iterations']:>6d} "
            f"{r['solve_time_ms']:>10.1f}"
        )

    # Save JSON
    out = ROOT / "validation" / "reports" / "psi_pointwise_rmse.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(asdict(summary), f, indent=2)
        f.write("\n")
    print(f"\nJSON report: {out}")

    print("\n" + "=" * 70)
    print("EFIT/GEQDSK aggregate psi_N RMSE benchmark gate")
    print("=" * 70)
    benchmark = validate_efit_nrmse_benchmark()
    print(f"Files validated:        {benchmark.count}")
    print(f"Minimum required files: {benchmark.min_required_files}")
    print(f"Threshold psi_N RMSE:   {benchmark.threshold:.6f}")
    print(f"Rows under threshold:   {benchmark.pass_count}/{benchmark.count}")
    print(
        f"Public gate rows under threshold: {benchmark.gate_pass_count}/{benchmark.gate_row_count}"
    )
    if benchmark.gate_worst_file:
        print(
            f"Worst public gate file: {benchmark.gate_worst_file} "
            f"(psi_N RMSE = {benchmark.gate_worst_psi_rmse_norm:.6f})"
        )
    print(
        f"Worst file:             {benchmark.worst_file} "
        f"(psi_N RMSE = {benchmark.worst_psi_rmse_norm:.6f})"
    )
    source_classes = ", ".join(
        f"{name}={count}" for name, count in sorted(benchmark.source_consistency_counts.items())
    )
    print(f"Source classes:        {source_classes}")
    print(
        f"Worst source residual: {benchmark.worst_source_alignment_file} "
        f"(relative L2 = {benchmark.worst_source_residual_l2:.6f})"
    )
    print(
        "Source-sum identity gate: "
        f"{'PASS' if benchmark.source_sum_identity_pass else 'FAIL'} "
        f"(max abs error = {benchmark.source_sum_identity_max_abs_error:.6e})"
    )
    print(
        "Operator-current closure gate: "
        f"{benchmark.operator_current_closure_pass_count}/{benchmark.count} rows, "
        f"{benchmark.gate_operator_current_closure_pass_count}/{benchmark.gate_row_count} public rows"
    )
    if benchmark.gate_worst_source_alignment_file:
        print(
            "Worst public source residual: "
            f"{benchmark.gate_worst_source_alignment_file} "
            f"(relative L2 = {benchmark.gate_worst_source_residual_l2:.6f})"
        )
    print(
        "Public operator-source gate: "
        f"{benchmark.gate_operator_source_pass_count}/{benchmark.gate_row_count} public rows "
        f"under psi_N RMSE <= {benchmark.operator_source_threshold:.6g}"
    )
    if benchmark.gate_operator_source_worst_file:
        print(
            "Worst public operator-source row: "
            f"{benchmark.gate_operator_source_worst_file} "
            f"(psi_N RMSE = {benchmark.gate_operator_source_worst_psi_rmse_norm:.6g})"
        )
    print(
        f"Adapted profile gate: {benchmark.adapted_profile_pass_count}/"
        f"{benchmark.source_convention_adapter_pass_count} accepted adapter rows "
        f"under psi_N RMSE <= {benchmark.adapted_profile_threshold:.6f}"
    )
    print(
        "Public adapted profile gate: "
        f"{benchmark.gate_adapted_profile_pass_count}/"
        f"{benchmark.gate_source_convention_adapter_pass_count} public accepted adapter rows"
    )
    if benchmark.adapted_profile_worst_file:
        print(
            f"Worst adapted profile: {benchmark.adapted_profile_worst_file} "
            f"(psi_N RMSE = {benchmark.adapted_profile_worst_psi_rmse_norm:.6f})"
        )
    worst_source_row = next(
        (row for row in benchmark.rows if row["file"] == benchmark.worst_source_alignment_file),
        None,
    )
    if worst_source_row is not None:
        print(
            "Worst source components: "
            f"pressure_fraction={worst_source_row['pressure_source_fraction']:.6f}, "
            f"ffprime_fraction={worst_source_row['ffprime_source_fraction']:.6f}, "
            f"plasma_mask_fraction={worst_source_row['plasma_mask_fraction']:.6f}"
        )
        print(
            "Worst source masked residuals: "
            f"plasma={worst_source_row['source_plasma_residual_l2']:.6f}, "
            f"vacuum={worst_source_row['source_vacuum_residual_l2']:.6f}"
        )
        print(
            "Best source candidate: "
            f"{worst_source_row['best_source_candidate']} "
            f"(relative L2 = {worst_source_row['best_source_candidate_residual_l2']:.6f}, "
            f"profile rank = {worst_source_row['profile_source_candidate_rank']})"
        )
        print(
            "Best operator candidate: "
            f"{worst_source_row['best_operator_candidate']} "
            f"(relative L2 = {worst_source_row['best_operator_candidate_residual_l2']:.6f}, "
            f"delta_star_psi rank = {worst_source_row['delta_star_psi_candidate_rank']})"
        )
    print(f"Gate status:            {'PASS' if benchmark.passes else 'FAIL'}")
    if benchmark.failure_reasons:
        print("Failure reasons:")
        for reason in benchmark.failure_reasons:
            print(f"  - {reason}")
    print("Provenance:")
    for machine, provenance in benchmark.provenance_by_machine.items():
        print(f"  - {machine}: {provenance}")

    benchmark_out = ROOT / "validation" / "reports" / "psi_efit_nrmse_benchmark.json"
    validate_efit_nrmse_benchmark_report(
        asdict(benchmark),
        load_efit_nrmse_benchmark_schema(),
    )
    with benchmark_out.open("w") as f:
        json.dump(asdict(benchmark), f, indent=2)
        f.write("\n")
    print(f"\nBenchmark JSON report: {benchmark_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
