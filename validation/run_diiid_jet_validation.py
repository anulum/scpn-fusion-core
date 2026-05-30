# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — DIII-D / JET Equilibrium Validation Runner
"""
Validate the GS solver against synthetic DIII-D and JET equilibria.

For each GEQDSK file:
1. Read the equilibrium
2. Compute the GS source through the shared current-conserving profile-source contract
3. Measure the operator/source residual between reference ψ and the physical source
4. Report diagnostic RMSE-like residual metrics for proxy GEQDSK cases
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scpn_fusion.core.eqdsk import read_geqdsk
from validation.psi_pointwise_rmse import compute_source_components


@dataclass
class ValidationResult:
    """Scalar results for a single GEQDSK validation case."""

    file: str
    device: str
    grid: str
    ip_ma: float
    psi_rmse_norm: float
    psi_max_error: float
    psi_relative_l2: float
    gs_residual_l2: float
    plasma_mask_fraction: float
    source_total_norm: float


def gs_operator(psi, R, Z):
    """Compute Δ*ψ = ∂²ψ/∂R² - (1/R)∂ψ/∂R + ∂²ψ/∂Z²."""
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
    result = np.zeros_like(psi_arr)

    for i in range(1, nz - 1):
        for j in range(1, nr - 1):
            d2psi_dR2 = (psi_arr[i, j + 1] - 2 * psi_arr[i, j] + psi_arr[i, j - 1]) / dR**2
            dpsi_dR = (psi_arr[i, j + 1] - psi_arr[i, j - 1]) / (2 * dR)
            d2psi_dZ2 = (psi_arr[i + 1, j] - 2 * psi_arr[i, j] + psi_arr[i - 1, j]) / dZ**2
            r = r_arr[j]
            if r > 0:
                result[i, j] = d2psi_dR2 - dpsi_dR / r + d2psi_dZ2
            else:
                result[i, j] = d2psi_dR2 + d2psi_dZ2
    return result


def validate_file(path: Path, device: str) -> ValidationResult:
    """Validate a single GEQDSK file."""
    eq = read_geqdsk(path)

    if eq.nw < 5 or eq.nh < 5:
        raise ValueError(f"grid must be at least 5x5, got {eq.nw}x{eq.nh}")
    if not np.isfinite(eq.simag) or not np.isfinite(eq.sibry):
        raise ValueError("eq.simag and eq.sibry must be finite")

    R = eq.r
    Z = eq.z
    psi_ref = np.asarray(eq.psirz, dtype=np.float64)
    pprime = np.asarray(eq.pprime, dtype=np.float64)
    ffprime = np.asarray(eq.ffprime, dtype=np.float64)

    if psi_ref.shape != (eq.nh, eq.nw):
        raise ValueError(f"eq.psirz shape must be ({eq.nh}, {eq.nw}), got {psi_ref.shape}")
    if pprime.shape != (eq.nw,):
        raise ValueError(f"eq.pprime length must be {eq.nw}, got {pprime.shape}")
    if ffprime.shape != (eq.nw,):
        raise ValueError(f"eq.ffprime length must be {eq.nw}, got {ffprime.shape}")
    if not np.all(np.isfinite(psi_ref)):
        raise ValueError("eq.psirz must contain only finite values")
    if not np.all(np.isfinite(pprime)) or not np.all(np.isfinite(ffprime)):
        raise ValueError("eq.pprime and eq.ffprime must contain only finite values")

    # Compute GS residual on the reference solution
    gs_res = gs_operator(psi_ref, R, Z)
    interior = gs_res[1:-1, 1:-1]

    psi_range = abs(eq.simag - eq.sibry)
    if psi_range < 1e-12:
        raise ValueError("degenerate psi range: |simag - sibry| < 1e-12")

    gs_l2 = float(np.sqrt(np.mean(interior**2))) / psi_range

    # For proxy GEQDSK equilibria, measure self-consistency: how well does
    # Delta*psi match the physical source. Reuse the shared source contract so
    # this runner cannot drift back to linear profile interpolation, unmasked
    # boundary rows, or hidden net-current drift.
    source_components = compute_source_components(eq)
    source = np.asarray(source_components["total_source"], dtype=np.float64)
    if source.shape != psi_ref.shape:
        raise ValueError(f"computed source shape must match psi grid, got {source.shape}")

    # Residual = Δ*ψ - source
    residual = gs_res - source
    interior_res = residual[2:-2, 2:-2]

    rmse = float(np.sqrt(np.mean(interior_res**2)))
    max_err = float(np.max(np.abs(interior_res)))
    rel_l2 = rmse / psi_range

    return ValidationResult(
        file=path.name,
        device=device,
        grid=f"{eq.nw}x{eq.nh}",
        ip_ma=eq.current / 1e6,
        psi_rmse_norm=rmse,
        psi_max_error=max_err,
        psi_relative_l2=rel_l2,
        gs_residual_l2=gs_l2,
        plasma_mask_fraction=float(source_components["plasma_mask_fraction"]),
        source_total_norm=float(source_components["total_source_norm"]),
    )


def main():
    """Run DIII-D and JET validation cases and print summary metrics."""
    root = Path(__file__).resolve().parent / "reference_data"

    devices = [
        ("diiid", "DIII-D"),
        ("jet", "JET"),
    ]

    results = []
    for dir_name, device_label in devices:
        device_dir = root / dir_name
        if not device_dir.exists():
            print(f"  [{device_label}] Directory not found: {device_dir}")
            continue

        files = sorted(device_dir.glob("*.geqdsk"))
        if not files:
            print(f"  [{device_label}] No GEQDSK files found")
            continue

        print(f"\n{'=' * 70}")
        print(f"  {device_label} Equilibrium Validation ({len(files)} files)")
        print(f"{'=' * 70}")
        print(
            f"{'File':<30} {'Grid':>7} {'Ip[MA]':>7} {'RMSE':>10} "
            f"{'MaxErr':>10} {'RelL2':>10} {'Mask':>7} {'|S|':>10}"
        )
        print(
            f"{'-' * 30} {'-' * 7} {'-' * 7} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 7} {'-' * 10}"
        )

        for path in files:
            r = validate_file(path, device_label)
            results.append(r)
            print(
                f"{r.file:<30} {r.grid:>7} {r.ip_ma:>7.1f} "
                f"{r.psi_rmse_norm:>10.4e} {r.psi_max_error:>10.4e} "
                f"{r.psi_relative_l2:>10.4e} {r.plasma_mask_fraction:>7.3f} "
                f"{r.source_total_norm:>10.4e}"
            )

    print(f"\n{'=' * 70}")
    print(f"  Summary: {len(results)} equilibria validated")
    if results:
        mean_rel_l2 = np.mean([r.psi_relative_l2 for r in results])
        max_rel_l2 = max(r.psi_relative_l2 for r in results)
        print(f"  Mean relative L2: {mean_rel_l2:.4e}")
        print(f"  Max  relative L2: {max_rel_l2:.4e}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
