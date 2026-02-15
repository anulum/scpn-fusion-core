# ----------------------------------------------------------------------
# SCPN Fusion Core -- Quasi-3D Contract Tests
# ----------------------------------------------------------------------
"""Tests for reusable quasi-3D contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scpn_fusion.core.quasi_3d_contracts import (
    build_divertor_profiles,
    build_quasi_3d_force_balance,
    calibrate_tbr_with_erosion,
    hall_mhd_zonal_ratio,
    load_jet_solps_reference_profile,
    solve_quasi_3d_force_residual,
)


def test_quasi_3d_contracts_smoke() -> None:
    root = Path(__file__).resolve().parents[1]
    jet_dir = root / "validation" / "reference_data" / "jet"

    quasi = build_quasi_3d_force_balance(seed=13, samples=192)
    residual = solve_quasi_3d_force_residual(
        asymmetry_index=float(quasi["asymmetry_index"]),
        n1_amp=float(quasi["n1_amp"]),
        n2_amp=float(quasi["n2_amp"]),
        poloidal_points=48,
        toroidal_points=32,
        iterations=12,
    )
    hall = hall_mhd_zonal_ratio(
        seed=31,
        grid=16,
        steps=24,
        fallback_asymmetry=float(quasi["asymmetry_index"]),
    )
    reference_profile, _ = load_jet_solps_reference_profile(
        reference_dir=jet_dir,
        toroidal_points=32,
    )
    divertor = build_divertor_profiles(
        n1_amp=float(quasi["n1_amp"]),
        n2_amp=float(quasi["n2_amp"]),
        z_n1_amp=float(quasi["z_n1_amp"]),
        zonal_ratio=float(hall["zonal_ratio"]),
        reference_profile_w_m2=reference_profile,
        toroidal_points=32,
    )
    tbr = calibrate_tbr_with_erosion(
        mean_heat_flux_w_m2=float(
            np.mean(np.asarray(divertor["predicted_profile_w_m2"], dtype=np.float64))
        ),
        thickness_cm=260.0,
        asdex_erosion_ref_mm_year=0.25,
    )

    assert float(quasi["force_balance_rmse_pct"]) >= 0.0
    assert float(residual["force_residual_p95_pct"]) >= 0.0
    assert 0.0 <= float(hall["zonal_ratio"]) <= 1.0
    assert float(divertor["cooling_gain_pct"]) > 0.0
    assert float(tbr["calibrated_tbr"]) > 0.0
