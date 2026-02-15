# ----------------------------------------------------------------------
# SCPN Fusion Core -- Task 4 Quasi-3D Modeling
# ----------------------------------------------------------------------
"""Task 4: quasi-3D force-balance, Hall-MHD divertor coupling, and TBR guard."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scpn_fusion.core.quasi_3d_contracts import (
    build_divertor_profiles,
    build_quasi_3d_force_balance,
    calibrate_tbr_with_erosion,
    hall_mhd_zonal_ratio,
    load_jet_solps_reference_profile,
    solve_quasi_3d_force_residual,
)


ROOT = Path(__file__).resolve().parents[1]
JET_REFERENCE_DIR = ROOT / "validation" / "reference_data" / "jet"


def _require_int(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    out = int(value)
    if out < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return out


def _require_finite(name: str, value: Any, minimum: float | None = None) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite.")
    if minimum is not None and out < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return out


def _rmse_percent(truth: np.ndarray, pred: np.ndarray) -> float:
    if truth.shape != pred.shape or truth.size == 0:
        raise ValueError("truth/pred arrays must be non-empty and same shape.")
    return float(
        100.0
        * np.sqrt(np.mean((pred - truth) ** 2))
        / max(float(np.mean(np.abs(truth))), 1e-12)
    )


def run_campaign(
    *,
    seed: int = 42,
    quasi_3d_samples: int = 720,
    hall_grid: int = 18,
    hall_steps: int = 36,
    toroidal_points: int = 48,
    tbr_thickness_cm: float = 260.0,
    asdex_erosion_ref_mm_year: float = 0.25,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    seed_i = _require_int("seed", seed, 0)
    samples = _require_int("quasi_3d_samples", quasi_3d_samples, 64)
    hall_grid_i = _require_int("hall_grid", hall_grid, 8)
    hall_steps_i = _require_int("hall_steps", hall_steps, 8)
    tor_pts = _require_int("toroidal_points", toroidal_points, 16)
    thickness_cm = _require_finite("tbr_thickness_cm", tbr_thickness_cm, 20.0)
    asdex_ref = _require_finite("asdex_erosion_ref_mm_year", asdex_erosion_ref_mm_year, 1e-6)

    quasi = build_quasi_3d_force_balance(seed=seed_i, samples=samples)
    force_residual = solve_quasi_3d_force_residual(
        asymmetry_index=float(quasi["asymmetry_index"]),
        n1_amp=float(quasi["n1_amp"]),
        n2_amp=float(quasi["n2_amp"]),
        poloidal_points=56,
        toroidal_points=tor_pts,
        iterations=14,
    )
    hall = hall_mhd_zonal_ratio(
        seed=seed_i + 911,
        grid=hall_grid_i,
        steps=hall_steps_i,
        fallback_asymmetry=float(quasi["asymmetry_index"]),
    )
    jet_reference_profile, jet_meta = load_jet_solps_reference_profile(
        reference_dir=JET_REFERENCE_DIR,
        toroidal_points=tor_pts,
    )
    divertor = build_divertor_profiles(
        n1_amp=float(quasi["n1_amp"]),
        n2_amp=float(quasi["n2_amp"]),
        z_n1_amp=float(quasi["z_n1_amp"]),
        zonal_ratio=float(hall["zonal_ratio"]),
        reference_profile_w_m2=jet_reference_profile,
        toroidal_points=tor_pts,
    )

    ref_profile = np.asarray(divertor["reference_profile_w_m2"], dtype=np.float64)
    pred_profile = np.asarray(divertor["predicted_profile_w_m2"], dtype=np.float64)
    jet_rmse_pct = _rmse_percent(ref_profile, pred_profile)
    two_fluid_index = float(
        float(hall["zonal_ratio"])
        * (1.0 + float(divertor["divertor_state"]["hartmann_number"]) / 300.0)
    )

    tbr = calibrate_tbr_with_erosion(
        mean_heat_flux_w_m2=float(np.mean(pred_profile)),
        thickness_cm=thickness_cm,
        asdex_erosion_ref_mm_year=asdex_ref,
    )

    thresholds = {
        "max_force_balance_rmse_pct": 8.0,
        "max_force_residual_p95_pct": 12.0,
        "min_two_fluid_index": 0.10,
        "min_temhd_cooling_gain_pct": 1.0,
        "max_jet_heat_flux_rmse_pct": 15.0,
        "max_erosion_curve_rmse_pct": 35.0,
        "max_calibrated_tbr": 1.10,
    }

    failure_reasons: list[str] = []
    if float(quasi["force_balance_rmse_pct"]) > thresholds["max_force_balance_rmse_pct"]:
        failure_reasons.append("force_balance_rmse_pct")
    if float(force_residual["force_residual_p95_pct"]) > thresholds["max_force_residual_p95_pct"]:
        failure_reasons.append("force_residual_p95_pct")
    if two_fluid_index < thresholds["min_two_fluid_index"]:
        failure_reasons.append("two_fluid_index")
    if float(divertor["cooling_gain_pct"]) < thresholds["min_temhd_cooling_gain_pct"]:
        failure_reasons.append("temhd_cooling_gain_pct")
    if jet_rmse_pct > thresholds["max_jet_heat_flux_rmse_pct"]:
        failure_reasons.append("jet_heat_flux_rmse_pct")
    if float(tbr["erosion_curve_rmse_pct"]) > thresholds["max_erosion_curve_rmse_pct"]:
        failure_reasons.append("erosion_curve_rmse_pct")
    if float(tbr["calibrated_tbr"]) > thresholds["max_calibrated_tbr"]:
        failure_reasons.append("calibrated_tbr")

    return {
        "seed": seed_i,
        "quasi_3d": {
            "nfp": int(quasi["nfp"]),
            "force_balance_rmse_pct": float(quasi["force_balance_rmse_pct"]),
            "force_residual_mean_pct": float(force_residual["force_residual_mean_pct"]),
            "force_residual_p95_pct": float(force_residual["force_residual_p95_pct"]),
            "asymmetry_index": float(quasi["asymmetry_index"]),
            "radial_spread_m": float(quasi["radial_spread_m"]),
        },
        "divertor_two_fluid": {
            "hall_backend": str(hall["backend"]),
            "hall_zonal_ratio": float(hall["zonal_ratio"]),
            "two_fluid_index": two_fluid_index,
            "two_fluid_temp_split_index": float(
                divertor["two_fluid_diag"]["two_fluid_temp_split_index"]
            ),
            "electron_temp_mean_kev": float(divertor["two_fluid_diag"]["electron_temp_mean_kev"]),
            "ion_temp_mean_kev": float(divertor["two_fluid_diag"]["ion_temp_mean_kev"]),
            "hartmann_number": float(divertor["divertor_state"]["hartmann_number"]),
            "temhd_cooling_gain_pct": float(divertor["cooling_gain_pct"]),
            "surface_temperature_c": float(divertor["divertor_state"]["surface_temperature_c"]),
            "surface_heat_flux_w_m2": float(divertor["divertor_state"]["surface_heat_flux_w_m2"]),
        },
        "jet_heat_flux_validation": {
            "jet_file_count": int(jet_meta["jet_file_count"]),
            "mean_q95_reference": float(jet_meta["mean_q95"]),
            "rmse_pct": float(jet_rmse_pct),
            "mean_reference_w_m2": float(np.mean(ref_profile)),
            "mean_predicted_w_m2": float(np.mean(pred_profile)),
            "max_reference_w_m2": float(np.max(ref_profile)),
            "max_predicted_w_m2": float(np.max(pred_profile)),
        },
        "pwi_tbr_calibration": tbr,
        "thresholds": thresholds,
        "failure_reasons": failure_reasons,
        "passes_thresholds": bool(len(failure_reasons) == 0),
        "runtime_seconds": float(time.perf_counter() - t0),
    }


def generate_report(**kwargs: Any) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task4_quasi_3d_modeling": run_campaign(**kwargs),
    }


def render_markdown(report: dict[str, Any]) -> str:
    g = report["task4_quasi_3d_modeling"]
    q = g["quasi_3d"]
    d = g["divertor_two_fluid"]
    j = g["jet_heat_flux_validation"]
    t = g["pwi_tbr_calibration"]
    th = g["thresholds"]
    lines = [
        "# Task 4 Quasi-3D Modeling Report",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Runtime: `{g['runtime_seconds']:.3f} s`",
        f"- Overall pass: `{'YES' if g['passes_thresholds'] else 'NO'}`",
        "",
        "## Quasi-3D Force Balance",
        "",
        f"- NFP: `{q['nfp']}`",
        f"- Force-balance RMSE: `{q['force_balance_rmse_pct']:.3f}%` (threshold `<= {th['max_force_balance_rmse_pct']:.1f}%`)",
        f"- Force-residual P95: `{q['force_residual_p95_pct']:.3f}%` (threshold `<= {th['max_force_residual_p95_pct']:.1f}%`)",
        f"- Asymmetry index: `{q['asymmetry_index']:.4f}`",
        "",
        "## Hall-MHD + TEMHD Coupling",
        "",
        f"- Hall backend: `{d['hall_backend']}`",
        f"- Hall zonal ratio: `{d['hall_zonal_ratio']:.4f}`",
        f"- Two-fluid index: `{d['two_fluid_index']:.4f}` (threshold `>= {th['min_two_fluid_index']:.2f}`)",
        f"- Two-fluid temperature split index: `{d['two_fluid_temp_split_index']:.4f}`",
        f"- TEMHD cooling gain: `{d['temhd_cooling_gain_pct']:.3f}%` (threshold `>= {th['min_temhd_cooling_gain_pct']:.1f}%`)",
        "",
        "## JET / SOLPS-ITER Proxy Heat Flux",
        "",
        f"- JET reference files: `{j['jet_file_count']}`",
        f"- RMSE: `{j['rmse_pct']:.3f}%` (threshold `<= {th['max_jet_heat_flux_rmse_pct']:.1f}%`)",
        f"- Mean reference heat flux: `{j['mean_reference_w_m2']:.3e} W/m^2`",
        f"- Mean predicted heat flux: `{j['mean_predicted_w_m2']:.3e} W/m^2`",
        "",
        "## Erosion-Calibrated TBR Guard",
        "",
        f"- Raw TBR: `{t['raw_tbr']:.4f}`",
        f"- Estimated erosion: `{t['estimated_erosion_mm_year']:.4f} mm/y`",
        f"- ASDEX reference erosion: `{t['asdex_reference_erosion_mm_year']:.4f} mm/y`",
        f"- Erosion-curve RMSE: `{t['erosion_curve_rmse_pct']:.3f}%` (threshold `<= {th['max_erosion_curve_rmse_pct']:.1f}%`)",
        f"- Calibrated TBR: `{t['calibrated_tbr']:.4f}` (threshold `<= {th['max_calibrated_tbr']:.2f}`)",
        "",
    ]
    if g["failure_reasons"]:
        lines.append(f"- Failure reasons: `{', '.join(g['failure_reasons'])}`")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quasi-3d-samples", type=int, default=720)
    parser.add_argument("--hall-grid", type=int, default=18)
    parser.add_argument("--hall-steps", type=int, default=36)
    parser.add_argument("--toroidal-points", type=int, default=48)
    parser.add_argument("--tbr-thickness-cm", type=float, default=260.0)
    parser.add_argument("--asdex-erosion-ref-mm-year", type=float, default=0.25)
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / "task4_quasi_3d_modeling.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "task4_quasi_3d_modeling.md"),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    report = generate_report(
        seed=args.seed,
        quasi_3d_samples=args.quasi_3d_samples,
        hall_grid=args.hall_grid,
        hall_steps=args.hall_steps,
        toroidal_points=args.toroidal_points,
        tbr_thickness_cm=args.tbr_thickness_cm,
        asdex_erosion_ref_mm_year=args.asdex_erosion_ref_mm_year,
    )
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    g = report["task4_quasi_3d_modeling"]
    print("Task 4 quasi-3D modeling validation complete.")
    print(
        "Summary -> "
        f"force_rmse={g['quasi_3d']['force_balance_rmse_pct']:.3f}%, "
        f"two_fluid_index={g['divertor_two_fluid']['two_fluid_index']:.4f}, "
        f"jet_rmse={g['jet_heat_flux_validation']['rmse_pct']:.3f}%, "
        f"calibrated_tbr={g['pwi_tbr_calibration']['calibrated_tbr']:.4f}, "
        f"passes_thresholds={g['passes_thresholds']}"
    )

    if args.strict and not g["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
