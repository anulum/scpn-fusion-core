# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Task 4 Quasi-3D Modeling
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Task 4: quasi-3D force-balance, Hall-MHD divertor coupling, and TBR guard."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.core.divertor_thermal_sim import DivertorLab
from scpn_fusion.core.equilibrium_3d import VMECStyleEquilibrium3D
from scpn_fusion.core.geometry_3d import Reactor3DBuilder
from scpn_fusion.nuclear.blanket_neutronics import BreedingBlanket
from scpn_fusion.nuclear.pwi_erosion import SputteringPhysics
from scpn_fusion.nuclear.temhd_peltier import TEMHD_Stabilizer


ROOT = Path(__file__).resolve().parents[1]


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


def _rmse_percent(truth: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    if truth.shape != pred.shape or truth.size == 0:
        raise ValueError("truth/pred arrays must be non-empty and same shape.")
    return float(
        100.0
        * np.sqrt(np.mean((pred - truth) ** 2))
        / max(float(np.mean(np.abs(truth))), 1e-12)
    )


def _build_quasi_3d_force_balance(
    *,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))

    base_eq = VMECStyleEquilibrium3D(
        r_axis=2.95,
        z_axis=0.0,
        a_minor=0.95,
        kappa=1.72,
        triangularity=0.32,
        nfp=1,
    )
    base_builder = Reactor3DBuilder(equilibrium_3d=base_eq, solve_equilibrium=False)
    stellar_eq = base_builder.build_stellarator_w7x_like_equilibrium(
        nfp=4,
        edge_ripple=0.06,
        vertical_ripple=0.04,
    )
    tracer = Reactor3DBuilder(
        equilibrium_3d=stellar_eq,
        solve_equilibrium=False,
    ).create_fieldline_tracer(
        rotational_transform=0.42,
        helical_coupling_scale=0.10,
        radial_coupling_scale=0.03,
        nfp=stellar_eq.nfp,
    )
    trace = tracer.trace_line(
        rho0=0.93,
        theta0=0.02,
        phi0=0.0,
        toroidal_turns=4,
        steps_per_turn=96,
    )
    asym = tracer.toroidal_asymmetry_observables(trace)

    theta = rng.uniform(0.0, 2.0 * np.pi, int(samples))
    phi = rng.uniform(0.0, 2.0 * np.pi, int(samples))
    rho = np.full_like(theta, 0.92, dtype=np.float64)
    x_b, y_b, z_b = base_eq.flux_to_cartesian(rho, theta, phi)
    x_s, y_s, z_s = stellar_eq.flux_to_cartesian(rho, theta, phi)
    dist = np.sqrt((x_s - x_b) ** 2 + (y_s - y_b) ** 2 + (z_s - z_b) ** 2)
    force_balance_rmse_pct = float(
        100.0 * np.sqrt(np.mean(dist**2)) / max(stellar_eq.a_minor, 1e-9)
    )

    return {
        "force_balance_rmse_pct": force_balance_rmse_pct,
        "nfp": int(stellar_eq.nfp),
        "asymmetry_index": float(asym.asymmetry_index),
        "radial_spread_m": float(asym.radial_spread),
        "n1_amp": float(asym.n1_amp),
        "n2_amp": float(asym.n2_amp),
        "z_n1_amp": float(asym.z_n1_amp),
    }


def _hall_mhd_zonal_ratio(
    *,
    seed: int,
    grid: int,
    steps: int,
    fallback_asymmetry: float,
) -> dict[str, Any]:
    try:
        from scpn_fusion.core.hall_mhd_discovery import HallMHD
    except Exception:
        ratio = float(np.clip(0.06 + 0.40 * abs(fallback_asymmetry), 0.0, 0.9))
        return {"backend": "proxy", "zonal_ratio": ratio}

    state = np.random.get_state()
    try:
        np.random.seed(int(seed))
        sim = HallMHD(N=int(grid))
        ratios: list[float] = []
        for _ in range(int(steps)):
            e_tot, e_zonal = sim.step()
            denom = max(float(e_tot), 1e-12)
            ratios.append(float(e_zonal) / denom)
    finally:
        np.random.set_state(state)

    ratio = float(np.mean(np.asarray(ratios, dtype=np.float64)))
    if not np.isfinite(ratio) or ratio <= 0.0:
        ratio = float(np.clip(0.06 + 0.40 * abs(fallback_asymmetry), 0.0, 0.9))
        return {"backend": "proxy", "zonal_ratio": ratio}
    return {"backend": "hall_mhd", "zonal_ratio": ratio}


def _build_divertor_profiles(
    *,
    n1_amp: float,
    n2_amp: float,
    z_n1_amp: float,
    zonal_ratio: float,
    toroidal_points: int,
) -> dict[str, Any]:
    with contextlib.redirect_stdout(io.StringIO()):
        lab = DivertorLab(P_sol_MW=42.0, R_major=2.95, B_pol=1.3)
        divertor_state = lab.simulate_temhd_liquid_metal(
            flow_velocity_m_s=5.0,
            expansion_factor=22.0,
        )

    q_base = float(divertor_state["surface_heat_flux_w_m2"])
    angles = np.linspace(0.0, 2.0 * np.pi, int(toroidal_points), endpoint=False)

    # JET/SOLPS-ITER proxy profile (deterministic reduced-order benchmark lane).
    jet_reference = q_base * (
        1.0
        + 0.10 * np.cos(angles - 0.30)
        + 0.04 * np.cos(2.0 * angles + 0.50)
    )
    predicted_raw = q_base * (
        1.0
        + 0.08 * float(n1_amp) * np.cos(angles)
        + 0.05 * float(n2_amp) * np.cos(2.0 * angles)
        - 0.03 * float(z_n1_amp) * np.sin(angles)
    )

    temhd = TEMHD_Stabilizer(layer_thickness_mm=7.0, B_field=11.5)
    predicted_cool: list[float] = []
    for value in predicted_raw:
        surf_t, _ = temhd.step(float(value) / 1e6, dt=0.05)
        sigmoid = 1.0 / (1.0 + np.exp(-(surf_t - 700.0) / 220.0))
        cooling_factor = float(1.0 - 0.05 * zonal_ratio - 0.06 * sigmoid)
        cooling_factor = float(np.clip(cooling_factor, 0.80, 1.00))
        predicted_cool.append(float(value) * cooling_factor)

    raw_arr = np.asarray(predicted_raw, dtype=np.float64)
    cool_arr = np.asarray(predicted_cool, dtype=np.float64)
    ref_arr = np.asarray(jet_reference, dtype=np.float64)
    cooling_gain_pct = float(
        100.0 * (np.mean(raw_arr) - np.mean(cool_arr)) / max(np.mean(raw_arr), 1e-12)
    )
    return {
        "reference_profile_w_m2": ref_arr,
        "predicted_profile_w_m2": cool_arr,
        "cooling_gain_pct": cooling_gain_pct,
        "divertor_state": {
            "hartmann_number": float(divertor_state["hartmann_number"]),
            "stability_index": float(divertor_state["stability_index"]),
            "surface_temperature_c": float(divertor_state["surface_temperature_c"]),
            "surface_heat_flux_w_m2": float(divertor_state["surface_heat_flux_w_m2"]),
        },
    }


def _calibrate_tbr_with_erosion(
    *,
    mean_heat_flux_w_m2: float,
    thickness_cm: float,
    asdex_erosion_ref_mm_year: float,
) -> dict[str, Any]:
    pwi = SputteringPhysics(material="Tungsten", redeposition_factor=0.97)
    particle_flux = max(1.0e21, (mean_heat_flux_w_m2 / 1.2e6) * 2.0e21)
    erosion = pwi.calculate_erosion_rate(
        flux_particles_m2_s=particle_flux,
        T_ion_eV=65.0,
        angle_deg=35.0,
    )
    erosion_mm_year = float(erosion["Erosion_mm_year"])

    blanket = BreedingBlanket(thickness_cm=float(thickness_cm), li6_enrichment=1.0)
    raw_tbr = float(
        blanket.calculate_volumetric_tbr(
            major_radius_m=6.2,
            minor_radius_m=2.0,
            elongation=1.7,
        ).tbr
    )
    calibration_factor = float(
        min(1.0, asdex_erosion_ref_mm_year / max(erosion_mm_year, 1e-12))
    )
    calibrated_tbr = float(raw_tbr * calibration_factor)

    return {
        "particle_flux_m2_s": float(particle_flux),
        "estimated_erosion_mm_year": erosion_mm_year,
        "asdex_reference_erosion_mm_year": float(asdex_erosion_ref_mm_year),
        "raw_tbr": raw_tbr,
        "calibration_factor": calibration_factor,
        "calibrated_tbr": calibrated_tbr,
        "calibration_triggered": bool(raw_tbr > 1.1),
    }


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
    asdex_ref = _require_finite(
        "asdex_erosion_ref_mm_year",
        asdex_erosion_ref_mm_year,
        1e-6,
    )

    quasi = _build_quasi_3d_force_balance(seed=seed_i, samples=samples)
    hall = _hall_mhd_zonal_ratio(
        seed=seed_i + 911,
        grid=hall_grid_i,
        steps=hall_steps_i,
        fallback_asymmetry=float(quasi["asymmetry_index"]),
    )
    divertor = _build_divertor_profiles(
        n1_amp=float(quasi["n1_amp"]),
        n2_amp=float(quasi["n2_amp"]),
        z_n1_amp=float(quasi["z_n1_amp"]),
        zonal_ratio=float(hall["zonal_ratio"]),
        toroidal_points=tor_pts,
    )

    ref_profile = np.asarray(divertor["reference_profile_w_m2"], dtype=np.float64)
    pred_profile = np.asarray(divertor["predicted_profile_w_m2"], dtype=np.float64)
    jet_rmse_pct = _rmse_percent(ref_profile, pred_profile)
    two_fluid_index = float(
        float(hall["zonal_ratio"])
        * (1.0 + float(divertor["divertor_state"]["hartmann_number"]) / 300.0)
    )

    tbr = _calibrate_tbr_with_erosion(
        mean_heat_flux_w_m2=float(np.mean(pred_profile)),
        thickness_cm=thickness_cm,
        asdex_erosion_ref_mm_year=asdex_ref,
    )

    thresholds = {
        "max_force_balance_rmse_pct": 8.0,
        "min_two_fluid_index": 0.10,
        "min_temhd_cooling_gain_pct": 1.0,
        "max_jet_heat_flux_rmse_pct": 15.0,
        "max_calibrated_tbr": 1.10,
    }

    failure_reasons: list[str] = []
    if float(quasi["force_balance_rmse_pct"]) > thresholds["max_force_balance_rmse_pct"]:
        failure_reasons.append("force_balance_rmse_pct")
    if two_fluid_index < thresholds["min_two_fluid_index"]:
        failure_reasons.append("two_fluid_index")
    if float(divertor["cooling_gain_pct"]) < thresholds["min_temhd_cooling_gain_pct"]:
        failure_reasons.append("temhd_cooling_gain_pct")
    if jet_rmse_pct > thresholds["max_jet_heat_flux_rmse_pct"]:
        failure_reasons.append("jet_heat_flux_rmse_pct")
    if float(tbr["calibrated_tbr"]) > thresholds["max_calibrated_tbr"]:
        failure_reasons.append("calibrated_tbr")

    return {
        "seed": seed_i,
        "quasi_3d": {
            "nfp": int(quasi["nfp"]),
            "force_balance_rmse_pct": float(quasi["force_balance_rmse_pct"]),
            "asymmetry_index": float(quasi["asymmetry_index"]),
            "radial_spread_m": float(quasi["radial_spread_m"]),
        },
        "divertor_two_fluid": {
            "hall_backend": str(hall["backend"]),
            "hall_zonal_ratio": float(hall["zonal_ratio"]),
            "two_fluid_index": two_fluid_index,
            "hartmann_number": float(divertor["divertor_state"]["hartmann_number"]),
            "temhd_cooling_gain_pct": float(divertor["cooling_gain_pct"]),
            "surface_temperature_c": float(divertor["divertor_state"]["surface_temperature_c"]),
            "surface_heat_flux_w_m2": float(divertor["divertor_state"]["surface_heat_flux_w_m2"]),
        },
        "jet_heat_flux_validation": {
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
        f"- Asymmetry index: `{q['asymmetry_index']:.4f}`",
        "",
        "## Hall-MHD + TEMHD Coupling",
        "",
        f"- Hall backend: `{d['hall_backend']}`",
        f"- Hall zonal ratio: `{d['hall_zonal_ratio']:.4f}`",
        f"- Two-fluid index: `{d['two_fluid_index']:.4f}` (threshold `>= {th['min_two_fluid_index']:.2f}`)",
        f"- TEMHD cooling gain: `{d['temhd_cooling_gain_pct']:.3f}%` (threshold `>= {th['min_temhd_cooling_gain_pct']:.1f}%`)",
        "",
        "## JET / SOLPS-ITER Proxy Heat Flux",
        "",
        f"- RMSE: `{j['rmse_pct']:.3f}%` (threshold `<= {th['max_jet_heat_flux_rmse_pct']:.1f}%`)",
        f"- Mean reference heat flux: `{j['mean_reference_w_m2']:.3e} W/m^2`",
        f"- Mean predicted heat flux: `{j['mean_predicted_w_m2']:.3e} W/m^2`",
        "",
        "## Erosion-Calibrated TBR Guard",
        "",
        f"- Raw TBR: `{t['raw_tbr']:.4f}`",
        f"- Estimated erosion: `{t['estimated_erosion_mm_year']:.4f} mm/y`",
        f"- ASDEX reference erosion: `{t['asdex_reference_erosion_mm_year']:.4f} mm/y`",
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
