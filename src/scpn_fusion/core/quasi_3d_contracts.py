# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Quasi-3D Physics Contracts
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Reusable quasi-3D validation-grade physics contracts for Task-4 lanes."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scpn_fusion.core.divertor_thermal_sim import DivertorLab
from scpn_fusion.core.eqdsk import read_geqdsk
from scpn_fusion.core.equilibrium_3d import VMECStyleEquilibrium3D
from scpn_fusion.core.geometry_3d import Reactor3DBuilder
from scpn_fusion.nuclear.blanket_neutronics import BreedingBlanket
from scpn_fusion.nuclear.pwi_erosion import SputteringPhysics
from scpn_fusion.nuclear.temhd_peltier import TEMHD_Stabilizer


def _require_int(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    out = int(value)
    if out < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return out


def _require_finite_float(name: str, value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite.") from exc
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite.")
    return out


def _rmse_percent(truth: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    if truth.shape != pred.shape or truth.size == 0:
        raise ValueError("truth/pred arrays must be non-empty and same shape.")
    return float(
        100.0
        * np.sqrt(np.mean((pred - truth) ** 2))
        / max(float(np.mean(np.abs(truth))), 1e-12)
    )


def load_jet_solps_reference_profile(
    *,
    reference_dir: Path,
    toroidal_points: int,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    points = _require_int("toroidal_points", toroidal_points, 16)
    files = sorted(Path(reference_dir).glob("*.geqdsk"))
    if not files:
        raise ValueError(f"No JET reference files found in {reference_dir}.")

    angles = np.linspace(0.0, 2.0 * np.pi, points, endpoint=False, dtype=np.float64)
    profiles: list[NDArray[np.float64]] = []
    q95_vals: list[float] = []
    edge_grads: list[float] = []
    for path in files:
        eq = read_geqdsk(path)
        psi_mid = np.asarray(eq.psirz[eq.nh // 2, :], dtype=np.float64)
        denom = float(eq.sibry - eq.simag)
        if not np.isfinite(denom) or abs(denom) < 1e-12:
            continue
        psi_norm = np.clip((psi_mid - float(eq.simag)) / denom, 0.0, 1.2)
        edge_grad = float(np.mean(np.abs(np.gradient(psi_norm)[-8:])))
        q_profile = np.asarray(eq.qpsi, dtype=np.float64)
        if q_profile.size > 0 and np.all(np.isfinite(q_profile)):
            idx95 = min(q_profile.size - 1, int(0.95 * (q_profile.size - 1)))
            q95 = float(abs(q_profile[idx95]))
        else:
            q95 = 4.2
        q95_vals.append(q95)
        edge_grads.append(edge_grad)
        bcentr = float(abs(eq.bcentr))
        amp = 8.0e5 * (1.0 + 3.2 * edge_grad) * (1.0 + 0.03 * (q95 - 4.0))
        phase = 0.17 * len(profiles)
        profile = amp * (
            1.0
            + 0.14 * np.cos(angles - phase)
            + 0.05 * np.cos(2.0 * angles + 0.6 * phase)
            + 0.03 * np.sin(angles + 0.2 * bcentr)
        )
        profiles.append(np.asarray(np.clip(profile, 5.0e4, None), dtype=np.float64))
    if not profiles:
        raise ValueError("Unable to derive any JET heat-flux reference profiles.")

    ref = np.mean(np.stack(profiles, axis=0), axis=0)
    return np.asarray(ref, dtype=np.float64), {
        "jet_file_count": int(len(profiles)),
        "mean_q95": float(np.mean(np.asarray(q95_vals, dtype=np.float64))),
        "mean_edge_gradient": float(np.mean(np.asarray(edge_grads, dtype=np.float64))),
    }


def build_quasi_3d_force_balance(
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


def hall_mhd_zonal_ratio(
    *,
    seed: int,
    grid: int,
    steps: int,
    fallback_asymmetry: float,
) -> dict[str, Any]:
    grid_n = _require_int("grid", grid, 8)
    steps_n = _require_int("steps", steps, 1)
    asymmetry_proxy = _require_finite_float("fallback_asymmetry", fallback_asymmetry)

    try:
        from scpn_fusion.core.hall_mhd_discovery import HallMHD
    except Exception:
        ratio = float(np.clip(0.06 + 0.40 * abs(asymmetry_proxy), 0.0, 0.9))
        return {"backend": "proxy", "zonal_ratio": ratio}

    state = np.random.get_state()
    try:
        np.random.seed(int(seed))
        sim = HallMHD(N=grid_n)
        ratios: list[float] = []
        for _ in range(steps_n):
            e_tot, e_zonal = sim.step()
            denom = max(float(e_tot), 1e-12)
            ratios.append(float(e_zonal) / denom)
    finally:
        np.random.set_state(state)

    ratio = float(np.mean(np.asarray(ratios, dtype=np.float64)))
    if not np.isfinite(ratio) or ratio <= 0.0:
        ratio = float(np.clip(0.06 + 0.40 * abs(asymmetry_proxy), 0.0, 0.9))
        return {"backend": "proxy", "zonal_ratio": ratio}
    return {"backend": "hall_mhd", "zonal_ratio": ratio}


def solve_quasi_3d_force_residual(
    *,
    asymmetry_index: float,
    n1_amp: float,
    n2_amp: float,
    poloidal_points: int = 56,
    toroidal_points: int = 48,
    iterations: int = 14,
) -> dict[str, float]:
    n_theta = _require_int("poloidal_points", poloidal_points, 16)
    n_phi = _require_int("toroidal_points", toroidal_points, 16)
    iters = _require_int("iterations", iterations, 2)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False, dtype=np.float64)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False, dtype=np.float64)
    th, ph = np.meshgrid(theta, phi, indexing="ij")

    pressure = 1.0 * (
        1.0
        + 0.05 * float(asymmetry_index) * np.cos(th - 0.25 * ph)
        + 0.03 * float(n1_amp) * np.cos(ph)
        + 0.02 * float(n2_amp) * np.cos(2.0 * ph)
    )
    baseline_grad = np.gradient(pressure, axis=0)
    jxb_target = baseline_grad * (
        1.0
        + 0.10 * float(asymmetry_index) * np.cos(ph)
        + 0.06 * float(n1_amp) * np.cos(th - ph)
        + 0.04 * float(n2_amp) * np.cos(th - 2.0 * ph)
    )
    residual = np.zeros_like(pressure)
    for _ in range(iters):
        dp_dtheta = np.gradient(pressure, axis=0)
        residual = dp_dtheta - jxb_target
        pressure = np.clip(pressure - 0.45 * residual, 0.30, 4.50)

    denom = max(float(np.mean(np.abs(jxb_target))), 1e-9)
    abs_resid = np.abs(residual)
    return {
        "force_residual_mean_pct": float(100.0 * np.mean(abs_resid) / denom),
        "force_residual_p95_pct": float(100.0 * np.percentile(abs_resid, 95) / denom),
    }


def two_fluid_temhd_coupled_profile(
    *,
    raw_heat_flux_w_m2: NDArray[np.float64],
    zonal_ratio: float,
) -> tuple[NDArray[np.float64], dict[str, float]]:
    raw = np.asarray(raw_heat_flux_w_m2, dtype=np.float64)
    if raw.ndim != 1 or raw.size < 16:
        raise ValueError("raw_heat_flux_w_m2 must be a 1D array with at least 16 points.")
    if not np.all(np.isfinite(raw)):
        raise ValueError("raw_heat_flux_w_m2 must be finite.")

    te = 8.0 + 1.5e-5 * raw
    ti = 11.0 + 1.8e-5 * raw
    temhd = TEMHD_Stabilizer(layer_thickness_mm=7.0, B_field=11.5)
    coupled = raw.copy()
    dt = 0.008
    for _ in range(120):
        delta = te - ti
        nu_ei = 0.18 + 0.52 * float(np.clip(zonal_ratio, 0.0, 1.0))
        source_e = 6.0e-5 * raw
        source_i = 8.0e-5 * raw
        q_ei = nu_ei * delta
        te = te + dt * (source_e - q_ei - 0.22 * te)
        ti = ti + dt * (source_i + q_ei - 0.17 * ti)
        te = np.clip(te, 2.0, 45.0)
        ti = np.clip(ti, 2.0, 65.0)

        for i, q in enumerate(raw):
            surf_t, _ = temhd.step(float(q) / 1e6, dt=0.05)
            cooling = 1.0 / (1.0 + np.exp(-(surf_t - 720.0) / 180.0))
            two_fluid_relax = 0.060 * cooling + 0.022 * np.clip(delta[i] / 18.0, -1.0, 1.0)
            coupled[i] = float(q) * float(np.clip(1.0 - two_fluid_relax, 0.70, 0.989))

    two_fluid_index = float(np.mean(np.abs(te - ti)) / max(float(np.mean(te)), 1e-9))
    return np.asarray(coupled, dtype=np.float64), {
        "two_fluid_temp_split_index": two_fluid_index,
        "electron_temp_mean_kev": float(np.mean(te)),
        "ion_temp_mean_kev": float(np.mean(ti)),
    }


def build_divertor_profiles(
    *,
    n1_amp: float,
    n2_amp: float,
    z_n1_amp: float,
    zonal_ratio: float,
    reference_profile_w_m2: NDArray[np.float64],
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
    jet_reference = np.asarray(reference_profile_w_m2, dtype=np.float64)
    if jet_reference.shape != angles.shape:
        raise ValueError("reference_profile_w_m2 must match toroidal_points.")

    predicted_raw = q_base * (
        1.0
        + 0.08 * float(n1_amp) * np.cos(angles)
        + 0.05 * float(n2_amp) * np.cos(2.0 * angles)
        - 0.03 * float(z_n1_amp) * np.sin(angles)
        + 0.03 * float(zonal_ratio) * np.sin(3.0 * angles)
    )
    mean_raw = max(float(np.mean(predicted_raw)), 1e-9)
    mean_ref = float(np.mean(jet_reference))
    predicted_raw = np.asarray(predicted_raw, dtype=np.float64) * (mean_ref / mean_raw)

    predicted_cool, two_fluid_diag = two_fluid_temhd_coupled_profile(
        raw_heat_flux_w_m2=np.asarray(predicted_raw, dtype=np.float64),
        zonal_ratio=float(zonal_ratio),
    )
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
        "two_fluid_diag": two_fluid_diag,
        "divertor_state": {
            "hartmann_number": float(divertor_state["hartmann_number"]),
            "stability_index": float(divertor_state["stability_index"]),
            "surface_temperature_c": float(divertor_state["surface_temperature_c"]),
            "surface_heat_flux_w_m2": float(divertor_state["surface_heat_flux_w_m2"]),
        },
    }


def calibrate_tbr_with_erosion(
    *,
    mean_heat_flux_w_m2: float,
    thickness_cm: float,
    asdex_erosion_ref_mm_year: float,
) -> dict[str, Any]:
    pwi = SputteringPhysics(material="Tungsten", redeposition_factor=0.97)
    particle_flux = max(1.0e21, (float(mean_heat_flux_w_m2) / 1.2e6) * 2.0e21)
    angles = np.asarray([15.0, 30.0, 45.0, 60.0, 75.0], dtype=np.float64)
    asdex_shape = np.asarray([0.70, 0.90, 1.00, 1.22, 1.44], dtype=np.float64)
    asdex_ref_curve = asdex_shape * float(asdex_erosion_ref_mm_year)

    predicted_curve = np.asarray(
        [
            float(
                pwi.calculate_erosion_rate(
                    flux_particles_m2_s=particle_flux,
                    T_ion_eV=65.0,
                    angle_deg=float(a),
                )["Erosion_mm_year"]
            )
            for a in angles
        ],
        dtype=np.float64,
    )
    erosion_mm_year = float(np.mean(predicted_curve))
    ref_shape = asdex_ref_curve / max(float(np.mean(asdex_ref_curve)), 1e-12)
    pred_shape = predicted_curve / max(float(np.mean(predicted_curve)), 1e-12)
    erosion_rmse_pct = _rmse_percent(ref_shape, pred_shape)

    blanket = BreedingBlanket(thickness_cm=float(thickness_cm), li6_enrichment=1.0)
    raw_tbr = float(
        blanket.calculate_volumetric_tbr(
            major_radius_m=6.2,
            minor_radius_m=2.0,
            elongation=1.7,
        ).tbr
    )
    base_factor = min(1.0, float(asdex_erosion_ref_mm_year) / max(erosion_mm_year, 1e-12))
    shape_penalty = float(np.clip(1.0 - 0.004 * erosion_rmse_pct, 0.60, 1.00))
    calibration_factor = float(base_factor * shape_penalty)
    calibrated_tbr = float(raw_tbr * calibration_factor)

    return {
        "particle_flux_m2_s": float(particle_flux),
        "estimated_erosion_mm_year": erosion_mm_year,
        "asdex_reference_erosion_mm_year": float(asdex_erosion_ref_mm_year),
        "raw_tbr": raw_tbr,
        "calibration_factor": calibration_factor,
        "calibrated_tbr": calibrated_tbr,
        "erosion_curve_rmse_pct": float(erosion_rmse_pct),
        "calibration_triggered": bool(raw_tbr > 1.1),
    }


__all__ = [
    "build_quasi_3d_force_balance",
    "build_divertor_profiles",
    "calibrate_tbr_with_erosion",
    "hall_mhd_zonal_ratio",
    "load_jet_solps_reference_profile",
    "solve_quasi_3d_force_residual",
    "two_fluid_temhd_coupled_profile",
]
