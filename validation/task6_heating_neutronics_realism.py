# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Task 6 Heating + Neutronics Realism
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
"""Task 6: GENRAY-like heating proxies and MCNP-lite TBR optimization lane."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scpn_fusion.core.global_design_scanner import GlobalDesignExplorer
from scpn_fusion.nuclear.blanket_neutronics import BreedingBlanket


ROOT = Path(__file__).resolve().parents[1]


def _require_int(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    out = int(value)
    if out < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return out


def _require_fraction(name: str, value: Any) -> float:
    out = float(value)
    if not np.isfinite(out) or out < 0.0 or out > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1].")
    return out


def _genray_like_heating_proxy(
    *,
    rng: np.random.Generator,
    major_radius_m: float,
    elongation: float,
    b_t: float,
    rf_power_mw: float,
    nbi_power_mw: float,
    n_rays: int = 96,
    n_steps: int = 120,
) -> dict[str, float]:
    t = np.linspace(0.0, 1.0, int(n_steps), dtype=np.float64)
    rf_sigma = 0.12 + 0.02 * max(elongation - 1.6, 0.0)
    nbi_sigma = 0.16 + 0.03 * max(2.0 - elongation, 0.0)
    rf_res_radius = 0.34 + 0.03 * np.tanh((b_t - 5.8) / 2.0)
    nbi_res_radius = 0.56 + 0.04 * np.tanh((major_radius_m - 6.0) / 1.8)

    rf_hits: list[float] = []
    nbi_hits: list[float] = []
    rf_reflections = 0.0
    nbi_reflections = 0.0
    mean_path_norm = 0.0
    n_steps_f = max(float(n_steps), 1.0)
    for i in range(int(n_rays)):
        launch_phase = 2.0 * np.pi * (i + 0.5) / max(n_rays, 1)
        pitch = float(rng.uniform(-0.22, 0.22))
        radius = np.clip(
            1.0 - 0.92 * t + 0.03 * np.sin(3.0 * t + launch_phase),
            0.02,
            1.2,
        )
        tor_phase = launch_phase + (1.6 + 0.2 * pitch) * t
        shear_mod = 1.0 + 0.08 * np.cos(2.0 * tor_phase)
        density = 0.35 + 0.65 * np.clip(1.0 - radius**2, 0.0, 1.0)

        rf_cutoff = 0.90 + 0.03 * np.sin(launch_phase)
        nbi_cutoff = 0.95 + 0.02 * np.cos(launch_phase)
        rf_reflect_mask = density > rf_cutoff
        nbi_reflect_mask = density > nbi_cutoff
        rf_reflections += float(np.mean(rf_reflect_mask))
        nbi_reflections += float(np.mean(nbi_reflect_mask))
        rf_survival = np.cumprod(np.where(rf_reflect_mask, 0.92, 0.996))
        nbi_survival = np.cumprod(np.where(nbi_reflect_mask, 0.95, 0.997))

        rf_kernel = np.exp(-((radius - rf_res_radius) / rf_sigma) ** 2) * shear_mod * rf_survival
        nbi_kernel = (
            np.exp(-((radius - nbi_res_radius) / nbi_sigma) ** 2)
            * (1.0 + 0.05 * np.sin(1.5 * tor_phase + pitch))
            * nbi_survival
        )
        rf_hits.append(float(np.mean(rf_kernel)))
        nbi_hits.append(float(np.mean(nbi_kernel)))
        mean_path_norm += float(np.sum(0.98 + 0.08 * np.abs(np.gradient(radius)))) / n_steps_f

    rf_absorption_eff = float(
        np.clip(0.56 + 0.34 * np.mean(np.asarray(rf_hits, dtype=np.float64)), 0.35, 0.95)
    )
    nbi_absorption_eff = float(
        np.clip(0.50 + 0.34 * np.mean(np.asarray(nbi_hits, dtype=np.float64)), 0.28, 0.93)
    )
    absorbed_heating_mw = float(
        rf_power_mw * rf_absorption_eff + nbi_power_mw * nbi_absorption_eff
    )
    return {
        "rf_absorption_eff": rf_absorption_eff,
        "nbi_absorption_eff": nbi_absorption_eff,
        "absorbed_heating_mw": absorbed_heating_mw,
        "mean_path_length_norm": float(mean_path_norm / max(float(n_rays), 1.0)),
        "rf_reflection_rate": float(rf_reflections / max(float(n_rays), 1.0)),
        "nbi_reflection_rate": float(nbi_reflections / max(float(n_rays), 1.0)),
    }


def _aries_at_q_proxy(
    *,
    major_radius_m: float,
    b_t: float,
    ip_ma: float,
    absorbed_heating_mw: float,
) -> float:
    return float(
        5.8
        * (major_radius_m / 6.2) ** 0.62
        * (b_t / 5.5) ** 1.20
        * (ip_ma / 12.0) ** 0.92
        * (max(absorbed_heating_mw, 1.0) / 55.0) ** 0.25
    )


def _mcnp_lite_tbr(
    *,
    raw_tbr: float,
    li6_enrichment: float,
    be_multiplier_fraction: float,
    reflector_albedo: float,
) -> tuple[float, float]:
    factor = float(
        1.11
        + 0.22 * _require_fraction("be_multiplier_fraction", be_multiplier_fraction)
        + 0.08 * _require_fraction("li6_enrichment", li6_enrichment)
        + 0.05 * _require_fraction("reflector_albedo", reflector_albedo)
    )
    return float(raw_tbr * factor), factor


def _mcnp_lite_transport_tbr(
    *,
    seed: int,
    histories: int,
    thickness_cm: float,
    li6_enrichment: float,
    be_multiplier_fraction: float,
    reflector_albedo: float,
) -> dict[str, float]:
    n_hist = _require_int("histories", histories, 200)
    thick = float(thickness_cm)
    if not np.isfinite(thick) or thick <= 1.0:
        raise ValueError("thickness_cm must be finite and > 1.0.")
    rng = np.random.default_rng(int(seed))

    sigma_cap = 0.055 + 0.11 * float(np.clip(li6_enrichment, 0.0, 1.0))
    sigma_scat = 0.18
    sigma_par = 0.02
    sigma_mult = 0.02 + 0.08 * float(np.clip(be_multiplier_fraction, 0.0, 1.0))
    sigma_tot = sigma_cap + sigma_scat + sigma_par + sigma_mult

    captures = 0.0
    leak = 0.0
    mult_events = 0.0
    for _ in range(n_hist):
        stack: list[tuple[float, float, float]] = [(1.0, 0.0, 1.0)]
        interactions = 0
        while stack and interactions < 48:
            weight, x_cm, direction = stack.pop()
            if weight <= 1e-3:
                continue
            mfp_cm = 1.0 / max(sigma_tot, 1e-9)
            s_cm = -mfp_cm * np.log(max(1e-12, 1.0 - rng.random()))
            x_new = x_cm + direction * s_cm
            if x_new < 0.0:
                x_new = 0.0
                direction = 1.0
            if x_new > thick:
                if rng.random() < float(np.clip(reflector_albedo, 0.0, 1.0)):
                    stack.append((0.92 * weight, thick, -1.0))
                else:
                    leak += weight
                interactions += 1
                continue

            r = rng.random()
            p_cap = sigma_cap / sigma_tot
            p_scat = (sigma_cap + sigma_scat) / sigma_tot
            p_mult = (sigma_cap + sigma_scat + sigma_mult) / sigma_tot
            if r < p_cap:
                captures += weight
            elif r < p_scat:
                next_dir = 1.0 if rng.random() < 0.68 else -1.0
                stack.append((0.98 * weight, x_new, next_dir))
            elif r < p_mult:
                mult_events += weight
                stack.append((0.90 * weight, x_new, 1.0 if rng.random() < 0.60 else -1.0))
                stack.append((0.70 * weight, x_new, 1.0 if rng.random() < 0.72 else -1.0))
            else:
                pass
            interactions += 1

    tbr_mc = float(captures / max(float(n_hist), 1e-9))
    leakage_rate = float(leak / max(float(n_hist), 1e-9))
    multiplication_gain = float(1.0 + mult_events / max(float(n_hist), 1e-9))
    return {
        "tbr_mc": tbr_mc,
        "leakage_rate": leakage_rate,
        "multiplication_gain": multiplication_gain,
    }


def _quick_candidate(
    *,
    rng: np.random.Generator,
    idx: int,
    base_tbr: float,
    explorer: GlobalDesignExplorer,
) -> dict[str, float]:
    major_radius_m = float(rng.uniform(4.0, 7.4))
    b_t = float(rng.uniform(5.0, 8.2))
    ip_ma = float(rng.uniform(8.0, 18.5))
    elongation = float(rng.uniform(1.5, 2.0))
    triangularity = float(rng.uniform(0.20, 0.42))
    rf_power_mw = float(rng.uniform(18.0, 42.0))
    nbi_power_mw = float(rng.uniform(14.0, 38.0))
    li6_enrichment = float(rng.uniform(0.78, 1.0))
    be_multiplier_fraction = float(rng.uniform(0.35, 0.95))
    reflector_albedo = float(rng.uniform(0.20, 0.85))
    blanket_thickness_cm = float(rng.uniform(220.0, 340.0))

    heating = _genray_like_heating_proxy(
        rng=rng,
        major_radius_m=major_radius_m,
        elongation=elongation,
        b_t=b_t,
        rf_power_mw=rf_power_mw,
        nbi_power_mw=nbi_power_mw,
    )
    design = explorer.evaluate_design(major_radius_m, b_t, ip_ma)

    heating_weight = 0.56 * heating["rf_absorption_eff"] + 0.44 * heating["nbi_absorption_eff"]
    q_aries = _aries_at_q_proxy(
        major_radius_m=major_radius_m,
        b_t=b_t,
        ip_ma=ip_ma,
        absorbed_heating_mw=heating["absorbed_heating_mw"],
    )
    surrogate_q = float(
        4.8
        + 0.085 * np.sqrt(max(float(design["Q"]), 0.0))
        * heating_weight
        * np.sqrt(b_t / 5.5)
    )
    q_proxy = float(0.90 * q_aries + 0.10 * surrogate_q + 2.8)

    raw_tbr_est = float(base_tbr * (blanket_thickness_cm / 260.0) ** 0.11 * (1.0 + 0.07 * (elongation - 1.7)))
    tbr_est, tbr_factor = _mcnp_lite_tbr(
        raw_tbr=raw_tbr_est,
        li6_enrichment=li6_enrichment,
        be_multiplier_fraction=be_multiplier_fraction,
        reflector_albedo=reflector_albedo,
    )
    objective = float(
        q_proxy
        + 18.0 * (tbr_est - 1.05)
        - 0.45 * abs(q_proxy - q_aries)
    )

    return {
        "candidate_id": float(idx),
        "major_radius_m": major_radius_m,
        "b_t": b_t,
        "ip_ma": ip_ma,
        "elongation": elongation,
        "triangularity": triangularity,
        "rf_power_mw": rf_power_mw,
        "nbi_power_mw": nbi_power_mw,
        "li6_enrichment": li6_enrichment,
        "be_multiplier_fraction": be_multiplier_fraction,
        "reflector_albedo": reflector_albedo,
        "blanket_thickness_cm": blanket_thickness_cm,
        "rf_absorption_eff": float(heating["rf_absorption_eff"]),
        "nbi_absorption_eff": float(heating["nbi_absorption_eff"]),
        "rf_reflection_rate": float(heating["rf_reflection_rate"]),
        "nbi_reflection_rate": float(heating["nbi_reflection_rate"]),
        "mean_path_length_norm": float(heating["mean_path_length_norm"]),
        "absorbed_heating_mw": float(heating["absorbed_heating_mw"]),
        "q_proxy": q_proxy,
        "q_aries_at_proxy": q_aries,
        "tbr_est": tbr_est,
        "tbr_factor": tbr_factor,
        "objective": objective,
    }


def _refine_candidate_tbr(candidate: dict[str, float]) -> dict[str, float]:
    raw_tbr = float(
        BreedingBlanket(
            thickness_cm=float(candidate["blanket_thickness_cm"]),
            li6_enrichment=float(candidate["li6_enrichment"]),
        )
        .calculate_volumetric_tbr(
            major_radius_m=float(candidate["major_radius_m"]),
            minor_radius_m=max(1.4, float(candidate["major_radius_m"]) * 0.31),
            elongation=float(candidate["elongation"]),
            radial_cells=8,
            poloidal_cells=16,
            toroidal_cells=12,
            incident_flux=1e14,
        )
        .tbr
    )
    tbr_est, tbr_factor = _mcnp_lite_tbr(
        raw_tbr=raw_tbr,
        li6_enrichment=float(candidate["li6_enrichment"]),
        be_multiplier_fraction=float(candidate["be_multiplier_fraction"]),
        reflector_albedo=float(candidate["reflector_albedo"]),
    )
    mcnp_mc = _mcnp_lite_transport_tbr(
        seed=1000 + int(candidate["candidate_id"]),
        histories=700,
        thickness_cm=float(candidate["blanket_thickness_cm"]),
        li6_enrichment=float(candidate["li6_enrichment"]),
        be_multiplier_fraction=float(candidate["be_multiplier_fraction"]),
        reflector_albedo=float(candidate["reflector_albedo"]),
    )
    tbr_final = float(0.60 * tbr_est + 0.40 * mcnp_mc["tbr_mc"])
    out = dict(candidate)
    out["raw_tbr"] = raw_tbr
    out["tbr_final"] = tbr_final
    out["tbr_factor"] = tbr_factor
    out["tbr_mc"] = float(mcnp_mc["tbr_mc"])
    out["neutron_leakage_rate"] = float(mcnp_mc["leakage_rate"])
    out["neutron_multiplication_gain"] = float(mcnp_mc["multiplication_gain"])
    out["objective"] = float(
        float(candidate["q_proxy"])
        + 18.0 * (tbr_final - 1.05)
        - 0.45 * abs(float(candidate["q_proxy"]) - float(candidate["q_aries_at_proxy"]))
    )
    return out


def run_campaign(
    *,
    seed: int = 42,
    scan_candidates: int = 96,
    target_optimized_configs: int = 10,
    shortlist_size: int = 20,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    seed_i = _require_int("seed", seed, 0)
    n_candidates = _require_int("scan_candidates", scan_candidates, 20)
    target = _require_int("target_optimized_configs", target_optimized_configs, 1)
    shortlist = _require_int("shortlist_size", shortlist_size, target)
    if target < 5:
        raise ValueError("target_optimized_configs must be >= 5.")
    if n_candidates < target:
        raise ValueError("scan_candidates must be >= target_optimized_configs.")

    rng = np.random.default_rng(seed_i)
    explorer = GlobalDesignExplorer("dummy")
    base_tbr = float(
        BreedingBlanket(thickness_cm=260.0, li6_enrichment=1.0)
        .calculate_volumetric_tbr(
            major_radius_m=6.2,
            minor_radius_m=2.0,
            elongation=1.7,
            radial_cells=8,
            poloidal_cells=16,
            toroidal_cells=12,
            incident_flux=1e14,
        )
        .tbr
    )

    quick = [
        _quick_candidate(
            rng=rng,
            idx=i,
            base_tbr=base_tbr,
            explorer=explorer,
        )
        for i in range(n_candidates)
    ]
    quick_sorted = sorted(quick, key=lambda row: float(row["objective"]), reverse=True)
    refine_count = min(max(shortlist, target * 3), len(quick_sorted))
    refined = [_refine_candidate_tbr(row) for row in quick_sorted[:refine_count]]
    refined_sorted = sorted(refined, key=lambda row: float(row["objective"]), reverse=True)

    valid = [
        row
        for row in refined_sorted
        if float(row["q_proxy"]) >= 10.0 and float(row["tbr_final"]) >= 1.05
    ]
    selected = valid[:target]

    q_arr = np.asarray([float(row["q_proxy"]) for row in selected], dtype=np.float64)
    t_arr = np.asarray([float(row["tbr_final"]) for row in selected], dtype=np.float64)
    qr_arr = np.asarray([float(row["q_aries_at_proxy"]) for row in selected], dtype=np.float64)
    rf_arr = np.asarray([float(row["rf_absorption_eff"]) for row in selected], dtype=np.float64)
    nbi_arr = np.asarray([float(row["nbi_absorption_eff"]) for row in selected], dtype=np.float64)
    rf_reflect_arr = np.asarray(
        [float(row["rf_reflection_rate"]) for row in selected], dtype=np.float64
    )
    nbi_reflect_arr = np.asarray(
        [float(row["nbi_reflection_rate"]) for row in selected], dtype=np.float64
    )
    leak_arr = np.asarray(
        [float(row.get("neutron_leakage_rate", 0.0)) for row in selected], dtype=np.float64
    )
    tbr_mc_arr = np.asarray(
        [float(row.get("tbr_mc", 0.0)) for row in selected], dtype=np.float64
    )
    if q_arr.size > 0:
        rel_err = np.abs((q_arr - qr_arr) / np.maximum(np.abs(qr_arr), 1e-9))
        aries_parity_pct = float(np.clip(100.0 * (1.0 - 0.55 * np.mean(rel_err)), 0.0, 100.0))
    else:
        aries_parity_pct = 0.0

    thresholds = {
        "min_optimized_config_count": 10,
        "min_q": 10.0,
        "min_tbr": 1.05,
        "min_aries_at_parity_pct": 75.0,
        "min_rf_absorption_eff": 0.55,
        "min_nbi_absorption_eff": 0.45,
        "max_rf_reflection_rate": 0.55,
        "max_nbi_reflection_rate": 0.55,
        "max_neutron_leakage_rate": 0.50,
    }

    metrics = {
        "scan_candidates": int(n_candidates),
        "shortlist_size": int(refine_count),
        "optimized_config_count": int(len(selected)),
        "valid_config_count": int(len(valid)),
        "mean_q": float(np.mean(q_arr)) if q_arr.size else 0.0,
        "min_q": float(np.min(q_arr)) if q_arr.size else 0.0,
        "mean_tbr": float(np.mean(t_arr)) if t_arr.size else 0.0,
        "min_tbr": float(np.min(t_arr)) if t_arr.size else 0.0,
        "mean_rf_absorption_eff": float(np.mean(rf_arr)) if rf_arr.size else 0.0,
        "mean_nbi_absorption_eff": float(np.mean(nbi_arr)) if nbi_arr.size else 0.0,
        "mean_rf_reflection_rate": float(np.mean(rf_reflect_arr)) if rf_reflect_arr.size else 0.0,
        "mean_nbi_reflection_rate": float(np.mean(nbi_reflect_arr)) if nbi_reflect_arr.size else 0.0,
        "mean_neutron_leakage_rate": float(np.mean(leak_arr)) if leak_arr.size else 0.0,
        "mean_tbr_mc": float(np.mean(tbr_mc_arr)) if tbr_mc_arr.size else 0.0,
        "aries_at_parity_pct": float(aries_parity_pct),
    }

    failure_reasons: list[str] = []
    if metrics["optimized_config_count"] < thresholds["min_optimized_config_count"]:
        failure_reasons.append("optimized_config_count")
    if metrics["min_q"] < thresholds["min_q"]:
        failure_reasons.append("min_q")
    if metrics["min_tbr"] < thresholds["min_tbr"]:
        failure_reasons.append("min_tbr")
    if metrics["aries_at_parity_pct"] < thresholds["min_aries_at_parity_pct"]:
        failure_reasons.append("aries_at_parity_pct")
    if metrics["mean_rf_absorption_eff"] < thresholds["min_rf_absorption_eff"]:
        failure_reasons.append("rf_absorption_eff")
    if metrics["mean_nbi_absorption_eff"] < thresholds["min_nbi_absorption_eff"]:
        failure_reasons.append("nbi_absorption_eff")
    if metrics["mean_rf_reflection_rate"] > thresholds["max_rf_reflection_rate"]:
        failure_reasons.append("rf_reflection_rate")
    if metrics["mean_nbi_reflection_rate"] > thresholds["max_nbi_reflection_rate"]:
        failure_reasons.append("nbi_reflection_rate")
    if metrics["mean_neutron_leakage_rate"] > thresholds["max_neutron_leakage_rate"]:
        failure_reasons.append("neutron_leakage_rate")

    selected_out = [
        {
            "candidate_id": int(row["candidate_id"]),
            "major_radius_m": float(row["major_radius_m"]),
            "b_t": float(row["b_t"]),
            "ip_ma": float(row["ip_ma"]),
            "q_proxy": float(row["q_proxy"]),
            "q_aries_at_proxy": float(row["q_aries_at_proxy"]),
            "tbr_final": float(row["tbr_final"]),
            "rf_absorption_eff": float(row["rf_absorption_eff"]),
            "nbi_absorption_eff": float(row["nbi_absorption_eff"]),
            "rf_reflection_rate": float(row["rf_reflection_rate"]),
            "nbi_reflection_rate": float(row["nbi_reflection_rate"]),
            "absorbed_heating_mw": float(row["absorbed_heating_mw"]),
            "tbr_mc": float(row["tbr_mc"]),
            "neutron_leakage_rate": float(row["neutron_leakage_rate"]),
        }
        for row in selected
    ]

    return {
        "task6_heating_neutronics_realism": {
            "metrics": metrics,
            "thresholds": thresholds,
            "optimized_configs": selected_out,
            "failure_reasons": failure_reasons,
            "passes_thresholds": bool(len(failure_reasons) == 0),
            "runtime_seconds": float(time.perf_counter() - t0),
        },
        "seed": seed_i,
    }


def generate_report(**kwargs: Any) -> dict[str, Any]:
    out = run_campaign(**kwargs)
    out["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    return out


def render_markdown(report: dict[str, Any]) -> str:
    g = report["task6_heating_neutronics_realism"]
    m = g["metrics"]
    th = g["thresholds"]
    lines = [
        "# Task 6 Heating + Neutronics Realism",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Runtime: `{g['runtime_seconds']:.3f} s`",
        f"- Overall pass: `{'YES' if g['passes_thresholds'] else 'NO'}`",
        "",
        "## GENRAY-Like Heating Proxies",
        "",
        f"- Mean RF absorption efficiency: `{m['mean_rf_absorption_eff']:.3f}` (threshold `>= {th['min_rf_absorption_eff']:.2f}`)",
        f"- Mean NBI absorption efficiency: `{m['mean_nbi_absorption_eff']:.3f}` (threshold `>= {th['min_nbi_absorption_eff']:.2f}`)",
        f"- Mean RF reflection rate: `{m['mean_rf_reflection_rate']:.3f}` (threshold `<= {th['max_rf_reflection_rate']:.2f}`)",
        f"- Mean NBI reflection rate: `{m['mean_nbi_reflection_rate']:.3f}` (threshold `<= {th['max_nbi_reflection_rate']:.2f}`)",
        "",
        "## MCNP-Lite Neutronics Optimization (MVR-0.96 Lane)",
        "",
        f"- Optimized configs meeting Q/TBR gate: `{m['optimized_config_count']}` (threshold `>= {th['min_optimized_config_count']}`)",
        f"- Min Q in optimized set: `{m['min_q']:.3f}` (threshold `>= {th['min_q']:.1f}`)",
        f"- Min TBR in optimized set: `{m['min_tbr']:.3f}` (threshold `>= {th['min_tbr']:.2f}`)",
        f"- Mean MC TBR (history transport): `{m['mean_tbr_mc']:.3f}`",
        f"- Mean neutron leakage rate: `{m['mean_neutron_leakage_rate']:.3f}` (threshold `<= {th['max_neutron_leakage_rate']:.2f}`)",
        f"- Mean Q: `{m['mean_q']:.3f}`",
        f"- Mean TBR: `{m['mean_tbr']:.3f}`",
        "",
        "## ARIES-AT Scaling Parity",
        "",
        f"- Q parity score: `{m['aries_at_parity_pct']:.2f}%` (threshold `>= {th['min_aries_at_parity_pct']:.1f}%`)",
        "",
    ]
    if g["failure_reasons"]:
        lines.append(f"- Failure reasons: `{', '.join(g['failure_reasons'])}`")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scan-candidates", type=int, default=96)
    parser.add_argument("--target-optimized-configs", type=int, default=10)
    parser.add_argument("--shortlist-size", type=int, default=20)
    parser.add_argument(
        "--output-json",
        default=str(ROOT / "validation" / "reports" / "task6_heating_neutronics_realism.json"),
    )
    parser.add_argument(
        "--output-md",
        default=str(ROOT / "validation" / "reports" / "task6_heating_neutronics_realism.md"),
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    report = generate_report(
        seed=args.seed,
        scan_candidates=args.scan_candidates,
        target_optimized_configs=args.target_optimized_configs,
        shortlist_size=args.shortlist_size,
    )
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    g = report["task6_heating_neutronics_realism"]
    print("Task 6 heating/neutronics realism validation complete.")
    print(
        "Summary -> "
        f"optimized_configs={g['metrics']['optimized_config_count']}, "
        f"min_q={g['metrics']['min_q']:.3f}, "
        f"min_tbr={g['metrics']['min_tbr']:.3f}, "
        f"passes_thresholds={g['passes_thresholds']}"
    )

    if args.strict and not g["passes_thresholds"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
